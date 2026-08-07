"""Misaki-native chunk planning for Kokoro 1.x English synthesis."""

import re
from dataclasses import dataclass
from typing import AsyncGenerator, Dict, List, Mapping, Optional, Sequence, Tuple

from kokoro import KPipeline

from ...core.config import settings
from ...structures.schemas import NormalizationOptions
from .normalizer import normalize_text
from .text_processor import (
    CUSTOM_PHONEMES,
    PAUSE_TAG_PATTERN,
    split_sentences,
)


@dataclass(frozen=True)
class KokoroPlannedChunk:
    """One frozen text or pause unit emitted by the Kokoro frontend."""

    text: str
    phonemes: str
    token_ids: Tuple[int, ...]
    misaki_tokens: Tuple[object, ...]
    pause_duration_s: Optional[float] = None

    @classmethod
    def pause(cls, duration_s: float) -> "KokoroPlannedChunk":
        """Create a pause-only plan unit."""
        return cls("", "", (), (), pause_duration_s=duration_s)

    @property
    def token_count(self) -> int:
        """Return the number of vocabulary IDs the model will consume."""
        return len(self.token_ids)


class _FrontendMaterializer:
    """Materialize and cache exact Misaki output within one planning pass."""

    def __init__(self, pipeline: KPipeline, vocab: Mapping[str, int]):
        self._pipeline = pipeline
        self._vocab = vocab
        self._cache: Dict[str, KokoroPlannedChunk] = {}

    def materialize(self, text: str) -> KokoroPlannedChunk:
        """Run the final frontend once for an exact candidate string."""
        cached = self._cache.get(text)
        if cached is not None:
            return cached

        phonemes, tokens = self._pipeline.g2p(text)
        phonemes = phonemes.strip()
        unknown = sorted(
            {character for character in phonemes if character not in self._vocab}
        )
        if unknown:
            raise ValueError(f"Kokoro vocabulary does not contain symbols: {unknown}")
        token_ids = tuple(self._vocab[character] for character in phonemes)
        planned = KokoroPlannedChunk(
            text=text,
            phonemes=phonemes,
            token_ids=token_ids,
            misaki_tokens=tuple(tokens),
        )
        self._cache[text] = planned
        return planned


class KokoroTextPlanner:
    """Pack normalized English prose using final Misaki/Kokoro token counts."""

    def __init__(
        self,
        pipeline: KPipeline,
        vocab: Mapping[str, int],
        lang_code: str,
        target_min: int = settings.target_min_tokens,
        target_max: int = settings.target_max_tokens,
        absolute_max: int = settings.absolute_max_tokens,
    ):
        if not 0 < target_min <= target_max <= absolute_max:
            raise ValueError("Invalid Kokoro token budget")
        self._pipeline = pipeline
        self._vocab = vocab
        self._lang_code = lang_code
        self._target_min = target_min
        self._target_max = target_max
        self._absolute_max = absolute_max

    async def plan(
        self,
        text: str,
        normalization_options: Optional[NormalizationOptions] = None,
    ) -> AsyncGenerator[KokoroPlannedChunk, None]:
        """Yield deterministic text and pause chunks for one request."""
        options = normalization_options or NormalizationOptions()
        materializer = _FrontendMaterializer(self._pipeline, self._vocab)

        for text_part, pause_duration in self._parts(text, options):
            if pause_duration is not None:
                yield KokoroPlannedChunk.pause(pause_duration)
                continue

            sentences = split_sentences(text_part, self._lang_code)
            current: Optional[KokoroPlannedChunk] = None
            for sentence in sentences:
                pieces = self._split_to_fit(sentence, materializer)
                if len(pieces) > 1 and current is not None:
                    yield current
                    current = None

                for piece in pieces:
                    if current is None:
                        current = piece
                        continue

                    combined = materializer.materialize(f"{current.text} {piece.text}")
                    if (
                        current.token_count >= self._target_min
                        and combined.token_count > self._target_max
                    ):
                        yield current
                        current = piece
                    elif combined.token_count <= self._target_max:
                        current = combined
                    elif (
                        combined.token_count <= self._absolute_max
                        and current.token_count < self._target_min
                    ):
                        current = combined
                    else:
                        yield current
                        current = piece

                if len(pieces) > 1 and current is not None:
                    yield current
                    current = None

            if current is not None:
                yield current

    def _parts(
        self,
        text: str,
        normalization_options: NormalizationOptions,
    ) -> Sequence[Tuple[str, Optional[float]]]:
        raw_parts = PAUSE_TAG_PATTERN.split(text)
        result: List[Tuple[str, Optional[float]]] = []
        index = 0
        while index < len(raw_parts):
            raw_text = raw_parts[index].strip()
            index += 1
            if raw_text:
                result.append((self._normalize(raw_text, normalization_options), None))

            if index < len(raw_parts) and re.fullmatch(
                r"\d+(?:\.\d+)?", raw_parts[index]
            ):
                duration = float(raw_parts[index])
                index += 1
                if duration > 0:
                    result.append(("", duration))
        return result

    def _normalize(self, text: str, options: NormalizationOptions) -> str:
        if not settings.advanced_text_normalization or not options.normalize:
            return text
        protected_parts = CUSTOM_PHONEMES.split(text)
        for index in range(0, len(protected_parts), 2):
            protected_parts[index] = normalize_text(protected_parts[index], options)
        return "".join(protected_parts).strip()

    def _split_to_fit(
        self,
        text: str,
        materializer: _FrontendMaterializer,
    ) -> List[KokoroPlannedChunk]:
        planned = materializer.materialize(text)
        if planned.token_count <= self._absolute_max:
            return [planned]

        clauses = self._comma_split(text)
        if len(clauses) > 1:
            return [
                piece
                for clause in clauses
                for piece in self._split_to_fit(clause, materializer)
            ]

        left, right = self._halve_at_whitespace(text)
        if not left or not right:
            raise ValueError("Unable to split oversized Kokoro token sequence")
        return self._split_to_fit(left, materializer) + self._split_to_fit(
            right, materializer
        )

    @staticmethod
    def _comma_split(text: str) -> List[str]:
        pieces: List[str] = []
        current = ""
        for character in text:
            current += character
            if character == ",":
                if current.strip():
                    pieces.append(current.strip())
                current = ""
        if current.strip():
            pieces.append(current.strip())
        return pieces

    @staticmethod
    def _halve_at_whitespace(text: str) -> Tuple[str, str]:
        if len(text) <= 1:
            return "", ""
        midpoint = len(text) // 2
        split_index = text.rfind(" ", 0, midpoint)
        if split_index < 0:
            split_index = text.find(" ", midpoint)
        if split_index < 0:
            split_index = midpoint
        return text[:split_index].strip(), text[split_index:].strip()
