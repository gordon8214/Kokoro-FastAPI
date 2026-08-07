#!/usr/bin/env python3
"""Export the exact English frontend used by Kokoro/Misaki.

The deployed Kokoro stack uses ``en_core_web_sm`` for tokenization and Penn
Treebank POS tags.  BetterTTS consumes this export in MisakiSwift so the local
frontend is measured against the same weights instead of Apple NaturalLanguage
heuristics.

Run this script from the pinned Kokoro environment::

    python scripts/export_spacy_parity.py model /tmp/spacy-parity
    python scripts/export_spacy_parity.py trace fixtures.txt trace.json
    python scripts/export_spacy_parity.py plan fixtures.txt kokoro-plan.json
"""

from __future__ import annotations

import argparse
import asyncio
import json
import shutil
from importlib.metadata import version
from pathlib import Path
from typing import Any, Iterable

import en_core_web_sm
import numpy as np
from safetensors.numpy import save_file
from spacy.attrs import IS_SPACE, NORM, ORTH, PREFIX, SHAPE, SPACY, SUFFIX
from spacy.lang.norm_exceptions import BASE_NORMS
from spacy.symbols import IDS

MODEL_VERSION = "en_core_web_sm-3.8.0"
FEATURE_NAMES = ("norm", "prefix", "suffix", "shape", "spacy", "is_space")
FEATURE_ROWS = (5000, 1000, 2500, 2500, 50, 50)
FEATURE_SEEDS = (8, 9, 10, 11, 12, 13)


def _regex_pattern(callable_value: Any) -> str | None:
    owner = getattr(callable_value, "__self__", None)
    return getattr(owner, "pattern", None)


def _model_dict(component: Any) -> dict[str, Any]:
    return component.model.to_dict()


def _params(model: dict[str, Any], node: int) -> dict[str, np.ndarray]:
    return model["params"][node]


def _copy_param(
    output: dict[str, np.ndarray],
    prefix: str,
    values: dict[str, np.ndarray],
) -> None:
    for name, value in values.items():
        output[f"{prefix}.{name}"] = np.ascontiguousarray(value, dtype=np.float32)


def export_model(output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    nlp = en_core_web_sm.load(enable=["tok2vec", "tagger"])
    tok2vec = _model_dict(nlp.get_pipe("tok2vec"))
    tagger = _model_dict(nlp.get_pipe("tagger"))

    weights: dict[str, np.ndarray] = {}
    for feature_index, node in enumerate((28, 30, 32, 34, 36, 38)):
        _copy_param(weights, f"embed.{feature_index}", _params(tok2vec, node))
    _copy_param(weights, "embed_projection", _params(tok2vec, 39))
    _copy_param(weights, "embed_layer_norm", _params(tok2vec, 40))
    for block_index, (maxout_node, norm_node) in enumerate(
        ((57, 58), (59, 60), (61, 62), (63, 64))
    ):
        _copy_param(
            weights, f"block.{block_index}.maxout", _params(tok2vec, maxout_node)
        )
        _copy_param(
            weights, f"block.{block_index}.layer_norm", _params(tok2vec, norm_node)
        )
    _copy_param(weights, "tagger", _params(tagger, 3))
    save_file(weights, output_dir / "spacy_tagger.safetensors")

    tokenizer = nlp.tokenizer
    special_cases: dict[str, list[dict[str, str]]] = {}
    for source, pieces in tokenizer.rules.items():
        exported: list[dict[str, str]] = []
        for piece in pieces:
            orth = piece.get(ORTH)
            if orth is None:
                continue
            item = {"orth": orth}
            norm = piece.get(NORM)
            if norm is not None:
                item["norm"] = norm
            exported.append(item)
        if exported:
            special_cases[source] = exported

    tokenizer_payload = {
        "version": MODEL_VERSION,
        "prefix": _regex_pattern(tokenizer.prefix_search),
        "suffix": _regex_pattern(tokenizer.suffix_search),
        "infix": _regex_pattern(tokenizer.infix_finditer),
        "url": _regex_pattern(tokenizer.url_match),
        "special_cases": special_cases,
        # Lookup tables serialize their orthography keys as StringStore IDs;
        # the reverse source string is intentionally not guaranteed to remain
        # in the vocabulary. Keep those IDs directly instead of attempting a
        # lossy reverse lookup.
        "norm_exceptions": {
            str(key): value
            for key, value in nlp.vocab.lookups.get_table("lexeme_norm").items()
        },
        "base_norms": BASE_NORMS,
        "string_ids": IDS,
    }
    (output_dir / "spacy_tokenizer.json").write_text(
        json.dumps(tokenizer_payload, ensure_ascii=False, separators=(",", ":")),
        encoding="utf-8",
    )

    metadata = {
        "version": MODEL_VERSION,
        "spacy": nlp.meta.get("spacy_version"),
        "language": "en",
        "features": [
            {"name": name, "rows": rows, "seed": seed}
            for name, rows, seed in zip(
                FEATURE_NAMES, FEATURE_ROWS, FEATURE_SEEDS, strict=True
            )
        ],
        "width": 96,
        "window": 1,
        "depth": 4,
        "maxout_pieces": 3,
        "labels": list(nlp.get_pipe("tagger").labels),
        # StringStore reserves stable small integers for these strings instead
        # of applying MurmurHash64A. `X` matters immediately for one-letter
        # uppercase shapes, but exporting the whole table keeps arbitrary
        # runtime text exact too (for example a literal `NOUN`).
        "string_ids": IDS,
    }
    (output_dir / "spacy_tagger.json").write_text(
        json.dumps(metadata, ensure_ascii=False, separators=(",", ":")),
        encoding="utf-8",
    )

    model_root = Path(en_core_web_sm.__file__).parent / MODEL_VERSION
    for filename in ("LICENSE", "LICENSES_SOURCES"):
        shutil.copy2(model_root / filename, output_dir / f"spacy_{filename}")


def _float_rows(array: np.ndarray) -> list[list[float]]:
    return np.asarray(array, dtype=np.float32).tolist()


def trace_texts(lines: Iterable[str], output: Path) -> None:
    nlp = en_core_web_sm.load(enable=["tok2vec", "tagger"])
    tok2vec = nlp.get_pipe("tok2vec")
    records: list[dict[str, Any]] = []
    attrs = [NORM, PREFIX, SUFFIX, SHAPE, SPACY, IS_SPACE]

    for raw in lines:
        text = raw.rstrip("\n")
        if not text:
            continue
        # Run the connected pipeline, not the tagger component in isolation.
        # The tagger's Tok2VecListener reads the tensor installed by the
        # preceding component; calling ``tagger(doc)`` directly leaves that
        # listener without its upstream value and collapses the trace to the
        # bias-only class (NN).
        doc = nlp(text)
        features = doc.to_array(attrs)
        encoded = doc.tensor
        records.append(
            {
                "text": text,
                "tokens": [
                    {
                        "text": token.text,
                        "whitespace": token.whitespace_,
                        "norm": token.norm_,
                        "prefix": token.prefix_,
                        "suffix": token.suffix_,
                        "shape": token.shape_,
                        "is_space": token.is_space,
                        "tag": token.tag_,
                    }
                    for token in doc
                ],
                "features": features.tolist(),
                "tok2vec": _float_rows(encoded),
            }
        )

    output.write_text(
        json.dumps(
            {"version": MODEL_VERSION, "records": records},
            ensure_ascii=False,
            separators=(",", ":"),
        ),
        encoding="utf-8",
    )


async def trace_kokoro_plans(
    lines: Iterable[str],
    output: Path,
    language: str,
) -> None:
    """Export final Misaki phonemes, Kokoro IDs, and planner boundaries."""
    from kokoro import KPipeline

    from api.src.services.text_processing.kokoro_planner import KokoroTextPlanner
    from api.src.services.text_processing.vocabulary import get_kokoro_vocab

    planner = KokoroTextPlanner(
        pipeline=KPipeline(lang_code=language, model=False),
        vocab=get_kokoro_vocab(),
        lang_code=language,
    )
    records = []
    for raw in lines:
        text = raw.rstrip("\n")
        if not text:
            continue
        chunks = [chunk async for chunk in planner.plan(text)]
        records.append(
            {
                "text": text,
                "chunks": [
                    {
                        "text": chunk.text,
                        "phonemes": chunk.phonemes,
                        "token_ids": list(chunk.token_ids),
                        "pause_duration_s": chunk.pause_duration_s,
                    }
                    for chunk in chunks
                ],
            }
        )

    output.write_text(
        json.dumps(
            {
                "kokoro": version("kokoro"),
                "misaki": version("misaki"),
                "language": language,
                "records": records,
            },
            ensure_ascii=False,
            separators=(",", ":"),
        ),
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)

    model_parser = subparsers.add_parser("model")
    model_parser.add_argument("output", type=Path)

    trace_parser = subparsers.add_parser("trace")
    trace_parser.add_argument("input", type=Path)
    trace_parser.add_argument("output", type=Path)

    plan_parser = subparsers.add_parser("plan")
    plan_parser.add_argument("input", type=Path)
    plan_parser.add_argument("output", type=Path)
    plan_parser.add_argument("--language", choices=("a", "b"), default="a")

    args = parser.parse_args()
    if args.command == "model":
        export_model(args.output)
    elif args.command == "trace":
        trace_texts(args.input.read_text(encoding="utf-8").splitlines(), args.output)
    else:
        asyncio.run(
            trace_kokoro_plans(
                args.input.read_text(encoding="utf-8").splitlines(),
                args.output,
                args.language,
            )
        )


if __name__ == "__main__":
    main()
