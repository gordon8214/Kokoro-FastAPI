"""Tests for final-Misaki Kokoro chunk planning."""

from dataclasses import dataclass

import pytest

from api.src.services.text_processing.kokoro_planner import KokoroTextPlanner
from api.src.services.text_processing.vocabulary import get_kokoro_vocab
from api.src.structures.schemas import NormalizationOptions


@dataclass
class FakeToken:
    text: str


class FakePipeline:
    def __init__(self, count):
        self._count = count

    def g2p(self, text):
        phonemes = "x" * self._count(text)
        return phonemes, [FakeToken(text)]


async def collect(planner, text):
    return [
        chunk
        async for chunk in planner.plan(
            text,
            NormalizationOptions(normalize=False),
        )
    ]


@pytest.mark.asyncio
async def test_remeasures_combined_sentence_frontend():
    first = "First sentence."
    second = "Second sentence."

    def count(candidate):
        return 451 if first in candidate and second in candidate else 100

    planner = KokoroTextPlanner(FakePipeline(count), {"x": 1}, "a")
    chunks = await collect(planner, f"{first} {second}")

    assert [chunk.text for chunk in chunks] == [first, second]
    assert [chunk.token_count for chunk in chunks] == [100, 100]


@pytest.mark.asyncio
async def test_runt_may_cross_target_max_but_not_absolute_max():
    first = "A short first sentence."
    second = "A longer second sentence."

    def count(candidate):
        if first in candidate and second in candidate:
            return 300
        return 100 if first in candidate else 200

    planner = KokoroTextPlanner(FakePipeline(count), {"x": 1}, "a")
    chunks = await collect(planner, f"{first} {second}")

    assert len(chunks) == 1
    assert chunks[0].token_count == 300


@pytest.mark.asyncio
async def test_full_chunk_stops_before_crossing_target_max():
    first = "A complete first sentence."
    second = "A second sentence."

    def count(candidate):
        if first in candidate and second in candidate:
            return 300
        return 180 if first in candidate else 120

    planner = KokoroTextPlanner(FakePipeline(count), {"x": 1}, "a")
    chunks = await collect(planner, f"{first} {second}")

    assert [chunk.text for chunk in chunks] == [first, second]


@pytest.mark.asyncio
async def test_oversized_sentence_recursively_splits_at_words():
    text = " ".join(["lorem"] * 80) + "."
    planner = KokoroTextPlanner(
        FakePipeline(len),
        {"x": 1},
        "a",
        target_min=30,
        target_max=40,
        absolute_max=50,
    )
    chunks = await collect(planner, text)

    assert len(chunks) > 1
    assert all(chunk.token_count <= 50 for chunk in chunks)
    assert " ".join(chunk.text for chunk in chunks).split() == text.split()


@pytest.mark.asyncio
async def test_pause_tags_survive_as_separate_plan_units():
    planner = KokoroTextPlanner(FakePipeline(len), {"x": 1}, "a")
    chunks = await collect(planner, "Hello. [pause:1.25s] Goodbye.")

    assert [chunk.text for chunk in chunks] == ["Hello.", "", "Goodbye."]
    assert chunks[1].pause_duration_s == 1.25
    assert chunks[1].token_ids == ()


@pytest.mark.asyncio
async def test_unknown_frontend_symbols_fail_instead_of_changing_the_plan():
    planner = KokoroTextPlanner(FakePipeline(lambda _: 1), {}, "a")

    with pytest.raises(
        ValueError,
        match="Kokoro vocabulary does not contain symbols",
    ):
        await collect(planner, "Hello.")


@pytest.mark.asyncio
async def test_real_misaki_frontend_and_model_vocab_are_materialized_exactly():
    from kokoro import KPipeline

    planner = KokoroTextPlanner(
        KPipeline(lang_code="a", model=False),
        get_kokoro_vocab(),
        "a",
    )
    chunks = await collect(
        planner,
        "They are content. The website content is useful.",
    )

    assert len(chunks) == 1
    assert chunks[0].phonemes == ("ðˌA ɑɹ kəntˈɛnt. ðə wˈɛbsˌIt kˈɑntɛnt ɪz jˈusfᵊl.")
    assert chunks[0].token_count == len(chunks[0].phonemes)
    assert chunks[0].token_ids[:8] == (81, 157, 24, 16, 69, 123, 16, 53)


@pytest.mark.asyncio
async def test_acronym_plural_and_possessive_reach_misaki_with_a_lowercase_s():
    from kokoro import KPipeline

    planner = KokoroTextPlanner(
        KPipeline(lang_code="a", model=False),
        get_kokoro_vocab(),
        "a",
    )

    for text in ["LEDs shine.", "LED's shine.", "LED’s shine."]:
        chunks = [chunk async for chunk in planner.plan(text, NormalizationOptions())]

        assert len(chunks) == 1
        assert chunks[0].phonemes == "ˌɛlˌidˈiz ʃˈIn."
