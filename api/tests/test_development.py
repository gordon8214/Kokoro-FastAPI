import base64
import json
from unittest.mock import MagicMock, patch

import pytest
import requests

from api.src.routers.development import phonemize_text, plan_kokoro_text
from api.src.structures.text_schemas import PhonemeRequest


def test_generate_captioned_speech():
    """Test the generate_captioned_speech function with mocked responses"""
    # Mock the API responses
    mock_audio_response = MagicMock()
    mock_audio_response.status_code = 200

    mock_timestamps_response = MagicMock()
    mock_timestamps_response.status_code = 200
    mock_timestamps_response.content = json.dumps(
        {
            "audio": base64.b64encode(b"mock audio data").decode("utf-8"),
            "timestamps": [{"word": "test", "start_time": 0.0, "end_time": 1.0}],
        }
    )

    # Patch the HTTP requests
    with patch("requests.post", return_value=mock_timestamps_response):
        # Import here to avoid module-level import issues
        from examples.captioned_speech_example import generate_captioned_speech

        # Test the function
        audio, timestamps = generate_captioned_speech("test text")

        # Verify we got both audio and timestamps
        assert audio == b"mock audio data"
        assert timestamps == [{"word": "test", "start_time": 0.0, "end_time": 1.0}]


@pytest.mark.asyncio
async def test_phonemize_returns_exact_kokoro_model_ids():
    response = await phonemize_text(
        PhonemeRequest(text="The website content is useful.", language="a")
    )

    assert response.phonemes == "ðə wˈɛbsˌIt kˈɑntɛnt ɪz jˈusfᵊl."
    assert len(response.tokens) == len(response.phonemes)
    assert response.tokens[response.phonemes.index("ᵊ")] == 42


@pytest.mark.asyncio
async def test_kokoro_plan_exposes_frozen_chunk_boundaries_and_ids():
    response = await plan_kokoro_text(
        PhonemeRequest(
            text="They are content. [pause:1.25s] Please continue.",
            language="a",
        )
    )

    assert [chunk.text for chunk in response.chunks] == [
        "They are content.",
        "",
        "Please continue.",
    ]
    assert response.chunks[0].phonemes == "ðˌA ɑɹ kəntˈɛnt."
    assert len(response.chunks[0].tokens) == len(response.chunks[0].phonemes)
    assert response.chunks[1].pause_duration_s == 1.25
