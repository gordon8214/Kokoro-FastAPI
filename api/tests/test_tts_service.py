from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest
import torch

from api.src.inference.base import AudioChunk
from api.src.inference.kokoro_v1 import KokoroV1
from api.src.services.text_processing import KokoroPlannedChunk
from api.src.services.tts_service import TTSService


@pytest.fixture
def mock_managers():
    """Mock model and voice managers."""

    async def _mock_managers():
        model_manager = AsyncMock()
        model_manager.get_backend.return_value = MagicMock()

        voice_manager = AsyncMock()
        voice_manager.get_voice_path.return_value = "/path/to/voice.pt"
        voice_manager.list_voices.return_value = ["voice1", "voice2"]

        with (
            patch("api.src.services.tts_service.get_model_manager") as mock_get_model,
            patch("api.src.services.tts_service.get_voice_manager") as mock_get_voice,
        ):
            mock_get_model.return_value = model_manager
            mock_get_voice.return_value = voice_manager
            return model_manager, voice_manager

    return _mock_managers()


@pytest.fixture
def tts_service(mock_managers):
    """Create TTSService instance with mocked dependencies."""

    async def _create_service():
        return await TTSService.create("test_output")

    return _create_service()


@pytest.mark.asyncio
async def test_service_creation():
    """Test service creation and initialization."""
    model_manager = AsyncMock()
    voice_manager = AsyncMock()

    with (
        patch("api.src.services.tts_service.get_model_manager") as mock_get_model,
        patch("api.src.services.tts_service.get_voice_manager") as mock_get_voice,
    ):
        mock_get_model.return_value = model_manager
        mock_get_voice.return_value = voice_manager

        service = await TTSService.create("test_output")
        assert service.output_dir == "test_output"
        assert service.model_manager is model_manager
        assert service._voice_manager is voice_manager


@pytest.mark.asyncio
async def test_get_voice_path_single():
    """Test getting path for single voice."""
    model_manager = AsyncMock()
    voice_manager = AsyncMock()
    voice_manager.get_voice_path.return_value = "/path/to/voice1.pt"

    with (
        patch("api.src.services.tts_service.get_model_manager") as mock_get_model,
        patch("api.src.services.tts_service.get_voice_manager") as mock_get_voice,
    ):
        mock_get_model.return_value = model_manager
        mock_get_voice.return_value = voice_manager

        service = await TTSService.create("test_output")
        name, path = await service._get_voices_path("voice1")
        assert name == "voice1"
        assert path == "/path/to/voice1.pt"
        voice_manager.get_voice_path.assert_called_once_with("voice1")


@pytest.mark.asyncio
async def test_get_voice_path_combined():
    """Test getting path for combined voices."""
    model_manager = AsyncMock()
    voice_manager = AsyncMock()
    voice_manager.get_voice_path.return_value = "/path/to/voice.pt"

    with (
        patch("api.src.services.tts_service.get_model_manager") as mock_get_model,
        patch("api.src.services.tts_service.get_voice_manager") as mock_get_voice,
        patch("torch.load") as mock_load,
        patch("torch.save") as mock_save,
        patch("tempfile.gettempdir") as mock_temp,
    ):
        mock_get_model.return_value = model_manager
        mock_get_voice.return_value = voice_manager
        mock_temp.return_value = "/tmp"
        mock_load.return_value = torch.ones(10)

        service = await TTSService.create("test_output")
        name, path = await service._get_voices_path("voice1+voice2")
        assert name == "voice1+voice2"
        # Verify the path points to a temporary file with expected format
        assert path.startswith("/tmp/")
        assert "voice1+voice2" in path
        assert path.endswith(".pt")
        mock_save.assert_called_once()


@pytest.mark.asyncio
async def test_list_voices():
    """Test listing available voices."""
    model_manager = AsyncMock()
    voice_manager = AsyncMock()
    voice_manager.list_voices.return_value = ["voice1", "voice2"]

    with (
        patch("api.src.services.tts_service.get_model_manager") as mock_get_model,
        patch("api.src.services.tts_service.get_voice_manager") as mock_get_voice,
    ):
        mock_get_model.return_value = model_manager
        mock_get_voice.return_value = voice_manager

        service = await TTSService.create("test_output")
        voices = await service.list_voices()
        assert voices == ["voice1", "voice2"]
        voice_manager.list_voices.assert_called_once()


@pytest.mark.asyncio
async def test_english_kokoro_stream_uses_one_frozen_frontend_plan():
    """The stream synthesizes the exact units emitted by Kokoro's planner."""
    planned = KokoroPlannedChunk(
        text="They are content.",
        phonemes="ðˌA ɑɹ kəntˈɛnt.",
        token_ids=(81, 157, 24),
        misaki_tokens=(MagicMock(),),
    )

    async def planned_chunks():
        yield planned

    backend = KokoroV1()
    backend.plan_text = MagicMock(return_value=planned_chunks())
    model_manager = MagicMock()
    model_manager.ensure_loaded = AsyncMock()
    model_manager.get_backend.return_value = backend

    service = TTSService()
    service.model_manager = model_manager
    service._get_voices_path = AsyncMock(return_value=("af_heart", "/voice.pt"))
    processed = []

    async def process_chunk(*args, **kwargs):
        processed.append((args, kwargs))
        yield AudioChunk(np.ones(24, dtype=np.float32), output=None)

    service._process_chunk = process_chunk

    with patch("api.src.services.tts_service.smart_split") as legacy_splitter:
        chunks = [
            chunk
            async for chunk in service.generate_audio_stream(
                "They are content.",
                "af_heart",
                MagicMock(),
                output_format=None,
                lang_code="a",
            )
        ]

    assert chunks
    legacy_splitter.assert_not_called()
    backend.plan_text.assert_called_once()
    assert processed[0][0][0] == planned.text
    assert processed[0][0][1] == list(planned.token_ids)
    assert processed[0][1]["prepared_chunk"] is planned
    assert processed[0][1]["is_last"] is False
    assert processed[1][1]["is_last"] is True


@pytest.mark.asyncio
async def test_prepared_chunk_cannot_succeed_without_audio():
    """A frozen plan mismatch or empty model result must fail the request."""
    planned = KokoroPlannedChunk(
        text="They are content.",
        phonemes="ðˌA ɑɹ kəntˈɛnt.",
        token_ids=(81, 157, 24),
        misaki_tokens=(MagicMock(),),
    )
    backend = KokoroV1()
    model_manager = MagicMock()
    model_manager.ensure_loaded = AsyncMock()
    model_manager.get_backend.return_value = backend

    async def no_audio(*args, **kwargs):
        if False:
            yield None

    model_manager.generate = no_audio
    service = TTSService()
    service.model_manager = model_manager

    with pytest.raises(
        RuntimeError,
        match="Kokoro produced no audio for the prepared chunk",
    ):
        async for _ in service._process_chunk(
            planned.text,
            list(planned.token_ids),
            "af_heart",
            "/voice.pt",
            1.0,
            MagicMock(),
            output_format=None,
            prepared_chunk=planned,
        ):
            pass
