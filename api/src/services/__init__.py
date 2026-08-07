"""Service package exports without eager inference/service import cycles."""

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .tts_service import TTSService


def __getattr__(name: str) -> Any:
    if name == "TTSService":
        from .tts_service import TTSService

        return TTSService
    raise AttributeError(name)


__all__ = ["TTSService"]
