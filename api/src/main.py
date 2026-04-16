"""
FastAPI OpenAI Compatible API
"""

import os
import sys
from contextlib import asynccontextmanager
from pathlib import Path

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from loguru import logger

from .core.config import settings
from .routers.debug import router as debug_router
from .routers.development import router as dev_router
from .routers.openai_compatible import router as openai_router
from .routers.web_player import router as web_router


def setup_logger():
    """Configure loguru logger with file rotation"""
    valid_levels = ["TRACE", "DEBUG", "INFO", "SUCCESS", "WARNING", "ERROR", "CRITICAL"]
    level = os.getenv("API_LOG_LEVEL", "DEBUG").upper()
    if level not in valid_levels:
        level = "DEBUG"

    # Determine log file path
    project_root = os.getenv("PROJECT_ROOT", str(Path(__file__).parent.parent.parent))
    log_file = Path(project_root) / "logs" / "kokoro.log"
    log_file.parent.mkdir(exist_ok=True)

    logger.remove()

    # File handler with rotation (single file, truncates at 10MB)
    logger.add(
        log_file,
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {module}:{line} | {message}",
        level=level,
        rotation="10 MB",
        retention=0,  # Delete old rotated files immediately (keeps only current)
    )

    # Also log to stdout for console/debug visibility
    logger.add(
        sys.stdout,
        format="<fg #2E8B57>{time:hh:mm:ss A}</fg #2E8B57> | "
        "{level: <8} | "
        "<fg #4169E1>{module}:{line}</fg #4169E1> | "
        "{message}",
        colorize=True,
        level=level,
    )

    print(f"Logging to {log_file} (level: {level})")


# Configure logger
setup_logger()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager — model loads lazily on first request."""
    from .services.temp_manager import cleanup_temp_files

    # Clean old temp files on startup
    await cleanup_temp_files()

    logger.info("Kokoro FastAPI started (model will load on first request)")

    yield


# Initialize FastAPI app
app = FastAPI(
    title=settings.api_title,
    description=settings.api_description,
    version=settings.api_version,
    lifespan=lifespan,
    openapi_url="/openapi.json",  # Explicitly enable OpenAPI schema
)

# Add CORS middleware if enabled
if settings.cors_enabled:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

# Include routers
app.include_router(openai_router, prefix="/v1")
app.include_router(dev_router)  # Development endpoints
app.include_router(debug_router)  # Debug endpoints
if settings.enable_web_player:
    app.include_router(web_router, prefix="/web")  # Web player static files


# Health check endpoint
@app.get("/health")
async def health_check():
    """Health check endpoint"""
    from .inference.model_manager import get_manager

    model_manager = await get_manager()
    return {"status": "healthy", "model_loaded": model_manager.is_loaded}


@app.post("/unload")
async def unload_model():
    """Unload model from VRAM to free GPU memory."""
    from .inference.model_manager import get_manager

    model_manager = await get_manager()
    if not model_manager.is_loaded:
        return {"status": "ok", "message": "No model loaded"}
    model_manager.unload_all()
    return {"status": "ok", "message": "Model unloaded"}


@app.get("/v1/test")
async def test_endpoint():
    """Test endpoint to verify routing"""
    return {"status": "ok"}


if __name__ == "__main__":
    uvicorn.run("api.src.main:app", host=settings.host, port=settings.port, reload=True)
