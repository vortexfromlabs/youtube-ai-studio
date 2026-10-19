"""
VortexLabs Pro Studio — Configuration Module
Loads environment variables from .env, defines typed settings,
and provides workspace path helpers with automatic directory creation.
"""

import os
import logging
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv

# ---------------------------------------------------------------------------
# Load .env from project root
# ---------------------------------------------------------------------------
_project_root = Path(__file__).resolve().parent.parent
_env_path = _project_root / ".env"
if _env_path.exists():
    load_dotenv(dotenv_path=_env_path)
else:
    load_dotenv()  # fall back to environment / shell exports


class _Settings:
    """
    Centralised, typed configuration object.
    Every attribute is resolved once at import-time so the rest of the
    codebase never touches os.environ directly.
    """

    # --- API Keys -----------------------------------------------------------
    GEMINI_API_KEY: str = os.getenv("GEMINI_API_KEY", "")
    ELEVENLABS_API_KEY: str = os.getenv("ELEVENLABS_API_KEY", "")
    HF_API_TOKEN: str = os.getenv("HF_API_TOKEN", "")
    YOUTUBE_CLIENT_SECRETS_FILE: str = os.getenv(
        "YOUTUBE_CLIENT_SECRETS_FILE", "client_secrets.json"
    )

    # --- Paths --------------------------------------------------------------
    WORKSPACE_ROOT: Path = Path(
        os.getenv("WORKSPACE_ROOT", str(_project_root / "workspace"))
    ).resolve()
    DATABASE_URL: str = os.getenv(
        "DATABASE_URL",
        f"sqlite:///{_project_root / 'vortex_pro.db'}",
    )
    FFMPEG_PATH: str = os.getenv("FFMPEG_PATH", "ffmpeg")

    # --- Logging ------------------------------------------------------------
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO").upper()

    # --- Gemini model names --------------------------------------------------
    # FIXED: Use universally valid free-tier model. gemini-2.5-flash is available
    # on v1beta and free tier, unlike gemini-1.5-pro-latest which 404s.
    GEMINI_MODEL: str = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")

    # --- HuggingFace models --------------------------------------------------
    HF_IMAGE_MODEL: str = os.getenv(
        "HF_IMAGE_MODEL", "black-forest-labs/FLUX.2-klein-4B"
    )
    HF_VIDEO_MODEL: str = os.getenv(
        "HF_VIDEO_MODEL", ""
    )

    # --- ElevenLabs defaults -------------------------------------------------
    ELEVENLABS_VOICE_ID: str = os.getenv("ELEVENLABS_VOICE_ID", "21m00Tcm4TlvDq8ikWAM")
    ELEVENLABS_MODEL_ID: str = os.getenv("ELEVENLABS_MODEL_ID", "eleven_multilingual_v2")

    # --- Chunk sizes for large-file I/O (bytes) ------------------------------
    FILE_CHUNK_SIZE: int = int(os.getenv("FILE_CHUNK_SIZE", str(8 * 1024 * 1024)))  # 8 MB

    # --- Subdirectory names inside a project workspace -----------------------
    DIR_SCRIPTS: str = "scripts"
    DIR_AUDIO: str = "audio"
    DIR_IMAGES: str = "images"
    DIR_IMAGES_RAW: str = "images/raw"
    DIR_IMAGES_TRANSPARENT: str = "images/transparent"
    DIR_BACKGROUNDS: str = "backgrounds"
    DIR_BACKGROUNDS_RAW: str = "backgrounds/raw"
    DIR_BACKGROUNDS_ANIMATED: str = "backgrounds/animated"
    DIR_LIPSYNC: str = "lipsync"
    DIR_COMPOSITED: str = "composited"
    DIR_FINAL: str = "final"
    DIR_QA: str = "qa"
    DIR_METADATA: str = "metadata"

    def validate(self) -> list[str]:
        """Return a list of human-readable warnings for missing keys."""
        warnings: list[str] = []
        if not self.GEMINI_API_KEY:
            warnings.append("GEMINI_API_KEY is not set — Agents 1-4 will use procedural fallback.")
        if not self.ELEVENLABS_API_KEY:
            warnings.append(
                "ELEVENLABS_API_KEY is not set — Agent 5 will use edge-tts fallback."
            )
        if not self.HF_API_TOKEN:
            warnings.append(
                "HF_API_TOKEN is not set — Agent 6/8 image/video generation will fail."
            )
        return warnings


settings = _Settings()

# ---------------------------------------------------------------------------
# Logging bootstrap
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=getattr(logging, settings.LOG_LEVEL, logging.INFO),
    format="%(asctime)s | %(name)-28s | %(levelname)-7s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("vortex.config")

_warnings = settings.validate()
for w in _warnings:
    logger.warning(w)


# ---------------------------------------------------------------------------
# Workspace helpers
# ---------------------------------------------------------------------------

def get_workspace_path(project_id: int, *sub: str) -> Path:
    """
    Build an absolute path inside a project's workspace and ensure every
    intermediate directory exists.

    Example
    -------
    >>> p = get_workspace_path(1, settings.DIR_AUDIO, "shot_03.mp3")
    >>> # → <WORKSPACE_ROOT>/project_1/audio/shot_03.mp3
    """
    parts = [settings.WORKSPACE_ROOT, f"project_{project_id}"] + list(sub)
    full = Path(*parts)
    # If the final component looks like a filename, ensure its parent exists;
    # otherwise ensure the directory itself exists.
    if full.suffix:
        full.parent.mkdir(parents=True, exist_ok=True)
    else:
        full.mkdir(parents=True, exist_ok=True)
    return full


def chunked_file_copy(src: Path, dst: Path, chunk_size: Optional[int] = None) -> Path:
    """
    Copy *src* → *dst* in fixed-size chunks so we never load a 20 GB video
    into memory at once.  Returns *dst*.
    """
    chunk_size = chunk_size or settings.FILE_CHUNK_SIZE
    dst.parent.mkdir(parents=True, exist_ok=True)
    with open(src, "rb") as fin, open(dst, "wb") as fout:
        while True:
            chunk = fin.read(chunk_size)
            if not chunk:
                break
            fout.write(chunk)
    return dst


def chunked_file_read(path: Path, chunk_size: Optional[int] = None):
    """
    Generator that yields bytes chunks from *path*.
    Useful for streaming large files to upload endpoints.
    """
    chunk_size = chunk_size or settings.FILE_CHUNK_SIZE
    with open(path, "rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            yield chunk
