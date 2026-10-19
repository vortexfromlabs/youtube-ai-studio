"""
Agent 5 — Audio Director
========================
Generates dialogue audio.
Primary: ElevenLabs.
Fallback 1: edge-tts (Microsoft).
Fallback 2: gTTS (Google Free TTS - Indestructible).
"""

import asyncio
import logging
import subprocess
from pathlib import Path
from typing import Optional

from core.config import settings, get_workspace_path
from core.database import managed_session, Shot, Character, STATUS_COMPLETED, STATUS_FAILED

logger = logging.getLogger("vortex.agent5_audio")

def _get_audio_duration(path: Path) -> float:
    try:
        ffmpeg_cmd = settings.FFMPEG_PATH.replace("ffmpeg", "ffprobe") if settings.FFMPEG_PATH != "ffmpeg" else "ffprobe"
        result = subprocess.run(
            [ffmpeg_cmd, "-v", "error", "-show_entries", "format=duration", "-of", "default=noprint_wrappers=1:nokey=1", str(path)],
            capture_output=True, text=True, timeout=30,
        )
        return float(result.stdout.strip())
    except Exception:
        return max(1.0, path.stat().st_size / 16000.0)

def _generate_elevenlabs(text: str, voice_id: str, output_path: Path) -> bool:
    if not settings.ELEVENLABS_API_KEY: return False
    try:
        from elevenlabs.client import ElevenLabs
        client = ElevenLabs(api_key=settings.ELEVENLABS_API_KEY)
        audio_stream = client.text_to_speech.convert(
            voice_id=voice_id or settings.ELEVENLABS_VOICE_ID,
            model_id=settings.ELEVENLABS_MODEL_ID,
            text=text,
        )
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "wb") as f:
            for chunk in audio_stream:
                if chunk: f.write(chunk)
        return output_path.stat().st_size > 0
    except Exception as exc:
        logger.warning(f"ElevenLabs TTS failed: {exc}")
        return False

def _generate_edge_tts(text: str, output_path: Path) -> bool:
    try:
        import edge_tts
        async def _do_tts():
            await edge_tts.Communicate(text, "hi-IN-SwaraNeural").save(str(output_path))
        asyncio.run(_do_tts())
        return output_path.exists() and output_path.stat().st_size > 0
    except Exception as exc:
        logger.warning(f"edge-tts fallback failed: {exc}")
        return False

def _generate_gtts(text: str, output_path: Path) -> bool:
    """Bulletproof Google TTS fallback."""
    try:
        from gtts import gTTS
        # Autodetect language based on text (defaults to Hindi if it looks like Kahaniya)
        lang = "hi" if any("\u0900" <= c <= "\u097F" for c in text) else "en"
        tts = gTTS(text=text, lang=lang)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        tts.save(str(output_path))
        logger.info(f"gTTS successfully generated audio: {output_path.name}")
        return output_path.exists() and output_path.stat().st_size > 0
    except Exception as exc:
        logger.error(f"gTTS fallback failed: {exc}")
        return False

_ELEVENLABS_VOICE_POOL = ["21m00Tcm4TlvDq8ikWAM", "AZnzlk1XvdvUeBnXmlld", "EXAVITQu4vr4xnSDxMaL"]

def run_audio_director(project_id: int) -> dict:
    logger.info("Agent 5 (Audio Director) starting for project %d", project_id)

    voice_map = {}
    with managed_session() as db:
        characters = db.query(Character).filter(Character.project_id == project_id).all()
        for idx, char in enumerate(characters):
            vid = _ELEVENLABS_VOICE_POOL[idx % len(_ELEVENLABS_VOICE_POOL)]
            voice_map[char.name] = vid

    with managed_session() as db:
        shots = db.query(Shot).filter(Shot.project_id == project_id).order_by(Shot.shot_number).all()
        shot_data = [{"id": s.id, "shot_number": s.shot_number, "dialogue": s.dialogue, "char": s.speaking_character_name} for s in shots]

    results = {"generated": 0, "skipped": 0, "failed": 0}

    for sdata in shot_data:
        dialogue = sdata["dialogue"]
        if not dialogue or not dialogue.strip():
            with managed_session() as db:
                shot = db.query(Shot).filter(Shot.id == sdata["id"]).first()
                if shot:
                    shot.duration_seconds = 5.0
                    shot.status = STATUS_COMPLETED
            results["skipped"] += 1
            continue

        output_path = get_workspace_path(project_id, settings.DIR_AUDIO, f"shot_{sdata['shot_number']:03d}.mp3")
        voice_id = voice_map.get(sdata["char"] or "", settings.ELEVENLABS_VOICE_ID)

        # Try API Waterfall: ElevenLabs -> EdgeTTS -> gTTS
        success = _generate_elevenlabs(dialogue, voice_id, output_path)
        if not success:
            success = _generate_edge_tts(dialogue, output_path)
        if not success:
            success = _generate_gtts(dialogue, output_path)

        if success:
            duration = _get_audio_duration(output_path)
            with managed_session() as db:
                shot = db.query(Shot).filter(Shot.id == sdata["id"]).first()
                if shot:
                    shot.audio_path = str(output_path)
                    shot.duration_seconds = duration
                    shot.status = STATUS_COMPLETED
            results["generated"] += 1
        else:
            with managed_session() as db:
                shot = db.query(Shot).filter(Shot.id == sdata["id"]).first()
                if shot: shot.status = STATUS_FAILED
            results["failed"] += 1

    logger.info("Agent 5 complete. Generated=%d, Skipped=%d, Failed=%d", results["generated"], results["skipped"], results["failed"])
    return results