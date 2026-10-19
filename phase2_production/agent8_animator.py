"""
Agent 8 — Background Animator
==============================
Animates static background images using:
  Primary: HuggingFace free video generation models (if available).
  Fallback: Procedural ffmpeg Ken Burns effect (pan/zoom) — robust local method.
"""

import logging
import subprocess
import time
from pathlib import Path
from typing import Optional

import requests

from core.config import settings, get_workspace_path
from core.database import (
    managed_session,
    Location,
    Shot,
    STATUS_COMPLETED,
    STATUS_FAILED,
)

logger = logging.getLogger("vortex.agent8_animator")

# ---------------------------------------------------------------------------
# FFmpeg Ken Burns effect (pan + zoom)
# ---------------------------------------------------------------------------

def _ken_burns_animate(
    image_path: Path,
    output_path: Path,
    duration: float = 8.0,
    fps: int = 24,
    width: int = 1920,
    height: int = 1080,
) -> bool:
    """
    Create a cinematic Ken Burns (slow zoom) animation from a static image.
    Uses simplified math to ensure Windows compatibility.
    """
    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)

        total_frames = int(duration * fps)
        
        # Windows-safe zoompan filter (straight zoom to center, no complex sin/cos math)
        zoompan_filter = (
            f"zoompan=z='min(zoom+0.001,1.3)'"
            f":x='iw/2-(iw/zoom/2)'"
            f":y='ih/2-(ih/zoom/2)'"
            f":d={total_frames}"
            f":s={width}x{height}"
            f":fps={fps}"
        )

        cmd = [
            settings.FFMPEG_PATH,
            "-y",
            "-loop", "1",
            "-i", str(image_path),
            "-vf", zoompan_filter,
            "-t", str(duration),
            "-c:v", "libx264",
            "-pix_fmt", "yuv420p",
            "-preset", "medium",
            "-crf", "23",
            str(output_path),
        ]

        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=120
        )

        if result.returncode != 0:
            logger.error("ffmpeg Ken Burns failed: %s", result.stderr[-1000:])
            return False

        if output_path.exists() and output_path.stat().st_size > 0:
            logger.info("Ken Burns animation saved: %s (%.1fs)", output_path.name, duration)
            return True
        return False

    except subprocess.TimeoutExpired:
        logger.error("ffmpeg Ken Burns timed out for %s", image_path.name)
        return False
    except Exception as exc:
        logger.error("Ken Burns animation failed: %s", exc)
        return False

# ---------------------------------------------------------------------------
# HuggingFace video generation (best-effort)
# ---------------------------------------------------------------------------

def _try_hf_video(image_path: Path, output_path: Path, duration: float) -> bool:
    if not settings.HF_VIDEO_MODEL or not settings.HF_API_TOKEN:
        return False

    try:
        url = f"[https://router.huggingface.co/hf-inference/models/](https://router.huggingface.co/hf-inference/models/){settings.HF_VIDEO_MODEL}"
        headers = {"Authorization": f"Bearer {settings.HF_API_TOKEN}"}

        with open(image_path, "rb") as f:
            image_bytes = f.read()

        response = requests.post(
            url,
            headers=headers,
            data=image_bytes,
            timeout=120,
        )

        if response.status_code == 200 and len(response.content) > 10000:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, "wb") as f:
                f.write(response.content)
            logger.info("HF video generation succeeded: %s", output_path.name)
            return True
        else:
            logger.info("HF video generation returned %d. Falling back to Ken Burns.", response.status_code)
            return False
    except Exception as exc:
        logger.info("HF video generation failed (%s). Using Ken Burns.", exc)
        return False

# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def run_animator(project_id: int) -> dict:
    logger.info("Agent 8 (Animator) starting for project %d", project_id)
    results = {"animated": 0, "failed": 0, "skipped": 0}

    with managed_session() as db:
        locations = db.query(Location).filter(Location.project_id == project_id).all()
        loc_list = [{"id": l.id, "name": l.name, "raw_bg_path": l.raw_bg_path} for l in locations]

    with managed_session() as db:
        shots = db.query(Shot).filter(Shot.project_id == project_id).all()
        location_max_duration: dict[str, float] = {}
        for s in shots:
            loc_name = s.location_name or ""
            dur = s.duration_seconds or 7.0
            if loc_name in location_max_duration:
                location_max_duration[loc_name] = max(location_max_duration[loc_name], dur)
            else:
                location_max_duration[loc_name] = dur

    animated_map: dict[str, str] = {}

    for ldata in loc_list:
        raw_path_str = ldata["raw_bg_path"]
        if not raw_path_str:
            logger.warning("Location '%s' has no background image. Skipping.", ldata["name"])
            results["skipped"] += 1
            continue

        raw_path = Path(raw_path_str)
        if not raw_path.exists():
            logger.warning("Background image not found: %s", raw_path)
            results["skipped"] += 1
            continue

        duration = location_max_duration.get(ldata["name"], 10.0) + 2.0
        duration = min(duration, 30.0)

        output_path = get_workspace_path(
            project_id,
            settings.DIR_BACKGROUNDS_ANIMATED,
            f"bg_{ldata['name'].replace(' ', '_').lower()}_animated.mp4",
        )

        success = _try_hf_video(raw_path, output_path, duration)
        if not success:
            success = _ken_burns_animate(raw_path, output_path, duration=duration)

        with managed_session() as db:
            loc = db.query(Location).filter(Location.id == ldata["id"]).first()
            if loc:
                if success:
                    loc.animated_bg_path = str(output_path)
                    loc.status = STATUS_COMPLETED
                    animated_map[ldata["name"]] = str(output_path)
                    results["animated"] += 1
                else:
                    loc.status = STATUS_FAILED
                    results["failed"] += 1

    with managed_session() as db:
        shots = db.query(Shot).filter(Shot.project_id == project_id).all()
        for shot in shots:
            loc_name = shot.location_name or ""
            if loc_name in animated_map:
                shot.animated_bg_path = animated_map[loc_name]
            elif animated_map:
                shot.animated_bg_path = list(animated_map.values())[0]

    logger.info("Agent 8 complete. Animated=%d, Failed=%d, Skipped=%d", results["animated"], results["failed"], results["skipped"])
    return results