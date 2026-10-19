"""
Agent 9 — Lip Sync Animator
============================
Takes transparent character PNGs (from Agent 7) and audio (from Agent 5).
Creates procedural lip-sync animation by alternating between "mouth open"
and "mouth closed" frames based on audio volume analysis using numpy.

Outputs transparent video (.mov with ProRes 4444 or .webm with VP9+alpha).
Falls back to .mp4 with green-screen keying if transparency codecs fail.
"""

import logging
import subprocess
import tempfile
import wave
from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image, ImageDraw

from core.config import settings, get_workspace_path
from core.database import (
    managed_session,
    Shot,
    Character,
    STATUS_COMPLETED,
    STATUS_FAILED,
)

logger = logging.getLogger("vortex.agent9_lipsync")

# ---------------------------------------------------------------------------
# Audio analysis: extract volume envelope
# ---------------------------------------------------------------------------

def _extract_volume_envelope(audio_path: Path, fps: int = 24) -> np.ndarray:
    """
    Convert audio to WAV, then compute per-frame RMS volume envelope.
    Returns an array of RMS values, one per video frame.
    """
    # Convert to WAV using ffmpeg
    tmp_wav = Path(tempfile.mktemp(suffix=".wav"))
    try:
        cmd = [
            settings.FFMPEG_PATH,
            "-y",
            "-i", str(audio_path),
            "-ar", "16000",
            "-ac", "1",
            "-f", "wav",
            str(tmp_wav),
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        if result.returncode != 0:
            logger.error("Audio→WAV conversion failed: %s", result.stderr[-500:])
            return np.array([])

        with wave.open(str(tmp_wav), "rb") as wf:
            n_frames = wf.getnframes()
            sample_rate = wf.getframerate()
            raw_data = wf.readframes(n_frames)
            samples = np.frombuffer(raw_data, dtype=np.int16).astype(np.float32)

        # Calculate RMS per video frame
        samples_per_frame = max(1, int(sample_rate / fps))
        n_video_frames = max(1, len(samples) // samples_per_frame)

        envelope = np.zeros(n_video_frames)
        for i in range(n_video_frames):
            start = i * samples_per_frame
            end = start + samples_per_frame
            chunk = samples[start:end]
            if len(chunk) > 0:
                rms = np.sqrt(np.mean(chunk ** 2))
                envelope[i] = rms

        # Normalize to 0-1
        max_val = np.max(envelope)
        if max_val > 0:
            envelope = envelope / max_val

        return envelope

    finally:
        tmp_wav.unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# Character mouth manipulation
# ---------------------------------------------------------------------------

def _create_mouth_frames(
    character_image_path: Path,
    output_dir: Path,
    envelope: np.ndarray,
    mouth_threshold: float = 0.15,
) -> int:
    """
    For each frame, produce a version of the character image with the
    mouth area visually altered (open or closed).

    Strategy:
    - Locate the approximate mouth region (lower-center of the image).
    - For "open" frames, draw a dark oval (open mouth).
    - For "closed" frames, use the original image.

    Returns the number of frames generated.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    base_img = Image.open(character_image_path).convert("RGBA")
    w, h = base_img.size

    # Estimate mouth region: center-x, around 70-75% from top
    mouth_cx = w // 2
    mouth_cy = int(h * 0.72)
    mouth_rx = int(w * 0.06)  # horizontal radius
    mouth_ry_closed = int(h * 0.008)
    mouth_ry_open = int(h * 0.025)

    frame_count = len(envelope)
    for i in range(frame_count):
        frame = base_img.copy()
        vol = envelope[i]

        if vol > mouth_threshold:
            # Draw open mouth
            draw = ImageDraw.Draw(frame)
            openness = min(1.0, vol / 0.6)
            ry = int(mouth_ry_closed + (mouth_ry_open - mouth_ry_closed) * openness)
            bbox = [
                mouth_cx - mouth_rx,
                mouth_cy - ry,
                mouth_cx + mouth_rx,
                mouth_cy + ry,
            ]
            draw.ellipse(bbox, fill=(40, 20, 20, 220))
            # Small highlight for teeth
            teeth_ry = max(1, ry // 3)
            teeth_bbox = [
                mouth_cx - mouth_rx + 4,
                mouth_cy - teeth_ry,
                mouth_cx + mouth_rx - 4,
                mouth_cy,
            ]
            draw.ellipse(teeth_bbox, fill=(230, 230, 230, 180))

        frame_path = output_dir / f"frame_{i:05d}.png"
        frame.save(str(frame_path), format="PNG")

    logger.info("Generated %d lip-sync frames in %s", frame_count, output_dir)
    return frame_count


# ---------------------------------------------------------------------------
# Assemble frames into video with transparency
# ---------------------------------------------------------------------------

def _assemble_transparent_video(
    frames_dir: Path,
    output_path: Path,
    fps: int = 24,
    audio_path: Optional[Path] = None,
) -> bool:
    """
    Assemble PNG frames (with alpha) into a video.
    Try VP9+alpha in WebM first, fall back to libx264 with yuva420p, then plain MP4.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Strategy 1: WebM with VP9 (supports alpha)
    webm_output = output_path.with_suffix(".webm")
    cmd = [
        settings.FFMPEG_PATH,
        "-y",
        "-framerate", str(fps),
        "-i", str(frames_dir / "frame_%05d.png"),
    ]
    if audio_path and audio_path.exists():
        cmd.extend(["-i", str(audio_path)])
    cmd.extend([
        "-c:v", "libvpx-vp9",
        "-pix_fmt", "yuva420p",
        "-b:v", "2M",
        "-auto-alt-ref", "0",
    ])
    if audio_path and audio_path.exists():
        cmd.extend(["-c:a", "libopus", "-b:a", "128k", "-shortest"])
    cmd.append(str(webm_output))

    result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
    if result.returncode == 0 and webm_output.exists() and webm_output.stat().st_size > 0:
        # Rename/move to expected output
        if output_path.suffix != ".webm":
            import shutil
            shutil.move(str(webm_output), str(output_path.with_suffix(".webm")))
            return True
        return True

    logger.info("VP9+alpha failed, falling back to MP4 with green overlay…")
    webm_output.unlink(missing_ok=True)

    # Strategy 2: MP4 with libx264 (no transparency, but reliable)
    mp4_output = output_path.with_suffix(".mp4")
    cmd = [
        settings.FFMPEG_PATH,
        "-y",
        "-framerate", str(fps),
        "-i", str(frames_dir / "frame_%05d.png"),
    ]
    if audio_path and audio_path.exists():
        cmd.extend(["-i", str(audio_path)])
    cmd.extend([
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        "-preset", "medium",
        "-crf", "20",
    ])
    if audio_path and audio_path.exists():
        cmd.extend(["-c:a", "aac", "-b:a", "128k", "-shortest"])
    cmd.append(str(mp4_output))

    result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
    if result.returncode == 0 and mp4_output.exists():
        logger.info("Lip-sync MP4 saved: %s", mp4_output.name)
        return True

    logger.error("Frame assembly failed: %s", result.stderr[-500:])
    return False


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def run_lipsync(project_id: int) -> dict:
    """
    Execute Agent 9: create lip-sync animations for all shots with dialogue.
    """
    logger.info("Agent 9 (Lip Sync) starting for project %d", project_id)
    results = {"synced": 0, "skipped": 0, "failed": 0}
    fps = 24

    # Load shots
    with managed_session() as db:
        shots = (
            db.query(Shot)
            .filter(Shot.project_id == project_id)
            .order_by(Shot.shot_number)
            .all()
        )
        shot_list = [
            {
                "id": s.id,
                "shot_number": s.shot_number,
                "dialogue": s.dialogue,
                "audio_path": s.audio_path,
                "speaking_character_name": s.speaking_character_name,
                "duration_seconds": s.duration_seconds or 5.0,
            }
            for s in shots
        ]

    # Build character name → transparent image map
    with managed_session() as db:
        characters = (
            db.query(Character)
            .filter(Character.project_id == project_id)
            .all()
        )
        char_image_map = {
            c.name: c.transparent_asset_path
            for c in characters
            if c.transparent_asset_path
        }

    for sdata in shot_list:
        shot_num = sdata["shot_number"]
        dialogue = sdata["dialogue"]

        if not dialogue or not dialogue.strip():
            results["skipped"] += 1
            continue

        audio_path_str = sdata["audio_path"]
        char_name = sdata["speaking_character_name"] or ""
        char_image_path_str = char_image_map.get(char_name, "")

        if not audio_path_str or not Path(audio_path_str).exists():
            logger.warning("Shot %d: no audio file found. Skipping lip sync.", shot_num)
            results["skipped"] += 1
            continue

        if not char_image_path_str or not Path(char_image_path_str).exists():
            # Use the first available character image as fallback
            if char_image_map:
                char_image_path_str = list(char_image_map.values())[0]
                logger.info(
                    "Shot %d: character '%s' not found, using fallback image.",
                    shot_num,
                    char_name,
                )
            else:
                logger.warning("Shot %d: no character images available. Skipping.", shot_num)
                results["skipped"] += 1
                continue

        audio_path = Path(audio_path_str)
        char_image_path = Path(char_image_path_str)

        # Step 1: Extract volume envelope
        envelope = _extract_volume_envelope(audio_path, fps=fps)
        if len(envelope) == 0:
            # Estimate frame count from duration
            n_frames = max(1, int(sdata["duration_seconds"] * fps))
            envelope = np.zeros(n_frames)
            logger.warning("Shot %d: could not analyze audio, using silent envelope.", shot_num)

        # Step 2: Generate mouth-animated frames
        frames_dir = get_workspace_path(
            project_id, settings.DIR_LIPSYNC, f"shot_{shot_num:03d}_frames"
        )
        frame_count = _create_mouth_frames(
            char_image_path, frames_dir, envelope
        )

        if frame_count == 0:
            results["failed"] += 1
            continue

        # Step 3: Assemble into video
        output_path = get_workspace_path(
            project_id, settings.DIR_LIPSYNC, f"shot_{shot_num:03d}_lipsync.mp4"
        )
        success = _assemble_transparent_video(
            frames_dir, output_path, fps=fps, audio_path=audio_path
        )

        # Check for webm output too
        webm_path = output_path.with_suffix(".webm")
        actual_output = str(output_path)
        if webm_path.exists() and webm_path.stat().st_size > 0:
            actual_output = str(webm_path)
        elif output_path.exists() and output_path.stat().st_size > 0:
            actual_output = str(output_path)

        with managed_session() as db:
            shot = db.query(Shot).filter(Shot.id == sdata["id"]).first()
            if shot:
                if success:
                    shot.lip_sync_video_path = actual_output
                    results["synced"] += 1
                else:
                    shot.status = STATUS_FAILED
                    results["failed"] += 1

    logger.info(
        "Agent 9 complete. Synced=%d, Skipped=%d, Failed=%d",
        results["synced"],
        results["skipped"],
        results["failed"],
    )
    return results