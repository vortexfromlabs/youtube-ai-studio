"""
agent10_compositor.py — Phase 3: Post-Production Compositor & Master Stitcher.

Responsibilities
----------------
1. Overlays the transparent lip-synced character animation onto the animated
   background for each individual shot using FFmpeg filter graphs.
2. Mixes the dialogue audio track with any background ambience.
3. Master-stitches all rendered shot clips into a final master video using
   FFmpeg's concat demuxer (-f concat -safe 0).
4. Updates the Shot and Project records in the database.
"""
from __future__ import annotations

import logging
import os
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Optional

from core.config import get_workspace_path, settings
from core.database import (
    STATUS_COMPLETED,
    STATUS_FAILED,
    Project,
    Shot,
    Location,
    managed_session,
)

logger = logging.getLogger("vortex.compositor")


def _get_ffmpeg_cmd() -> str:
    """Return the configured or default FFmpeg binary path."""
    return settings.FFMPEG_PATH if settings.FFMPEG_PATH else "ffmpeg"


def _composite_single_shot(
    shot: Shot,
    project_id: int,
    output_dir: Path,
) -> Optional[Path]:
    """Composite character lip-sync video over background for a single shot."""
    ffmpeg_bin = _get_ffmpeg_cmd()
    shot_output_path = output_dir / f"shot_{shot.shot_number:04d}_composited.mp4"

    bg_path = Path(shot.animated_bg_path) if hasattr(shot, "animated_bg_path") and shot.animated_bg_path else None
    if not bg_path or not bg_path.exists():
        if hasattr(shot, "background_image_path") and shot.background_image_path:
            bg_path = Path(shot.background_image_path)

    char_path = Path(shot.lip_sync_video_path) if shot.lip_sync_video_path else None
    audio_path = Path(shot.audio_path) if shot.audio_path else None
    duration = shot.duration_seconds if shot.duration_seconds and shot.duration_seconds > 0 else 5.0

    # Case 1: Both character overlay and background video/image exist
    if char_path and char_path.exists() and bg_path and bg_path.exists():
        cmd = [
            ffmpeg_bin,
            "-y",
            "-i", str(bg_path),
            "-i", str(char_path),
            "-filter_complex",
            "[0:v]scale=1920:1080:force_original_aspect_ratio=decrease,pad=1920:1080:(ow-iw)/2:(oh-ih)/2[bg];"
            "[1:v]scale=1920:1080[fg];"
            "[bg][fg]overlay=0:0:shortest=1[outv]",
            "-map", "[outv]",
        ]
        if audio_path and audio_path.exists():
            cmd.extend(["-i", str(audio_path), "-map", "2:a", "-c:a", "aac", "-b:a", "192k"])
        cmd.extend([
            "-c:v", "libx264",
            "-pix_fmt", "yuv420p",
            "-t", str(duration),
            str(shot_output_path),
        ])

    # Case 2: Only background exists with audio
    elif bg_path and bg_path.exists() and audio_path and audio_path.exists():
        cmd = [
            ffmpeg_bin,
            "-y",
            "-loop", "1",
            "-i", str(bg_path),
            "-i", str(audio_path),
            "-c:v", "libx264",
            "-tune", "stillimage",
            "-c:a", "aac",
            "-b:a", "192k",
            "-pix_fmt", "yuv420p",
            "-shortest",
            "-t", str(duration),
            str(shot_output_path),
        ]

    # Case 3: Only character video exists
    elif char_path and char_path.exists():
        cmd = [
            ffmpeg_bin,
            "-y",
            "-i", str(char_path),
        ]
        if audio_path and audio_path.exists():
            cmd.extend(["-i", str(audio_path), "-c:a", "aac", "-b:a", "192k"])
        cmd.extend([
            "-c:v", "libx264",
            "-pix_fmt", "yuv420p",
            "-t", str(duration),
            str(shot_output_path),
        ])

    # Fallback Case: Generate clean solid test frame
    else:
        cmd = [
            ffmpeg_bin,
            "-y",
            "-f", "lavfi",
            "-i", f"color=c=0x1E293B:s=1920x1080:d={duration}",
        ]
        if audio_path and audio_path.exists():
            cmd.extend(["-i", str(audio_path), "-c:a", "aac", "-b:a", "192k"])
        else:
            cmd.extend(["-f", "lavfi", "-i", "anullsrc=r=44100:cl=stereo", "-t", str(duration)])
        cmd.extend([
            "-c:v", "libx264",
            "-pix_fmt", "yuv420p",
            str(shot_output_path),
        ])

    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=90)
        if proc.returncode == 0 and shot_output_path.exists() and shot_output_path.stat().st_size > 500:
            return shot_output_path
        logger.warning("FFmpeg command failed for shot %d: %s", shot.shot_number, proc.stderr[:300])
    except Exception as exc:
        logger.error("Error compositing shot %d: %s", shot.shot_number, exc)

    return None


def _stitch_master_video(shot_paths: List[Path], master_output_path: Path) -> bool:
    """Concatenate all composited shot MP4s into a single master episode video."""
    ffmpeg_bin = _get_ffmpeg_cmd()
    concat_txt = master_output_path.parent / "concat_list.txt"

    lines = [f"file '{p.resolve().as_posix().replace('\'', '\'\\\'\'')}'" for p in shot_paths]
    concat_txt.write_text("\n".join(lines), encoding="utf-8")

    cmd = [
        ffmpeg_bin,
        "-y",
        "-f", "concat",
        "-safe", "0",
        "-i", str(concat_txt),
        "-c:v", "libx264",
        "-c:a", "aac",
        "-b:a", "192k",
        "-pix_fmt", "yuv420p",
        str(master_output_path),
    ]

    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=180)
        if proc.returncode == 0 and master_output_path.exists() and master_output_path.stat().st_size > 1000:
            return True
        logger.error("FFmpeg concat failed: %s", proc.stderr[:500])
    except Exception as exc:
        logger.error("Exception during master video stitching: %s", exc)

    return False


def run_compositor(project_id: int) -> Dict[str, Any]:
    """Main Agent 10 entry point: composites all shots and stitches master video."""
    logger.info("Agent 10 (Compositor) starting for Project #%d", project_id)

    workspace_shots = get_workspace_path(project_id, "shots")
    workspace_exports = get_workspace_path(project_id, "exports")
    workspace_shots.mkdir(parents=True, exist_ok=True)
    workspace_exports.mkdir(parents=True, exist_ok=True)

    master_video_path = workspace_exports / f"project_{project_id}_master.mp4"

    composited_paths: List[Path] = []

    # Load all shots
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
                "lip_sync_video_path": s.lip_sync_video_path,
                "animated_bg_path": s.animated_bg_path,
                "audio_path": s.audio_path,
                "duration_seconds": s.duration_seconds or 5.0,
                "location_name": s.location_name,
            }
            for s in shots
        ]

    # Fallback: find any animated background
    fallback_bg = None
    with managed_session() as db:
        locs = (
            db.query(Location)
            .filter(
                Location.project_id == project_id,
                Location.animated_bg_path.isnot(None),
            )
            .all()
        )
        if locs:
            fallback_bg = locs[0].animated_bg_path

    results = {"composited": 0, "failed": 0, "master_created": False}

    for sdata in shot_list:
        shot_num = sdata["shot_number"]

        # Determine background
        bg_path_str = sdata["animated_bg_path"] or fallback_bg
        if not bg_path_str or not Path(bg_path_str).exists():
            bg_path = get_workspace_path(
                project_id, settings.DIR_BACKGROUNDS_ANIMATED, "black_bg.mp4"
            )
            if not bg_path.exists():
                cmd = [
                    _get_ffmpeg_cmd(),
                    "-y",
                    "-f", "lavfi",
                    "-i", "color=c=black:s=1920x1080:d=30:r=24",
                    "-c:v", "libx264",
                    "-pix_fmt", "yuv420p",
                    str(bg_path),
                ]
                subprocess.run(cmd, capture_output=True, timeout=60)
            bg_path_str = str(bg_path)

        bg_path = Path(bg_path_str)

        # Character lip sync video
        char_video_path = None
        if sdata["lip_sync_video_path"]:
            p = Path(sdata["lip_sync_video_path"])
            if p.exists():
                char_video_path = p
            else:
                webm_p = p.with_suffix(".webm")
                if webm_p.exists():
                    char_video_path = webm_p

        # Audio
        audio_path = None
        if sdata["audio_path"]:
            p = Path(sdata["audio_path"])
            if p.exists():
                audio_path = p

        output_path = get_workspace_path(
            project_id,
            settings.DIR_COMPOSITED,
            f"shot_{shot_num:03d}_composited.mp4",
        )

        success = False
        with managed_session() as db:
            shot = db.query(Shot).filter(Shot.id == sdata["id"]).first()
            if shot:
                comp_path = _composite_single_shot(shot, project_id, workspace_shots)
                if comp_path:
                    shot.composited_video_path = str(comp_path.resolve())
                    shot.status = STATUS_COMPLETED
                    composited_paths.append(comp_path)
                    results["composited"] += 1
                    success = True
                else:
                    shot.status = STATUS_FAILED
                    results["failed"] += 1

    # Stitch all composited shots
    if composited_paths:
        stitch_ok = _stitch_master_video(composited_paths, master_video_path)
        if stitch_ok:
            results["master_created"] = True
            with managed_session() as db:
                project = db.query(Project).filter(Project.id == project_id).first()
                if project:
                    project.final_video_path = str(master_video_path)
                    project.status = STATUS_COMPLETED
            logger.info("Master video created: %s", master_video_path)
        else:
            logger.error("Master video stitching failed.")
    else:
        logger.error("No composited shots available for stitching.")

    logger.info(
        "Agent 10 complete. Composited=%d, Failed=%d, Master=%s",
        results["composited"],
        results["failed"],
        results["master_created"],
    )
    return results