"""
Agent 11 — Quality Assurance
=============================
Uses faster-whisper to transcribe the master video's audio track.
Compares the transcript against the original script dialogue.
If dialogue is missing or significantly different, logs BugReports
targeting Agent 5 (audio) or Agent 10 (compositing) and resets the
affected Shot status to 'pending'.

FIXED: All FFmpeg calls now use settings.FFMPEG_PATH which safely defaults to "ffmpeg"
even when .env contains FFMPEG_PATH="" (previously caused [WinError 87]).
"""

import logging
import subprocess
import tempfile
from pathlib import Path
from typing import Optional

from core.config import settings, get_workspace_path
from core.database import (
    managed_session,
    Project,
    Shot,
    BugReport,
    STATUS_PENDING,
    STATUS_COMPLETED,
)

logger = logging.getLogger("vortex.agent11_qa")

# ---------------------------------------------------------------------------
# Transcript extraction using faster-whisper
# ---------------------------------------------------------------------------

def _get_ffmpeg_cmd() -> str:
    """Return the configured FFmpeg binary path - never empty (fixes WinError 87)."""
    return settings.FFMPEG_PATH  # Guaranteed "ffmpeg" if env blank


def _transcribe_video(video_path: Path) -> str:
    """
    Extract audio from video, transcribe using faster-whisper.
    Returns the full transcript text.
    """
    # Extract audio to temp WAV using fixed FFmpeg path
    tmp_wav = Path(tempfile.mktemp(suffix=".wav"))
    try:
        ffmpeg_bin = _get_ffmpeg_cmd()
        cmd = [
            ffmpeg_bin,
            "-y",
            "-i", str(video_path),
            "-vn",
            "-ar", "16000",
            "-ac", "1",
            "-f", "wav",
            str(tmp_wav),
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        if result.returncode != 0:
            logger.error("Audio extraction failed: %s", result.stderr[-500:])
            return ""

        if not tmp_wav.exists() or tmp_wav.stat().st_size < 100:
            logger.warning("Extracted audio file is too small or missing.")
            return ""

        # Transcribe with faster-whisper
        try:
            from faster_whisper import WhisperModel

            model = WhisperModel("base", device="cpu", compute_type="int8")
            segments, info = model.transcribe(str(tmp_wav), language="en")

            transcript_parts = []
            for segment in segments:
                transcript_parts.append(segment.text.strip())

            full_transcript = " ".join(transcript_parts)
            logger.info(
                "Transcription complete. %d chars, detected language: %s (prob=%.2f)",
                len(full_transcript),
                info.language,
                info.language_probability,
            )
            return full_transcript

        except ImportError:
            logger.error("faster-whisper is not installed.")
            return ""
        except Exception as exc:
            logger.error("Transcription failed: %s", exc)
            return ""

    finally:
        tmp_wav.unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# Dialogue comparison
# ---------------------------------------------------------------------------

def _normalize_text(text: str) -> str:
    """Lowercase, strip punctuation, collapse whitespace."""
    import re
    text = text.lower()
    text = re.sub(r"[^\w\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _word_overlap_ratio(reference: str, hypothesis: str) -> float:
    """
    Compute the fraction of reference words found in the hypothesis.
    Simple bag-of-words approach for QA purposes.
    """
    ref_words = set(_normalize_text(reference).split())
    hyp_words = set(_normalize_text(hypothesis).split())

    if not ref_words:
        return 1.0  # no reference words means nothing to miss

    overlap = ref_words.intersection(hyp_words)
    return len(overlap) / len(ref_words)


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def run_qa(project_id: int) -> dict:
    """
    Execute Agent 11: QA the master video against the script.

    1. Transcribe the master video.
    2. Compare against each shot's expected dialogue.
    3. Log BugReports for missing dialogue and reset shot status.
    """
    logger.info("Agent 11 (QA) starting for project %d", project_id)
    results = {
        "transcript_length": 0,
        "shots_checked": 0,
        "shots_passed": 0,
        "bugs_filed": 0,
    }

    # Get master video path
    with managed_session() as db:
        project = db.query(Project).filter(Project.id == project_id).first()
        if not project or not project.final_video_path:
            logger.error("No master video found for project %d.", project_id)
            return results
        video_path = Path(project.final_video_path)

    if not video_path.exists():
        logger.error("Master video file not found: %s", video_path)
        return results

    # Step 1: Transcribe
    logger.info("Transcribing master video: %s", video_path.name)
    transcript = _transcribe_video(video_path)
    results["transcript_length"] = len(transcript)

    if not transcript:
        logger.warning("Transcription returned empty. Filing global bug report.")
        with managed_session() as db:
            bug = BugReport(
                project_id=project_id,
                source_agent="agent11_qa",
                target_agent="agent5_audio",
                severity="high",
                description=(
                    "Master video transcription returned empty. "
                    "Possible issues: no audio track, corrupt audio, or "
                    "all shots have silent audio."
                ),
            )
            db.add(bug)
        results["bugs_filed"] = 1
        return results

    # Save transcript
    transcript_path = get_workspace_path(
        project_id, settings.DIR_QA, "transcript.txt"
    )
    transcript_path.write_text(transcript, encoding="utf-8")
    logger.info("Transcript saved to %s", transcript_path)

    # Step 2: Compare against shot dialogue
    with managed_session() as db:
        shots = (
            db.query(Shot)
            .filter(Shot.project_id == project_id)
            .order_by(Shot.shot_number)
            .all()
        )
        shot_data = [
            {"id": s.id, "shot_number": s.shot_number, "dialogue": s.dialogue}
            for s in shots
        ]

    missing_dialogue_shots = []
    qa_threshold = 0.3  # At least 30% of words should appear in transcript

    for sdata in shot_data:
        dialogue = sdata["dialogue"]
        if not dialogue or not dialogue.strip():
            continue

        results["shots_checked"] += 1
        overlap = _word_overlap_ratio(dialogue, transcript)

        if overlap >= qa_threshold:
            results["shots_passed"] += 1
            logger.debug(
                "Shot %d: QA PASSED (overlap=%.2f)",
                sdata["shot_number"],
                overlap,
            )
        else:
            missing_dialogue_shots.append(
                {
                    "shot_id": sdata["id"],
                    "shot_number": sdata["shot_number"],
                    "expected": dialogue[:100],
                    "overlap": overlap,
                }
            )
            logger.warning(
                "Shot %d: QA FAILED (overlap=%.2f < %.2f). Expected: '%.60s…'",
                sdata["shot_number"],
                overlap,
                qa_threshold,
                dialogue,
            )

    # Step 3: File bug reports
    with managed_session() as db:
        for miss in missing_dialogue_shots:
            # Determine target agent
            shot = db.query(Shot).filter(Shot.id == miss["shot_id"]).first()
            if shot:
                if not shot.audio_path or not Path(shot.audio_path).exists():
                    target_agent = "agent5_audio"
                    desc = (
                        f"Shot {miss['shot_number']}: Missing or no audio file. "
                        f"Expected dialogue: '{miss['expected']}…' "
                        f"Word overlap with transcript: {miss['overlap']:.2f}"
                    )
                else:
                    target_agent = "agent10_compositor"
                    desc = (
                        f"Shot {miss['shot_number']}: Audio exists but dialogue "
                        f"not detected in master video. Possible compositing issue. "
                        f"Expected: '{miss['expected']}…' "
                        f"Overlap: {miss['overlap']:.2f}"
                    )

                bug = BugReport(
                    project_id=project_id,
                    shot_id=miss["shot_id"],
                    source_agent="agent11_qa",
                    target_agent=target_agent,
                    severity="medium",
                    description=desc,
                )
                db.add(bug)

                # Reset shot status to pending
                shot.status = STATUS_PENDING

            results["bugs_filed"] += 1

    # Save QA report
    qa_report = {
        "transcript_length": results["transcript_length"],
        "shots_checked": results["shots_checked"],
        "shots_passed": results["shots_passed"],
        "bugs_filed": results["bugs_filed"],
        "missing_shots": missing_dialogue_shots,
    }
    import json
    qa_report_path = get_workspace_path(
        project_id, settings.DIR_QA, "qa_report.json"
    )
    qa_report_path.write_text(
        json.dumps(qa_report, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    logger.info(
        "Agent 11 complete. Checked=%d, Passed=%d, Bugs=%d",
        results["shots_checked"],
        results["shots_passed"],
        results["bugs_filed"],
    )
    return results
