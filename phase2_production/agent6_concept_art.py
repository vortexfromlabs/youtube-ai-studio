"""
Agent 6 — Concept Art Generator
================================
Uses the Hugging Face Serverless Inference API to generate static images
for Characters (front-facing, full-body on plain background) and
Background Locations (wide cinematic establishing shots).

Uses FLUX.schnell or other free models via the standard inference endpoint.
"""

import io
import logging
import time
from pathlib import Path
from typing import Optional

import requests
from PIL import Image

from core.config import settings, get_workspace_path
from core.database import (
    managed_session,
    Character,
    Location,
    STATUS_COMPLETED,
    STATUS_FAILED,
    STATUS_IN_PROGRESS,
)

logger = logging.getLogger("vortex.agent6_concept_art")

# ---------------------------------------------------------------------------
# Hugging Face Inference API helper
# ---------------------------------------------------------------------------

_HF_API_URL_TEMPLATE = "https://router.huggingface.co/hf-inference/models/{model}"
_MAX_RETRIES = 3
_RETRY_DELAY = 15  # seconds (model loading can take time)


def _generate_image_hf(
    prompt: str,
    output_path: Path,
    model: Optional[str] = None,
    width: int = 1024,
    height: int = 1024,
) -> bool:
    """
    Call HuggingFace Inference API for text-to-image generation.
    Returns True on success.
    """
    model = model or settings.HF_IMAGE_MODEL
    url = _HF_API_URL_TEMPLATE.format(model=model)
    headers = {
        "Authorization": f"Bearer {settings.HF_API_TOKEN}",
        "Content-Type": "application/json",
    }
    payload = {
        "inputs": prompt,
        "parameters": {
            "width": width,
            "height": height,
        },
    }

    for attempt in range(1, _MAX_RETRIES + 1):
        try:
            logger.info(
                "HF image request attempt %d/%d for: %.80s…",
                attempt,
                _MAX_RETRIES,
                prompt,
            )
            response = requests.post(
                url, headers=headers, json=payload, timeout=120
            )

            if response.status_code == 503:
                # Model is loading
                wait_time = _RETRY_DELAY * attempt
                logger.info("Model loading, waiting %ds…", wait_time)
                time.sleep(wait_time)
                continue

            if response.status_code == 200:
                content_type = response.headers.get("content-type", "")
                if "image" in content_type or len(response.content) > 1000:
                    output_path.parent.mkdir(parents=True, exist_ok=True)
                    img = Image.open(io.BytesIO(response.content))
                    img.save(str(output_path), format="PNG")
                    logger.info("Image saved: %s (%dx%d)", output_path.name, img.width, img.height)
                    return True
                else:
                    logger.warning(
                        "Unexpected response content-type: %s, body: %.500s",
                        content_type,
                        response.text,
                    )
                    return False

            logger.warning(
                "HF API returned %d: %.500s",
                response.status_code,
                response.text,
            )
            if attempt < _MAX_RETRIES:
                time.sleep(_RETRY_DELAY)

        except requests.exceptions.Timeout:
            logger.warning("HF API timeout on attempt %d", attempt)
            if attempt < _MAX_RETRIES:
                time.sleep(_RETRY_DELAY)
        except Exception as exc:
            logger.error("HF API unexpected error: %s", exc)
            if attempt < _MAX_RETRIES:
                time.sleep(_RETRY_DELAY)

    return False


# ---------------------------------------------------------------------------
# Fallback: generate a coloured placeholder image with text
# ---------------------------------------------------------------------------

def _generate_placeholder_image(
    label: str, output_path: Path, size: tuple[int, int] = (1024, 1024)
) -> bool:
    """Create a simple labelled placeholder PNG when the API fails."""
    try:
        from PIL import ImageDraw, ImageFont

        img = Image.new("RGB", size, color=(45, 45, 65))
        draw = ImageDraw.Draw(img)

        # Try to use a default font; fall back to ImageFont.load_default()
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 28)
        except (IOError, OSError):
            font = ImageFont.load_default()

        # Wrap text
        max_chars_per_line = 40
        words = label.split()
        lines = []
        current_line = ""
        for word in words:
            if len(current_line) + len(word) + 1 <= max_chars_per_line:
                current_line = f"{current_line} {word}".strip()
            else:
                lines.append(current_line)
                current_line = word
        if current_line:
            lines.append(current_line)

        y_start = size[1] // 2 - len(lines) * 18
        for i, line in enumerate(lines):
            bbox = draw.textbbox((0, 0), line, font=font)
            text_width = bbox[2] - bbox[0]
            x = (size[0] - text_width) // 2
            draw.text((x, y_start + i * 36), line, fill=(200, 200, 220), font=font)

        output_path.parent.mkdir(parents=True, exist_ok=True)
        img.save(str(output_path), format="PNG")
        logger.info("Placeholder image saved: %s", output_path.name)
        return True
    except Exception as exc:
        logger.error("Even placeholder generation failed: %s", exc)
        return False


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def run_concept_art(project_id: int) -> dict:
    """
    Execute Agent 6 for the given project.

    1. Generate images for all Characters (front-facing, full-body).
    2. Generate images for all Locations (wide establishing shot).
    3. Update database paths.
    """
    logger.info("Agent 6 (Concept Art) starting for project %d", project_id)
    results = {"characters_ok": 0, "characters_fail": 0, "locations_ok": 0, "locations_fail": 0}

    # --- Characters ---
    with managed_session() as db:
        characters = (
            db.query(Character)
            .filter(Character.project_id == project_id)
            .all()
        )
        char_list = [
            {"id": c.id, "name": c.name, "visual_prompt": c.visual_prompt or c.description or c.name}
            for c in characters
        ]

    for cdata in char_list:
        output_path = get_workspace_path(
            project_id,
            settings.DIR_IMAGES_RAW,
            f"char_{cdata['name'].replace(' ', '_').lower()}.png",
        )

        prompt = cdata["visual_prompt"]
        success = _generate_image_hf(prompt, output_path, width=768, height=1024)

        if not success:
            logger.warning("Falling back to placeholder for character '%s'", cdata["name"])
            success = _generate_placeholder_image(
                f"CHARACTER: {cdata['name']}\n{prompt[:120]}", output_path, (768, 1024)
            )

        with managed_session() as db:
            char = db.query(Character).filter(Character.id == cdata["id"]).first()
            if char:
                if success:
                    char.raw_image_path = str(output_path)
                    char.status = STATUS_COMPLETED
                    results["characters_ok"] += 1
                else:
                    char.status = STATUS_FAILED
                    results["characters_fail"] += 1

    # --- Locations ---
    with managed_session() as db:
        locations = (
            db.query(Location)
            .filter(Location.project_id == project_id)
            .all()
        )
        loc_list = [
            {"id": l.id, "name": l.name, "visual_prompt": l.visual_prompt or l.description or l.name}
            for l in locations
        ]

    for ldata in loc_list:
        output_path = get_workspace_path(
            project_id,
            settings.DIR_BACKGROUNDS_RAW,
            f"bg_{ldata['name'].replace(' ', '_').lower()}.png",
        )

        prompt = ldata["visual_prompt"]
        success = _generate_image_hf(prompt, output_path, width=1920, height=1080)

        if not success:
            success = _generate_placeholder_image(
                f"BACKGROUND: {ldata['name']}\n{prompt[:120]}",
                output_path,
                (1920, 1080),
            )

        with managed_session() as db:
            loc = db.query(Location).filter(Location.id == ldata["id"]).first()
            if loc:
                if success:
                    loc.raw_bg_path = str(output_path)
                    loc.cinematic_bg_path = str(output_path)  # same until animated
                    loc.status = STATUS_COMPLETED
                    results["locations_ok"] += 1
                else:
                    loc.status = STATUS_FAILED
                    results["locations_fail"] += 1

    logger.info(
        "Agent 6 complete. Chars OK=%d Fail=%d | Locs OK=%d Fail=%d",
        results["characters_ok"],
        results["characters_fail"],
        results["locations_ok"],
        results["locations_fail"],
    )
    return results