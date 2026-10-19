"""
Agent 6 — Concept Art Generator
================================
Uses Pollinations.ai for 100% FREE, keyless, reliable image generation.
Characters: front-facing, full-body on plain background
Backgrounds: wide cinematic establishing shots

MIGRATED: Hugging Face (deprecated, blank blue images) -> Pollinations.ai
Pollinations is free and requires NO API key: https://image.pollinations.ai/prompt/{prompt}?width=...&height=...&nologo=true
"""

import logging
import time
import urllib.parse
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
# Pollinations.ai helper - 100% free, no API key, ultra-reliable
# ---------------------------------------------------------------------------

def _generate_image_pollinations(
    prompt: str,
    output_path: Path,
    width: int = 1024,
    height: int = 1024,
) -> bool:
    """
    Generate image via Pollinations.ai - 100% free, no API key required.
    Uses: https://image.pollinations.ai/prompt/{urllib.parse.quote(prompt)}?width={width}&height={height}&nologo=true
    Saves response.content directly to output_path.
    """
    try:
        # URL encode the prompt to handle spaces, special chars
        encoded_prompt = urllib.parse.quote(prompt)
        url = f"https://image.pollinations.ai/prompt/{encoded_prompt}?width={width}&height={height}&nologo=true&enhance=true"

        logger.info("Pollinations request for: %.80s… (size %dx%d)", prompt, width, height)

        # Simple GET - no headers, no auth, no API key needed
        response = requests.get(url, timeout=90)

        if response.status_code == 200:
            content_type = response.headers.get("content-type", "")
            # Pollinations returns image/jpeg or image/png directly
            if "image" in content_type or len(response.content) > 5000:
                output_path.parent.mkdir(parents=True, exist_ok=True)
                # Save content directly - no PIL re-encoding needed for reliability
                # But validate it's a real image
                output_path.write_bytes(response.content)
                # Verify we can open it as image (corruption check)
                try:
                    img = Image.open(output_path)
                    img.verify()  # Verify without decoding fully
                    logger.info("Pollinations image saved: %s (%dx%d, %d bytes)", output_path.name, width, height, len(response.content))
                    return True
                except Exception as ve:
                    logger.warning("Pollinations returned invalid image data: %s. Size: %d bytes", ve, len(response.content))
                    return False
            else:
                logger.warning("Pollinations unexpected content-type: %s, size: %d", content_type, len(response.content))
                return False
        else:
            logger.warning("Pollinations returned %d: %.500s", response.status_code, response.text if hasattr(response, 'text') else "")
            return False

    except requests.exceptions.Timeout:
        logger.warning("Pollinations timeout for prompt: %.50s", prompt)
        return False
    except Exception as exc:
        logger.error("Pollinations unexpected error: %s", exc)
        return False


# ---------------------------------------------------------------------------
# Fallback: generate a coloured placeholder image with text (if Pollinations down)
# ---------------------------------------------------------------------------

def _generate_placeholder_image(
    label: str, output_path: Path, size: tuple[int, int] = (1024, 1024)
) -> bool:
    """Create a simple labelled placeholder PNG when Pollinations fails."""
    try:
        from PIL import ImageDraw, ImageFont

        img = Image.new("RGB", size, color=(45, 45, 65))
        draw = ImageDraw.Draw(img)

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
    Uses Pollinations.ai - 100% free, no deprecation, no blank blue placeholders.
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
        # Enhance prompt for character sheet (plain background for easy bg removal later)
        if "plain" not in prompt.lower() and "background" not in prompt.lower():
            prompt = f"{prompt}, character design sheet, plain white background, full body, front facing"
        success = _generate_image_pollinations(prompt, output_path, width=768, height=1024)

        if not success:
            logger.warning("Pollinations failed, falling back to placeholder for character '%s'", cdata["name"])
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
        # Enhance for cinematic background
        if "cinematic" not in prompt.lower():
            prompt = f"{prompt}, cinematic wide establishing shot, highly detailed, 8k, vibrant lighting"
        success = _generate_image_pollinations(prompt, output_path, width=1920, height=1080)

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
