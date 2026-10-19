"""
Agent 7 — Background Remover
=============================
Uses the Python `rembg` library to process Agent 6's raw character images,
producing transparent PNGs suitable for compositing over animated backgrounds.
"""

import logging
from pathlib import Path

from PIL import Image

from core.config import settings, get_workspace_path
from core.database import (
    managed_session,
    Character,
    STATUS_COMPLETED,
    STATUS_FAILED,
)

logger = logging.getLogger("vortex.agent7_bg_remover")


def _remove_background(input_path: Path, output_path: Path) -> bool:
    """
    Remove background from *input_path* and save transparent PNG to *output_path*.
    Uses rembg with the default u2net model.
    """
    try:
        from rembg import remove

        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Read input in chunks-safe manner (rembg.remove accepts bytes)
        with open(input_path, "rb") as f:
            input_bytes = f.read()

        output_bytes = remove(input_bytes)

        with open(output_path, "wb") as f:
            f.write(output_bytes)

        # Verify the output is a valid image with alpha
        img = Image.open(output_path)
        if img.mode != "RGBA":
            img = img.convert("RGBA")
            img.save(str(output_path), format="PNG")

        logger.info(
            "Background removed: %s → %s (%dx%d, mode=%s)",
            input_path.name,
            output_path.name,
            img.width,
            img.height,
            img.mode,
        )
        return True

    except ImportError:
        logger.error("rembg is not installed. Install with: pip install rembg[gpu]")
        return False
    except Exception as exc:
        logger.error("Background removal failed for %s: %s", input_path.name, exc)
        return False


def _fallback_simple_removal(input_path: Path, output_path: Path) -> bool:
    """
    Ultra-simple fallback: convert to RGBA and make near-white pixels transparent.
    Not great quality, but keeps the pipeline running.
    """
    try:
        import numpy as np

        img = Image.open(input_path).convert("RGBA")
        data = np.array(img)

        # Make pixels that are close to white (or very light) transparent
        r, g, b, a = data[:, :, 0], data[:, :, 1], data[:, :, 2], data[:, :, 3]
        white_mask = (r > 220) & (g > 220) & (b > 220)
        data[white_mask, 3] = 0

        result = Image.fromarray(data, "RGBA")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        result.save(str(output_path), format="PNG")
        logger.info("Fallback BG removal saved: %s", output_path.name)
        return True
    except Exception as exc:
        logger.error("Fallback BG removal also failed: %s", exc)
        return False


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def run_bg_remover(project_id: int) -> dict:
    """
    Execute Agent 7: remove backgrounds from all character images.
    """
    logger.info("Agent 7 (BG Remover) starting for project %d", project_id)
    results = {"processed": 0, "failed": 0, "skipped": 0}

    with managed_session() as db:
        characters = (
            db.query(Character)
            .filter(Character.project_id == project_id)
            .all()
        )
        char_list = [
            {
                "id": c.id,
                "name": c.name,
                "raw_image_path": c.raw_image_path,
            }
            for c in characters
        ]

    for cdata in char_list:
        raw_path_str = cdata["raw_image_path"]
        if not raw_path_str:
            logger.warning("Character '%s' has no raw image. Skipping.", cdata["name"])
            results["skipped"] += 1
            continue

        raw_path = Path(raw_path_str)
        if not raw_path.exists():
            logger.warning("Raw image not found: %s. Skipping.", raw_path)
            results["skipped"] += 1
            continue

        output_path = get_workspace_path(
            project_id,
            settings.DIR_IMAGES_TRANSPARENT,
            f"char_{cdata['name'].replace(' ', '_').lower()}_transparent.png",
        )

        success = _remove_background(raw_path, output_path)

        if not success:
            logger.info("Trying fallback simple removal for '%s'…", cdata["name"])
            success = _fallback_simple_removal(raw_path, output_path)

        with managed_session() as db:
            char = db.query(Character).filter(Character.id == cdata["id"]).first()
            if char:
                if success:
                    char.transparent_asset_path = str(output_path)
                    char.status = STATUS_COMPLETED
                    results["processed"] += 1
                else:
                    char.status = STATUS_FAILED
                    results["failed"] += 1

    logger.info(
        "Agent 7 complete. Processed=%d, Failed=%d, Skipped=%d",
        results["processed"],
        results["failed"],
        results["skipped"],
    )
    return results