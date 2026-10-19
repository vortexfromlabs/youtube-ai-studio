"""
Agent 3 — World & Cast Architect
=================================
Reads the full script from Agent 2 and outputs highly detailed visual
generation prompts via Groq (Llama 3.3).

MIGRATED: google-genai -> groq (JSON mode)
FIXED: No longer passes SQLAlchemy Project objects outside managed_session.
"""

import json
import logging
import re
from typing import Any

from groq import Groq

from core.config import settings, get_workspace_path
from core.database import managed_session, Project, Character, Location, STATUS_PENDING

logger = logging.getLogger("vortex.agent3_architect")

_ARCHITECT_SYSTEM_PROMPT = """You are a Visual Development Artist.
Output detailed TEXT-TO-IMAGE prompts based on the screenplay.
OUTPUT STRICT JSON ONLY:
{
  "art_style": "global style",
  "characters": [{"name": "Exact Name", "visual_prompt": "prompt", "voice_tone": "warm", "style_keywords": "tags"}],
  "locations": [{"name": "Location Name", "visual_prompt": "prompt", "mood": "mood"}]
}
Return ONLY valid JSON."""

# ---------------------------------------------------------------------------
# Fallback helpers - PURE PYTHON, NO SQLALCHEMY OBJECTS
# ---------------------------------------------------------------------------

def _fallback_visuals(blueprint: dict[str, Any], script: dict[str, Any]) -> dict[str, Any]:
    """
    Generate procedural visual prompts safely.
    IMPORTANT: Only uses plain dicts/lists (blueprint/script), NEVER SQLAlchemy objects.
    """
    logger.warning("Using procedural fallback for Architect visuals...")

    art_style = blueprint.get("art_style") or blueprint.get("visual_style") or "3D Pixar cartoon style"
    if not isinstance(art_style, str) or not art_style.strip():
        art_style = "3D Pixar cartoon style"

    raw_characters = blueprint.get("characters", [])
    if not isinstance(raw_characters, list):
        raw_characters = []

    if not raw_characters:
        script_chars = set()
        for beat in script.get("beats", []) if isinstance(script, dict) else []:
            if isinstance(beat, dict):
                for line in beat.get("lines", []):
                    if isinstance(line, dict) and line.get("character"):
                        script_chars.add(line["character"])
        if script_chars:
            raw_characters = [{"name": n, "species_or_type": "character"} for n in script_chars]
        else:
            raw_characters = [
                {"name": "Aria", "species_or_type": "human", "personality_traits": ["brave"], "visual_style_notes": "detailed"},
                {"name": "Orion", "species_or_type": "human", "personality_traits": ["loyal"], "visual_style_notes": "detailed"}
            ]

    chars = []
    for c in raw_characters:
        if not isinstance(c, dict):
            continue
        name = str(c.get("name", "Unknown")).strip() or "Unknown"
        species = c.get("species_or_type", c.get("species", "character"))
        personality = ", ".join(c.get("personality_traits", ["brave", "kind"])) if isinstance(c.get("personality_traits"), list) else str(c.get("personality_traits", "brave"))
        visual_notes = c.get("visual_style_notes", c.get("description", "detailed and expressive"))
        voice = c.get("voice_tone", "warm friendly voice")

        chars.append({
            "name": name,
            "visual_prompt": f"{art_style}, detailed character sheet of {name}, a {species}, {visual_notes}, {personality}, plain white background, ultra detailed, 8k, expressive eyes, vibrant colors",
            "voice_tone": voice,
            "style_keywords": f"detailed, expressive, {species}, Pixar style"
        })

    if not chars:
        chars = [{
            "name": "Hero",
            "visual_prompt": f"{art_style}, detailed character sheet of Hero, brave adventurer, plain white background, ultra detailed, Pixar style",
            "voice_tone": "warm",
            "style_keywords": "detailed, expressive"
        }]

    raw_locations = blueprint.get("locations", [])
    if not isinstance(raw_locations, list):
        raw_locations = []

    if not raw_locations:
        potential_locs = set()
        for beat in script.get("beats", []) if isinstance(script, dict) else []:
            if isinstance(beat, dict):
                sd = beat.get("stage_direction", "")
                if "forest" in sd.lower():
                    potential_locs.add("Enchanted Forest")
                elif "cave" in sd.lower():
                    potential_locs.add("Crystal Cave")
        if potential_locs:
            raw_locations = [{"name": n, "description": f"a beautiful {n}", "mood": "mystical"} for n in potential_locs]
        else:
            raw_locations = [
                {"name": "Enchanted Forest", "description": "lush magical forest with glowing mushrooms", "mood": "mystical"},
                {"name": "Crystal Cave", "description": "sparkling cave with crystals", "mood": "mysterious"}
            ]

    locs = []
    for l in raw_locations:
        if not isinstance(l, dict):
            continue
        name = str(l.get("name", "Unknown Location")).strip() or "Unknown Location"
        desc = l.get("description", f"beautiful cinematic location {name}")
        mood = l.get("mood", "cinematic")
        locs.append({
            "name": name,
            "visual_prompt": f"{art_style}, cinematic wide shot of {name}, {desc}, {mood} lighting, high quality, ultra detailed, wide angle, Pixar style, 8k",
            "mood": mood
        })

    if not locs:
        locs = [{
            "name": "Main Area",
            "visual_prompt": f"{art_style}, cinematic wide shot of Main Area, beautiful lighting, high quality, Pixar style",
            "mood": "cinematic"
        }]

    return {"art_style": art_style, "characters": chars, "locations": locs}


def _emergency_fallback_blueprint(raw_prompt: str) -> dict[str, Any]:
    """Create minimal blueprint if none exists in DB."""
    try:
        from phase1_brain.agent1_researcher import _fallback_blueprint as fb1
        return fb1(raw_prompt or "An animated adventure")
    except Exception:
        return {
            "title": "The Adventure",
            "art_style": "3D Pixar cartoon style",
            "characters": [
                {"name": "Aria", "species_or_type": "human", "role": "protagonist", "personality_traits": ["brave"], "visual_style_notes": "Pixar style", "voice_tone": "warm"},
                {"name": "Orion", "species_or_type": "human", "role": "sidekick", "personality_traits": ["loyal"], "visual_style_notes": "Pixar style", "voice_tone": "playful"}
            ],
            "locations": [
                {"name": "Enchanted Forest", "description": "magical forest", "mood": "mystical"},
                {"name": "Crystal Cave", "description": "sparkling cave", "mood": "mysterious"}
            ],
            "plot_summary": "An adventure",
            "themes": ["friendship"],
            "genre": "Adventure"
        }


def _emergency_fallback_script(blueprint: dict[str, Any]) -> dict[str, Any]:
    """Create minimal script if none exists."""
    try:
        from phase1_brain.agent2_scriptwriter import _fallback_script as fb2
        return fb2(blueprint)
    except Exception:
        char_names = [c.get("name", "Hero") for c in blueprint.get("characters", [])[:2] if isinstance(c, dict)]
        if not char_names:
            char_names = ["Aria", "Orion"]
        return {
            "title": blueprint.get("title", "Adventure"),
            "estimated_duration_minutes": 5.0,
            "beats": [
                {
                    "timestamp": "00:00 - 01:00",
                    "stage_direction": "Opening shot, characters meet",
                    "lines": [{"character": char_names[0], "emotion": "wonder", "dialogue": "What an adventure awaits!"}]
                }
            ]
        }


def _call_groq_for_visuals(blueprint: dict[str, Any], script: dict[str, Any]) -> dict[str, Any]:
    """
    Call Groq for visual prompts with JSON mode.
    BULLETPROOF: Catches ALL exceptions and returns fallback.
    """
    try:
        if not settings.GROQ_API_KEY:
            raise RuntimeError("GROQ_API_KEY not set - using fallback")

        client = Groq(api_key=settings.GROQ_API_KEY)
        user_message = f"BLUEPRINT:\n{json.dumps(blueprint, ensure_ascii=False)}\n\nSCRIPT:\n{json.dumps(script, ensure_ascii=False)}"

        response = client.chat.completions.create(
            model=settings.GROQ_MODEL,
            messages=[
                {"role": "system", "content": _ARCHITECT_SYSTEM_PROMPT},
                {"role": "user", "content": user_message},
            ],
            temperature=0.7,
            max_tokens=4096,
            response_format={"type": "json_object"},
        )

        raw_text = response.choices[0].message.content
        if not raw_text or not raw_text.strip():
            logger.warning("Groq returned empty response for visuals, using fallback.")
            return _fallback_visuals(blueprint, script)

        cleaned = raw_text.strip()
        parsed = json.loads(cleaned)

        if not isinstance(parsed, dict):
            raise ValueError("Groq visuals not a dict")

        if not parsed.get("characters") or not isinstance(parsed["characters"], list) or len(parsed["characters"]) == 0:
            logger.warning("Groq visuals missing characters, supplementing with fallback.")
            fallback = _fallback_visuals(blueprint, script)
            parsed["characters"] = fallback["characters"]
            if not parsed.get("art_style"):
                parsed["art_style"] = fallback["art_style"]

        if not parsed.get("locations") or not isinstance(parsed["locations"], list) or len(parsed["locations"]) == 0:
            logger.warning("Groq visuals missing locations, supplementing.")
            fallback = _fallback_visuals(blueprint, script)
            parsed["locations"] = fallback["locations"]
            if not parsed.get("art_style"):
                parsed["art_style"] = fallback["art_style"]

        if not parsed.get("art_style"):
            parsed["art_style"] = blueprint.get("art_style", "3D Pixar cartoon style")

        return parsed

    except Exception as exc:
        logger.error(f"Groq API failed in Agent 3 ({type(exc).__name__}): {exc}", exc_info=True)
        return _fallback_visuals(blueprint, script)


def run_architect(project_id: int) -> dict[str, Any]:
    """
    Execute Agent 3 - 100% BULLETPROOF.
    """
    logger.info("Agent 3 (Architect) starting for project %d", project_id)

    blueprint: dict[str, Any] = {}
    script: dict[str, Any] = {}
    raw_prompt: str = ""
    blueprint_json_raw: str = ""
    script_json_raw: str = ""

    with managed_session() as db:
        project = db.query(Project).filter(Project.id == project_id).first()
        if not project:
            raise ValueError(f"Project {project_id} not found.")
        raw_prompt = project.raw_prompt or ""
        blueprint_json_raw = project.anti_cliche_blueprint or ""
        script_json_raw = project.full_script or ""
        try:
            if blueprint_json_raw:
                blueprint = json.loads(blueprint_json_raw)
            else:
                blueprint = {}
        except Exception as exc:
            logger.warning("Failed to parse blueprint JSON for project %d: %s", project_id, exc)
            blueprint = {}
        try:
            if script_json_raw:
                script = json.loads(script_json_raw)
            else:
                script = {}
        except Exception as exc:
            logger.warning("Failed to parse script JSON for project %d: %s", project_id, exc)
            script = {}

    if not blueprint or not blueprint.get("characters"):
        logger.warning("No valid blueprint for project %d, creating emergency fallback.", project_id)
        blueprint = _emergency_fallback_blueprint(raw_prompt)

    if not script or not script.get("beats"):
        logger.warning("No valid script for project %d, creating emergency fallback script.", project_id)
        script = _emergency_fallback_script(blueprint)

    visuals = _call_groq_for_visuals(blueprint, script)

    if not isinstance(visuals, dict) or not visuals.get("characters"):
        logger.warning("Visuals invalid after Groq, forcing fallback.")
        visuals = _fallback_visuals(blueprint, script)

    try:
        visuals_path = get_workspace_path(project_id, settings.DIR_METADATA, "visual_prompts.json")
        visuals_path.write_text(json.dumps(visuals, indent=2, ensure_ascii=False), encoding="utf-8")
        logger.info("Visuals saved to %s", visuals_path)
    except Exception as exc:
        logger.warning("Failed to write visuals to workspace: %s", exc)

    try:
        with managed_session() as db:
            for char_data in visuals.get("characters", []):
                if not isinstance(char_data, dict):
                    continue
                char_name = str(char_data.get("name", "Unknown")).strip() or "Unknown"
                visual_prompt = char_data.get("visual_prompt", "")
                voice_tone = char_data.get("voice_tone", "")
                existing = db.query(Character).filter(Character.project_id == project_id, Character.name == char_name).first()
                if existing:
                    existing.visual_prompt = visual_prompt
                    existing.voice_name = voice_tone
                    existing.status = STATUS_PENDING
                else:
                    db.add(Character(project_id=project_id, name=char_name, visual_prompt=visual_prompt, voice_name=voice_tone, status=STATUS_PENDING))
            for loc_data in visuals.get("locations", []):
                if not isinstance(loc_data, dict):
                    continue
                loc_name = str(loc_data.get("name", "Unknown")).strip() or "Unknown"
                visual_prompt = loc_data.get("visual_prompt", "")
                existing = db.query(Location).filter(Location.project_id == project_id, Location.name == loc_name).first()
                if existing:
                    existing.visual_prompt = visual_prompt
                    existing.status = STATUS_PENDING
                else:
                    db.add(Location(project_id=project_id, name=loc_name, visual_prompt=visual_prompt, status=STATUS_PENDING))
        logger.info("Agent 3 persisted %d characters and %d locations.", len(visuals.get("characters", [])), len(visuals.get("locations", [])))
    except Exception as exc:
        logger.error("Failed to persist visuals to DB: %s", exc, exc_info=True)
        logger.warning("Continuing pipeline despite DB persistence error in Agent 3.")

    return visuals
