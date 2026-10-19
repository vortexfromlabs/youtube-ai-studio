"""
Agent 2 — Scriptwriter
======================
Reads the AntiClicheBlueprint from Agent 1 and uses Gemini to write a
detailed, minute-by-minute script with exact dialogue, emotional cues,
stage directions, and shot hints for every single line.

BULLETPROOF: If Gemini fails (404, 429, 503, etc.) or blueprint is missing,
generates a complete procedural fallback script so pipeline continues.
"""

import json
import logging
import re
from typing import Any

from google import genai
from google.genai import types as genai_types

from core.config import settings, get_workspace_path
from core.database import (
    managed_session,
    Project,
    STATUS_IN_PROGRESS,
    STATUS_COMPLETED,
    STATUS_FAILED,
)

logger = logging.getLogger("vortex.agent2_scriptwriter")

# ---------------------------------------------------------------------------
# Gemini system prompt for scriptwriting
# ---------------------------------------------------------------------------

_SCRIPT_SYSTEM_PROMPT = """\
You are an award-winning animation screenwriter. You receive an
AntiClicheBlueprint JSON describing a unique animated story: its characters,
locations, plot, tone, and themes.

Write a COMPLETE, minute-by-minute screenplay suitable for a 3-to-8 minute
animated video. Follow these rules EXACTLY:

1. **Use ONLY the character names from the blueprint.** Do not rename or
   add characters.
2. Every line of dialogue must include:
   - The character name (in CAPS).
   - An emotional/delivery cue in parentheses, e.g. (excitedly), (whispering).
   - The actual dialogue in quotes.
3. Between dialogue blocks, include brief **stage directions** describing
   action, camera suggestions, and environment.
4. Mark approximate timestamps at the start of each "beat" like
   `[00:00 - 00:30]`, `[00:30 - 01:00]`, etc.
5. Cover an intro, rising action, climax, and resolution.
6. End with "[END]".

OUTPUT FORMAT — return a JSON object:
{
  "title": "<story title>",
  "estimated_duration_minutes": <float>,
  "beats": [
    {
      "timestamp": "<start - end>",
      "stage_direction": "<action and setting description>",
      "lines": [
        {
          "character": "<CHARACTER NAME>",
          "emotion": "<emotion cue>",
          "dialogue": "<the spoken line>"
        }
      ]
    }
  ]
}

Return ONLY valid JSON. No markdown fences, no commentary.
"""

# ---------------------------------------------------------------------------
# Fallback helpers
# ---------------------------------------------------------------------------

def _extract_character_names(blueprint: dict[str, Any]) -> list[str]:
    """Safely extract character names from blueprint dict."""
    chars = blueprint.get("characters", [])
    names = []
    if isinstance(chars, list):
        for c in chars:
            if isinstance(c, dict) and c.get("name"):
                names.append(str(c["name"]).strip())
            elif isinstance(c, str) and c.strip():
                names.append(c.strip())
    return [n for n in names if n]


def _fallback_script(blueprint: dict[str, Any]) -> dict[str, Any]:
    """
    Generate a complete, valid procedural script when Gemini fails.
    Always returns valid JSON with title, estimated_duration_minutes, beats.
    """
    logger.warning("Using procedural fallback for Scriptwriter (Gemini unavailable).")

    title = blueprint.get("title", "The Adventure") if isinstance(blueprint, dict) else "The Adventure"
    plot = blueprint.get("plot_summary", "An adventure of friendship and courage.") if isinstance(blueprint, dict) else "An adventure of friendship and courage."
    char_names = _extract_character_names(blueprint) if isinstance(blueprint, dict) else []

    if not char_names:
        char_names = ["Aria", "Orion"]

    # Ensure at least 2 for dialogue variety
    if len(char_names) == 1:
        char_names.append("Narrator")

    locations = blueprint.get("locations", []) if isinstance(blueprint, dict) else []
    loc_names = []
    if isinstance(locations, list):
        for loc in locations:
            if isinstance(loc, dict) and loc.get("name"):
                loc_names.append(loc["name"])
            elif isinstance(loc, str):
                loc_names.append(loc)
    if not loc_names:
        loc_names = ["Enchanted Forest", "Crystal Cave"]

    primary_loc = loc_names[0]
    secondary_loc = loc_names[1] if len(loc_names) > 1 else loc_names[0]

    # Build 5 beats covering narrative arc
    beats: list[dict[str, Any]] = []

    # Beat 1: Intro - establish world and characters
    beats.append({
        "timestamp": "00:00 - 00:45",
        "stage_direction": f"Opening wide shot of {primary_loc}. Morning light filters through the trees. {char_names[0]} stands at the edge, looking out with curiosity. Camera slowly pushes in.",
        "lines": [
            {
                "character": char_names[0],
                "emotion": "wonder",
                "dialogue": f"I've always wondered what's beyond {primary_loc}. Today feels like the day to find out."
            },
            {
                "character": char_names[1] if len(char_names) > 1 else char_names[0],
                "emotion": "playful",
                "dialogue": "Are you thinking what I'm thinking? Adventure awaits!"
            }
        ]
    })

    # Beat 2: Rising Action - call to adventure
    beats.append({
        "timestamp": "00:45 - 01:45",
        "stage_direction": f"Medium shot, both characters walking through {primary_loc}. They discover a mysterious glowing object or map. Camera circles them.",
        "lines": [
            {
                "character": char_names[0],
                "emotion": "excited",
                "dialogue": "Look at this! It's never been here before. Do you think it leads to the legend?"
            },
            {
                "character": char_names[1] if len(char_names) > 1 else char_names[0],
                "emotion": "curious",
                "dialogue": "There's only one way to find out. Together?"
            },
            {
                "character": char_names[0],
                "emotion": "determined",
                "dialogue": "Together. Let's go."
            }
        ]
    })

    # Beat 3: Rising Action / Challenge - midpoint obstacle
    beats.append({
        "timestamp": "01:45 - 03:00",
        "stage_direction": f"Wide shot of {secondary_loc}. Shadows lengthen. The characters face their first real obstacle — a puzzle, a rift, or a guardian. Tense music builds.",
        "lines": [
            {
                "character": char_names[1] if len(char_names) > 1 else char_names[0],
                "emotion": "worried",
                "dialogue": "This is more dangerous than I thought. Maybe we should turn back?"
            },
            {
                "character": char_names[0],
                "emotion": "encouraging",
                "dialogue": f"We've come so far. I believe in us. Remember why we started — for {primary_loc}, for everyone we care about."
            },
            {
                "character": char_names[1] if len(char_names) > 1 else char_names[0],
                "emotion": "inspired",
                "dialogue": "You're right. We can do this — if we work together."
            }
        ]
    })

    # Beat 4: Climax - confrontation/resolution
    beats.append({
        "timestamp": "03:00 - 04:15",
        "stage_direction": f"Close-ups alternating between characters. They combine strengths to overcome the climax challenge in {secondary_loc}. Bright light, emotional music swell.",
        "lines": [
            {
                "character": char_names[0],
                "emotion": "triumphant",
                "dialogue": "Now! Use what we learned — we have to trust each other!"
            },
            {
                "character": char_names[1] if len(char_names) > 1 else char_names[0],
                "emotion": "joyful",
                "dialogue": "It's working! We did it! I knew we could!"
            }
        ]
    })

    # Beat 5: Resolution - heartwarming conclusion
    beats.append({
        "timestamp": "04:15 - 05:00",
        "stage_direction": f"Wide pull-back shot showing {primary_loc} at sunset, peaceful and beautiful. Characters sit together, reflecting. Camera rises to drone shot. Fade to warm glow.",
        "lines": [
            {
                "character": char_names[0],
                "emotion": "warm",
                "dialogue": "We started this journey looking for treasure, but we found something better."
            },
            {
                "character": char_names[1] if len(char_names) > 1 else char_names[0],
                "emotion": "grateful",
                "dialogue": "Friendship like ours? That's the greatest adventure of all."
            },
            {
                "character": char_names[0],
                "emotion": "hopeful",
                "dialogue": "And our next adventure is just beginning."
            }
        ]
    })

    # Calculate duration ~ 5 minutes
    total_lines = sum(len(b.get("lines", [])) for b in beats)
    estimated = round(4.5 + total_lines * 0.15, 1)

    return {
        "title": title,
        "estimated_duration_minutes": estimated,
        "beats": beats,
        "plot_summary": plot,
        "notes": "Procedural fallback script generated due to Gemini API unavailability. Covers full narrative arc with dialogue for all characters."
    }


def _fallback_blueprint_from_prompt(raw_prompt: str) -> dict[str, Any]:
    """Emergency fallback blueprint when even blueprint is missing."""
    from phase1_brain.agent1_researcher import _fallback_blueprint as _fb1
    try:
        return _fb1(raw_prompt)
    except Exception:
        # Ultra minimal fallback if import fails
        return {
            "title": "The Adventure",
            "genre": "Adventure",
            "tone": "whimsical",
            "target_audience": "general",
            "unique_selling_point": "A heartfelt adventure",
            "themes": ["friendship", "courage"],
            "characters": [
                {"name": "Aria", "species_or_type": "human", "role": "protagonist", "personality_traits": ["brave"], "visual_style_notes": "Pixar style", "voice_tone": "warm"},
                {"name": "Orion", "species_or_type": "human", "role": "sidekick", "personality_traits": ["loyal"], "visual_style_notes": "Pixar style", "voice_tone": "playful"}
            ],
            "locations": [
                {"name": "Enchanted Forest", "description": "magical forest", "mood": "mystical"},
                {"name": "Crystal Cave", "description": "sparkling cave", "mood": "mysterious"}
            ],
            "plot_summary": "An adventure of friendship.",
            "anti_cliche_notes": "Fallback"
        }


# ---------------------------------------------------------------------------
# Gemini call - raises on failure
# ---------------------------------------------------------------------------

def _call_gemini_for_script(blueprint: dict[str, Any]) -> dict[str, Any]:
    """Call Gemini with the blueprint to generate the full script JSON. Raises on any failure."""
    if not settings.GEMINI_API_KEY:
        raise RuntimeError("GEMINI_API_KEY not set")

    client = genai.Client(api_key=settings.GEMINI_API_KEY)

    user_message = (
        "Here is the AntiClicheBlueprint. Write the full screenplay.\n\n"
        + json.dumps(blueprint, indent=2, ensure_ascii=False)
    )

    response = client.models.generate_content(
        model=settings.GEMINI_MODEL,
        contents=[user_message],
        config=genai_types.GenerateContentConfig(
            system_instruction=_SCRIPT_SYSTEM_PROMPT,
            temperature=0.85,
            max_output_tokens=8192,
        ),
    )

    if not response or not hasattr(response, "text") or not response.text:
        raise ValueError("Gemini returned empty response for script")

    raw_text = response.text.strip()
    logger.debug("Gemini raw script response length: %d chars", len(raw_text))

    # Strip markdown fences
    cleaned = raw_text
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
        cleaned = re.sub(r"\s*```$", "", cleaned)

    try:
        script_data = json.loads(cleaned)
    except json.JSONDecodeError as exc:
        logger.error("Failed to parse script JSON: %s", exc)
        logger.error("Raw text (first 3000 chars): %s", raw_text[:3000])
        raise ValueError(f"Gemini returned invalid JSON for script: {exc}") from exc

    # Basic validation
    if not isinstance(script_data, dict) or "beats" not in script_data:
        raise ValueError("Gemini script missing 'beats' field")

    return script_data


# ---------------------------------------------------------------------------
# Public entry point - 100% BULLETPROOF
# ---------------------------------------------------------------------------

def run_scriptwriter(project_id: int) -> dict[str, Any]:
    """
    Execute Agent 2 for the given project.
    100% bulletproof: handles missing blueprint and any Gemini error.

    1. Load the AntiClicheBlueprint from the database (or create fallback).
    2. Call Gemini to generate the full screenplay (or fallback).
    3. Persist the script as both a JSON column and a file.
    4. Return the script dict.
    """
    logger.info("Agent 2 (Scriptwriter) starting for project %d", project_id)

    # Load blueprint inside session - extract to plain dict immediately
    blueprint: dict[str, Any] = {}
    raw_prompt: str = ""
    with managed_session() as db:
        project = db.query(Project).filter(Project.id == project_id).first()
        if project is None:
            raise ValueError(f"Project {project_id} not found.")
        raw_prompt = project.raw_prompt or ""
        if not project.anti_cliche_blueprint:
            logger.warning("Project %d has no blueprint. Will use fallback blueprint.", project_id)
            blueprint = {}
        else:
            try:
                blueprint = json.loads(project.anti_cliche_blueprint)
            except Exception as exc:
                logger.warning("Failed to parse blueprint JSON for project %d: %s. Using fallback.", project_id, exc)
                blueprint = {}

    # If blueprint missing or invalid, create fallback blueprint from raw_prompt
    if not blueprint or not blueprint.get("characters"):
        logger.warning("No valid blueprint for project %d, generating emergency fallback blueprint.", project_id)
        if raw_prompt:
            blueprint = _fallback_blueprint_from_prompt(raw_prompt)
        else:
            blueprint = _fallback_blueprint_from_prompt("Create an animated adventure about friendship")
        # Persist this repaired blueprint for downstream agents
        try:
            with managed_session() as db:
                project = db.query(Project).filter(Project.id == project_id).first()
                if project:
                    project.anti_cliche_blueprint = json.dumps(blueprint, indent=2, ensure_ascii=False)
        except Exception as exc:
            logger.warning("Failed to persist fallback blueprint: %s", exc)

    # Generate script with Gemini + fallback
    logger.info("Calling Gemini for full screenplay (model=%s)…", settings.GEMINI_MODEL)
    script_data: dict[str, Any] = {}
    try:
        script_data = _call_gemini_for_script(blueprint)
        # Validate beats
        if not script_data.get("beats") or not isinstance(script_data["beats"], list) or len(script_data["beats"]) == 0:
            logger.warning("Gemini script has no beats, using fallback.")
            script_data = _fallback_script(blueprint)
    except Exception as exc:
        logger.error("Script generation failed (%s): %s. Using procedural fallback.", type(exc).__name__, exc, exc_info=True)
        script_data = _fallback_script(blueprint)

    # Ensure script_data is valid
    if not isinstance(script_data, dict) or "beats" not in script_data:
        logger.warning("Script data invalid after generation, forcing fallback.")
        script_data = _fallback_script(blueprint)

    # Save to workspace file
    script_json_str = json.dumps(script_data, indent=2, ensure_ascii=False)
    try:
        script_path = get_workspace_path(
            project_id, settings.DIR_SCRIPTS, "full_script.json"
        )
        script_path.write_text(script_json_str, encoding="utf-8")
        logger.info("Script saved to %s", script_path)

        # Also save to metadata for consistency
        meta_path = get_workspace_path(project_id, settings.DIR_METADATA, "script.json")
        meta_path.write_text(script_json_str, encoding="utf-8")
    except Exception as exc:
        logger.warning("Failed to write script to workspace: %s", exc)

    # Persist to DB
    try:
        with managed_session() as db:
            project = db.query(Project).filter(Project.id == project_id).first()
            if project:
                project.full_script = script_json_str
    except Exception as exc:
        logger.error("Failed to persist script to DB: %s", exc)

    beat_count = len(script_data.get("beats", []))
    total_lines = sum(
        len(b.get("lines", [])) for b in script_data.get("beats", []) if isinstance(b, dict)
    )
    logger.info(
        "Agent 2 complete. Script has %d beats and %d dialogue lines.",
        beat_count,
        total_lines,
    )
    return script_data
