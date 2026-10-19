"""
Agent 2 — Scriptwriter
======================
Reads the AntiClicheBlueprint from Agent 1 and uses Gemini to write a
detailed, minute-by-minute script with exact dialogue, emotional cues,
stage directions, and shot hints for every single line.
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


def _call_gemini_for_script(blueprint: dict[str, Any]) -> dict[str, Any]:
    """Call Gemini with the blueprint to generate the full script JSON."""
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

    return script_data


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def run_scriptwriter(project_id: int) -> dict[str, Any]:
    """
    Execute Agent 2 for the given project.

    1. Load the AntiClicheBlueprint from the database.
    2. Call Gemini to generate the full screenplay.
    3. Persist the script as both a JSON column and a file.
    4. Return the script dict.
    """
    logger.info("Agent 2 (Scriptwriter) starting for project %d", project_id)

    # Load blueprint
    with managed_session() as db:
        project = db.query(Project).filter(Project.id == project_id).first()
        if project is None:
            raise ValueError(f"Project {project_id} not found.")
        if not project.anti_cliche_blueprint:
            raise ValueError(
                f"Project {project_id} has no blueprint. Run Agent 1 first."
            )
        blueprint = json.loads(project.anti_cliche_blueprint)

    # Generate script
    logger.info("Calling Gemini for full screenplay…")
    try:
        script_data = _call_gemini_for_script(blueprint)
    except Exception as exc:
        logger.error("Script generation failed: %s", exc)
        with managed_session() as db:
            project = db.query(Project).filter(Project.id == project_id).first()
            if project:
                project.status = STATUS_FAILED
        raise

    # Save to file
    script_json_str = json.dumps(script_data, indent=2, ensure_ascii=False)
    script_path = get_workspace_path(
        project_id, settings.DIR_SCRIPTS, "full_script.json"
    )
    script_path.write_text(script_json_str, encoding="utf-8")
    logger.info("Script saved to %s", script_path)

    # Persist to DB
    with managed_session() as db:
        project = db.query(Project).filter(Project.id == project_id).first()
        if project:
            project.full_script = script_json_str

    beat_count = len(script_data.get("beats", []))
    total_lines = sum(
        len(b.get("lines", [])) for b in script_data.get("beats", [])
    )
    logger.info(
        "Agent 2 complete. Script has %d beats and %d dialogue lines.",
        beat_count,
        total_lines,
    )
    return script_data