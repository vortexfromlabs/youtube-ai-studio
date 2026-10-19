"""
Agent 4 — Storyboarder
======================
Breaks the script into Shots safely. Includes procedural fallback if API fails.
"""

import json
import logging
import re
from typing import Any

from google import genai
from google.genai import types as genai_types

from core.config import settings, get_workspace_path
from core.database import managed_session, Project, Episode, Shot, STATUS_PENDING

logger = logging.getLogger("vortex.agent4_storyboarder")

_STORYBOARD_SYSTEM_PROMPT = """\
You are a professional storyboard artist. Break the screenplay into SHOTS.
CRITICAL: MAX 15 SHOTS TOTAL.
OUTPUT STRICT JSON ONLY:
{
  "episode_title": "<title>",
  "shots": [
    {
      "shot_number": 1,
      "description": "<ultra brief visual action>",
      "camera_angle": "Wide Shot",
      "location_name": "<exact location>",
      "speaking_character_name": "<character or null>",
      "dialogue": "<dialogue or null>",
      "emotion": "happy",
      "duration_seconds": 8
    }
  ]
}
"""

def _fallback_storyboard(script: dict[str, Any], loc_names: list[str]) -> dict[str, Any]:
    """If Gemini API fails, auto-generate procedural shots from script beats."""
    logger.warning("Using procedural fallback for Storyboarder...")
    beats = script.get("beats", [])
    shots = []
    default_loc = loc_names[0] if loc_names else "Main Area"
    
    shot_idx = 1
    for beat in beats[:15]: # Max 15 shots
        lines = beat.get("lines", [])
        if lines:
            for line in lines:
                shots.append({
                    "shot_number": shot_idx,
                    "description": beat.get("stage_direction", "Action occurs.")[:50],
                    "camera_angle": "Medium Shot",
                    "location_name": default_loc,
                    "speaking_character_name": line.get("character"),
                    "dialogue": line.get("dialogue"),
                    "emotion": line.get("emotion", "neutral"),
                    "duration_seconds": max(5, len(str(line.get("dialogue", "")).split()) * 0.5)
                })
                shot_idx += 1
        else:
            shots.append({
                "shot_number": shot_idx,
                "description": beat.get("stage_direction", "Action occurs.")[:50],
                "camera_angle": "Wide Shot",
                "location_name": default_loc,
                "speaking_character_name": None,
                "dialogue": None,
                "emotion": "neutral",
                "duration_seconds": 6
            })
            shot_idx += 1
            
    return {"episode_title": script.get("title", "Storyboard"), "shots": shots}

def _call_gemini_for_storyboard(script: dict[str, Any], loc_names: list[str]) -> dict[str, Any]:
    try:
        client = genai.Client(api_key=settings.GEMINI_API_KEY)
        beats_only = script.get("beats", [])
        user_message = f"LOCATIONS: {', '.join(loc_names)}\n\nSCRIPT BEATS:\n{json.dumps(beats_only, ensure_ascii=False)}"

        response = client.models.generate_content(
            model=settings.GEMINI_MODEL,
            contents=[user_message],
            config=genai_types.GenerateContentConfig(
                system_instruction=_STORYBOARD_SYSTEM_PROMPT,
                temperature=0.4,
                max_output_tokens=8192,
            ),
        )

        if not response or not response.text:
            return _fallback_storyboard(script, loc_names)

        cleaned = response.text.strip()
        if cleaned.startswith("```"):
            cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
            cleaned = re.sub(r"\s*```$", "", cleaned)
            
        # Fix truncated JSON safely
        if not cleaned.endswith("}"):
            if not cleaned.endswith("]"): cleaned += '"}]}'
            else: cleaned += "}"
            
        return json.loads(cleaned)
    except Exception as exc:
        logger.error(f"Gemini API failed in Agent 4: {exc}")
        return _fallback_storyboard(script, loc_names)

def run_storyboarder(project_id: int) -> dict[str, Any]:
    logger.info("Agent 4 (Storyboarder) starting for project %d", project_id)

    with managed_session() as db:
        project = db.query(Project).filter(Project.id == project_id).first()
        if not project or not project.full_script: raise ValueError("No script found.")
        script = json.loads(project.full_script)
        loc_names = [l.name for l in project.locations if l.name]

    storyboard = _call_gemini_for_storyboard(script, loc_names)

    sb_path = get_workspace_path(project_id, settings.DIR_METADATA, "storyboard.json")
    sb_path.write_text(json.dumps(storyboard, indent=2, ensure_ascii=False), encoding="utf-8")

    with managed_session() as db:
        episode = Episode(project_id=project_id, episode_number=1, title=storyboard.get("episode_title", "Ep 1"), status=STATUS_PENDING)
        db.add(episode)
        db.flush()

        db.query(Shot).filter(Shot.project_id == project_id).delete()
        db.flush()

        created_count = 0
        for s in storyboard.get("shots", []):
            db.add(Shot(
                project_id=project_id, episode_id=episode.id, shot_number=s.get("shot_number", created_count+1),
                description=s.get("description", ""), dialogue=s.get("dialogue"), speaking_character_name=s.get("speaking_character_name"),
                emotion=s.get("emotion"), camera_angle=s.get("camera_angle", "Wide"), location_name=s.get("location_name"),
                duration_seconds=float(s.get("duration_seconds", 8)), status=STATUS_PENDING,
            ))
            created_count += 1

    return {"status": "success", "shots_created": created_count}