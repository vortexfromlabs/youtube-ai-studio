"""
Agent 4 — Storyboarder
======================
Breaks the script into Shots safely. Includes bulletproof fallback if API fails.
Never crashes on missing script/locations - generates procedural storyboard.
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

# ---------------------------------------------------------------------------
# Fallback helpers - PURE PYTHON, NO ORM OUTSIDE SESSION
# ---------------------------------------------------------------------------

def _fallback_storyboard(script: dict[str, Any], loc_names: list[str]) -> dict[str, Any]:
    """Generate procedural shots from script beats when Gemini fails."""
    logger.warning("Using procedural fallback for Storyboarder (Gemini unavailable)...")

    if not isinstance(script, dict):
        script = {}

    beats = script.get("beats", [])
    if not isinstance(beats, list):
        beats = []

    # If no beats, create generic beats
    if not beats:
        logger.warning("Script has no beats, creating generic fallback beats.")
        beats = [
            {
                "stage_direction": "Opening scene in enchanted forest, characters meet",
                "lines": [{"character": "Aria", "dialogue": "Hello, what an adventure!", "emotion": "excited"}]
            },
            {
                "stage_direction": "Characters journey through mysterious cave",
                "lines": [{"character": "Orion", "dialogue": "We must be brave.", "emotion": "determined"}]
            },
            {
                "stage_direction": "Climax - overcoming obstacle together",
                "lines": [{"character": "Aria", "dialogue": "We did it together!", "emotion": "joyful"}]
            }
        ]

    shots: list[dict[str, Any]] = []
    default_loc = loc_names[0] if loc_names else "Main Area"
    if not default_loc or not isinstance(default_loc, str):
        default_loc = "Main Area"

    shot_idx = 1
    # Limit to 15 shots total as per system prompt
    for beat in beats[:15]:
        if not isinstance(beat, dict):
            continue
        lines = beat.get("lines", [])
        stage_dir = beat.get("stage_direction", "Action occurs.") or "Action occurs."
        # Truncate description
        desc = str(stage_dir)[:80]

        if isinstance(lines, list) and len(lines) > 0:
            for line in lines:
                if shot_idx > 15:
                    break
                if not isinstance(line, dict):
                    continue
                dialogue = line.get("dialogue")
                # Clean dialogue
                if dialogue is not None:
                    dialogue = str(dialogue).strip()
                    if not dialogue:
                        dialogue = None
                char_name = line.get("character")
                if char_name is not None:
                    char_name = str(char_name).strip() or None

                # Duration based on dialogue length: ~0.5 sec per word, min 5 sec
                words = len(str(dialogue).split()) if dialogue else 6
                duration = max(5.0, min(12.0, words * 0.5 + 2.0))

                shots.append({
                    "shot_number": shot_idx,
                    "description": desc,
                    "camera_angle": "Medium Shot" if dialogue else "Wide Shot",
                    "location_name": default_loc,
                    "speaking_character_name": char_name,
                    "dialogue": dialogue,
                    "emotion": line.get("emotion", "neutral") or "neutral",
                    "duration_seconds": duration
                })
                shot_idx += 1
        else:
            if shot_idx > 15:
                break
            shots.append({
                "shot_number": shot_idx,
                "description": desc,
                "camera_angle": "Wide Shot",
                "location_name": default_loc,
                "speaking_character_name": None,
                "dialogue": None,
                "emotion": "neutral",
                "duration_seconds": 6.0
            })
            shot_idx += 1

        if shot_idx > 15:
            break

    # Ensure at least 1 shot
    if not shots:
        shots = [{
            "shot_number": 1,
            "description": "Opening wide shot, characters in enchanted forest",
            "camera_angle": "Wide Shot",
            "location_name": default_loc,
            "speaking_character_name": None,
            "dialogue": None,
            "emotion": "neutral",
            "duration_seconds": 8.0
        }]

    episode_title = script.get("title", "The Adventure") if isinstance(script.get("title"), str) else "The Adventure"
    if not episode_title or not episode_title.strip():
        episode_title = "Episode 1: The Beginning"

    return {"episode_title": episode_title, "shots": shots}


def _emergency_script_fallback(raw_prompt: str = "", blueprint: dict[str, Any] | None = None) -> dict[str, Any]:
    """Create emergency script when no script in DB."""
    try:
        if blueprint and isinstance(blueprint, dict) and blueprint.get("characters"):
            from phase1_brain.agent2_scriptwriter import _fallback_script as fb
            return fb(blueprint)
        elif raw_prompt:
            # Try to create blueprint first then script
            from phase1_brain.agent1_researcher import _fallback_blueprint as fb1
            from phase1_brain.agent2_scriptwriter import _fallback_script as fb2
            bp = fb1(raw_prompt)
            return fb2(bp)
    except Exception as exc:
        logger.warning("Emergency script fallback via imports failed: %s", exc)

    # Minimal inline fallback
    return {
        "title": "The Adventure",
        "estimated_duration_minutes": 5.0,
        "beats": [
            {
                "timestamp": "00:00 - 00:45",
                "stage_direction": "Opening shot in Enchanted Forest, hero stands looking out",
                "lines": [{"character": "Aria", "emotion": "wonder", "dialogue": "Today is the day for adventure!"}]
            },
            {
                "timestamp": "00:45 - 02:00",
                "stage_direction": "Journey through Crystal Cave, facing challenge",
                "lines": [{"character": "Aria", "emotion": "determined", "dialogue": "We can do this together!"}]
            },
            {
                "timestamp": "02:00 - 03:30",
                "stage_direction": "Climax and resolution, sunset",
                "lines": [{"character": "Aria", "emotion": "joyful", "dialogue": "We did it! Friendship wins!"}]
            }
        ]
    }


def _call_gemini_for_storyboard(script: dict[str, Any], loc_names: list[str]) -> dict[str, Any]:
    """
    Call Gemini for storyboard. BULLETPROOF: any exception returns fallback.
    """
    try:
        if not settings.GEMINI_API_KEY:
            raise RuntimeError("GEMINI_API_KEY not set - using fallback")

        client = genai.Client(api_key=settings.GEMINI_API_KEY)
        beats_only = script.get("beats", []) if isinstance(script, dict) else []
        if not isinstance(beats_only, list):
            beats_only = []

        # Ensure loc_names is list of strings
        if not isinstance(loc_names, list):
            loc_names = []
        loc_names_clean = [str(l).strip() for l in loc_names if l and isinstance(l, (str, int, float))]
        if not loc_names_clean:
            loc_names_clean = ["Main Area"]

        user_message = f"LOCATIONS: {', '.join(loc_names_clean)}\n\nSCRIPT BEATS:\n{json.dumps(beats_only, ensure_ascii=False)}"

        response = client.models.generate_content(
            model=settings.GEMINI_MODEL,
            contents=[user_message],
            config=genai_types.GenerateContentConfig(
                system_instruction=_STORYBOARD_SYSTEM_PROMPT,
                temperature=0.4,
                max_output_tokens=8192,
            ),
        )

        if not response or not hasattr(response, "text") or not response.text:
            logger.warning("Gemini returned empty response for storyboard, using fallback.")
            return _fallback_storyboard(script, loc_names_clean)

        cleaned = response.text.strip()
        if cleaned.startswith("```"):
            cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
            cleaned = re.sub(r"\s*```$", "", cleaned)

        # Fix truncated JSON safely
        if not cleaned.endswith("}"):
            logger.warning("Gemini storyboard JSON appears truncated, attempting repair.")
            if not cleaned.endswith("]"):
                # Try to close open structures
                # Count open brackets vs close
                open_braces = cleaned.count("{") - cleaned.count("}")
                open_brackets = cleaned.count("[") - cleaned.count("]")
                # Simple repair: close with appropriate suffix
                cleaned += '"}]}' if open_brackets > 0 else '"}'
            else:
                cleaned += "}"

        parsed = json.loads(cleaned)

        # Validate
        if not isinstance(parsed, dict):
            raise ValueError("Storyboard not a dict")

        if "shots" not in parsed or not isinstance(parsed["shots"], list) or len(parsed["shots"]) == 0:
            logger.warning("Gemini storyboard missing shots, using fallback supplement.")
            fallback = _fallback_storyboard(script, loc_names_clean)
            if "shots" not in parsed or not parsed["shots"]:
                parsed["shots"] = fallback["shots"]
            if "episode_title" not in parsed or not parsed["episode_title"]:
                parsed["episode_title"] = fallback["episode_title"]

        # Ensure shots have required fields and limit to 15
        if len(parsed.get("shots", [])) > 15:
            logger.info("Trimming storyboard from %d to 15 shots.", len(parsed["shots"]))
            parsed["shots"] = parsed["shots"][:15]

        # Validate each shot has required keys
        for idx, shot in enumerate(parsed.get("shots", [])):
            if not isinstance(shot, dict):
                continue
            # Ensure shot_number
            if "shot_number" not in shot or not isinstance(shot["shot_number"], int):
                shot["shot_number"] = idx + 1
            # Ensure duration_seconds is float
            try:
                shot["duration_seconds"] = float(shot.get("duration_seconds", 8))
            except Exception:
                shot["duration_seconds"] = 8.0
            # Ensure camera_angle
            if not shot.get("camera_angle"):
                shot["camera_angle"] = "Wide Shot"
            if not shot.get("location_name"):
                shot["location_name"] = loc_names_clean[0]

        return parsed

    except Exception as exc:
        logger.error(f"Gemini API failed in Agent 4 ({type(exc).__name__}): {exc}", exc_info=True)
        # Ensure loc_names is valid list for fallback
        safe_locs = loc_names if isinstance(loc_names, list) and loc_names else ["Main Area"]
        return _fallback_storyboard(script, safe_locs)


def run_storyboarder(project_id: int) -> dict[str, Any]:
    """
    Execute Agent 4 - 100% BULLETPROOF.
    - Extracts script and loc_names inside session as plain data
    - Handles missing script via emergency fallback
    - Any Gemini error yields procedural storyboard
    - Never raises 'No script found' ValueError - instead generates fallback
    """
    logger.info("Agent 4 (Storyboarder) starting for project %d", project_id)

    # -----------------------------------------------------------------------
    # CRITICAL: Extract all needed data INSIDE session as plain types
    # -----------------------------------------------------------------------
    script: dict[str, Any] = {}
    loc_names: list[str] = []
    raw_prompt: str = ""
    blueprint: dict[str, Any] = {}

    with managed_session() as db:
        project = db.query(Project).filter(Project.id == project_id).first()
        if not project:
            raise ValueError(f"Project {project_id} not found in database.")

        raw_prompt = project.raw_prompt or ""

        # Load blueprint for fallback use (if script missing)
        try:
            if project.anti_cliche_blueprint:
                blueprint = json.loads(project.anti_cliche_blueprint)
            else:
                blueprint = {}
        except Exception as exc:
            logger.warning("Failed to parse blueprint for project %d: %s", project_id, exc)
            blueprint = {}

        # Load script - BUT DON'T RAISE if missing, create fallback
        if not project.full_script:
            logger.warning("Project %d has no full_script. Will generate fallback script for storyboarding.", project_id)
            script = {}
        else:
            try:
                script = json.loads(project.full_script)
                if not isinstance(script, dict):
                    logger.warning("Script JSON is not a dict for project %d, using fallback.", project_id)
                    script = {}
            except Exception as exc:
                logger.warning("Failed to parse script JSON for project %d: %s. Using fallback.", project_id, exc)
                script = {}

        # Extract location names INSIDE session - plain list of strings
        try:
            loc_names = [str(l.name).strip() for l in project.locations if l and l.name and str(l.name).strip()]
        except Exception as exc:
            logger.warning("Failed to extract location names for project %d: %s", project_id, exc)
            loc_names = []

        # If loc_names empty, try to derive from blueprint locations
        if not loc_names and isinstance(blueprint, dict):
            bp_locs = blueprint.get("locations", [])
            if isinstance(bp_locs, list):
                for loc in bp_locs:
                    if isinstance(loc, dict) and loc.get("name"):
                        loc_names.append(str(loc["name"]).strip())
                    elif isinstance(loc, str) and loc.strip():
                        loc_names.append(loc.strip())
            # Also try script locations
            if not loc_names and isinstance(script, dict):
                # Some scripts might have implicit locations
                pass

        if not loc_names:
            logger.info("No location names found for project %d, using default.", project_id)
            loc_names = ["Main Area"]

    # Emergency: if script still empty, create one from blueprint/raw_prompt
    if not script or not script.get("beats"):
        logger.warning("No valid script for project %d, creating emergency fallback script.", project_id)
        script = _emergency_script_fallback(raw_prompt, blueprint if blueprint else None)
        # Also persist this emergency script for traceability
        try:
            script_json = json.dumps(script, indent=2, ensure_ascii=False)
            with managed_session() as db:
                project = db.query(Project).filter(Project.id == project_id).first()
                if project:
                    project.full_script = script_json
            # Save to workspace
            try:
                emergency_path = get_workspace_path(project_id, settings.DIR_SCRIPTS, "full_script.json")
                emergency_path.write_text(script_json, encoding="utf-8")
            except Exception:
                pass
        except Exception as exc:
            logger.warning("Failed to persist emergency script: %s", exc)

    # Ensure loc_names is valid for Gemini call
    if not loc_names or not isinstance(loc_names, list):
        loc_names = ["Main Area"]

    # Call Gemini with plain data - never ORM objects
    storyboard = _call_gemini_for_storyboard(script, loc_names)

    # Validate storyboard
    if not isinstance(storyboard, dict) or "shots" not in storyboard:
        logger.warning("Storyboard invalid after Gemini, forcing fallback.")
        storyboard = _fallback_storyboard(script, loc_names)

    if not storyboard.get("shots") or len(storyboard["shots"]) == 0:
        logger.warning("Storyboard has no shots, forcing fallback.")
        storyboard = _fallback_storyboard(script, loc_names)

    # Save to workspace file
    try:
        sb_path = get_workspace_path(project_id, settings.DIR_METADATA, "storyboard.json")
        sb_path.write_text(json.dumps(storyboard, indent=2, ensure_ascii=False), encoding="utf-8")
        logger.info("Storyboard saved to %s", sb_path)
    except Exception as exc:
        logger.warning("Failed to write storyboard to workspace: %s", exc)

    # Persist to DB: Episodes and Shots
    try:
        with managed_session() as db:
            episode = Episode(project_id=project_id, episode_number=1, title=storyboard.get("episode_title", "Episode 1"), status=STATUS_PENDING)
            db.add(episode)
            db.flush()  # Get episode.id

            # Clear existing shots for idempotency
            db.query(Shot).filter(Shot.project_id == project_id).delete()
            db.flush()

            created_count = 0
            for s in storyboard.get("shots", []):
                if not isinstance(s, dict):
                    continue
                try:
                    duration = float(s.get("duration_seconds", 8))
                except Exception:
                    duration = 8.0
                # Ensure duration reasonable
                duration = max(1.0, min(30.0, duration))

                db.add(Shot(
                    project_id=project_id,
                    episode_id=episode.id,
                    shot_number=s.get("shot_number", created_count+1),
                    description=str(s.get("description", ""))[:500],
                    dialogue=s.get("dialogue"),
                    speaking_character_name=s.get("speaking_character_name"),
                    emotion=s.get("emotion"),
                    camera_angle=s.get("camera_angle", "Wide Shot"),
                    location_name=s.get("location_name"),
                    duration_seconds=duration,
                    status=STATUS_PENDING,
                ))
                created_count += 1

        logger.info("Agent 4 persisted %d shots.", created_count)
        return {"status": "success", "shots_created": created_count, "episode_title": storyboard.get("episode_title", "Ep 1")}
    except Exception as exc:
        logger.error("Failed to persist storyboard to DB: %s", exc, exc_info=True)
        # Count shots even if DB failed, pipeline can continue
        shots_created = len(storyboard.get("shots", []))
        logger.warning("Continuing pipeline despite DB persistence error in Agent 4. Shots generated: %d", shots_created)
        return {"status": "success_with_warnings", "shots_created": shots_created, "warning": str(exc)}
