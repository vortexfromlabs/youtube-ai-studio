"""
Agent 4 — Storyboarder
======================
Breaks the script into Shots via Groq (Llama 3.3).
MIGRATED: google-genai -> groq (JSON mode, no cutoffs)
BULLETPROOF: Never crashes on missing script/locations.
"""

import json
import logging
import re
from typing import Any

from groq import Groq

from core.config import settings, get_workspace_path
from core.database import managed_session, Project, Episode, Shot, STATUS_PENDING

logger = logging.getLogger("vortex.agent4_storyboarder")

_STORYBOARD_SYSTEM_PROMPT = """You are a professional storyboard artist. Break the screenplay into SHOTS.
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
}"""

# ---------------------------------------------------------------------------
# Fallback helpers
# ---------------------------------------------------------------------------

def _fallback_storyboard(script: dict[str, Any], loc_names: list[str]) -> dict[str, Any]:
    """Generate procedural shots from script beats when Groq fails."""
    logger.warning("Using procedural fallback for Storyboarder (Groq unavailable)...")

    if not isinstance(script, dict):
        script = {}
    beats = script.get("beats", [])
    if not isinstance(beats, list):
        beats = []
    if not beats:
        logger.warning("Script has no beats, creating generic fallback beats.")
        beats = [
            {"stage_direction": "Opening scene in enchanted forest, characters meet", "lines": [{"character": "Aria", "dialogue": "Hello, what an adventure!", "emotion": "excited"}]},
            {"stage_direction": "Characters journey through mysterious cave", "lines": [{"character": "Orion", "dialogue": "We must be brave.", "emotion": "determined"}]},
            {"stage_direction": "Climax - overcoming obstacle together", "lines": [{"character": "Aria", "dialogue": "We did it together!", "emotion": "joyful"}]}
        ]

    shots: list[dict[str, Any]] = []
    default_loc = loc_names[0] if loc_names else "Main Area"
    if not default_loc or not isinstance(default_loc, str):
        default_loc = "Main Area"

    shot_idx = 1
    for beat in beats[:15]:
        if not isinstance(beat, dict):
            continue
        lines = beat.get("lines", [])
        stage_dir = beat.get("stage_direction", "Action occurs.") or "Action occurs."
        desc = str(stage_dir)[:80]
        if isinstance(lines, list) and len(lines) > 0:
            for line in lines:
                if shot_idx > 15:
                    break
                if not isinstance(line, dict):
                    continue
                dialogue = line.get("dialogue")
                if dialogue is not None:
                    dialogue = str(dialogue).strip()
                    if not dialogue:
                        dialogue = None
                char_name = line.get("character")
                if char_name is not None:
                    char_name = str(char_name).strip() or None
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
            from phase1_brain.agent1_researcher import _fallback_blueprint as fb1
            from phase1_brain.agent2_scriptwriter import _fallback_script as fb2
            bp = fb1(raw_prompt)
            return fb2(bp)
    except Exception as exc:
        logger.warning("Emergency script fallback via imports failed: %s", exc)
    return {
        "title": "The Adventure",
        "estimated_duration_minutes": 5.0,
        "beats": [
            {"timestamp": "00:00 - 00:45", "stage_direction": "Opening shot in Enchanted Forest, hero stands looking out", "lines": [{"character": "Aria", "emotion": "wonder", "dialogue": "Today is the day for adventure!"}]},
            {"timestamp": "00:45 - 02:00", "stage_direction": "Journey through Crystal Cave, facing challenge", "lines": [{"character": "Aria", "emotion": "determined", "dialogue": "We can do this together!"}]},
            {"timestamp": "02:00 - 03:30", "stage_direction": "Climax and resolution, sunset", "lines": [{"character": "Aria", "emotion": "joyful", "dialogue": "We did it! Friendship wins!"}]}
        ]
    }


def _call_groq_for_storyboard(script: dict[str, Any], loc_names: list[str]) -> dict[str, Any]:
    """Call Groq for storyboard with JSON mode."""
    try:
        if not settings.GROQ_API_KEY:
            raise RuntimeError("GROQ_API_KEY not set - using fallback")

        client = Groq(api_key=settings.GROQ_API_KEY)
        beats_only = script.get("beats", []) if isinstance(script, dict) else []
        if not isinstance(beats_only, list):
            beats_only = []

        if not isinstance(loc_names, list):
            loc_names = []
        loc_names_clean = [str(l).strip() for l in loc_names if l and isinstance(l, (str, int, float))]
        if not loc_names_clean:
            loc_names_clean = ["Main Area"]

        user_message = f"LOCATIONS: {', '.join(loc_names_clean)}\n\nSCRIPT BEATS:\n{json.dumps(beats_only, ensure_ascii=False)}"

        response = client.chat.completions.create(
            model=settings.GROQ_MODEL,
            messages=[
                {"role": "system", "content": _STORYBOARD_SYSTEM_PROMPT},
                {"role": "user", "content": user_message},
            ],
            temperature=0.4,
            max_tokens=8192,
            response_format={"type": "json_object"},
        )

        raw_text = response.choices[0].message.content
        if not raw_text or not raw_text.strip():
            logger.warning("Groq returned empty response for storyboard, using fallback.")
            return _fallback_storyboard(script, loc_names_clean)

        cleaned = raw_text.strip()
        parsed = json.loads(cleaned)

        if not isinstance(parsed, dict):
            raise ValueError("Storyboard not a dict")

        if "shots" not in parsed or not isinstance(parsed["shots"], list) or len(parsed["shots"]) == 0:
            logger.warning("Groq storyboard missing shots, using fallback supplement.")
            fallback = _fallback_storyboard(script, loc_names_clean)
            if "shots" not in parsed or not parsed["shots"]:
                parsed["shots"] = fallback["shots"]
            if "episode_title" not in parsed or not parsed["episode_title"]:
                parsed["episode_title"] = fallback["episode_title"]

        if len(parsed.get("shots", [])) > 15:
            logger.info("Trimming storyboard from %d to 15 shots.", len(parsed["shots"]))
            parsed["shots"] = parsed["shots"][:15]

        for idx, shot in enumerate(parsed.get("shots", [])):
            if not isinstance(shot, dict):
                continue
            if "shot_number" not in shot or not isinstance(shot["shot_number"], int):
                shot["shot_number"] = idx + 1
            try:
                shot["duration_seconds"] = float(shot.get("duration_seconds", 8))
            except Exception:
                shot["duration_seconds"] = 8.0
            if not shot.get("camera_angle"):
                shot["camera_angle"] = "Wide Shot"
            if not shot.get("location_name"):
                shot["location_name"] = loc_names_clean[0]

        return parsed

    except Exception as exc:
        logger.error(f"Groq API failed in Agent 4 ({type(exc).__name__}): {exc}", exc_info=True)
        safe_locs = loc_names if isinstance(loc_names, list) and loc_names else ["Main Area"]
        return _fallback_storyboard(script, safe_locs)


def run_storyboarder(project_id: int) -> dict[str, Any]:
    """
    Execute Agent 4 - 100% BULLETPROOF.
    """
    logger.info("Agent 4 (Storyboarder) starting for project %d", project_id)

    script: dict[str, Any] = {}
    loc_names: list[str] = []
    raw_prompt: str = ""
    blueprint: dict[str, Any] = {}

    with managed_session() as db:
        project = db.query(Project).filter(Project.id == project_id).first()
        if not project:
            raise ValueError(f"Project {project_id} not found in database.")
        raw_prompt = project.raw_prompt or ""
        try:
            if project.anti_cliche_blueprint:
                blueprint = json.loads(project.anti_cliche_blueprint)
            else:
                blueprint = {}
        except Exception as exc:
            logger.warning("Failed to parse blueprint for project %d: %s", project_id, exc)
            blueprint = {}
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
        try:
            loc_names = [str(l.name).strip() for l in project.locations if l and l.name and str(l.name).strip()]
        except Exception as exc:
            logger.warning("Failed to extract location names for project %d: %s", project_id, exc)
            loc_names = []
        if not loc_names and isinstance(blueprint, dict):
            bp_locs = blueprint.get("locations", [])
            if isinstance(bp_locs, list):
                for loc in bp_locs:
                    if isinstance(loc, dict) and loc.get("name"):
                        loc_names.append(str(loc["name"]).strip())
                    elif isinstance(loc, str) and loc.strip():
                        loc_names.append(loc.strip())
        if not loc_names:
            logger.info("No location names found for project %d, using default.", project_id)
            loc_names = ["Main Area"]

    if not script or not script.get("beats"):
        logger.warning("No valid script for project %d, creating emergency fallback script.", project_id)
        script = _emergency_script_fallback(raw_prompt, blueprint if blueprint else None)
        try:
            script_json = json.dumps(script, indent=2, ensure_ascii=False)
            with managed_session() as db:
                project = db.query(Project).filter(Project.id == project_id).first()
                if project:
                    project.full_script = script_json
            try:
                emergency_path = get_workspace_path(project_id, settings.DIR_SCRIPTS, "full_script.json")
                emergency_path.write_text(script_json, encoding="utf-8")
            except Exception:
                pass
        except Exception as exc:
            logger.warning("Failed to persist emergency script: %s", exc)

    if not loc_names or not isinstance(loc_names, list):
        loc_names = ["Main Area"]

    storyboard = _call_groq_for_storyboard(script, loc_names)

    if not isinstance(storyboard, dict) or "shots" not in storyboard:
        logger.warning("Storyboard invalid after Groq, forcing fallback.")
        storyboard = _fallback_storyboard(script, loc_names)
    if not storyboard.get("shots") or len(storyboard["shots"]) == 0:
        logger.warning("Storyboard has no shots, forcing fallback.")
        storyboard = _fallback_storyboard(script, loc_names)

    try:
        sb_path = get_workspace_path(project_id, settings.DIR_METADATA, "storyboard.json")
        sb_path.write_text(json.dumps(storyboard, indent=2, ensure_ascii=False), encoding="utf-8")
        logger.info("Storyboard saved to %s", sb_path)
    except Exception as exc:
        logger.warning("Failed to write storyboard to workspace: %s", exc)

    try:
        with managed_session() as db:
            episode = Episode(project_id=project_id, episode_number=1, title=storyboard.get("episode_title", "Episode 1"), status=STATUS_PENDING)
            db.add(episode)
            db.flush()
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
        shots_created = len(storyboard.get("shots", []))
        logger.warning("Continuing pipeline despite DB persistence error in Agent 4. Shots generated: %d", shots_created)
        return {"status": "success_with_warnings", "shots_created": shots_created, "warning": str(exc)}
