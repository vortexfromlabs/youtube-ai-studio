"""
Agent 3 — World & Cast Architect
=================================
Reads the full script from Agent 2 and outputs highly detailed visual
generation prompts. Includes a bulletproof fallback if the API fails.
"""

import json
import logging
import re
from typing import Any

from google import genai
from google.genai import types as genai_types

from core.config import settings, get_workspace_path
from core.database import managed_session, Project, Character, Location, STATUS_PENDING

logger = logging.getLogger("vortex.agent3_architect")

_ARCHITECT_SYSTEM_PROMPT = """\
You are a Visual Development Artist.
Output detailed TEXT-TO-IMAGE prompts based on the screenplay.
OUTPUT STRICT JSON ONLY:
{
  "art_style": "global style",
  "characters": [{"name": "Exact Name", "visual_prompt": "prompt", "voice_tone": "warm", "style_keywords": "tags"}],
  "locations": [{"name": "Location Name", "visual_prompt": "prompt", "mood": "mood"}]
}
Return ONLY valid JSON.
"""

def _fallback_visuals(blueprint: dict[str, Any], project: Project) -> dict[str, Any]:
    """If Gemini API fails, auto-generate procedural visual prompts safely."""
    logger.warning("Using procedural fallback for Architect visuals...")
    
    # FIX: Pull art_style from the blueprint dictionary, NOT the project DB object
    art_style = blueprint.get("art_style", "3D Pixar cartoon style")
    
    chars = []
    for c in project.characters:
        chars.append({
            "name": c.name,
            "visual_prompt": f"{art_style}, detailed character sheet of {c.name}, plain white background.",
            "voice_tone": "warm",
            "style_keywords": "detailed, expressive"
        })
    locs = []
    for l in project.locations:
        locs.append({
            "name": l.name,
            "visual_prompt": f"{art_style}, cinematic wide shot of {l.name}, beautiful lighting, high quality.",
            "mood": "cinematic"
        })
    return {"art_style": art_style, "characters": chars, "locations": locs}

def _call_gemini_for_visuals(blueprint: dict[str, Any], script: dict[str, Any], project: Project) -> dict[str, Any]:
    try:
        client = genai.Client(api_key=settings.GEMINI_API_KEY)
        user_message = f"BLUEPRINT:\n{json.dumps(blueprint, ensure_ascii=False)}\n\nSCRIPT:\n{json.dumps(script, ensure_ascii=False)}"
        
        response = client.models.generate_content(
            model=settings.GEMINI_MODEL,
            contents=[user_message],
            config=genai_types.GenerateContentConfig(
                system_instruction=_ARCHITECT_SYSTEM_PROMPT,
                temperature=0.7,
                max_output_tokens=4096,
            ),
        )

        if not response or not response.text:
            return _fallback_visuals(blueprint, project)

        cleaned = response.text.strip()
        if cleaned.startswith("```"):
            cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
            cleaned = re.sub(r"\s*```$", "", cleaned)
        return json.loads(cleaned)
        
    except Exception as exc:
        logger.error(f"Gemini API failed in Agent 3: {exc}")
        return _fallback_visuals(blueprint, project)

def run_architect(project_id: int) -> dict[str, Any]:
    logger.info("Agent 3 (Architect) starting for project %d", project_id)

    with managed_session() as db:
        project = db.query(Project).filter(Project.id == project_id).first()
        if not project: raise ValueError("Project not found.")
        
        # Load JSON safely
        blueprint = json.loads(project.anti_cliche_blueprint or "{}")
        script = json.loads(project.full_script or "{}")

    visuals = _call_gemini_for_visuals(blueprint, script, project)

    visuals_path = get_workspace_path(project_id, settings.DIR_METADATA, "visual_prompts.json")
    visuals_path.write_text(json.dumps(visuals, indent=2, ensure_ascii=False), encoding="utf-8")

    with managed_session() as db:
        for char_data in visuals.get("characters", []):
            char_name = char_data.get("name", "Unknown")
            existing = db.query(Character).filter(Character.project_id == project_id, Character.name == char_name).first()
            if existing:
                existing.visual_prompt = char_data.get("visual_prompt", "")
                existing.voice_name = char_data.get("voice_tone", "")
                existing.status = STATUS_PENDING
            else:
                db.add(Character(project_id=project_id, name=char_name, visual_prompt=char_data.get("visual_prompt", ""), status=STATUS_PENDING))

        for loc_data in visuals.get("locations", []):
            loc_name = loc_data.get("name", "Unknown")
            existing = db.query(Location).filter(Location.project_id == project_id, Location.name == loc_name).first()
            if existing:
                existing.visual_prompt = loc_data.get("visual_prompt", "")
                existing.status = STATUS_PENDING
            else:
                db.add(Location(project_id=project_id, name=loc_name, visual_prompt=loc_data.get("visual_prompt", ""), status=STATUS_PENDING))

    return visuals