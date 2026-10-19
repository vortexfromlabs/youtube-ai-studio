"""
Agent 1 — Researcher
====================
Takes the raw user prompt, searches for 3 similar animated stories via
DuckDuckGo, feeds them alongside the prompt to Gemini 2.5 Pro, and
produces an *AntiClicheBlueprint* JSON that guarantees uniqueness.

The blueprint dynamically extracts EXACT character names, settings, and
themes from whatever the user asked for — no hardcoded names ever.
"""

import json
import logging
import re
import time
from typing import Any, Optional

from google import genai
from google.genai import types as genai_types
from ddgs import DDGS

from core.config import settings
from core.database import managed_session, Project, STATUS_IN_PROGRESS, STATUS_COMPLETED, STATUS_FAILED

logger = logging.getLogger("vortex.agent1_researcher")

# ---------------------------------------------------------------------------
# DuckDuckGo search helper
# ---------------------------------------------------------------------------

def _search_competitors(query: str, max_results: int = 3) -> list[dict[str, str]]:
    """
    Search DuckDuckGo for similar animated stories and return a list of
    dicts with keys: title, href, body.
    """
    results: list[dict[str, str]] = []
    search_query = f"animated story similar to: {query}"
    try:
        with DDGS() as ddgs:
            for r in ddgs.text(search_query, max_results=max_results):
                results.append(
                    {
                        "title": r.get("title", ""),
                        "href": r.get("href", ""),
                        "body": r.get("body", ""),
                    }
                )
        logger.info("DuckDuckGo returned %d competitor results.", len(results))
    except Exception as exc:
        logger.warning("DuckDuckGo search failed (%s). Continuing without competitor data.", exc)
    return results


# ---------------------------------------------------------------------------
# Gemini call: produce AntiClicheBlueprint
# ---------------------------------------------------------------------------

_BLUEPRINT_SYSTEM_PROMPT = """\
You are a senior animation story consultant. You receive:
1. A user's original story prompt.
2. A list of existing similar animated stories found online.

Your job is to produce an **AntiClicheBlueprint** — a JSON document that
ensures the final animation is fresh and unique while honouring the user's
exact request.

CRITICAL RULES:
- Extract the EXACT character names mentioned by the user. Do NOT invent or
  substitute names. If the user says "a fox named Kira and a dragon named Blaze",
  the characters list MUST contain "Kira" and "Blaze".
- If the user gives no names, invent fitting original ones.
- Do NOT recycle any character names, plot beats, or visual styles from
  the competitor list.

OUTPUT strictly valid JSON with this schema:
{
  "title": "<unique story title>",
  "genre": "<genre>",
  "tone": "<e.g. whimsical, dark, comedic>",
  "target_audience": "<e.g. children 5-10, young adults>",
  "unique_selling_point": "<one sentence>",
  "themes": ["<theme1>", "<theme2>"],
  "characters": [
    {
      "name": "<exact name from user or invented>",
      "species_or_type": "<e.g. human, fox, robot>",
      "role": "<protagonist / antagonist / sidekick / mentor / etc.>",
      "personality_traits": ["<trait1>", "<trait2>"],
      "visual_style_notes": "<brief appearance description>",
      "voice_tone": "<e.g. warm baritone, squeaky high-pitched>"
    }
  ],
  "locations": [
    {
      "name": "<location name>",
      "description": "<cinematic visual description>",
      "mood": "<e.g. eerie, sunny, mystical>"
    }
  ],
  "plot_summary": "<3-5 sentence unique plot>",
  "anti_cliche_notes": "<what clichés from competitors you avoided and how>"
}

Return ONLY the JSON. No markdown fences, no extra text.
"""


def _call_gemini_for_blueprint(
    user_prompt: str,
    competitor_data: list[dict[str, str]],
) -> dict[str, Any]:
    """
    Call Gemini 2.5 Pro to generate the AntiClicheBlueprint JSON.
    """
    client = genai.Client(api_key=settings.GEMINI_API_KEY)

    competitor_text = ""
    if competitor_data:
        lines = []
        for i, c in enumerate(competitor_data, 1):
            lines.append(f"{i}. **{c['title']}** — {c['body']}")
        competitor_text = "\n".join(lines)
    else:
        competitor_text = "(No competitor data available — ensure maximum originality.)"

    user_message = (
        f"USER PROMPT:\n{user_prompt}\n\n"
        f"EXISTING SIMILAR STORIES FOUND ONLINE:\n{competitor_text}"
    )

    response = client.models.generate_content(
        model=settings.GEMINI_MODEL,
        contents=[user_message],
        config=genai_types.GenerateContentConfig(
            system_instruction=_BLUEPRINT_SYSTEM_PROMPT,
            temperature=0.9,
            max_output_tokens=4096,
        ),
    )

    raw_text = response.text.strip()
    logger.debug("Gemini raw blueprint response length: %d chars", len(raw_text))

    # Strip markdown code fences if present
    cleaned = raw_text
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
        cleaned = re.sub(r"\s*```$", "", cleaned)

    try:
        blueprint = json.loads(cleaned)
    except json.JSONDecodeError as exc:
        logger.error("Failed to parse blueprint JSON: %s", exc)
        logger.error("Raw text was: %s", raw_text[:2000])
        raise ValueError(f"Gemini returned invalid JSON for blueprint: {exc}") from exc

    return blueprint


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def run_researcher(project_id: int) -> dict[str, Any]:
    """
    Execute Agent 1 for the given project.

    1. Read the raw prompt from the database.
    2. Search competitors via DuckDuckGo.
    3. Call Gemini for the AntiClicheBlueprint.
    4. Persist the blueprint JSON back to the Project row.
    5. Return the blueprint dict.
    """
    logger.info("Agent 1 (Researcher) starting for project %d", project_id)

    with managed_session() as db:
        project = db.query(Project).filter(Project.id == project_id).first()
        if project is None:
            raise ValueError(f"Project {project_id} not found in database.")
        raw_prompt = project.raw_prompt
        project.status = STATUS_IN_PROGRESS
        db.commit()

    # Step 1: competitor search
    logger.info("Searching for competitor animated stories…")
    competitors = _search_competitors(raw_prompt, max_results=3)

    # Step 2: Gemini blueprint generation
    logger.info("Calling Gemini for AntiClicheBlueprint…")
    try:
        blueprint = _call_gemini_for_blueprint(raw_prompt, competitors)
    except Exception as exc:
        logger.error("Blueprint generation failed: %s", exc)
        with managed_session() as db:
            project = db.query(Project).filter(Project.id == project_id).first()
            if project:
                project.status = STATUS_FAILED
                db.commit()
        raise

    # Step 3: persist
    blueprint_json_str = json.dumps(blueprint, indent=2, ensure_ascii=False)
    with managed_session() as db:
        project = db.query(Project).filter(Project.id == project_id).first()
        if project:
            project.anti_cliche_blueprint = blueprint_json_str
            db.commit()

    logger.info(
        "Agent 1 complete. Blueprint has %d characters, %d locations.",
        len(blueprint.get("characters", [])),
        len(blueprint.get("locations", [])),
    )
    return blueprint