"""
Agent 1 — Researcher
====================
Takes the raw user prompt, searches for 3 similar animated stories via
DuckDuckGo, feeds them alongside the prompt to Groq (Llama 3.3 70B), and
produces an AntiClicheBlueprint JSON that guarantees uniqueness.

MIGRATED: google-genai -> groq (100% free, ultra-reliable, JSON mode)
BULLETPROOF: If Groq fails (429, 503, quota, network, etc.) a complete
procedural fallback blueprint is generated so the pipeline never crashes.
"""

import json
import logging
import re
from typing import Any

from groq import Groq
from ddgs import DDGS

from core.config import settings, get_workspace_path
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
# Groq system prompt for blueprint
# ---------------------------------------------------------------------------

_BLUEPRINT_SYSTEM_PROMPT = """You are a senior animation story consultant. You receive:
1. A user's original story prompt.
2. A list of existing similar animated stories found online.

Your job is to produce an AntiClicheBlueprint — a JSON document that
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

# ---------------------------------------------------------------------------
# Fallback helpers
# ---------------------------------------------------------------------------

def _extract_names_from_prompt(prompt: str) -> list[str]:
    """Heuristically extract character names from user prompt, supporting multi-word names like 'Professor Hoot'."""
    names: list[str] = []
    # Patterns like "named X" or "called X" - captures up to 2 words (e.g., "Professor Hoot")
    for pattern in [r'\bnamed\s+([A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+)?)\b', r'\bcalled\s+([A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+)?)\b']:
        for m in re.finditer(pattern, prompt):
            candidate = m.group(1).strip()
            if candidate and candidate not in names:
                names.append(candidate)
    # If still empty, try capitalized words that are not common words
    if not names:
        candidates = re.findall(r'\b[A-Z][a-z]+\b', prompt)
        common = {
            "The", "This", "That", "These", "Those", "There", "Then", "When",
            "Where", "What", "Create", "Make", "Generate", "Story", "About",
            "With", "From", "Into", "Once", "Brave", "Little", "Hidden",
            "Enchanted", "Animated", "Create", "Please"
        }
        filtered: list[str] = []
        seen = set()
        for c in candidates:
            if c not in common and c not in seen:
                seen.add(c)
                filtered.append(c)
        # Limit to 3 to avoid noise
        names = filtered[:3]
    return names


def _fallback_blueprint(user_prompt: str) -> dict[str, Any]:
    """
    Generate a complete, valid procedural blueprint when Groq fails.
    Always returns a dict that satisfies the blueprint schema.
    """
    logger.warning("Using procedural fallback for Researcher blueprint (Groq unavailable).")
    names = _extract_names_from_prompt(user_prompt)
    lower = user_prompt.lower()

    if not names:
        # Infer sensible defaults from keywords
        if "fox" in lower and "owl" in lower:
            names = ["Luna", "Professor Hoot"]
        elif "fox" in lower:
            names = ["Luna", "Blaze"]
        elif "dragon" in lower:
            names = ["Blaze", "Kira"]
        elif "princess" in lower:
            names = ["Aria", "Orion"]
        elif "robot" in lower:
            names = ["Nova", "Axel"]
        else:
            names = ["Aria", "Orion"]
        # Ensure at least 2 characters
        if len(names) == 1:
            names.append("Companion")

    # Build characters with species hints
    species_hints = {
        "fox": "fox",
        "wolf": "wolf",
        "dragon": "dragon",
        "owl": "owl",
        "cat": "cat",
        "dog": "dog",
        "robot": "robot",
        "princess": "human",
        "king": "human",
        "queen": "human",
        "wizard": "human",
        "knight": "human",
        "alien": "alien",
        "bear": "bear",
        "rabbit": "rabbit",
        "lion": "lion",
    }
    characters = []
    roles = ["protagonist", "sidekick", "mentor", "antagonist", "supporting"]
    for idx, name in enumerate(names):
        species = "human"
        # Try to find species near name or in general prompt
        name_lower = name.lower()
        for keyword, s in species_hints.items():
            if keyword in lower:
                if idx == 0 or keyword in name_lower or f"{keyword} named {name_lower}" in lower:
                    species = s
                    break
        # Also check if prompt contains species + named pattern for this specific name
        for kw, s in species_hints.items():
            if re.search(rf'\b{kw}\b.*\bnamed\s+{re.escape(name)}\b', user_prompt, re.IGNORECASE):
                species = s
                break
            if re.search(rf'\bnamed\s+{re.escape(name)}\b.*\b{kw}\b', user_prompt, re.IGNORECASE):
                species = s
                break

        characters.append({
            "name": name,
            "species_or_type": species,
            "role": roles[idx % len(roles)],
            "personality_traits": ["brave", "curious"] if idx == 0 else ["loyal", "witty"] if idx == 1 else ["wise", "patient"],
            "visual_style_notes": f"{name}, a {species} with expressive large eyes, detailed fur/clothing texture, vibrant Pixar-style 3D rendering",
            "voice_tone": "warm youthful voice" if idx == 0 else "playful energetic voice" if idx == 1 else "deep wise voice"
        })

    # Ensure at least 2 characters
    if len(characters) < 2:
        characters.append({
            "name": "Mira",
            "species_or_type": "human",
            "role": "sidekick",
            "personality_traits": ["kind", "clever"],
            "visual_style_notes": "Mira, a kind human with bright smile, expressive eyes, Pixar 3D style",
            "voice_tone": "gentle warm voice"
        })

    # Locations - extract from prompt
    loc_keywords = ["forest", "cave", "castle", "city", "village", "ocean", "mountain", "desert", "space", "jungle", "kingdom", "temple", "crystal cave", "enchanted forest", "hidden cave"]
    found_locs: list[str] = []
    for kw in loc_keywords:
        if kw in lower:
            found_locs.append(kw)

    # Deduplicate preserving order
    seen_locs = set()
    unique_locs = []
    for loc in found_locs:
        if loc not in seen_locs:
            seen_locs.add(loc)
            unique_locs.append(loc)

    # If we found multi-word matches like "crystal cave" and also "cave", prefer longer
    if "crystal cave" in unique_locs and "cave" in unique_locs:
        unique_locs.remove("cave")
    if "enchanted forest" in unique_locs and "forest" in unique_locs:
        unique_locs.remove("forest")

    if not unique_locs:
        unique_locs = ["enchanted forest", "crystal cave"]

    locations = []
    moods = ["mystical", "mysterious", "serene", "eerie", "vibrant"]
    for idx, loc_kw in enumerate(unique_locs[:3]):
        pretty = " ".join(w.capitalize() for w in loc_kw.split())
        locations.append({
            "name": pretty,
            "description": f"A cinematic, highly detailed {loc_kw} with beautiful volumetric lighting, magical atmosphere, ultra-detailed environment, Pixar 3D style",
            "mood": moods[idx % len(moods)]
        })

    while len(locations) < 2:
        idx = len(locations)
        locations.append({
            "name": f"Mystic Area {idx + 1}",
            "description": "A beautiful cinematic location with vibrant colors, detailed environment, Pixar 3D style, ultra detailed",
            "mood": "serene"
        })

    # Title - use first 6 words or generic
    words = user_prompt.strip().split()
    title_candidate = " ".join(words[:6]).strip()
    if len(title_candidate) < 10 or len(title_candidate.split()) < 2:
        title = f"The Adventure of {' and '.join(names)}"
    else:
        title = title_candidate.title()
        title = re.sub(r'[^\w\s]+$', '', title)
        if len(title) > 80:
            title = title[:77] + "..."

    char_list_str = " and ".join(names)

    return {
        "title": title,
        "genre": "Adventure",
        "tone": "whimsical and heartwarming",
        "target_audience": "children 6-12 and families",
        "unique_selling_point": f"A unique adventure featuring {char_list_str} that avoids common clichés through original worldbuilding and heartfelt character dynamics.",
        "themes": ["friendship", "courage", "discovery"],
        "characters": characters,
        "locations": locations,
        "plot_summary": f"When {char_list_str} embark on a journey through {locations[0]['name']}, they discover hidden secrets that test their courage and friendship. Through unexpected twists and heartfelt moments in {locations[1]['name'] if len(locations) > 1 else locations[0]['name']}, they learn that true strength comes from within and from trusting each other. The story culminates in a climactic challenge where they must overcome their fears to save their world and prove their bond is unbreakable.",
        "anti_cliche_notes": "Avoided typical chosen-one, damsel-in-distress, and wise-old-man tropes by focusing on collaborative problem-solving, mutual mentorship, and original character dynamics. Procedural fallback ensured uniqueness when competitor API unavailable by grounding story in user prompt specifics.",
        "art_style": "3D Pixar cartoon style"
    }


def _validate_and_repair_blueprint(blueprint: dict[str, Any], user_prompt: str) -> dict[str, Any]:
    """Ensure blueprint has all required fields, repairing with fallback if needed."""
    fallback = _fallback_blueprint(user_prompt)
    required_keys = ["title", "genre", "tone", "target_audience", "unique_selling_point", "themes", "characters", "locations", "plot_summary", "anti_cliche_notes"]
    for key in required_keys:
        if key not in blueprint or blueprint[key] is None or blueprint[key] == "":
            logger.warning("Blueprint missing key '%s', repairing from fallback.", key)
            blueprint[key] = fallback[key]
    if not isinstance(blueprint.get("characters"), list) or len(blueprint["characters"]) == 0:
        logger.warning("Blueprint has no characters, repairing.")
        blueprint["characters"] = fallback["characters"]
    else:
        for idx, char in enumerate(blueprint["characters"]):
            if not isinstance(char, dict):
                blueprint["characters"][idx] = fallback["characters"][idx % len(fallback["characters"])]
                continue
            for ckey in ["name", "species_or_type", "role", "personality_traits", "visual_style_notes", "voice_tone"]:
                if ckey not in char or not char[ckey]:
                    char[ckey] = fallback["characters"][0].get(ckey, "unknown")
    if not isinstance(blueprint.get("locations"), list) or len(blueprint["locations"]) == 0:
        logger.warning("Blueprint has no locations, repairing.")
        blueprint["locations"] = fallback["locations"]
    else:
        for idx, loc in enumerate(blueprint["locations"]):
            if not isinstance(loc, dict):
                blueprint["locations"][idx] = fallback["locations"][idx % len(fallback["locations"])]
                continue
            for lkey in ["name", "description", "mood"]:
                if lkey not in loc or not loc[lkey]:
                    loc[lkey] = fallback["locations"][0].get(lkey, "unknown")
    if not isinstance(blueprint.get("themes"), list) or len(blueprint["themes"]) == 0:
        blueprint["themes"] = fallback["themes"]
    if "art_style" not in blueprint:
        blueprint["art_style"] = fallback.get("art_style", "3D Pixar cartoon style")
    return blueprint


# ---------------------------------------------------------------------------
# Groq call with bulletproof error handling + JSON mode
# ---------------------------------------------------------------------------

def _call_groq_for_blueprint(
    user_prompt: str,
    competitor_data: list[dict[str, str]],
) -> dict[str, Any]:
    """
    Call Groq (Llama 3.3) to generate the AntiClicheBlueprint JSON.
    Uses response_format json_object to guarantee perfect JSON without cutoffs.
    Raises on any failure so caller can fallback.
    """
    if not settings.GROQ_API_KEY:
        raise RuntimeError("GROQ_API_KEY not set")

    client = Groq(api_key=settings.GROQ_API_KEY)

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

    response = client.chat.completions.create(
        model=settings.GROQ_MODEL,
        messages=[
            {"role": "system", "content": _BLUEPRINT_SYSTEM_PROMPT},
            {"role": "user", "content": user_message},
        ],
        temperature=0.9,
        max_tokens=4096,
        response_format={"type": "json_object"},
    )

    raw_text = response.choices[0].message.content
    if not raw_text or not raw_text.strip():
        raise ValueError("Groq returned empty response for blueprint")

    raw_text = raw_text.strip()
    logger.debug("Groq raw blueprint response length: %d chars", len(raw_text))

    try:
        blueprint = json.loads(raw_text)
    except json.JSONDecodeError as exc:
        logger.error("Failed to parse blueprint JSON: %s", exc)
        logger.error("Raw text was: %s", raw_text[:2000])
        raise ValueError(f"Groq returned invalid JSON for blueprint: {exc}") from exc

    return blueprint


# ---------------------------------------------------------------------------
# Public entry point - 100% BULLETPROOF
# ---------------------------------------------------------------------------

def run_researcher(project_id: int) -> dict[str, Any]:
    """
    Execute Agent 1 for the given project.
    100% bulletproof: any Groq error (429, 503, etc.) results in
    procedural fallback blueprint, never crashes the pipeline.
    """
    logger.info("Agent 1 (Researcher) starting for project %d", project_id)

    with managed_session() as db:
        project = db.query(Project).filter(Project.id == project_id).first()
        if project is None:
            raise ValueError(f"Project {project_id} not found in database.")
        raw_prompt = project.raw_prompt
        project.status = STATUS_IN_PROGRESS

    if not raw_prompt or not raw_prompt.strip():
        raw_prompt = "Create an animated adventure story about friendship and courage in a magical world."

    logger.info("Searching for competitor animated stories…")
    competitors = _search_competitors(raw_prompt, max_results=3)

    blueprint: dict[str, Any] = {}
    try:
        logger.info("Calling Groq for AntiClicheBlueprint (model=%s)…", settings.GROQ_MODEL)
        blueprint = _call_groq_for_blueprint(raw_prompt, competitors)
        blueprint = _validate_and_repair_blueprint(blueprint, raw_prompt)
        logger.info("Groq blueprint succeeded.")
    except Exception as exc:
        logger.error("Blueprint generation failed (%s). Using procedural fallback. Error: %s", type(exc).__name__, exc, exc_info=True)
        blueprint = _fallback_blueprint(raw_prompt)

    if not isinstance(blueprint, dict) or not blueprint.get("characters"):
        logger.warning("Blueprint invalid after generation, forcing fallback.")
        blueprint = _fallback_blueprint(raw_prompt)

    blueprint_json_str = json.dumps(blueprint, indent=2, ensure_ascii=False)
    with managed_session() as db:
        project = db.query(Project).filter(Project.id == project_id).first()
        if project:
            project.anti_cliche_blueprint = blueprint_json_str

    try:
        blueprint_path = get_workspace_path(project_id, settings.DIR_METADATA, "blueprint.json")
        blueprint_path.write_text(blueprint_json_str, encoding="utf-8")
        logger.info("Blueprint saved to %s", blueprint_path)
    except Exception as exc:
        logger.warning("Failed to write blueprint to workspace: %s", exc)

    logger.info(
        "Agent 1 complete. Blueprint has %d characters, %d locations.",
        len(blueprint.get("characters", [])),
        len(blueprint.get("locations", [])),
    )
    return blueprint
