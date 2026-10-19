"""
Agent 12 — YouTube Publisher
=============================
Generates SEO metadata (title, description, tags) using Gemini,
then uses the Google YouTube Data API v3 to chunk-upload the final
master video (safe for 20GB+ files via resumable upload).
"""

import json
import logging
import re
from pathlib import Path
from typing import Any, Optional

from core.config import settings, get_workspace_path, chunked_file_read
from core.database import managed_session, Project, STATUS_COMPLETED

logger = logging.getLogger("vortex.agent12_youtube")

# ---------------------------------------------------------------------------
# SEO metadata generation via Gemini
# ---------------------------------------------------------------------------

_SEO_SYSTEM_PROMPT = """\
You are a YouTube SEO expert for animated content. Given a story title,
plot summary, and character names, generate optimised YouTube metadata.

OUTPUT strictly valid JSON:
{
  "youtube_title": "<catchy title, max 100 chars>",
  "youtube_description": "<detailed description with keywords, 500-2000 chars>",
  "tags": ["<tag1>", "<tag2>", "…up to 20 tags"],
  "category_id": "24",
  "default_language": "en"
}

Return ONLY valid JSON.
"""


def _generate_seo_metadata(
    title: str,
    blueprint: dict[str, Any],
) -> dict[str, Any]:
    """Generate YouTube SEO metadata using Gemini."""
    try:
        from google import genai
        from google.genai import types as genai_types

        client = genai.Client(api_key=settings.GEMINI_API_KEY)

        characters_str = ", ".join(
            c.get("name", "") for c in blueprint.get("characters", [])
        )
        plot = blueprint.get("plot_summary", "An animated story.")
        genre = blueprint.get("genre", "Animation")
        themes = ", ".join(blueprint.get("themes", []))

        user_msg = (
            f"Title: {title}\n"
            f"Genre: {genre}\n"
            f"Characters: {characters_str}\n"
            f"Themes: {themes}\n"
            f"Plot: {plot}"
        )

        response = client.models.generate_content(
            model=settings.GEMINI_MODEL,
            contents=[user_msg],
            config=genai_types.GenerateContentConfig(
                system_instruction=_SEO_SYSTEM_PROMPT,
                temperature=0.7,
                max_output_tokens=2048,
            ),
        )

        raw = response.text.strip()
        if raw.startswith("```"):
            raw = re.sub(r"^```(?:json)?\s*", "", raw)
            raw = re.sub(r"\s*```$", "", raw)

        metadata = json.loads(raw)
        return metadata

    except Exception as exc:
        logger.warning("SEO metadata generation failed: %s. Using defaults.", exc)
        return {
            "youtube_title": title[:100],
            "youtube_description": f"Watch the animated story: {title}",
            "tags": ["animation", "story", "animated short"],
            "category_id": "24",
            "default_language": "en",
        }


# ---------------------------------------------------------------------------
# YouTube resumable upload
# ---------------------------------------------------------------------------

def _upload_to_youtube(
    video_path: Path,
    metadata: dict[str, Any],
) -> Optional[str]:
    """
    Upload video to YouTube using resumable upload (handles 20GB+ files).
    Returns the YouTube video ID on success, None on failure.
    """
    try:
        from google_auth_oauthlib.flow import InstalledAppFlow
        from googleapiclient.discovery import build
        from googleapiclient.http import MediaFileUpload
        import google.auth.transport.requests

        SCOPES = ["https://www.googleapis.com/auth/youtube.upload"]

        secrets_path = Path(settings.YOUTUBE_CLIENT_SECRETS_FILE)
        if not secrets_path.exists():
            logger.error(
                "YouTube client secrets file not found: %s",
                secrets_path,
            )
            return None

        # OAuth2 flow
        flow = InstalledAppFlow.from_client_secrets_file(
            str(secrets_path), SCOPES
        )
        credentials = flow.run_local_server(port=0)

        youtube = build("youtube", "v3", credentials=credentials)

        body = {
            "snippet": {
                "title": metadata.get("youtube_title", "Animated Story"),
                "description": metadata.get("youtube_description", ""),
                "tags": metadata.get("tags", []),
                "categoryId": metadata.get("category_id", "24"),
                "defaultLanguage": metadata.get("default_language", "en"),
            },
            "status": {
                "privacyStatus": "private",  # Start as private for safety
                "selfDeclaredMadeForKids": False,
            },
        }

        # Use resumable upload with chunked transfer for large files
        chunk_size = settings.FILE_CHUNK_SIZE  # 8MB chunks
        media = MediaFileUpload(
            str(video_path),
            mimetype="video/mp4",
            resumable=True,
            chunksize=chunk_size,
        )

        request = youtube.videos().insert(
            part=",".join(body.keys()),
            body=body,
            media_body=media,
        )

        logger.info("Starting YouTube upload: %s (%.1f MB)",
                     video_path.name,
                     video_path.stat().st_size / (1024 * 1024))

        response = None
        while response is None:
            status, response = request.next_chunk()
            if status:
                progress = int(status.progress() * 100)
                logger.info("Upload progress: %d%%", progress)

        video_id = response.get("id", "")
        logger.info("YouTube upload complete! Video ID: %s", video_id)
        logger.info("URL: https://www.youtube.com/watch?v=%s", video_id)
        return video_id

    except ImportError as exc:
        logger.error("Google API libraries not available: %s", exc)
        return None
    except Exception as exc:
        logger.error("YouTube upload failed: %s", exc)
        return None


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def run_youtube_publisher(project_id: int, skip_upload: bool = False) -> dict:
    """
    Execute Agent 12:
    1. Generate SEO metadata.
    2. Upload to YouTube (unless skip_upload=True).
    3. Persist metadata and video ID.
    """
    logger.info("Agent 12 (YouTube Publisher) starting for project %d", project_id)
    results = {"seo_generated": False, "uploaded": False, "video_id": None}

    # Load project data
    with managed_session() as db:
        project = db.query(Project).filter(Project.id == project_id).first()
        if not project:
            raise ValueError(f"Project {project_id} not found.")
        title = project.title
        video_path_str = project.final_video_path
        blueprint_str = project.anti_cliche_blueprint

    blueprint = {}
    if blueprint_str:
        try:
            blueprint = json.loads(blueprint_str)
        except json.JSONDecodeError:
            blueprint = {}

    # Step 1: SEO metadata
    logger.info("Generating YouTube SEO metadata…")
    metadata = _generate_seo_metadata(title, blueprint)
    results["seo_generated"] = True

    # Save metadata
    metadata_path = get_workspace_path(
        project_id, settings.DIR_METADATA, "youtube_seo.json"
    )
    metadata_path.write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    logger.info("SEO metadata saved to %s", metadata_path)

    with managed_session() as db:
        project = db.query(Project).filter(Project.id == project_id).first()
        if project:
            project.seo_metadata = json.dumps(metadata, ensure_ascii=False)

    # Step 2: Upload
    if skip_upload:
        logger.info("YouTube upload skipped (skip_upload=True).")
        return results

    if not video_path_str or not Path(video_path_str).exists():
        logger.error("No master video available for upload.")
        return results

    video_path = Path(video_path_str)
    video_id = _upload_to_youtube(video_path, metadata)

    if video_id:
        results["uploaded"] = True
        results["video_id"] = video_id
        with managed_session() as db:
            project = db.query(Project).filter(Project.id == project_id).first()
            if project:
                project.youtube_video_id = video_id
                project.status = STATUS_COMPLETED

    logger.info("Agent 12 complete. Results: %s", results)
    return results