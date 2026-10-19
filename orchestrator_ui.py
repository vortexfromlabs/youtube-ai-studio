"""
VortexLabs Pro Studio — Orchestrator & Streamlit UI
====================================================
Terminal 4: The master orchestrator that accepts a user prompt and
sequentially triggers all 12 agents, displaying a real-time progress
log in a Streamlit interface.

Usage:
    streamlit run orchestrator_ui.py
"""

import json
import logging
import sys
import time
import traceback
from datetime import datetime
from io import StringIO
from pathlib import Path
from typing import Any, Callable, Optional

import streamlit as st

# ---------------------------------------------------------------------------
# Must import core first to initialise config and database
# ---------------------------------------------------------------------------
sys.path.insert(0, str(Path(__file__).resolve().parent))

from core.config import settings, get_workspace_path
from core.database import (
    managed_session,
    Project,
    Shot,
    Character,
    Location,
    BugReport,
    STATUS_PENDING,
    STATUS_COMPLETED,
    init_db,
)

# Agent imports
from phase1_brain.agent1_researcher import run_researcher
from phase1_brain.agent2_scriptwriter import run_scriptwriter
from phase1_brain.agent3_architect import run_architect
from phase1_brain.agent4_storyboarder import run_storyboarder
from phase2_production.agent5_audio import run_audio_director
from phase2_production.agent6_concept_art import run_concept_art
from phase2_production.agent7_bg_remover import run_bg_remover
from phase2_production.agent8_animator import run_animator
from phase3_post.agent9_lipsync import run_lipsync
from phase3_post.agent10_compositor import run_compositor
from phase3_post.agent11_qa import run_qa
from phase3_post.agent12_youtube import run_youtube_publisher

logger = logging.getLogger("vortex.orchestrator")

# ---------------------------------------------------------------------------
# Page Configuration
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="VortexLabs Pro Studio",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Custom log handler that captures to a Streamlit-compatible buffer
# ---------------------------------------------------------------------------

class StreamlitLogHandler(logging.Handler):
    """Captures log records into a list for display in the UI."""

    def __init__(self):
        super().__init__()
        self.records: list[str] = []

    def emit(self, record: logging.LogRecord):
        try:
            msg = self.format(record)
            self.records.append(msg)
        except Exception:
            self.handleError(record)

    def get_logs(self) -> str:
        return "\n".join(self.records)

    def clear(self):
        self.records.clear()


# Install the custom handler on root logger
if "log_handler" not in st.session_state:
    handler = StreamlitLogHandler()
    handler.setFormatter(
        logging.Formatter("%(asctime)s | %(name)-28s | %(levelname)-7s | %(message)s")
    )
    logging.getLogger().addHandler(handler)
    st.session_state.log_handler = handler


# ---------------------------------------------------------------------------
# Agent Pipeline Definition
# ---------------------------------------------------------------------------

AGENT_PIPELINE = [
    {
        "number": 1,
        "name": "🔍 Agent 1 — Researcher",
        "description": "Searching for similar stories & generating AntiClicheBlueprint…",
        "function": run_researcher,
        "phase": "Pre-Production",
    },
    {
        "number": 2,
        "name": "✍️ Agent 2 — Scriptwriter",
        "description": "Writing detailed minute-by-minute screenplay…",
        "function": run_scriptwriter,
        "phase": "Pre-Production",
    },
    {
        "number": 3,
        "name": "🎨 Agent 3 — Architect",
        "description": "Generating visual prompts for characters & locations…",
        "function": run_architect,
        "phase": "Pre-Production",
    },
    {
        "number": 4,
        "name": "🎬 Agent 4 — Storyboarder",
        "description": "Breaking script into individual shots…",
        "function": run_storyboarder,
        "phase": "Pre-Production",
    },
    {
        "number": 5,
        "name": "🔊 Agent 5 — Audio Director",
        "description": "Generating dialogue audio for all shots…",
        "function": run_audio_director,
        "phase": "Production",
    },
    {
        "number": 6,
        "name": "🖼️ Agent 6 — Concept Art",
        "description": "Generating character & background images via AI…",
        "function": run_concept_art,
        "phase": "Production",
    },
    {
        "number": 7,
        "name": "✂️ Agent 7 — BG Remover",
        "description": "Removing backgrounds from character images…",
        "function": run_bg_remover,
        "phase": "Production",
    },
    {
        "number": 8,
        "name": "🎥 Agent 8 — Animator",
        "description": "Animating backgrounds with Ken Burns / AI video…",
        "function": run_animator,
        "phase": "Production",
    },
    {
        "number": 9,
        "name": "👄 Agent 9 — Lip Sync",
        "description": "Creating lip-sync animations from audio…",
        "function": run_lipsync,
        "phase": "Post-Production",
    },
    {
        "number": 10,
        "name": "🎞️ Agent 10 — Compositor",
        "description": "Compositing shots & stitching master video…",
        "function": run_compositor,
        "phase": "Post-Production",
    },
    {
        "number": 11,
        "name": "🔍 Agent 11 — QA",
        "description": "Transcribing & validating master video…",
        "function": run_qa,
        "phase": "Post-Production",
    },
    {
        "number": 12,
        "name": "📺 Agent 12 — YouTube Publisher",
        "description": "Generating SEO metadata & uploading…",
        "function": None,  # Special handling
        "phase": "Post-Production",
    },
]


# ---------------------------------------------------------------------------
# Helper: create a new project in the database
# ---------------------------------------------------------------------------

def _create_project(prompt: str) -> int:
    """Create a Project row and return its ID."""
    init_db()
    with managed_session() as db:
        project = Project(
            title=prompt[:200],
            raw_prompt=prompt,
            status=STATUS_PENDING,
        )
        db.add(project)
        db.flush()
        project_id = project.id
    return project_id


# ---------------------------------------------------------------------------
# Main Streamlit UI
# ---------------------------------------------------------------------------

def main():
    """Render the VortexLabs Pro Studio interface."""

    # -- Sidebar --
    with st.sidebar:
        st.image(
            "https://img.icons8.com/color/96/movie-projector.png",
            width=64,
        )
        st.title("VortexLabs Pro Studio")
        st.markdown("---")
        st.markdown("### 🏗️ 12-Agent Pipeline")
        st.markdown(
            """
            **Phase 1 — Pre-Production**
            1. Researcher
            2. Scriptwriter
            3. Architect
            4. Storyboarder

            **Phase 2 — Production**
            5. Audio Director
            6. Concept Art
            7. BG Remover
            8. Animator

            **Phase 3 — Post-Production**
            9. Lip Sync
            10. Compositor
            11. QA
            12. YouTube Publisher
            """
        )
        st.markdown("---")

        # Config status
        warnings = settings.validate()
        if warnings:
            st.warning("⚠️ Configuration warnings:")
            for w in warnings:
                st.caption(f"• {w}")
        else:
            st.success("✅ All API keys configured!")

        st.caption(f"Workspace: `{settings.WORKSPACE_ROOT}`")

    # -- Main content --
    st.title("🎬 VortexLabs Pro Studio")
    st.subheader("Distributed 12-Agent Animation Pipeline")
    st.markdown("---")

    # Prompt input
    prompt = st.text_area(
        "📝 Enter your animation story prompt:",
        height=120,
        placeholder=(
            "Example: Create a 5-minute animated story about a brave little fox "
            "named Luna who discovers a hidden crystal cave guarded by a wise old "
            "owl named Professor Hoot. Set in an enchanted forest with glowing "
            "mushrooms and floating fireflies."
        ),
    )

    col1, col2, col3 = st.columns([1, 1, 2])
    with col1:
        skip_youtube = st.checkbox("Skip YouTube Upload", value=True)
    with col2:
        start_button = st.button(
            "🚀 Launch Pipeline",
            type="primary",
            use_container_width=True,
        )

    st.markdown("---")

    if start_button and prompt.strip():
        # Clear previous logs
        st.session_state.log_handler.clear()

        # Create project
        project_id = _create_project(prompt.strip())
        st.success(f"📁 Project #{project_id} created!")

        # Progress tracking
        total_agents = len(AGENT_PIPELINE)
        progress_bar = st.progress(0, text="Initialising pipeline…")
        status_container = st.container()
        log_expander = st.expander("📋 Real-time Pipeline Log", expanded=True)
        log_placeholder = log_expander.empty()

        results_all: dict[int, Any] = {}
        errors: list[str] = []

        for idx, agent_info in enumerate(AGENT_PIPELINE):
            agent_num = agent_info["number"]
            agent_name = agent_info["name"]
            description = agent_info["description"]
            func = agent_info["function"]
            phase = agent_info["phase"]

            # Update progress
            progress = (idx) / total_agents
            progress_bar.progress(progress, text=f"{agent_name}: {description}")

            with status_container:
                agent_status = st.status(
                    f"{agent_name} — {phase}",
                    expanded=True,
                    state="running",
                )
                agent_status.write(f"⏳ {description}")

            start_time = time.time()

            try:
                if agent_num == 12:
                    # Special handling for YouTube publisher
                    result = run_youtube_publisher(
                        project_id, skip_upload=skip_youtube
                    )
                elif func is not None:
                    result = func(project_id)
                else:
                    result = {"status": "skipped"}

                elapsed = time.time() - start_time
                results_all[agent_num] = result

                with status_container:
                    agent_status.update(
                        label=f"{agent_name} — ✅ Complete ({elapsed:.1f}s)",
                        state="complete",
                        expanded=False,
                    )
                    agent_status.write(f"Result: `{json.dumps(result, default=str)[:500]}`")

            except Exception as exc:
                elapsed = time.time() - start_time
                error_msg = f"{agent_name} failed: {exc}"
                errors.append(error_msg)
                logger.error("%s\n%s", error_msg, traceback.format_exc())

                with status_container:
                    agent_status.update(
                        label=f"{agent_name} — ❌ Failed ({elapsed:.1f}s)",
                        state="error",
                        expanded=True,
                    )
                    agent_status.write(f"Error: `{str(exc)[:500]}`")
                    agent_status.write("Pipeline continuing to next agent…")

            # Update log display
            log_placeholder.code(
                st.session_state.log_handler.get_logs()[-5000:],
                language="text",
            )

        # Final progress
        progress_bar.progress(1.0, text="🎉 Pipeline Complete!")

        st.markdown("---")

        # Results summary
        st.header("📊 Pipeline Results")

        col_pass, col_fail, col_total = st.columns(3)
        with col_pass:
            st.metric("✅ Agents Succeeded", len(results_all))
        with col_fail:
            st.metric("❌ Agents Failed", len(errors))
        with col_total:
            st.metric("📊 Total Agents", total_agents)

        if errors:
            with st.expander("❌ Error Details", expanded=True):
                for err in errors:
                    st.error(err)

        # Show final video if available
        with managed_session() as db:
            project = db.query(Project).filter(Project.id == project_id).first()
            if project and project.final_video_path:
                video_path = Path(project.final_video_path)
                if video_path.exists():
                    st.header("🎬 Final Video")
                    st.video(str(video_path))

                    # Download button (chunked for large files)
                    with open(video_path, "rb") as vf:
                        st.download_button(
                            label="⬇️ Download Master Video",
                            data=vf,
                            file_name=f"vortex_project_{project_id}_master.mp4",
                            mime="video/mp4",
                        )

        # Show database summary
        with st.expander("🗃️ Database Summary"):
            with managed_session() as db:
                shot_count = (
                    db.query(Shot)
                    .filter(Shot.project_id == project_id)
                    .count()
                )
                char_count = (
                    db.query(Character)
                    .filter(Character.project_id == project_id)
                    .count()
                )
                loc_count = (
                    db.query(Location)
                    .filter(Location.project_id == project_id)
                    .count()
                )
                bug_count = (
                    db.query(BugReport)
                    .filter(BugReport.project_id == project_id)
                    .count()
                )

            st.write(f"**Shots:** {shot_count}")
            st.write(f"**Characters:** {char_count}")
            st.write(f"**Locations:** {loc_count}")
            st.write(f"**Bug Reports:** {bug_count}")

            # Show shots table
            with managed_session() as db:
                shots = (
                    db.query(Shot)
                    .filter(Shot.project_id == project_id)
                    .order_by(Shot.shot_number)
                    .all()
                )
                if shots:
                    shot_table = []
                    for s in shots:
                        shot_table.append(
                            {
                                "#": s.shot_number,
                                "Camera": s.camera_angle or "—",
                                "Character": s.speaking_character_name or "—",
                                "Duration": f"{s.duration_seconds:.1f}s"
                                if s.duration_seconds
                                else "—",
                                "Status": s.status,
                                "Audio": "✅" if s.audio_path else "❌",
                                "LipSync": "✅" if s.lip_sync_video_path else "❌",
                                "Composited": "✅"
                                if s.composited_video_path
                                else "❌",
                            }
                        )
                    st.dataframe(shot_table, use_container_width=True)

            # Show bug reports
            with managed_session() as db:
                bugs = (
                    db.query(BugReport)
                    .filter(BugReport.project_id == project_id)
                    .all()
                )
                if bugs:
                    st.subheader("🐛 Bug Reports (from QA)")
                    for b in bugs:
                        severity_icon = {
                            "high": "🔴",
                            "medium": "🟡",
                            "low": "🟢",
                        }.get(b.severity, "⚪")
                        st.write(
                            f"{severity_icon} **{b.target_agent}** ← {b.source_agent}: "
                            f"{b.description}"
                        )

    elif start_button:
        st.warning("Please enter a story prompt before launching the pipeline.")

    # -- Footer --
    st.markdown("---")
    st.caption(
        "VortexLabs Pro Studio v1.0 — 12-Agent Distributed Animation Pipeline  |  "
        "Built with Streamlit, Gemini, ElevenLabs, HuggingFace, FFmpeg, and ❤️"
    )


if __name__ == "__main__":
    main()