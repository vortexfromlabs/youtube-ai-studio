"""
VortexLabs Pro Studio — Core Package
Provides configuration loading, database management, and shared utilities
used across all 12 agent microservices.
"""

from core.config import settings, get_workspace_path
from core.database import (
    engine,
    SessionLocal,
    Base,
    get_db,
    init_db,
    Project,
    Character,
    Location,
    Episode,
    Shot,
    BugReport,
)

__all__ = [
    "settings",
    "get_workspace_path",
    "engine",
    "SessionLocal",
    "Base",
    "get_db",
    "init_db",
    "Project",
    "Character",
    "Location",
    "Episode",
    "Shot",
    "BugReport",
]