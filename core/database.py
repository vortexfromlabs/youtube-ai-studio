"""
VortexLabs Pro Studio — Database Layer
SQLAlchemy ORM models for the full 12-Agent pipeline plus session helpers.
Uses SQLite in WAL mode for concurrent reads from multiple agent processes.
"""

import datetime
import json
import logging
from contextlib import contextmanager
from typing import Generator, Optional

from sqlalchemy import (
    Column,
    Integer,
    String,
    Text,
    Float,
    DateTime,
    ForeignKey,
    Enum as SAEnum,
    create_engine,
    event,
    JSON,
)
from sqlalchemy.orm import (
    DeclarativeBase,
    relationship,
    sessionmaker,
    Session,
)

from core.config import settings

logger = logging.getLogger("vortex.database")

# ---------------------------------------------------------------------------
# Engine with WAL pragma for SQLite
# ---------------------------------------------------------------------------
engine = create_engine(
    settings.DATABASE_URL,
    connect_args={"check_same_thread": False},
    echo=False,
    pool_pre_ping=True,
)


@event.listens_for(engine, "connect")
def _set_sqlite_pragma(dbapi_connection, connection_record):
    """Enable WAL mode and foreign keys on every new SQLite connection."""
    cursor = dbapi_connection.cursor()
    cursor.execute("PRAGMA journal_mode=WAL;")
    cursor.execute("PRAGMA foreign_keys=ON;")
    cursor.execute("PRAGMA busy_timeout=5000;")
    cursor.close()


SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


# ---------------------------------------------------------------------------
# Base
# ---------------------------------------------------------------------------
class Base(DeclarativeBase):
    """Shared declarative base for all ORM models."""
    pass


# ---------------------------------------------------------------------------
# Status Enums (stored as strings for portability)
# ---------------------------------------------------------------------------
STATUS_PENDING = "pending"
STATUS_IN_PROGRESS = "in_progress"
STATUS_COMPLETED = "completed"
STATUS_FAILED = "failed"
STATUS_RETRYING = "retrying"

VALID_STATUSES = (
    STATUS_PENDING,
    STATUS_IN_PROGRESS,
    STATUS_COMPLETED,
    STATUS_FAILED,
    STATUS_RETRYING,
)


# ---------------------------------------------------------------------------
# ORM Models
# ---------------------------------------------------------------------------
class Project(Base):
    __tablename__ = "projects"

    id = Column(Integer, primary_key=True, autoincrement=True)
    title = Column(String(512), nullable=False)
    raw_prompt = Column(Text, nullable=False)
    anti_cliche_blueprint = Column(Text, nullable=True)
    full_script = Column(Text, nullable=True)
    status = Column(String(32), default=STATUS_PENDING, nullable=False)
    final_video_path = Column(String(1024), nullable=True)
    youtube_video_id = Column(String(128), nullable=True)
    seo_metadata = Column(Text, nullable=True)
    created_at = Column(DateTime, default=datetime.datetime.utcnow)
    updated_at = Column(
        DateTime,
        default=datetime.datetime.utcnow,
        onupdate=datetime.datetime.utcnow,
    )

    # relationships
    characters = relationship(
        "Character", back_populates="project", cascade="all, delete-orphan"
    )
    locations = relationship(
        "Location", back_populates="project", cascade="all, delete-orphan"
    )
    episodes = relationship(
        "Episode", back_populates="project", cascade="all, delete-orphan"
    )
    shots = relationship(
        "Shot", back_populates="project", cascade="all, delete-orphan"
    )
    bug_reports = relationship(
        "BugReport", back_populates="project", cascade="all, delete-orphan"
    )

    def __repr__(self):
        return f"<Project id={self.id} title={self.title!r} status={self.status}>"


class Character(Base):
    __tablename__ = "characters"

    id = Column(Integer, primary_key=True, autoincrement=True)
    project_id = Column(Integer, ForeignKey("projects.id"), nullable=False)
    name = Column(String(256), nullable=False)
    description = Column(Text, nullable=True)
    visual_prompt = Column(Text, nullable=True)
    voice_id = Column(String(256), nullable=True)
    voice_name = Column(String(256), nullable=True)
    raw_image_path = Column(String(1024), nullable=True)
    transparent_asset_path = Column(String(1024), nullable=True)
    style_keywords = Column(Text, nullable=True)
    status = Column(String(32), default=STATUS_PENDING, nullable=False)
    created_at = Column(DateTime, default=datetime.datetime.utcnow)

    project = relationship("Project", back_populates="characters")

    def __repr__(self):
        return f"<Character id={self.id} name={self.name!r}>"


class Location(Base):
    __tablename__ = "locations"

    id = Column(Integer, primary_key=True, autoincrement=True)
    project_id = Column(Integer, ForeignKey("projects.id"), nullable=False)
    name = Column(String(256), nullable=False)
    description = Column(Text, nullable=True)
    visual_prompt = Column(Text, nullable=True)
    raw_bg_path = Column(String(1024), nullable=True)
    cinematic_bg_path = Column(String(1024), nullable=True)
    animated_bg_path = Column(String(1024), nullable=True)
    status = Column(String(32), default=STATUS_PENDING, nullable=False)
    created_at = Column(DateTime, default=datetime.datetime.utcnow)

    project = relationship("Project", back_populates="locations")

    def __repr__(self):
        return f"<Location id={self.id} name={self.name!r}>"


class Episode(Base):
    __tablename__ = "episodes"

    id = Column(Integer, primary_key=True, autoincrement=True)
    project_id = Column(Integer, ForeignKey("projects.id"), nullable=False)
    episode_number = Column(Integer, nullable=False, default=1)
    title = Column(String(512), nullable=True)
    synopsis = Column(Text, nullable=True)
    status = Column(String(32), default=STATUS_PENDING, nullable=False)
    created_at = Column(DateTime, default=datetime.datetime.utcnow)

    project = relationship("Project", back_populates="episodes")
    shots = relationship(
        "Shot", back_populates="episode", cascade="all, delete-orphan"
    )

    def __repr__(self):
        return f"<Episode id={self.id} number={self.episode_number}>"


class Shot(Base):
    __tablename__ = "shots"

    id = Column(Integer, primary_key=True, autoincrement=True)
    project_id = Column(Integer, ForeignKey("projects.id"), nullable=False)
    episode_id = Column(Integer, ForeignKey("episodes.id"), nullable=True)
    shot_number = Column(Integer, nullable=False)
    description = Column(Text, nullable=True)
    dialogue = Column(Text, nullable=True)
    speaking_character_name = Column(String(256), nullable=True)
    emotion = Column(String(128), nullable=True)
    camera_angle = Column(String(256), nullable=True)
    location_name = Column(String(256), nullable=True)
    duration_seconds = Column(Float, nullable=True)
    audio_path = Column(String(1024), nullable=True)
    character_image_path = Column(String(1024), nullable=True)
    background_image_path = Column(String(1024), nullable=True)
    animated_bg_path = Column(String(1024), nullable=True)
    lip_sync_video_path = Column(String(1024), nullable=True)
    composited_video_path = Column(String(1024), nullable=True)
    status = Column(String(32), default=STATUS_PENDING, nullable=False)
    created_at = Column(DateTime, default=datetime.datetime.utcnow)
    updated_at = Column(
        DateTime,
        default=datetime.datetime.utcnow,
        onupdate=datetime.datetime.utcnow,
    )

    project = relationship("Project", back_populates="shots")
    episode = relationship("Episode", back_populates="shots")

    def __repr__(self):
        return (
            f"<Shot id={self.id} number={self.shot_number} "
            f"status={self.status}>"
        )


class BugReport(Base):
    __tablename__ = "bug_reports"

    id = Column(Integer, primary_key=True, autoincrement=True)
    project_id = Column(Integer, ForeignKey("projects.id"), nullable=False)
    shot_id = Column(Integer, ForeignKey("shots.id"), nullable=True)
    source_agent = Column(String(64), nullable=False)
    target_agent = Column(String(64), nullable=False)
    severity = Column(String(32), default="medium", nullable=False)
    description = Column(Text, nullable=False)
    resolved = Column(Integer, default=0, nullable=False)  # 0=open, 1=resolved
    created_at = Column(DateTime, default=datetime.datetime.utcnow)

    project = relationship("Project", back_populates="bug_reports")

    def __repr__(self):
        return (
            f"<BugReport id={self.id} source={self.source_agent} "
            f"target={self.target_agent} resolved={self.resolved}>"
        )


# ---------------------------------------------------------------------------
# Initialisation helpers
# ---------------------------------------------------------------------------

def init_db():
    """Create all tables if they don't exist yet."""
    Base.metadata.create_all(bind=engine)
    logger.info("Database tables initialised.")


def get_db() -> Generator[Session, None, None]:
    """
    FastAPI-compatible dependency that yields a session and guarantees
    cleanup. Also works as a plain context manager via contextlib.
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


@contextmanager
def managed_session() -> Generator[Session, None, None]:
    """
    Context-manager wrapper around get_db() for use outside FastAPI routes.
    Automatically commits on success and rolls back on exception.
    """
    db = SessionLocal()
    try:
        yield db
        db.commit()
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()


# ---------------------------------------------------------------------------
# Auto-init on import (safe idempotent call)
# ---------------------------------------------------------------------------
init_db()