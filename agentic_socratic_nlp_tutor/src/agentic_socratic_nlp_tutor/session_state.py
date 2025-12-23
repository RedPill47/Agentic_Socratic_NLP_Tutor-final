"""
Session State Data Model

Defines the SessionState dataclass for managing tutoring session state.
"""

from dataclasses import dataclass, field
from typing import Optional, List, Dict
from datetime import datetime


@dataclass
class SessionState:
    """Minimal session state for tutoring."""
    session_id: str
    current_topic: Optional[str] = None
    mastered_concepts: List[str] = field(default_factory=list)
    learning_style: Optional[str] = None
    difficulty: str = "intermediate"
    conversation_history: List[Dict[str, str]] = field(default_factory=list)
    onboarding_complete: bool = False
    interaction_count: int = 0
    stated_goal: Optional[str] = None
    stated_level: Optional[str] = None
    created_at: datetime = field(default_factory=datetime.now)
    last_updated: datetime = field(default_factory=datetime.now)
    # Performance tracking for difficulty adaptation
    understanding_scores: List[float] = field(default_factory=list)  # Recent scores (0-1)
    performance_trend: Optional[str] = None  # "improving", "declining", "stable"
    # Database IDs for persistence
    session_db_id: Optional[str] = None
    user_id: Optional[str] = None
    # RAG source files used in current session
    source_files: List[str] = field(default_factory=list)  # List of source files from RAG

