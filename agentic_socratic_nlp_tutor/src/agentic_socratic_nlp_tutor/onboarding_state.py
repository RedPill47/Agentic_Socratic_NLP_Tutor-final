"""
Onboarding State Management

Explicit state structure and transitions for multi-stage onboarding flow.
"""

from dataclasses import dataclass, field
from typing import Optional, List, Dict, Tuple, Any
from enum import Enum


class OnboardingStage(Enum):
    """Onboarding stages."""
    WELCOME = "welcome"
    LEARNING_STYLE = "learning_style"  # NEW: Learning style assessment
    GOAL_CAPTURED = "goal_captured"  # DEPRECATED: Will be merged into WELCOME
    KNOWLEDGE_ASSESSMENT = "knowledge_assessment"  # Renamed from LEVEL_PROBING
    LEVEL_PROBING = "level_probing"  # DEPRECATED: Use KNOWLEDGE_ASSESSMENT
    PREFERENCES = "preferences"  # NEW: Learning preferences
    CONFIRMATION = "confirmation"  # NEW: Summary and confirmation
    COMPLETE = "complete"


@dataclass
class DiagnosticQuestion:
    """Represents a diagnostic question asked during onboarding."""
    concept: str  # Concept being probed
    question: str  # The question text
    response: Optional[str] = None  # Student's response
    strength_score: Optional[float] = None  # 0-1 score
    strength_category: Optional[str] = None  # "strong", "weak", "moderate"


@dataclass
class OnboardingState:
    """
    Explicit state structure for onboarding flow.
    
    Tracks progress through multi-stage calibration:
    1. Welcome - Initial greeting
    2. Goal Captured - User's goal extracted
    3. Level Probing - Binary search calibration (2-3 questions)
    4. Complete - Calibration done, ready for normal tutoring
    """
    stage: OnboardingStage = OnboardingStage.WELCOME
    
    # Goal information
    stated_goal: Optional[str] = None
    implied_level: Optional[str] = None  # "beginner", "intermediate", "advanced"
    
    # Binary search state
    diagnostic_questions: List[DiagnosticQuestion] = field(default_factory=list)
    current_probing_concept: Optional[str] = None
    binary_search_bounds: Optional[Tuple[str, str]] = None  # (lower_bound, upper_bound)
    
    # Calibration results
    calibrated_difficulty: Optional[str] = None
    inferred_mastered_concepts: List[str] = field(default_factory=list)
    remaining_gaps: List[str] = field(default_factory=list)
    
    # NEW: Learning style (explicitly captured)
    learning_style_question_asked: bool = False
    learning_style_preference: Optional[str] = None  # User's stated preference
    learning_style_detected: Optional[Dict[str, Any]] = None  # Detected from response
    
    # NEW: Learning preferences
    preferences_question_asked: bool = False
    teaching_pace: Optional[str] = None  # fast/detailed/balanced
    practice_preference: Optional[str] = None  # examples/deep_dives/applications
    
    # NEW: Stage tracking
    stages_completed: List[OnboardingStage] = field(default_factory=list)
    
    def transition_to_goal_captured(self, goal: str, implied_level: Optional[str] = None):
        """Transition to goal captured stage."""
        self.stage = OnboardingStage.GOAL_CAPTURED
        self.stated_goal = goal
        self.implied_level = implied_level
    
    def transition_to_level_probing(self):
        """Transition to level probing (binary search) stage."""
        self.stage = OnboardingStage.LEVEL_PROBING
    
    def transition_to_complete(
        self,
        difficulty: str,
        mastered_concepts: List[str],
        gaps: List[str]
    ):
        """Transition to complete stage with calibration results."""
        self.stage = OnboardingStage.COMPLETE
        self.calibrated_difficulty = difficulty
        self.inferred_mastered_concepts = mastered_concepts
        self.remaining_gaps = gaps
    
    def add_diagnostic_question(
        self,
        concept: str,
        question: str,
        response: Optional[str] = None,
        strength_score: Optional[float] = None,
        strength_category: Optional[str] = None
    ):
        """Add a diagnostic question to the history."""
        diagnostic = DiagnosticQuestion(
            concept=concept,
            question=question,
            response=response,
            strength_score=strength_score,
            strength_category=strength_category
        )
        self.diagnostic_questions.append(diagnostic)
        self.current_probing_concept = concept
    
    def update_binary_search_bounds(self, lower: Optional[str] = None, upper: Optional[str] = None):
        """Update binary search bounds."""
        if self.binary_search_bounds is None:
            self.binary_search_bounds = (lower, upper)
        else:
            current_lower, current_upper = self.binary_search_bounds
            self.binary_search_bounds = (
                lower if lower else current_lower,
                upper if upper else current_upper
            )
    
    def get_question_count(self) -> int:
        """Get number of diagnostic questions asked."""
        return len(self.diagnostic_questions)
    
    def should_continue_probing(self, max_questions: int = 4) -> bool:
        """Check if we should continue probing."""
        # Support both old and new stage names for backward compatibility
        if self.stage not in [OnboardingStage.LEVEL_PROBING, OnboardingStage.KNOWLEDGE_ASSESSMENT]:
            return False
        # Require at least 2 questions, allow up to max_questions
        return len(self.diagnostic_questions) < max_questions and len(self.diagnostic_questions) >= 0
    
    def is_complete(self) -> bool:
        """Check if onboarding is complete."""
        return self.stage == OnboardingStage.COMPLETE

