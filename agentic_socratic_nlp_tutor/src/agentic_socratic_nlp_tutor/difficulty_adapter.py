"""
Automatic Difficulty Adaptation

Automatically adjusts teaching difficulty based on student performance.
Uses recent performance scores and trends to determine if difficulty should change.
"""

from typing import Optional, List
from dataclasses import dataclass


@dataclass
class DifficultyAdjustment:
    """Result of difficulty adjustment check."""
    should_adjust: bool
    direction: Optional[str]  # "increase", "decrease", or None
    reason: str
    new_difficulty: Optional[str] = None


class DifficultyAdapter:
    """
    Automatically adjusts difficulty based on student performance.
    
    Algorithm:
    - Track recent scores (last 5 interactions)
    - If avg > 0.8 and trend "improving" → increase difficulty
    - If avg < 0.4 and trend "declining" → decrease difficulty
    - Otherwise maintain current level
    """
    
    DIFFICULTY_LEVELS = ["beginner", "intermediate", "advanced"]
    
    # Thresholds
    HIGH_SCORE_THRESHOLD = 0.8
    LOW_SCORE_THRESHOLD = 0.4
    MIN_INTERACTIONS = 3  # Need at least 3 interactions to adjust
    
    def check_adjustment(
        self,
        current_difficulty: str,
        recent_scores: List[float],
        performance_trend: Optional[str] = None
    ) -> DifficultyAdjustment:
        """
        Check if difficulty should be adjusted.
        
        Args:
            current_difficulty: Current difficulty level
            recent_scores: List of recent understanding scores (0-1)
            performance_trend: "improving", "declining", "stable", or None
            
        Returns:
            DifficultyAdjustment with recommendation
        """
        # Need minimum interactions
        if len(recent_scores) < self.MIN_INTERACTIONS:
            return DifficultyAdjustment(
                should_adjust=False,
                direction=None,
                reason=f"Need at least {self.MIN_INTERACTIONS} interactions (have {len(recent_scores)})"
            )
        
        # Calculate average score
        avg_score = sum(recent_scores) / len(recent_scores)
        
        # Determine trend if not provided
        if performance_trend is None:
            performance_trend = self._calculate_trend(recent_scores)
        
        # Check for increase
        if avg_score > self.HIGH_SCORE_THRESHOLD and performance_trend == "improving":
            new_difficulty = self._raise_difficulty(current_difficulty)
            if new_difficulty != current_difficulty:
                return DifficultyAdjustment(
                    should_adjust=True,
                    direction="increase",
                    reason=f"High performance (avg={avg_score:.2f}) with improving trend",
                    new_difficulty=new_difficulty
                )
        
        # Check for decrease
        if avg_score < self.LOW_SCORE_THRESHOLD and performance_trend == "declining":
            new_difficulty = self._lower_difficulty(current_difficulty)
            if new_difficulty != current_difficulty:
                return DifficultyAdjustment(
                    should_adjust=True,
                    direction="decrease",
                    reason=f"Low performance (avg={avg_score:.2f}) with declining trend",
                    new_difficulty=new_difficulty
                )
        
        # No adjustment needed
        return DifficultyAdjustment(
            should_adjust=False,
            direction=None,
            reason=f"Performance stable (avg={avg_score:.2f}, trend={performance_trend})"
        )
    
    def _calculate_trend(self, scores: List[float]) -> str:
        """
        Calculate performance trend from scores.
        
        Args:
            scores: List of scores (most recent last)
            
        Returns:
            "improving", "declining", or "stable"
        """
        if len(scores) < 2:
            return "stable"
        
        # Compare first half vs second half
        mid = len(scores) // 2
        first_half_avg = sum(scores[:mid]) / len(scores[:mid])
        second_half_avg = sum(scores[mid:]) / len(scores[mid:])
        
        diff = second_half_avg - first_half_avg
        
        if diff > 0.1:  # Significant improvement
            return "improving"
        elif diff < -0.1:  # Significant decline
            return "declining"
        else:
            return "stable"
    
    def _raise_difficulty(self, current: str) -> str:
        """Raise difficulty level."""
        if current not in self.DIFFICULTY_LEVELS:
            return current
        
        current_idx = self.DIFFICULTY_LEVELS.index(current)
        if current_idx < len(self.DIFFICULTY_LEVELS) - 1:
            return self.DIFFICULTY_LEVELS[current_idx + 1]
        return current  # Already at max
    
    def _lower_difficulty(self, current: str) -> str:
        """Lower difficulty level."""
        if current not in self.DIFFICULTY_LEVELS:
            return current
        
        current_idx = self.DIFFICULTY_LEVELS.index(current)
        if current_idx > 0:
            return self.DIFFICULTY_LEVELS[current_idx - 1]
        return current  # Already at min
    
    def apply_adjustment(
        self,
        state,
        adjustment: DifficultyAdjustment
    ) -> bool:
        """
        Apply difficulty adjustment to session state.
        
        Args:
            state: SessionState object
            adjustment: DifficultyAdjustment result
            
        Returns:
            True if adjustment was applied, False otherwise
        """
        if not adjustment.should_adjust or not adjustment.new_difficulty:
            return False
        
        old_difficulty = state.difficulty
        state.difficulty = adjustment.new_difficulty
        
        # Log adjustment (if logging available)
        print(f"📊 Difficulty adjusted: {old_difficulty} → {adjustment.new_difficulty} ({adjustment.reason})")
        
        return True

