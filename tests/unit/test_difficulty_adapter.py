"""
Unit Tests for Difficulty Adapter

Tests automatic difficulty adjustment logic.
"""

import pytest
import sys
import os

# Add project root to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.insert(0, os.path.join(project_root, "agentic_socratic_nlp_tutor", "src"))

from agentic_socratic_nlp_tutor.difficulty_adapter import DifficultyAdapter, DifficultyAdjustment


class TestDifficultyAdapter:
    """Test suite for DifficultyAdapter."""
    
    @pytest.fixture
    def adapter(self):
        """Create adapter instance."""
        return DifficultyAdapter()
    
    def test_increase_difficulty_high_scores_improving(self, adapter):
        """Test increasing difficulty when scores are high and improving."""
        adjustment = adapter.check_adjustment(
            current_difficulty="intermediate",
            recent_scores=[0.85, 0.90, 0.88, 0.92, 0.87],
            performance_trend="improving"
        )
        
        assert adjustment.should_adjust == True
        assert adjustment.direction == "increase"
        assert adjustment.new_difficulty == "advanced"
        assert "High performance" in adjustment.reason
    
    def test_decrease_difficulty_low_scores_declining(self, adapter):
        """Test decreasing difficulty when scores are low and declining."""
        adjustment = adapter.check_adjustment(
            current_difficulty="intermediate",
            recent_scores=[0.35, 0.30, 0.28, 0.32, 0.25],
            performance_trend="declining"
        )
        
        assert adjustment.should_adjust == True
        assert adjustment.direction == "decrease"
        assert adjustment.new_difficulty == "beginner"
        assert "Low performance" in adjustment.reason
    
    def test_maintain_difficulty_stable(self, adapter):
        """Test maintaining difficulty when performance is stable."""
        adjustment = adapter.check_adjustment(
            current_difficulty="intermediate",
            recent_scores=[0.65, 0.70, 0.68, 0.72, 0.69],
            performance_trend="stable"
        )
        
        assert adjustment.should_adjust == False
        assert adjustment.direction is None
        assert "stable" in adjustment.reason.lower()
    
    def test_insufficient_data(self, adapter):
        """Test that adjustment requires minimum interactions."""
        adjustment = adapter.check_adjustment(
            current_difficulty="intermediate",
            recent_scores=[0.85, 0.90],  # Only 2 scores
            performance_trend="improving"
        )
        
        assert adjustment.should_adjust == False
        assert "Need at least" in adjustment.reason
    
    def test_already_at_max_difficulty(self, adapter):
        """Test that difficulty doesn't increase beyond advanced."""
        adjustment = adapter.check_adjustment(
            current_difficulty="advanced",
            recent_scores=[0.95, 0.98, 0.96, 0.97, 0.99],
            performance_trend="improving"
        )
        
        # Should not adjust (already at max)
        assert adjustment.should_adjust == False or adjustment.new_difficulty == "advanced"
    
    def test_already_at_min_difficulty(self, adapter):
        """Test that difficulty doesn't decrease below beginner."""
        adjustment = adapter.check_adjustment(
            current_difficulty="beginner",
            recent_scores=[0.15, 0.10, 0.12, 0.08, 0.11],
            performance_trend="declining"
        )
        
        # Should not adjust (already at min)
        assert adjustment.should_adjust == False or adjustment.new_difficulty == "beginner"
    
    def test_calculate_trend_improving(self, adapter):
        """Test trend calculation for improving performance."""
        scores = [0.5, 0.6, 0.7, 0.8, 0.9]
        trend = adapter._calculate_trend(scores)
        assert trend == "improving"
    
    def test_calculate_trend_declining(self, adapter):
        """Test trend calculation for declining performance."""
        scores = [0.9, 0.8, 0.7, 0.6, 0.5]
        trend = adapter._calculate_trend(scores)
        assert trend == "declining"
    
    def test_calculate_trend_stable(self, adapter):
        """Test trend calculation for stable performance."""
        scores = [0.7, 0.68, 0.72, 0.69, 0.71]
        trend = adapter._calculate_trend(scores)
        assert trend == "stable"
    
    def test_raise_difficulty(self, adapter):
        """Test raising difficulty level."""
        assert adapter._raise_difficulty("beginner") == "intermediate"
        assert adapter._raise_difficulty("intermediate") == "advanced"
        assert adapter._raise_difficulty("advanced") == "advanced"  # Max
    
    def test_lower_difficulty(self, adapter):
        """Test lowering difficulty level."""
        assert adapter._lower_difficulty("advanced") == "intermediate"
        assert adapter._lower_difficulty("intermediate") == "beginner"
        assert adapter._lower_difficulty("beginner") == "beginner"  # Min


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

