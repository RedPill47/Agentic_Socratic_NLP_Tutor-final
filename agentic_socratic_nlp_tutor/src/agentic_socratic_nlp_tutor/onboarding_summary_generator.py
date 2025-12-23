"""
Onboarding Summary Generator

Generates calibration summary and writes back to session state.
"""

from typing import List, Optional
from agentic_socratic_nlp_tutor.onboarding_state import OnboardingState


class OnboardingSummaryGenerator:
    """
    Generates personalized calibration summary after onboarding.
    
    Creates a message that:
    1. Acknowledges student's foundation
    2. Identifies mastered concepts
    3. Sets expectations for learning
    4. Writes back to session state
    """
    
    def generate_summary(self, onboarding_state: OnboardingState) -> str:
        """
        Generate calibration summary message.
        
        Args:
            onboarding_state: Completed onboarding state
            
        Returns:
            Personalized summary message
        """
        if not onboarding_state.is_complete():
            return "Onboarding not yet complete."
        
        mastered = onboarding_state.inferred_mastered_concepts
        goal = onboarding_state.stated_goal or "NLP"
        difficulty = onboarding_state.calibrated_difficulty or "intermediate"
        
        # Build summary
        parts = []
        
        # Opening
        if mastered:
            # Show top 3-5 mastered concepts
            shown_concepts = mastered[:5]
            if len(mastered) > 5:
                concept_list = ", ".join(shown_concepts) + f", and {len(mastered) - 5} more"
            else:
                concept_list = ", ".join(shown_concepts)
            
            parts.append(f"Great! I have a good sense of where you're at.")
            parts.append(f"You have a solid foundation in {concept_list}.")
        else:
            parts.append("Great! I understand your learning goals.")
        
        # Goal reference
        parts.append(f"We'll build on that as we explore {goal}.")
        
        # Difficulty mention (if relevant)
        if difficulty != "intermediate":
            parts.append(f"I'll adjust the complexity to match your {difficulty} level.")
        
        # Closing - don't ask a question, just confirm readiness
        parts.append("Let's begin!")
        
        return " ".join(parts)
    
    def write_to_session_state(
        self,
        onboarding_state: OnboardingState,
        session_state
    ):
        """
        Write calibration results back to session state.
        
        Args:
            onboarding_state: Completed onboarding state
            session_state: SessionState object to update
        """
        if not onboarding_state.is_complete():
            return
        
        # Update difficulty
        if onboarding_state.calibrated_difficulty:
            session_state.difficulty = onboarding_state.calibrated_difficulty
        
        # Update mastered concepts
        if onboarding_state.inferred_mastered_concepts:
            # Merge with existing (avoid duplicates)
            existing = set(session_state.mastered_concepts)
            new_concepts = [
                c for c in onboarding_state.inferred_mastered_concepts
                if c not in existing
            ]
            session_state.mastered_concepts.extend(new_concepts)
        
        # Update prerequisite gaps (if tracked in session state)
        if hasattr(session_state, 'prerequisite_gaps'):
            session_state.prerequisite_gaps = onboarding_state.remaining_gaps
        
        # Mark onboarding complete
        session_state.onboarding_complete = True
        
        # Update stated goal if available
        if onboarding_state.stated_goal:
            session_state.stated_goal = onboarding_state.stated_goal

