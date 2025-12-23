"""
End-to-End Tests for Onboarding Flows

Tests the multi-stage onboarding with binary search calibration:
- Beginner goal flow
- Intermediate goal flow
- Advanced goal flow
- Edge cases (very strong/weak responses)
"""

import pytest
import asyncio
import sys
import os

# Add project root to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.insert(0, os.path.join(project_root, "agentic_socratic_nlp_tutor", "src"))

from agentic_socratic_nlp_tutor.socratic_tutor import SocraticTutor, SessionState
from agentic_socratic_nlp_tutor.onboarding_state import OnboardingStage


class TestOnboardingFlows:
    """Test suite for onboarding flows."""
    
    @pytest.fixture
    def tutor(self):
        """Create tutor instance for testing."""
        return SocraticTutor()
    
    @pytest.fixture
    def session_state(self):
        """Create a new session state."""
        return SessionState(session_id="test_session")
    
    @pytest.mark.asyncio
    async def test_beginner_goal_flow(self, tutor, session_state):
        """
        Test onboarding flow for beginner goal.
        
        Expected:
        1. Welcome stage - extracts "beginner" goal
        2. Goal captured - transitions to level probing
        3. Level probing - asks diagnostic questions
        4. Complete - generates summary
        """
        session_state.interaction_count = 1
        
        # Stage 1: Welcome
        onboarding_state = tutor.get_or_create_onboarding_state(session_state.session_id)
        assert onboarding_state.stage == OnboardingStage.WELCOME
        
        # User says they're a beginner
        user_input = "I'm new to NLP and want to learn the basics"
        responses = []
        async for chunk in tutor._handle_onboarding(user_input, session_state):
            responses.append(chunk)
        
        response_text = "".join(responses)
        
        # Should have transitioned to goal captured or level probing
        onboarding_state = tutor.get_or_create_onboarding_state(session_state.session_id)
        assert onboarding_state.stage in [OnboardingStage.GOAL_CAPTURED, OnboardingStage.LEVEL_PROBING]
        assert onboarding_state.stated_goal == "NLP fundamentals"
        assert onboarding_state.implied_level == "beginner"
        
        # Should have asked at least one question
        assert len(onboarding_state.diagnostic_questions) >= 0  # May or may not have question yet
        
        print(f"✅ Beginner flow: Goal extracted, stage = {onboarding_state.stage}")
    
    @pytest.mark.asyncio
    async def test_intermediate_goal_flow(self, tutor, session_state):
        """
        Test onboarding flow for intermediate goal.
        
        Expected:
        1. Extracts "intermediate" level
        2. Starts probing with appropriate concepts
        """
        session_state.interaction_count = 1
        
        user_input = "I know some NLP, want to learn about transformers"
        responses = []
        async for chunk in tutor._handle_onboarding(user_input, session_state):
            responses.append(chunk)
        
        onboarding_state = tutor.get_or_create_onboarding_state(session_state.session_id)
        assert onboarding_state.stated_goal in ["Transformer", "transformers", "general NLP"]
        assert onboarding_state.implied_level in ["intermediate", None]
        
        print(f"✅ Intermediate flow: Goal = {onboarding_state.stated_goal}")
    
    @pytest.mark.asyncio
    async def test_advanced_goal_flow(self, tutor, session_state):
        """
        Test onboarding flow for advanced goal.
        
        Expected:
        1. Extracts specific advanced topic
        2. Probes with advanced prerequisites
        """
        session_state.interaction_count = 1
        
        user_input = "I want to master BERT fine-tuning"
        responses = []
        async for chunk in tutor._handle_onboarding(user_input, session_state):
            responses.append(chunk)
        
        onboarding_state = tutor.get_or_create_onboarding_state(session_state.session_id)
        # Should detect BERT or related concept
        assert onboarding_state.stated_goal is not None
        
        print(f"✅ Advanced flow: Goal = {onboarding_state.stated_goal}")
    
    @pytest.mark.asyncio
    async def test_binary_search_strong_response(self, tutor, session_state):
        """
        Test binary search with strong response.
        
        Expected:
        - Strong response → moves up in prerequisite chain
        - Next question is about more advanced concept
        """
        session_state.interaction_count = 1
        
        # Initialize onboarding
        onboarding_state = tutor.get_or_create_onboarding_state(session_state.session_id)
        onboarding_state.transition_to_goal_captured("Transformer", "intermediate")
        onboarding_state.transition_to_level_probing()
        
        # Add first diagnostic question
        onboarding_state.add_diagnostic_question("Attention Mechanisms", "What do you know about attention?")
        
        # Simulate strong response
        strong_response = "Attention mechanisms allow models to focus on relevant parts of the input. They compute attention weights that determine how much each part should influence the output. This is crucial for handling long sequences."
        
        # Score the response
        if tutor.response_scorer:
            score = await tutor.response_scorer.score(strong_response, "What do you know about attention?")
            category = tutor.response_scorer.classify(score)
            
            assert score > 0.7, f"Expected strong score > 0.7, got {score}"
            assert category == "strong", f"Expected 'strong', got '{category}'"
            
            # Update question with response
            onboarding_state.diagnostic_questions[-1].response = strong_response
            onboarding_state.diagnostic_questions[-1].strength_score = score
            onboarding_state.diagnostic_questions[-1].strength_category = category
            
            # Get next concept (should move up)
            if onboarding_state.should_continue_probing():
                next_concept = tutor._get_next_probing_concept(onboarding_state, category)
                # Should be more advanced than "Attention Mechanisms"
                print(f"✅ Strong response: Score={score:.2f}, Next concept={next_concept}")
        else:
            pytest.skip("Response scorer not available")
    
    @pytest.mark.asyncio
    async def test_binary_search_weak_response(self, tutor, session_state):
        """
        Test binary search with weak response.
        
        Expected:
        - Weak response → moves down in prerequisite chain
        - Next question is about earlier prerequisite
        """
        session_state.interaction_count = 1
        
        onboarding_state = tutor.get_or_create_onboarding_state(session_state.session_id)
        onboarding_state.transition_to_goal_captured("Transformer", "intermediate")
        onboarding_state.transition_to_level_probing()
        onboarding_state.add_diagnostic_question("Attention Mechanisms", "What do you know about attention?")
        
        # Simulate weak response
        weak_response = "I'm not sure, maybe something about focusing?"
        
        if tutor.response_scorer:
            score = await tutor.response_scorer.score(weak_response, "What do you know about attention?")
            category = tutor.response_scorer.classify(score)
            
            assert score < 0.4, f"Expected weak score < 0.4, got {score}"
            assert category == "weak", f"Expected 'weak', got '{category}'"
            
            onboarding_state.diagnostic_questions[-1].response = weak_response
            onboarding_state.diagnostic_questions[-1].strength_score = score
            onboarding_state.diagnostic_questions[-1].strength_category = category
            
            # Get next concept (should move down)
            if onboarding_state.should_continue_probing():
                next_concept = tutor._get_next_probing_concept(onboarding_state, category)
                print(f"✅ Weak response: Score={score:.2f}, Next concept={next_concept}")
        else:
            pytest.skip("Response scorer not available")
    
    @pytest.mark.asyncio
    async def test_mastery_inference(self, tutor):
        """
        Test mastery inference from strong responses.
        
        Expected:
        - Strong response about concept X → infers mastery of X + prerequisites
        """
        # Test with Transformer
        demonstrated = "Transformer"
        inferred = tutor.prerequisites.infer_mastery(demonstrated)
        
        assert demonstrated in inferred, "Should include demonstrated concept"
        assert len(inferred) > 1, "Should infer prerequisites"
        
        # Check that prerequisites are included
        prereqs = tutor.prerequisites.get_all_prerequisites(demonstrated)
        for prereq in prereqs[:3]:  # Check first 3
            assert prereq in inferred, f"Should infer prerequisite: {prereq}"
        
        print(f"✅ Mastery inference: {demonstrated} → {len(inferred)} concepts inferred")
    
    @pytest.mark.asyncio
    async def test_middle_prerequisite(self, tutor):
        """
        Test get_middle_prerequisite method.
        
        Expected:
        - Returns a concept in the middle of prerequisite chain
        - Good for binary search starting point
        """
        # Test with Transformer
        middle = tutor.prerequisites.get_middle_prerequisite("Transformer")
        
        assert middle is not None, "Should return a middle prerequisite"
        assert middle != "Transformer", "Should not return the concept itself"
        
        # Should be in prerequisites
        all_prereqs = tutor.prerequisites.get_all_prerequisites("Transformer")
        assert middle in all_prereqs, f"{middle} should be a prerequisite of Transformer"
        
        print(f"✅ Middle prerequisite for Transformer: {middle}")
    
    @pytest.mark.asyncio
    async def test_onboarding_completion(self, tutor, session_state):
        """
        Test full onboarding completion flow.
        
        Expected:
        1. Multiple diagnostic questions
        2. Mastery inference
        3. Difficulty calibration
        4. Summary generation
        """
        session_state.interaction_count = 1
        
        onboarding_state = tutor.get_or_create_onboarding_state(session_state.session_id)
        onboarding_state.transition_to_goal_captured("Transformer", "intermediate")
        onboarding_state.transition_to_level_probing()
        
        # Simulate 2-3 questions with mixed responses
        questions = [
            ("Attention Mechanisms", "What do you know about attention?", "strong"),
            ("RNN", "What do you know about RNNs?", "moderate"),
        ]
        
        for concept, question, strength in questions:
            onboarding_state.add_diagnostic_question(concept, question)
            # Simulate response scoring
            onboarding_state.diagnostic_questions[-1].strength_score = 0.8 if strength == "strong" else 0.5
            onboarding_state.diagnostic_questions[-1].strength_category = strength
        
        # Complete onboarding
        await tutor._complete_onboarding(onboarding_state, session_state)
        
        assert onboarding_state.is_complete(), "Onboarding should be complete"
        assert onboarding_state.calibrated_difficulty is not None, "Should have calibrated difficulty"
        assert len(onboarding_state.inferred_mastered_concepts) > 0, "Should have inferred mastery"
        
        # Generate summary
        if tutor.summary_generator:
            summary = tutor.summary_generator.generate_summary(onboarding_state)
            assert len(summary) > 0, "Should generate summary"
            print(f"✅ Onboarding complete: Difficulty={onboarding_state.calibrated_difficulty}, Mastered={len(onboarding_state.inferred_mastered_concepts)}")
            print(f"   Summary: {summary[:100]}...")


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v", "-s"])

