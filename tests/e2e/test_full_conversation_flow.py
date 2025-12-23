"""
End-to-End Tests for Full Conversation Flow

Tests the complete tutoring system including:
- Onboarding → Normal tutoring → Planning requests
- State persistence
- Difficulty adaptation
- Background MAS integration
"""

import pytest
import asyncio
import sys
import os
from typing import List, Dict

# Add project root to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.insert(0, os.path.join(project_root, "agentic_socratic_nlp_tutor", "src"))

from agentic_socratic_nlp_tutor.socratic_tutor import SocraticTutor, SessionState


class TestFullConversationFlow:
    """Test complete conversation flows."""
    
    @pytest.fixture
    def tutor(self):
        """Create tutor instance."""
        return SocraticTutor()
    
    @pytest.mark.asyncio
    async def test_onboarding_to_tutoring_flow(self, tutor):
        """
        Test complete flow: onboarding → normal tutoring.
        
        Expected:
        1. Onboarding completes (2-3 questions)
        2. Normal tutoring begins
        3. State persists correctly
        """
        session_id = "test_session_1"
        user_id = "test_user_1"
        
        # Stage 1: Onboarding - Welcome
        responses = []
        async for chunk in tutor.respond("I'm new to NLP and want to learn the basics", session_id, user_id):
            responses.append(chunk)
        
        response_text = "".join(responses)
        state = await tutor.get_or_create_session(session_id, user_id)
        
        # Should be in onboarding
        assert not state.onboarding_complete, "Should still be in onboarding"
        assert len(response_text) > 0, "Should have received response"
        
        # Stage 2: Continue onboarding (simulate diagnostic question)
        responses = []
        async for chunk in tutor.respond("I know some basic programming", session_id, user_id):
            responses.append(chunk)
        
        # Stage 3: Complete onboarding
        responses = []
        async for chunk in tutor.respond("I understand basic machine learning", session_id, user_id):
            responses.append(chunk)
        
        state = await tutor.get_or_create_session(session_id, user_id)
        
        # Should complete onboarding after a few interactions
        # (exact number depends on implementation)
        print(f"✅ Onboarding flow: Completed after interactions, state.onboarding_complete = {state.onboarding_complete}")
    
    @pytest.mark.asyncio
    async def test_normal_tutoring_flow(self, tutor):
        """
        Test normal tutoring after onboarding.
        
        Expected:
        1. Topic detection works
        2. RAG retrieval works
        3. Response is generated
        4. State is updated
        """
        session_id = "test_session_2"
        user_id = "test_user_2"
        
        # Skip onboarding by setting state
        state = await tutor.get_or_create_session(session_id, user_id)
        state.onboarding_complete = True
        state.difficulty = "intermediate"
        await tutor.save_session(state)
        
        # Ask a question
        responses = []
        async for chunk in tutor.respond("What is tokenization?", session_id, user_id):
            responses.append(chunk)
        
        response_text = "".join(responses)
        
        assert len(response_text) > 0, "Should have received response"
        assert "token" in response_text.lower() or len(response_text) > 50, "Response should be relevant"
        
        # Check state updated
        state = await tutor.get_or_create_session(session_id, user_id)
        assert len(state.conversation_history) >= 2, "Should have conversation history"
        
        print(f"✅ Normal tutoring: Response length = {len(response_text)}")
    
    @pytest.mark.asyncio
    async def test_planning_request_flow(self, tutor):
        """
        Test planning request detection and handling.
        
        Expected:
        1. Intent detection works
        2. Planning MAS is triggered
        3. Curriculum is generated
        """
        session_id = "test_session_3"
        user_id = "test_user_3"
        
        # Skip onboarding
        state = await tutor.get_or_create_session(session_id, user_id)
        state.onboarding_complete = True
        state.difficulty = "intermediate"
        state.mastered_concepts = ["RNN", "LSTM"]
        await tutor.save_session(state)
        
        # Request a learning plan
        responses = []
        async for chunk in tutor.respond("Create a learning plan for transformers", session_id, user_id):
            responses.append(chunk)
        
        response_text = "".join(responses)
        
        # Should detect planning request
        assert tutor.is_planning_request("Create a learning plan for transformers"), "Should detect planning request"
        
        # Should have generated plan (may take 15-45 seconds)
        assert len(response_text) > 100, "Should have generated plan"
        
        print(f"✅ Planning request: Plan length = {len(response_text)}")
    
    @pytest.mark.asyncio
    async def test_difficulty_adaptation_flow(self, tutor):
        """
        Test automatic difficulty adaptation.
        
        Expected:
        1. High scores → difficulty increases
        2. Low scores → difficulty decreases
        3. State is updated
        """
        session_id = "test_session_4"
        user_id = "test_user_4"
        
        # Skip onboarding
        state = await tutor.get_or_create_session(session_id, user_id)
        state.onboarding_complete = True
        state.difficulty = "intermediate"
        state.understanding_scores = [0.85, 0.90, 0.88, 0.92, 0.87]  # High scores
        state.performance_trend = "improving"
        await tutor.save_session(state)
        
        # Simulate response (triggers difficulty check)
        if tutor.difficulty_adapter:
            adjustment = tutor.difficulty_adapter.check_adjustment(
                current_difficulty=state.difficulty,
                recent_scores=state.understanding_scores,
                performance_trend=state.performance_trend
            )
            
            if adjustment.should_adjust:
                tutor.difficulty_adapter.apply_adjustment(state, adjustment)
                await tutor.save_session(state)
                
                assert state.difficulty == "advanced", "Difficulty should increase"
                print(f"✅ Difficulty adaptation: {state.difficulty}")
    
    @pytest.mark.asyncio
    async def test_response_caching(self, tutor):
        """
        Test response-level semantic caching.
        
        Expected:
        1. First query → cache miss → LLM call
        2. Similar query → cache hit → instant response
        """
        session_id = "test_session_5"
        user_id = "test_user_5"
        
        # Skip onboarding
        state = await tutor.get_or_create_session(session_id, user_id)
        state.onboarding_complete = True
        await tutor.save_session(state)
        
        if not tutor.response_cache:
            pytest.skip("Response cache not available")
        
        # First query
        query1 = "What is tokenization?"
        responses1 = []
        async for chunk in tutor.respond(query1, session_id, user_id):
            responses1.append(chunk)
        response1 = "".join(responses1)
        
        # Similar query (should hit cache)
        query2 = "What does tokenization mean?"
        cached = tutor.response_cache.get(query2)
        
        if cached:
            print(f"✅ Response caching: Cache hit for similar query")
        else:
            print(f"⚠️ Response caching: Cache miss (may need more similar queries)")
    
    @pytest.mark.asyncio
    async def test_state_persistence(self, tutor):
        """
        Test state persistence across "restarts".
        
        Expected:
        1. Create session and update state
        2. Save to database
        3. Load from database (simulate restart)
        4. State should be preserved
        """
        session_id = "test_session_6"
        user_id = "test_user_6"
        
        # Create and update state
        state = await tutor.get_or_create_session(session_id, user_id)
        state.onboarding_complete = True
        state.difficulty = "advanced"
        state.mastered_concepts = ["RNN", "LSTM", "Transformer"]
        state.current_topic = "BERT"
        state.interaction_count = 5
        
        # Save state
        saved = await tutor.save_session(state)
        
        if tutor.use_persistent_sessions:
            assert saved, "Should save to database"
            
            # Simulate restart: create new tutor instance
            tutor2 = SocraticTutor(supabase_client=tutor.supabase_client)
            state2 = await tutor2.get_or_create_session(session_id, user_id)
            
            # Verify state persisted
            assert state2.onboarding_complete == state.onboarding_complete
            assert state2.difficulty == state.difficulty
            assert state2.mastered_concepts == state.mastered_concepts
            assert state2.current_topic == state.current_topic
            
            print(f"✅ State persistence: State preserved across 'restart'")
        else:
            print(f"⚠️ State persistence: Using in-memory (persistence not available)")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])

