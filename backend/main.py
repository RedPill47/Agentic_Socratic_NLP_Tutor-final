"""
FastAPI Backend for Agentic Socratic NLP Tutor - WITH SUPABASE INTEGRATION

Provides REST API endpoints with:
- JWT Authentication
- Supabase persistence for messages and sessions
- CrewAI agent execution
- Real-time streaming
- Progress tracking
"""

from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
import os
import sys
import json
import asyncio
from datetime import datetime as dt
import logging
import signal

# Setup enhanced logging with pretty formatting
from lib.logger import setup_logging, get_logger

# Setup logging with colors and structured output
setup_logging(level=logging.INFO, use_colors=True)

# Create main logger
logger = get_logger("backend.main")

# Add the agentic_socratic_nlp_tutor package to Python path
backend_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(backend_dir)

package_paths = [
    os.path.join(project_root, 'agentic_socratic_nlp_tutor', 'src'),
]

for package_src in package_paths:
    if os.path.exists(package_src):
        if package_src not in sys.path:
            sys.path.insert(0, package_src)
        break

# Import Supabase and auth utilities
from lib.supabase_client import get_supabase_client
from lib.auth import get_current_user

# Import simplified SocraticTutor (new, faster approach)
try:
    from agentic_socratic_nlp_tutor.socratic_tutor import SocraticTutor
    SOCRATIC_TUTOR_AVAILABLE = True
except ImportError as e:
    SOCRATIC_TUTOR_AVAILABLE = False
    logger.warning("SocraticTutor not available", data={"error": str(e)})

# Singleton pattern for SocraticTutor to avoid reinitializing on every request
_tutor_instance = None
_background_sync = None

def get_tutor_instance():
    """Get or create singleton SocraticTutor instance."""
    global _tutor_instance
    if _tutor_instance is None:
        # Pass Supabase client for MAS persistence
        supabase = get_supabase_client()
        _tutor_instance = SocraticTutor(supabase_client=supabase)
    return _tutor_instance

def get_background_sync():
    """Get or create background sync instance."""
    global _background_sync
    if _background_sync is None:
        try:
            from agentic_socratic_nlp_tutor.background_sync import BackgroundSync
            sync_interval = int(os.getenv("BACKGROUND_SYNC_INTERVAL_HOURS", "24"))
            _background_sync = BackgroundSync(sync_interval_hours=sync_interval)
        except ImportError:
            logger.warning("BackgroundSync not available")
            _background_sync = None
    return _background_sync

# Initialize FastAPI app
app = FastAPI(
    title="Agentic Socratic NLP Tutor API",
    description="REST API for AI-powered NLP tutoring system with Supabase",
    version="2.0.0"
)

# CORS middleware for Next.js frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://localhost:3001",
        "http://127.0.0.1:3000",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ==================== Pydantic Models ====================

class ChatMessage(BaseModel):
    content: str
    session_id: Optional[str] = None


class ChatResponse(BaseModel):
    content: str
    session_id: str
    student_id: str
    metadata: Optional[Dict[str, Any]] = None


class StateSummary(BaseModel):
    session_id: str
    difficulty: str
    current_topic: Optional[str]
    mastered_concepts: List[str]
    learning_style: Optional[str]
    interaction_count: int
    onboarding_complete: bool
    performance_trend: Optional[str]
    understanding_scores: List[float]
    source_files: List[str] = []  # RAG source files used
    last_adaptation_event: Optional[str] = None


class ProgressMetrics(BaseModel):
    """Progress tracking metrics."""
    concepts_mastered: int
    concepts_in_progress: int
    total_concepts: int
    mastery_percentage: float
    interactions_total: int
    avg_understanding_score: float
    time_spent_minutes: int
    current_streak: int
    best_streak: int
    strengths: List[str]
    focus_areas: List[str]


# ==================== Helper Functions ====================

async def get_or_create_session(session_id: str, user_id: str):
    """Get existing session or create new one in Supabase"""
    supabase = get_supabase_client()

    # Try to get session by session_id and user_id
    result = supabase.table('sessions').select('*').eq('session_id', session_id).eq('user_id', user_id).execute()

    if result.data and len(result.data) > 0:
        return result.data[0]

    # Create new session (only fields that exist in sessions table)
    # Note: difficulty, mastered_concepts, learning_style, onboarding_complete are NOT in sessions table
    # (they're in profiles table). source_files IS in sessions table (for RAG traceability)
    new_session = {
        "user_id": user_id,
        "session_id": session_id,
        "interaction_count": 0,
        "current_topic": None,
        "stated_goal": None,
        "stated_level": None,
        "performance_trend": None,
        "understanding_scores": "[]",
        "source_files": "[]"  # Initialize empty array for RAG source files
    }

    create_result = supabase.table('sessions').insert(new_session).execute()

    if not create_result.data:
        raise HTTPException(status_code=500, detail="Failed to create session")

    return create_result.data[0]


async def get_conversation_history(session_db_id: str, limit: int = 10):
    """Get recent conversation messages from Supabase"""
    supabase = get_supabase_client()

    result = supabase.table('messages') \
        .select('*') \
        .eq('session_id', session_db_id) \
        .order('created_at', desc=False) \
        .limit(limit) \
        .execute()

    return result.data if result.data else []


async def store_message(session_db_id: str, role: str, content: str, metadata: dict = None):
    """Store message in Supabase"""
    supabase = get_supabase_client()

    message_data = {
        "session_id": session_db_id,
        "role": role,
        "content": content,
        "metadata": metadata or {}
    }

    result = supabase.table('messages').insert(message_data).execute()
    return result.data[0] if result.data else None


async def update_session_activity(session_db_id: str, topic: str = None):
    """Update session last activity and interaction count"""
    supabase = get_supabase_client()

    # Get current session
    session_result = supabase.table('sessions').select('*').eq('id', session_db_id).single().execute()

    if not session_result.data:
        return

    session = session_result.data

    update_data = {
        "last_activity": dt.now().isoformat(),
        "interaction_count": session.get("interaction_count", 0) + 1
    }

    if topic:
        update_data["current_topic"] = topic

    supabase.table('sessions').update(update_data).eq('id', session_db_id).execute()


# ==================== API Endpoints ====================

@app.get("/")
async def root():
    """Health check endpoint."""
    return {
        "status": "ok",
        "service": "Agentic Socratic NLP Tutor API (Supabase Integrated)",
        "version": "2.0.0",
        "socratic_tutor_available": SOCRATIC_TUTOR_AVAILABLE,
        "supabase_connected": True
    }


@app.post("/api/chat/stream")
async def chat_stream(message: ChatMessage, user: dict = Depends(get_current_user)):
    """
    Stream chat response in real-time with SSE.
    Uses simplified SocraticTutor (fast, direct LLM approach).
    Requires authentication.
    Stores all messages in Supabase.
    """

    # Get or create session
    session_id = message.session_id or f"session_{dt.now().timestamp()}"
    session = await get_or_create_session(session_id, user["id"])

    async def generate():
        """Generator function for streaming response."""
        try:
            # Log incoming request
            import time
            start_time = time.time()
            logger.request("POST", f"/api/chat/stream", user_id=user["id"], data={
                "session_id": session_id,
                "message_length": len(message.content),
                "message_preview": message.content[:50] + "..." if len(message.content) > 50 else message.content
            })
            
            # Store user message
            await store_message(session["id"], "user", message.content)
            logger.info("💾 User message stored in database")

            # Use simplified tutor (fast, direct LLM approach)
            if SOCRATIC_TUTOR_AVAILABLE:
                logger.section("CHAT REQUEST PROCESSING", {
                    "session_id": session_id[:20] + "...",
                    "user_id": user["id"][:20] + "...",
                    "message_length": len(message.content)
                })
                
                # New simplified approach - single LLM call, true streaming
                # Use singleton to avoid reinitializing on every request
                tutor = get_tutor_instance()
                full_response = ""
                
                # Get session state BEFORE responding (to check conversation history)
                logger.subsection("Loading Session State")
                tutor_state = await tutor.get_or_create_session(session_id, user_id=user["id"])
                logger.info("Session loaded", data={
                    "conversation_history_length": len(tutor_state.conversation_history),
                    "current_topic": tutor_state.current_topic,
                    "onboarding_complete": tutor_state.onboarding_complete,
                    "mastered_concepts_count": len(tutor_state.mastered_concepts),
                    "learning_style": tutor_state.learning_style,
                    "difficulty": tutor_state.difficulty
                })
                
                # Stream response chunks directly from LLM
                logger.subsection("Generating Response (Streaming)")
                logger.info("Starting LLM response generation...")
                chunk_count = 0
                async for chunk in tutor.respond(message.content, session_id, user_id=user["id"]):
                    if chunk:  # Only yield non-empty chunks
                        full_response += chunk
                        chunk_count += 1
                        chunk_data = {
                            "type": "chunk",
                            "content": chunk,
                            "done": False
                        }
                        yield f"data: {json.dumps(chunk_data)}\n\n"
                
                logger.success("Response generated", data={
                    "total_chunks": chunk_count,
                    "response_length": len(full_response),
                    "response_preview": full_response[:100] + "..." if len(full_response) > 100 else full_response
                })
                
                # Get the actual state object from tutor (it's already updated in respond())
                # The tutor's respond() method updates state.current_topic and saves it
                # We need to access the same state object that was just updated
                # Note: The state is saved in respond() after _normal_respond completes
                # So we can safely reload from DB to get the latest saved version
                tutor_state = await tutor.get_or_create_session(session_id, user_id=user["id"])
                
                logger.subsection("Session State After Response")
                logger.info("Session state after response", data={
                    "conversation_history_length": len(tutor_state.conversation_history),
                    "current_topic": tutor_state.current_topic,
                    "interaction_count": tutor_state.interaction_count,
                    "source_files_count": len(tutor_state.source_files),
                    "source_files": tutor_state.source_files[:5] if tutor_state.source_files else []
                })
                
                # Verify topic was detected
                if tutor_state.current_topic:
                    logger.success(f"✅ Topic detected: {tutor_state.current_topic}")
                else:
                    logger.warning("⚠️ No topic detected in session state")
                
                # Store session_db_id and user_id in state for MAS persistence (if not already set)
                if not tutor_state.session_db_id:
                    tutor_state.session_db_id = session["id"]
                if not tutor_state.user_id:
                    tutor_state.user_id = user["id"]
                
                # Save session state to ensure everything is persisted
                logger.subsection("Saving Session State")
                await tutor.save_session(tutor_state)
                logger.success("Session saved to database", data={
                    "session_db_id": session["id"],
                    "current_topic": tutor_state.current_topic,
                    "conversation_history_length": len(tutor_state.conversation_history),
                    "source_files_count": len(tutor_state.source_files)
                })
                
                # Try to load cached analysis from database on first interaction
                if tutor.background_mas and len(tutor_state.conversation_history) <= 2:
                    logger.subsection("Loading MAS Analysis from Database")
                    try:
                        # Load latest analysis from database
                        cached_analysis = await tutor.background_mas.load_from_database(
                            session_db_id=session["id"],
                            user_id=user["id"]
                        )
                        if cached_analysis:
                            # Cache in memory for fast access
                            tutor.background_mas.analysis_cache[session_id] = cached_analysis
                            logger.success("MAS analysis loaded from database", data={
                                "session_id": session_id[:20] + "...",
                                "has_learning_style": bool(cached_analysis.learning_style),
                                "has_performance": bool(cached_analysis.performance_assessment),
                                "knowledge_gaps_count": len(cached_analysis.knowledge_gaps)
                            })
                        else:
                            logger.info("No cached MAS analysis found")
                    except Exception as e:
                        logger.warning("Failed to load MAS analysis from database", data={"error": str(e)})
                
                # Store assistant response with RAG chunks in metadata
                logger.subsection("Storing Response in Database")
                metadata = {
                    "state": "learning",
                    "topic": tutor_state.current_topic
                }
                # Include RAG chunks if available
                if hasattr(tutor_state, '_last_rag_chunks') and tutor_state._last_rag_chunks:
                    metadata["rag_sources"] = tutor_state._last_rag_chunks
                    logger.info("RAG sources included in metadata", data={
                        "rag_chunks_count": len(tutor_state._last_rag_chunks),
                        "source_files": [chunk.get("source_file", "unknown") for chunk in tutor_state._last_rag_chunks[:3]]
                    })
                    # Clear after storing
                    tutor_state._last_rag_chunks = None
                await store_message(session["id"], "assistant", full_response, metadata)
                await update_session_activity(session["id"], tutor_state.current_topic)
                logger.success("Assistant response stored in database")
                
                # Log completion
                duration = time.time() - start_time
                logger.response(200, "/api/chat/stream", duration=duration, data={
                    "session_id": session_id[:20] + "...",
                    "response_length": len(full_response),
                    "topic": tutor_state.current_topic
                })
                logger.end_section()
                
                # Send completion
                completion = {
                    "type": "done",
                    "metadata": {
                        "session_id": session["session_id"],
                        "student_id": user["id"],
                        "state": "learning",
                        "topic": tutor_state.current_topic
                    },
                    "done": True
                }
                yield f"data: {json.dumps(completion)}\n\n"
                return
            
            # If SocraticTutor is not available, return error
            if not SOCRATIC_TUTOR_AVAILABLE:
                error_chunk = {
                    "type": "error",
                    "content": "Tutor system not available",
                    "done": True
                }
                yield f"data: {json.dumps(error_chunk)}\n\n"
                return

        except Exception as e:
            import traceback
            logger.error("Error in chat_stream", error=e, data={
                "session_id": session_id,
                "user_id": user["id"][:20] + "..." if user.get("id") else None,
                "error_type": type(e).__name__,
                "traceback": traceback.format_exc()
            })
            error_chunk = {
                "type": "error",
                "content": f"Error: {str(e)}",
                "done": True
            }
            yield f"data: {json.dumps(error_chunk)}\n\n"

    return StreamingResponse(generate(), media_type="text/event-stream")


@app.get("/api/state/{session_id}", response_model=StateSummary)
async def get_state(session_id: str, user: dict = Depends(get_current_user)):
    """Get current session state."""
    if not SOCRATIC_TUTOR_AVAILABLE:
        raise HTTPException(status_code=503, detail="SocraticTutor not available")
    
    tutor = get_tutor_instance()
    state = await tutor.get_or_create_session(session_id, user_id=user["id"])
    
    return StateSummary(
        session_id=state.session_id,
        difficulty=state.difficulty,
        current_topic=state.current_topic,
        mastered_concepts=state.mastered_concepts,
        learning_style=state.learning_style,
        interaction_count=state.interaction_count,
        onboarding_complete=state.onboarding_complete,
        performance_trend=state.performance_trend,
        understanding_scores=state.understanding_scores,
        source_files=getattr(state, 'source_files', []),  # Get source files from session state
        last_adaptation_event=None  # Could track this if needed
    )


@app.post("/api/sessions/{session_id}/welcome")
async def get_welcome_message(session_id: str, user: dict = Depends(get_current_user)):
    """
    Get welcome message for a new session.
    Automatically generates and stores the welcome message.
    """
    if not SOCRATIC_TUTOR_AVAILABLE:
        raise HTTPException(status_code=503, detail="SocraticTutor not available")
    
    try:
        # Get or create session
        session = await get_or_create_session(session_id, user["id"])
        logger.section("WELCOME MESSAGE REQUEST", {"session_id": session_id[:20] + "..."})
        
        # Check if welcome message already exists (check all messages, not just first)
        logger.subsection("Checking for Existing Welcome Message")
        existing_messages = await get_conversation_history(session["id"], limit=10)
        if existing_messages:
            # Check if there's already an assistant message (could be welcome or other)
            assistant_messages = [msg for msg in existing_messages if msg.get("role") == "assistant"]
            if assistant_messages:
                # Return the first assistant message (likely the welcome)
                welcome_msg = assistant_messages[0]
                logger.success("Found existing welcome message", data={
                    "message_length": len(welcome_msg.get("content", "")),
                    "message_preview": welcome_msg.get("content", "")[:100] + "..." if len(welcome_msg.get("content", "")) > 100 else welcome_msg.get("content", "")
                })
                logger.end_section()
                return {
                    "content": welcome_msg.get("content", ""),
                    "session_id": session["session_id"]
                }
        
        # No existing messages - generate welcome message
        logger.info("No existing welcome message found, generating new one")
        
        # Check if user has completed onboarding in profiles table (user-level)
        logger.subsection("Checking User Onboarding Status")
        user_has_completed_onboarding = False
        try:
            from agentic_socratic_nlp_tutor.user_profile_manager import UserProfileManager
            supabase = get_supabase_client()
            if supabase:
                # Check profiles table directly for user-level onboarding status
                user_profile_manager = UserProfileManager(supabase_client=supabase)
                user_profile = await user_profile_manager.get_user_profile(user["id"])
                if user_profile and user_profile.onboarding_complete:
                    user_has_completed_onboarding = True
                    logger.success("User has completed onboarding", data={
                        "user_id": user["id"][:20] + "...",
                        "onboarding_completed_at": str(user_profile.onboarding_completed_at) if hasattr(user_profile, 'onboarding_completed_at') else None
                    })
                else:
                    logger.info("User has not completed onboarding")
        except Exception as e:
            logger.warning("Could not check user onboarding status", data={"error": str(e)})
        
        # Get tutor instance
        tutor = get_tutor_instance()
        state = await tutor.get_or_create_session(session_id, user_id=user["id"])
        
        # Generate welcome message based on user's onboarding status
        # Only show full welcome message if user has NEVER completed onboarding
        if user_has_completed_onboarding:
            # User has completed onboarding before - use simple welcome
            welcome_content = "Hello! Welcome back! What would you like to learn about today?"
        elif not state.onboarding_complete:
            # New user or session without onboarding - use full welcome message
            welcome_content = tutor._welcome_message()
        else:
            # Session has onboarding complete but user check failed - use simple welcome
            welcome_content = "Hello! Welcome back! What would you like to learn about today?"
        
        # Store welcome message in database
        await store_message(session["id"], "assistant", welcome_content, {
            "type": "welcome",
            "state": "welcome"
        })
        
        logger.success("Welcome message stored", data={
            "message_length": len(welcome_content),
            "message_type": "returning_user" if user_has_completed_onboarding else "new_user"
        })
        logger.end_section()
        
        # Update session activity
        await update_session_activity(session["id"])
        
        return {
            "content": welcome_content,
            "session_id": session["session_id"]
        }
    except Exception as e:
        logger.error("Error getting welcome message", error=e, data={
            "session_id": session_id,
            "user_id": user["id"][:20] + "..." if user.get("id") else None
        })
        logger.end_section()
        raise HTTPException(status_code=500, detail=f"Error generating welcome message: {str(e)}")


@app.get("/api/progress/{session_id}", response_model=ProgressMetrics)
async def get_progress(session_id: str, user: dict = Depends(get_current_user)):
    """
    Get progress metrics for a session.
    
    Calculates:
    - Concepts mastered
    - Mastery percentage
    - Average understanding score
    - Current/best streaks
    - Strengths and focus areas
    """
    if not SOCRATIC_TUTOR_AVAILABLE:
        raise HTTPException(status_code=503, detail="SocraticTutor not available")
    
    tutor = get_tutor_instance()
    state = await tutor.get_or_create_session(session_id, user_id=user["id"])
    
    # Get all concepts from knowledge graph
    if hasattr(tutor.prerequisites, 'get_all_concepts'):
        all_concepts = tutor.prerequisites.get_all_concepts()
        total_concepts = len(all_concepts)
    else:
        # Fallback: estimate from graph
        total_concepts = len(tutor.prerequisites.graph) if hasattr(tutor.prerequisites, 'graph') else 30
    
    # Calculate metrics
    concepts_mastered = len(state.mastered_concepts)
    concepts_in_progress = len([c for c in state.mastered_concepts if c == state.current_topic]) if state.current_topic else 0
    mastery_percentage = (concepts_mastered / total_concepts * 100) if total_concepts > 0 else 0.0
    
    # Understanding scores
    avg_understanding_score = sum(state.understanding_scores) / len(state.understanding_scores) if state.understanding_scores else 0.0
    
    # Calculate streaks
    current_streak = 0
    best_streak = 0
    if state.understanding_scores:
        current_streak_count = 0
        best_streak_count = 0
        for score in reversed(state.understanding_scores):
            if score >= 0.7:  # Good score threshold
                current_streak_count += 1
                best_streak_count = max(best_streak_count, current_streak_count)
            else:
                current_streak = current_streak_count
                current_streak_count = 0
        current_streak = current_streak_count
        best_streak = best_streak_count
    
    # Strengths and focus areas (simplified)
    strengths = state.mastered_concepts[:3] if state.mastered_concepts else []
    
    # Focus areas: concepts with low scores or struggling
    focus_areas = []
    if state.current_topic and state.current_topic not in state.mastered_concepts:
        focus_areas.append(state.current_topic)
    
    # Estimate time spent (simplified: interactions * 2 minutes)
    time_spent_minutes = state.interaction_count * 2
    
    return ProgressMetrics(
        concepts_mastered=concepts_mastered,
        concepts_in_progress=concepts_in_progress,
        total_concepts=total_concepts,
        mastery_percentage=round(mastery_percentage, 2),
        interactions_total=state.interaction_count,
        avg_understanding_score=round(avg_understanding_score, 2),
        time_spent_minutes=time_spent_minutes,
        current_streak=current_streak,
        best_streak=best_streak,
        strengths=strengths,
        focus_areas=focus_areas
    )


@app.get("/api/sessions")
async def get_sessions(user: dict = Depends(get_current_user)):
    """Get all sessions for the current user"""
    supabase = get_supabase_client()

    result = supabase.table('sessions') \
        .select('*') \
        .eq('user_id', user["id"]) \
        .order('last_activity', desc=True) \
        .execute()

    return {"sessions": result.data if result.data else []}


@app.get("/api/sessions/{session_id}/messages")
async def get_session_messages(session_id: str, user: dict = Depends(get_current_user)):
    """Get all messages for a specific session"""
    supabase = get_supabase_client()

    # Verify session belongs to user
    session_result = supabase.table('sessions') \
        .select('*') \
        .eq('session_id', session_id) \
        .eq('user_id', user["id"]) \
        .execute()

    if not session_result.data:
        raise HTTPException(status_code=404, detail="Session not found")

    session = session_result.data[0]

    # Get messages
    messages_result = supabase.table('messages') \
        .select('*') \
        .eq('session_id', session["id"]) \
        .order('created_at', desc=False) \
        .execute()

    return {"messages": messages_result.data if messages_result.data else []}


@app.get("/api/sessions/{session_id}/rag-materials")
async def get_rag_materials(session_id: str, user: dict = Depends(get_current_user)):
    """Get RAG materials (chunks) used in this session"""
    if not SOCRATIC_TUTOR_AVAILABLE:
        raise HTTPException(status_code=503, detail="SocraticTutor not available")
    
    tutor = get_tutor_instance()
    state = await tutor.get_or_create_session(session_id, user_id=user["id"])
    
    # Get source files from session state
    source_files = getattr(state, 'source_files', [])
    
    if not source_files:
        return {"chunks": []}
    
    # Retrieve chunks from RAG system for the source files
    chunks = []
    try:
        # Get the RAG instance
        rag = tutor.rag
        
        # For each source file, query the vector store to get chunks
        # We'll use the current topic or a general query
        query = state.current_topic or "NLP concepts"
        
        # Get chunks from vector store
        results = rag.vector_store.similarity_search(query, k=20)
        
        # Filter to only include chunks from our source files
        for doc in results:
            doc_source = (
                doc.metadata.get('source_file') or 
                doc.metadata.get('source') or 
                doc.metadata.get('filename', '')
            )
            
            if doc_source in source_files:
                chunks.append({
                    "content": doc.page_content,
                    "source_file": doc_source,
                    "page_number": doc.metadata.get('page_number') or doc.metadata.get('page'),
                    "slide_title": doc.metadata.get('slide_title'),
                    "topic": doc.metadata.get('topic'),
                    "difficulty": doc.metadata.get('difficulty'),
                })
        
        # Deduplicate chunks (by content hash)
        seen_content = set()
        unique_chunks = []
        for chunk in chunks:
            content_hash = chunk["content"][:100]  # First 100 chars as hash
            if content_hash not in seen_content:
                seen_content.add(content_hash)
                unique_chunks.append(chunk)
        
        return {"chunks": unique_chunks}
    
    except Exception as e:
        logger.error("Error retrieving RAG materials", error=e)
        return {"chunks": []}


@app.delete("/api/sessions/{session_id}")
async def delete_session(session_id: str, user: dict = Depends(get_current_user)):
    """Delete a session (and all its messages via cascade)"""
    supabase = get_supabase_client()

    # Verify session belongs to user
    session_result = supabase.table('sessions') \
        .select('id') \
        .eq('session_id', session_id) \
        .eq('user_id', user["id"]) \
        .execute()

    if not session_result.data:
        raise HTTPException(status_code=404, detail="Session not found")

    # Delete session (messages will cascade delete)
    supabase.table('sessions').delete().eq('session_id', session_id).execute()

    return {"status": "deleted", "session_id": session_id}


@app.on_event("startup")
async def startup_event():
    """Startup event - initialize background sync."""
    background_sync = get_background_sync()
    if background_sync:
        await background_sync.start()
        logger.success("Background sync started")

@app.on_event("shutdown")
async def shutdown_event():
    """Shutdown event - stop background sync."""
    background_sync = get_background_sync()
    if background_sync:
        await background_sync.stop()
        logger.info("🛑 Background sync stopped")

if __name__ == "__main__":
    import uvicorn
    
    # Suppress KeyboardInterrupt traceback during graceful shutdown
    def handle_exit(*args):
        """Handle graceful shutdown."""
        logger.section("SERVER SHUTDOWN", {"reason": "SIGTERM received"})
        sys.exit(0)
    
    # Register signal handlers for graceful shutdown
    signal.signal(signal.SIGINT, handle_exit)
    signal.signal(signal.SIGTERM, handle_exit)
    
    # Enable auto-reload in development (watches for file changes)
    # Note: CancelledError during reload is normal and harmless
    try:
        uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)
    except KeyboardInterrupt:
        logger.info("🛑 Server stopped.")
        sys.exit(0)
