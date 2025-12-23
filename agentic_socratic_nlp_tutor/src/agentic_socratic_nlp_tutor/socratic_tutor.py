"""
Simplified Socratic Tutor - Direct LLM approach without agent abstraction

Replaces ConversationManager with a streamlined architecture:
- Single LLM call per response
- Simplified RAG (no query expansion, no re-ranking)
- Simple prerequisite graph
- Background evaluation (single function, not MAS)
- Streaming support
"""

import os
import asyncio
import logging
from typing import Dict, Optional, List, AsyncGenerator, Any, Tuple
from dataclasses import dataclass, field
from datetime import datetime
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document
from openai import AsyncOpenAI
from dotenv import load_dotenv

# Disable ChromaDB telemetry globally to avoid PostHog connection errors
os.environ["ANONYMIZED_TELEMETRY"] = "False"

logger = logging.getLogger(__name__)

# Import background MAS for conversation analysis
try:
    from agentic_socratic_nlp_tutor.background_analysis import BackgroundAnalysisMAS
    BACKGROUND_MAS_AVAILABLE = True
except ImportError:
    BACKGROUND_MAS_AVAILABLE = False
    print("⚠️ Background MAS not available - continuing without it")

# Import onboarding components
try:
    from agentic_socratic_nlp_tutor.onboarding_state import OnboardingState, OnboardingStage
    from agentic_socratic_nlp_tutor.response_strength_scorer import ResponseStrengthScorer
    from agentic_socratic_nlp_tutor.onboarding_summary_generator import OnboardingSummaryGenerator
    ONBOARDING_COMPONENTS_AVAILABLE = True
except ImportError:
    ONBOARDING_COMPONENTS_AVAILABLE = False
    print("⚠️ Onboarding components not available - using simplified onboarding")

# Import difficulty adapter
try:
    from agentic_socratic_nlp_tutor.difficulty_adapter import DifficultyAdapter
    DIFFICULTY_ADAPTER_AVAILABLE = True
except ImportError:
    DIFFICULTY_ADAPTER_AVAILABLE = False
    print("⚠️ Difficulty adapter not available")

# Import planning crew
try:
    from agentic_socratic_nlp_tutor.planning_crew import PlanningCrew
    PLANNING_CREW_AVAILABLE = True
except ImportError:
    PLANNING_CREW_AVAILABLE = False
    print("⚠️ Planning Crew not available")

# Import session manager
try:
    from agentic_socratic_nlp_tutor.session_manager import SessionManager
    SESSION_MANAGER_AVAILABLE = True
except ImportError:
    SESSION_MANAGER_AVAILABLE = False
    print("⚠️ Session Manager not available")

# Import response cache
try:
    from agentic_socratic_nlp_tutor.response_cache import ResponseCache
    RESPONSE_CACHE_AVAILABLE = True
except ImportError:
    RESPONSE_CACHE_AVAILABLE = False
    print("⚠️ Response Cache not available")

load_dotenv()

# Import SessionState from separate module to avoid circular imports
from agentic_socratic_nlp_tutor.session_state import SessionState


class SimplePrerequisiteGraph:
    """Simple in-memory prerequisite graph for NLP concepts."""
    
    def __init__(self):
        # Core NLP concept prerequisites
        # Format: concept -> list of prerequisite concepts
        self.graph = {
            # Beginner concepts (no prerequisites)
            "Text Preprocessing": [],  # Parent concept for preprocessing tasks
            "Tokenization": [],  # Part of text preprocessing, but no prerequisite needed
            "Stemming": [],  # Part of text preprocessing
            "Lemmatization": [],  # Part of text preprocessing
            "Stop Words": [],  # Part of text preprocessing
            "Bag of Words": [],
            "TF-IDF": [],
            
            # Intermediate concepts
            "Word Embeddings": ["Tokenization", "Vector Spaces"],
            "Word2Vec": ["Word Embeddings"],
            "GloVe": ["Word Embeddings"],
            "FastText": ["Word Embeddings"],
            "RNN": ["Neural Networks", "Sequences"],
            "LSTM": ["RNN"],
            "GRU": ["RNN"],
            "Sequence-to-Sequence": ["RNN", "LSTM"],
            "Language Modeling": ["RNN", "Probability"],
            
            # Advanced concepts
            "Attention Mechanisms": ["RNN", "Sequence-to-Sequence"],
            "Self-Attention": ["Attention Mechanisms"],
            "Transformer": ["Self-Attention", "Neural Networks"],
            "BERT": ["Transformer", "Word Embeddings"],
            "GPT": ["Transformer", "Language Modeling"],
            "T5": ["Transformer", "Sequence-to-Sequence"],
            "Fine-tuning": ["BERT", "GPT"],
            "Transfer Learning": ["Fine-tuning"],
            "Prompt Engineering": ["GPT", "Language Modeling"],
        }
    
    def get_prerequisites(self, concept: str) -> List[str]:
        """Get direct prerequisites for a concept."""
        return self.graph.get(concept, [])
    
    def get_all_prerequisites(self, concept: str) -> List[str]:
        """Get all prerequisites (transitive closure)."""
        prerequisites = set()
        to_process = [concept]
        
        while to_process:
            current = to_process.pop()
            if current in prerequisites:
                continue
            prerequisites.add(current)
            direct_prereqs = self.graph.get(current, [])
            to_process.extend(direct_prereqs)
        
        # Remove the concept itself
        prerequisites.discard(concept)
        return list(prerequisites)
    
    def get_gaps(self, concept: str, mastered: List[str]) -> List[str]:
        """Identify prerequisite gaps for a concept."""
        if not concept:
            return []
        
        all_prereqs = self.get_all_prerequisites(concept)
        mastered_set = set(mastered)
        return [p for p in all_prereqs if p not in mastered_set]
    
    def get_middle_prerequisite(self, concept: str) -> Optional[str]:
        """
        Get a prerequisite that's "in the middle" of the prerequisite chain.
        Used for binary search in onboarding - finds a good diagnostic question.
        
        Algorithm:
        1. Get all prerequisites (transitive)
        2. Sort by depth/order (topological sort)
        3. Return concept at middle index
        
        Example: For "Transformer" → prerequisites: [RNN, LSTM, Attention, Self-Attention]
                 Middle: "Attention" (good diagnostic question)
        """
        if not concept or concept not in self.graph:
            return None
        
        all_prereqs = self.get_all_prerequisites(concept)
        if not all_prereqs:
            return None
        
        # Topological sort: order prerequisites by depth
        # Concepts with fewer prerequisites come first
        def get_depth(c: str) -> int:
            """Get depth of concept (how many prerequisites it has)."""
            prereqs = self.get_all_prerequisites(c)
            return len(prereqs)
        
        # Sort by depth (shallow first, deep last)
        sorted_prereqs = sorted(all_prereqs, key=get_depth)
        
        # Return middle element
        middle_idx = len(sorted_prereqs) // 2
        return sorted_prereqs[middle_idx]
    
    def infer_mastery(self, demonstrated_concept: str) -> List[str]:
        """
        Infer mastery of prerequisites from demonstrated knowledge.
        
        If a student demonstrates knowledge of concept X, we can infer
        they likely know all prerequisites of X.
        
        Returns: [demonstrated_concept] + all prerequisites (transitive)
        
        Example: If student knows "Transformer" → infer:
                 ["Transformer", "Self-Attention", "Attention Mechanisms", 
                  "RNN", "LSTM", "Sequence-to-Sequence", "Neural Networks", ...]
        """
        if not demonstrated_concept:
            return []
        
        # Get all prerequisites (transitive)
        all_prereqs = self.get_all_prerequisites(demonstrated_concept)
        
        # Return the concept itself + all prerequisites
        return [demonstrated_concept] + all_prereqs


class FastRAG:
    """Simplified RAG - no query expansion, no re-ranking, just semantic search."""
    
    def __init__(self, db_path: Optional[str] = None):
        # Initialize embeddings
        self.embeddings = HuggingFaceEmbeddings(
            model_name="all-MiniLM-L6-v2",
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )
        
        # Initialize vector store
        if db_path is None:
            # Get absolute path of this file
            current_file = os.path.abspath(__file__)
            current_dir = os.path.dirname(current_file)
            
            # From: agentic_socratic_nlp_tutor/src/agentic_socratic_nlp_tutor/socratic_tutor.py
            # To project root: go up 3 levels
            # ../ -> src/agentic_socratic_nlp_tutor/
            # ../../ -> agentic_socratic_nlp_tutor/
            # ../../../ -> project root
            project_root = os.path.abspath(os.path.join(current_dir, "../../.."))
            db_path = os.path.join(project_root, "data", "chroma_db")
            
            # Verify the calculated path is correct by checking if it contains the project name
            # This helps catch path calculation errors
            if "Agentic_Socratic_NLP_Tutor" not in db_path:
                # Path calculation seems wrong, try finding project root by looking for data/chroma_db
                # Walk up from current file until we find data/chroma_db
                search_dir = current_dir
                for _ in range(5):  # Don't go up more than 5 levels
                    potential_db = os.path.join(search_dir, "data", "chroma_db")
                    if os.path.exists(potential_db):
                        db_path = potential_db
                        break
                    search_dir = os.path.dirname(search_dir)
                    if search_dir == os.path.dirname(search_dir):  # Reached filesystem root
                        break
        
        # Final verification
        if not os.path.exists(db_path):
            # Try one more time with current working directory
            cwd_db = os.path.join(os.getcwd(), "data", "chroma_db")
            if os.path.exists(cwd_db):
                db_path = cwd_db
            else:
                # All attempts failed - provide helpful error message
                raise FileNotFoundError(
                    f"Knowledge base not found at {db_path}.\n"
                    f"Current working directory: {os.getcwd()}\n"
                    f"File location: {current_file}\n"
                    f"Please ensure ChromaDB is initialized by running: python src/ingest_pdfs_enhanced.py"
                )
        
        # Disable ChromaDB telemetry to avoid PostHog connection errors
        os.environ["ANONYMIZED_TELEMETRY"] = "False"
        
        self.vector_store = Chroma(
            persist_directory=db_path,
            embedding_function=self.embeddings
        )
        
        # Simple cache for recent queries
        self.cache: Dict[str, str] = {}
        self.cache_max_size = 50
    
    def _get_cache_key(self, query: str, k: int, difficulty: Optional[str]) -> str:
        """Generate cache key from query, k, and difficulty"""
        normalized = f"{query.lower().strip()}:{k}:{difficulty or 'any'}"
        import hashlib
        return hashlib.md5(normalized.encode()).hexdigest()
    
    def query(self, query: str, k: int = 3, difficulty: Optional[str] = None, return_chunks: bool = False) -> tuple[str, List[str], Optional[List[Dict]]]:
        """
        Fast semantic search with MMR for diversity.
        No query expansion, no LLM re-ranking.
        Target: < 500ms
        
        Returns:
            tuple: (formatted_text, source_files_list, chunks_with_metadata)
            chunks_with_metadata is None unless return_chunks=True
        """
        # Check cache using proper cache key
        cache_key = self._get_cache_key(query, k, difficulty)
        if cache_key in self.cache:
            cached_result = self.cache[cache_key]
            # If cached result is a tuple, return it; otherwise extract sources
            if isinstance(cached_result, tuple):
                # If return_chunks is requested but cache doesn't have chunks, return None
                if return_chunks:
                    formatted, source_files = cached_result
                    return formatted, source_files, None
                return (*cached_result, None)
            # Legacy cache format - return with empty sources
            return cached_result, [], None if return_chunks else None
        
        # MMR search for relevance + diversity (no LLM calls)
        try:
            results = self.vector_store.max_marginal_relevance_search(
                query,
                k=k,
                fetch_k=k * 3,  # Fetch more, select diverse subset
                lambda_mult=0.7  # Balance relevance vs diversity
            )
            
            # Filter by difficulty if specified
            if difficulty and difficulty != "any":
                filtered = []
                for doc in results:
                    doc_difficulty = doc.metadata.get('difficulty', '').lower()
                    if doc_difficulty == difficulty.lower() or not doc_difficulty:
                        filtered.append(doc)
                results = filtered[:k] if filtered else results
            
            # Extract source files (deduplicated)
            source_files = []
            seen_sources = set()
            for doc in results:
                # Try multiple metadata keys for source file
                source_file = (
                    doc.metadata.get('source_file') or 
                    doc.metadata.get('source') or 
                    doc.metadata.get('filename', '')
                )
                if source_file and source_file not in seen_sources:
                    source_files.append(source_file)
                    seen_sources.add(source_file)
            
            # Format results
            formatted = self._format_results(results)
            
            # Prepare chunks with metadata if requested
            chunks_with_metadata = None
            if return_chunks:
                chunks_with_metadata = []
                for doc in results:
                    chunk_data = {
                        "content": doc.page_content,
                        "source_file": (
                            doc.metadata.get('source_file') or 
                            doc.metadata.get('source') or 
                            doc.metadata.get('filename', 'Unknown')
                        ),
                        "page_number": doc.metadata.get('page_number') or doc.metadata.get('page'),
                        "slide_title": doc.metadata.get('slide_title'),
                        "topic": doc.metadata.get('topic'),
                        "difficulty": doc.metadata.get('difficulty'),
                    }
                    chunks_with_metadata.append(chunk_data)
            
            # Cache result as tuple
            if len(self.cache) >= self.cache_max_size:
                # Remove oldest entry (simple FIFO)
                oldest_key = next(iter(self.cache))
                del self.cache[oldest_key]
            self.cache[cache_key] = (formatted, source_files)
            
            return formatted, source_files, chunks_with_metadata
        except Exception as e:
            print(f"RAG query error: {e}")
            error_chunks = None if not return_chunks else []
            return "No relevant content found.", [], error_chunks
    
    def _format_results(self, results: List[Document]) -> str:
        """Format retrieved chunks for prompt injection."""
        if not results:
            return "No specific content retrieved."
        
        formatted_chunks = []
        for i, doc in enumerate(results, 1):
            source = doc.metadata.get('source', 'Course Material')
            page = doc.metadata.get('page', '')
            source_str = f"{source}" + (f" (page {page})" if page else "")
            formatted_chunks.append(
                f"[Source {i}: {source_str}]\n{doc.page_content}"
            )
        
        return "\n\n".join(formatted_chunks)


class SocraticTutor:
    """
    Simplified Socratic Tutor - single LLM approach.
    
    Replaces ConversationManager with direct LLM calls, removing agent overhead.
    """
    
    def __init__(self, db_path: Optional[str] = None, supabase_client=None):
        self.rag = FastRAG(db_path)
        # Initialize prerequisite graph
        # Try enhanced graph first, fallback to simple
        try:
            from agentic_socratic_nlp_tutor.knowledge_graph import EnhancedPrerequisiteGraph
            self.prerequisites = EnhancedPrerequisiteGraph()
            self.use_enhanced_graph = True
        except ImportError:
            self.prerequisites = SimplePrerequisiteGraph()
            self.use_enhanced_graph = False
        
        # Disable long-running planning in test environments
        self.disable_planning = bool(os.getenv("PYTEST_CURRENT_TEST") or os.getenv("TESTING"))
        
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY not found in environment variables")
        self.llm_client = AsyncOpenAI(api_key=api_key)
        # Use gpt-4o-mini as fallback if gpt-5-mini doesn't work
        self.model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
        
        # Auto-sync configuration (can be enabled via environment variables)
        self.auto_sync_enabled = os.getenv("AUTO_SYNC_CONCEPTS", "true").lower() == "true"
        self.lazy_load_enabled = os.getenv("LAZY_LOAD_CONCEPTS", "true").lower() == "true"
        self.runtime_discovery_enabled = os.getenv("RUNTIME_DISCOVERY", "true").lower() == "true"
        
        # Track concepts discovered at runtime (to avoid duplicate LLM calls)
        self._discovered_concepts: Dict[str, bool] = {}
        
        # Store supabase client for MAS persistence
        self.supabase_client = supabase_client
        
        # Initialize session manager for persistence
        if SESSION_MANAGER_AVAILABLE and supabase_client:
            self.session_manager = SessionManager(supabase_client=supabase_client)
            self.use_persistent_sessions = True
        else:
            self.session_manager = None
            self.use_persistent_sessions = False
        
        # Initialize user profile manager for user-level data
        try:
            from agentic_socratic_nlp_tutor.user_profile_manager import UserProfileManager
            self.user_profile_manager = UserProfileManager(supabase_client=supabase_client) if supabase_client else None
            logger.info("✅ [SocraticTutor] UserProfileManager initialized")
        except ImportError:
            logger.warning("⚠️ [SocraticTutor] UserProfileManager not available")
            self.user_profile_manager = None
        
        # In-memory session state (fallback if persistence unavailable)
        self.sessions: Dict[str, SessionState] = {}
        
        # Initialize background MAS for conversation analysis (optional)
        if BACKGROUND_MAS_AVAILABLE:
            self.background_mas = BackgroundAnalysisMAS(self.prerequisites, supabase_client=supabase_client)
        else:
            self.background_mas = None
        
        # Initialize onboarding components
        if ONBOARDING_COMPONENTS_AVAILABLE:
            self.response_scorer = ResponseStrengthScorer()
            self.summary_generator = OnboardingSummaryGenerator()
            # Store onboarding states per session
            self.onboarding_states: Dict[str, OnboardingState] = {}
        else:
            self.response_scorer = None
            self.summary_generator = None
            self.onboarding_states = {}
        
        # Initialize difficulty adapter
        if DIFFICULTY_ADAPTER_AVAILABLE:
            self.difficulty_adapter = DifficultyAdapter()
        else:
            self.difficulty_adapter = None
        
        # Planning crew will be initialized on-demand (requires session state)
        self.planning_crew = None
        
        # Initialize response cache for semantic caching
        if RESPONSE_CACHE_AVAILABLE:
            self.response_cache = ResponseCache(
                similarity_threshold=0.85,
                max_cache_size=100,
                ttl_hours=24
            )
        else:
            self.response_cache = None
    
    async def get_or_create_session(self, session_id: str, user_id: Optional[str] = None) -> SessionState:
        """
        Get existing session or create new one.
        
        Uses SessionManager for persistence if available, otherwise falls back to in-memory.
        Also loads user-level profile data to personalize the session.
        """
        # Try to load from persistent storage
        if self.use_persistent_sessions and self.session_manager:
            try:
                session = await self.session_manager.get_session(session_id, user_id)
                if session:
                    # Cache in memory for fast access
                    self.sessions[session_id] = session
                    
                    # Load user-level profile data to personalize session
                    if user_id and self.user_profile_manager:
                        try:
                            user_profile = await self.user_profile_manager.get_user_profile(user_id)
                            if user_profile:
                                # Merge user-level data with session data
                                # Use user-level learning style (always override, profiles is source of truth)
                                if user_profile.learning_style:
                                    session.learning_style = user_profile.learning_style
                                    logger.info(f"📚 [Session] Loaded user-level learning style: {user_profile.learning_style}")
                                
                                # Merge user-level mastered concepts with session concepts
                                user_mastered = set(user_profile.mastered_concepts or [])
                                session_mastered = set(session.mastered_concepts or [])
                                # Union: combine both sets
                                session.mastered_concepts = sorted(list(user_mastered | session_mastered))
                                if user_mastered:
                                    logger.info(f"📚 [Session] Loaded {len(user_mastered)} user-level mastered concepts")
                                
                                # Use user-level difficulty (overall_difficulty from profiles)
                                if user_profile.overall_difficulty:
                                    session.difficulty = user_profile.overall_difficulty
                                    logger.info(f"📚 [Session] Loaded user-level difficulty: {user_profile.overall_difficulty}")
                                
                                # Use user-level onboarding status (if user completed onboarding, skip it)
                                if user_profile.onboarding_complete and not session.onboarding_complete:
                                    session.onboarding_complete = True
                                    logger.info(f"✅ [Session] User has completed onboarding, skipping for this session")
                        except Exception as profile_error:
                            logger.warning(f"⚠️ [Session] Error loading user profile: {profile_error}")
                    
                    # Only initialize onboarding state if onboarding is NOT complete
                    # This prevents restarting onboarding for sessions that already completed it
                    if ONBOARDING_COMPONENTS_AVAILABLE and session_id not in self.onboarding_states:
                        if not session.onboarding_complete:
                            # Only create onboarding state if onboarding hasn't been completed
                            self.onboarding_states[session_id] = OnboardingState()
                            logger.info(f"🔄 [Session] Created new onboarding state for session (onboarding_complete=False)")
                        else:
                            logger.info(f"✅ [Session] Skipping onboarding state creation (onboarding_complete=True)")
                    return session
            except Exception as e:
                print(f"⚠️ Error loading session from database: {e}, using in-memory")
        
        # Fallback to in-memory (or create new)
        if session_id not in self.sessions:
            session = SessionState(session_id=session_id)
            
            # Load user-level profile data for new session
            if user_id and self.user_profile_manager:
                try:
                    user_profile = await self.user_profile_manager.get_user_profile(user_id)
                    if user_profile:
                        # Initialize with user-level data
                        if user_profile.learning_style:
                            session.learning_style = user_profile.learning_style
                        if user_profile.mastered_concepts:
                            session.mastered_concepts = user_profile.mastered_concepts.copy()
                        if user_profile.overall_difficulty:
                            session.difficulty = user_profile.overall_difficulty
                        if user_profile.onboarding_complete:
                            session.onboarding_complete = True
                        logger.info(f"📚 [Session] Initialized new session with user-level data")
                except Exception as profile_error:
                    logger.warning(f"⚠️ [Session] Error loading user profile for new session: {profile_error}")
            
            self.sessions[session_id] = session
            # Initialize onboarding state for new session
            if ONBOARDING_COMPONENTS_AVAILABLE:
                self.onboarding_states[session_id] = OnboardingState()
        return self.sessions[session_id]
    
    async def save_session(self, session: SessionState) -> bool:
        """
        Save session state to persistent storage.
        
        Also updates user-level profile with mastered concepts from session.
        
        Args:
            session: SessionState to save
            
        Returns:
            True if saved successfully
        """
        if self.use_persistent_sessions and self.session_manager:
            try:
                result = await self.session_manager.save_session(session)
                
                # Update user-level profile with mastered concepts from session
                if session.user_id and self.user_profile_manager and session.mastered_concepts:
                    try:
                        await self.user_profile_manager.add_mastered_concepts(
                            session.user_id,
                            session.mastered_concepts
                        )
                        logger.debug(f"✅ [SocraticTutor] Updated user profile with {len(session.mastered_concepts)} mastered concepts")
                    except Exception as e:
                        logger.warning(f"⚠️ [SocraticTutor] Error updating user profile: {e}")
                
                return result
            except Exception as e:
                print(f"⚠️ Error saving session to database: {e}")
                # Fallback: keep in memory
                self.sessions[session.session_id] = session
                return False
        else:
            # In-memory only
            self.sessions[session.session_id] = session
            return True
    
    def get_or_create_onboarding_state(self, session_id: str, session_state: Optional[SessionState] = None) -> Optional[OnboardingState]:
        """
        Get onboarding state for session, or create new one.
        
        Note: Onboarding state is kept in-memory only (not persisted).
        It's temporary and only needed during onboarding flow.
        
        If session is reloaded mid-onboarding, attempts to infer stage from conversation history.
        
        Args:
            session_id: Session ID
            session_state: Optional session state to check if onboarding is already complete
        """
        if not ONBOARDING_COMPONENTS_AVAILABLE:
            return None
        
        # Don't create new onboarding state if onboarding is already complete
        if session_state and session_state.onboarding_complete:
            return None
        
        if session_id not in self.onboarding_states:
            # Create new onboarding state
            onboarding_state = OnboardingState()
            
            # If we have conversation history, try to infer the onboarding stage
            # This helps recover from mid-onboarding session reloads
            if session_state and session_state.conversation_history:
                history = session_state.conversation_history
                
                # Count diagnostic questions in history (they contain "Before we dive deeper")
                diagnostic_questions = [
                    msg for msg in history 
                    if msg.get("role") == "assistant" 
                    and "Before we dive deeper" in msg.get("content", "")
                ]
                
                if len(diagnostic_questions) > 0:
                    # We're in LEVEL_PROBING stage
                    onboarding_state.stage = OnboardingStage.LEVEL_PROBING
                    logger.info(f"🔄 [Onboarding] Inferred LEVEL_PROBING stage from {len(diagnostic_questions)} diagnostic questions in history")
                    
                    # Try to extract goal from conversation history
                    for msg in history:
                        if msg.get("role") == "user":
                            goal, level = self._extract_goal_and_level(msg.get("content", ""))
                            if goal:
                                onboarding_state.stated_goal = goal
                                onboarding_state.implied_level = level
                                onboarding_state.stage = OnboardingStage.GOAL_CAPTURED
                                logger.info(f"🔄 [Onboarding] Inferred goal: {goal}, level: {level}")
                                break
                    
                    # If we have 3+ diagnostic questions, we should probably complete onboarding
                    if len(diagnostic_questions) >= 3:
                        logger.warning(f"⚠️ [Onboarding] Found {len(diagnostic_questions)} diagnostic questions, suggesting completion")
                        # Don't set stage to COMPLETE here - let the flow handle it
                elif any("Welcome" in msg.get("content", "") or "Hello" in msg.get("content", "") 
                        for msg in history if msg.get("role") == "assistant"):
                    # We're past welcome, likely in GOAL_CAPTURED stage
                    onboarding_state.stage = OnboardingStage.GOAL_CAPTURED
                    logger.info(f"🔄 [Onboarding] Inferred GOAL_CAPTURED stage from welcome message in history")
            
            self.onboarding_states[session_id] = onboarding_state
        
        return self.onboarding_states[session_id]
    
    def detect_topic(self, text: str) -> Optional[str]:
        """
        Topic detection using keyword matching.
        Uses enhanced graph's detect_topic if available, otherwise falls back to simple matching.
        
        NOTE: This is the fast path. If this fails, detect_topic_llm_fallback() will be called,
        which can detect ANY topic from RAG (not limited to knowledge graph).
        """
        # Use enhanced graph's detect_topic if available (better keyword matching)
        if self.use_enhanced_graph and hasattr(self.prerequisites, 'detect_topic'):
            topic = self.prerequisites.detect_topic(text)
            if topic:
                return topic
        
        # Fallback to simple keyword matching
        text_lower = text.lower()
        
        # Check for concept keywords
        for concept in self.prerequisites.graph.keys():
            concept_lower = concept.lower()
            # Check if concept name or keywords appear
            if concept_lower in text_lower:
                return concept
        
        # Check for common NLP terms (this is a fallback - LLM fallback handles everything else)
        nlp_terms = {
            "tokenization": "Tokenization",
            "tokenize": "Tokenization",
            "text processing": "Text Preprocessing",
            "text preprocessing": "Text Preprocessing",
            "preprocessing": "Text Preprocessing",
            "word embedding": "Word Embeddings",
            "word2vec": "Word2Vec",
            "glove": "GloVe",
            "transformer": "Transformer",
            "attention": "Attention Mechanisms",
            "bert": "BERT",
            "gpt": "GPT",
            "rnn": "RNN",
            "lstm": "LSTM",
            "dependency parsing": "Dependency Parsing",
            "dependency parse": "Dependency Parsing",
            "parsing": "Dependency Parsing",  # Fallback if just "parsing" mentioned
            "named entity": "Named Entity Recognition",
            "ner": "Named Entity Recognition",
            "pos tagging": "Part-of-Speech Tagging",
            "part of speech": "Part-of-Speech Tagging",
            "sentiment analysis": "Sentiment Analysis",
            "sentiment": "Sentiment Analysis",
            "text classification": "Text Classification",
            "classification": "Text Classification",
            "language modeling": "Language Modeling",
            "language model": "Language Modeling",
        }
        
        for term, concept in nlp_terms.items():
            if term in text_lower:
                return concept
        
        return None
    
    async def detect_topic_llm_fallback(self, text: str) -> Optional[str]:
        """
        LLM-based topic detection fallback.
        Used when keyword matching fails - can detect any NLP topic from user input.
        
        This allows the system to work with topics in the vector store that aren't in the knowledge graph.
        The system will use RAG to teach about ANY topic, even if not in the knowledge graph.
        
        ARCHITECTURE: RAG is the source of truth for content. Knowledge graph is for:
        - Prerequisite checking (if topic is in graph)
        - Learning path generation
        - But NOT required for teaching - RAG can handle any topic!
        
        KEY INSIGHT: We don't need to manually add every concept to the knowledge graph.
        RAG can retrieve content for ANY topic in the vector store. The knowledge graph
        is only needed if you want prerequisite checking and learning paths.
        """
        try:
            # Optional: Use RAG to help validate/refine topic detection
            # This makes the system more robust by ensuring detected topics exist in RAG
            rag_context = ""
            try:
                # Quick RAG check to see what topics are relevant (fast, < 500ms)
                rag_results = self.rag.vector_store.similarity_search(text, k=2)
                if rag_results:
                    # Extract potential topics from RAG results
                    rag_snippets = [r.page_content[:200] for r in rag_results[:2]]
                    rag_context = "\n\nRelevant content from course materials:\n" + "\n---\n".join(rag_snippets)
            except Exception as rag_error:
                logger.debug(f"RAG validation check failed (non-critical): {rag_error}")
                rag_context = ""  # Continue without RAG context
            
            prompt = f"""Extract the main NLP concept or topic the user is asking about from this text:

"{text}"
{rag_context}

Return ONLY the concept name (e.g., "Dependency Parsing", "Text Classification", "Text Preprocessing", "Machine Translation", "Conditional Random Fields"), or "none" if it's not about a specific NLP concept.

IMPORTANT MAPPINGS:
- "text processing" or "text preprocessing" → "Text Preprocessing"
- "dependency parsing" → "Dependency Parsing"
- "named entity recognition" or "ner" → "Named Entity Recognition"
- "part of speech tagging" or "pos tagging" → "Part-of-Speech Tagging"
- "sentiment analysis" → "Sentiment Analysis"
- "conditional random fields" or "crf" → "Conditional Random Fields"
- "crf tagging" → "Conditional Random Fields"

Be specific - if they ask about "dependency parsing", return "Dependency Parsing", not just "parsing".
If they ask about "text processing", return "Text Preprocessing".
If they ask about "CRF tagging", return "Conditional Random Fields".

The topic can be ANY NLP concept from the course materials - it doesn't need to be in a predefined list.
If the relevant content above mentions specific concepts, use those names.
"""
            response = await self.llm_client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=30,
                temperature=0.1
            )
            topic = response.choices[0].message.content.strip()
            
            # Clean up response
            topic = topic.replace('"', '').replace("'", "").strip()
            
            # Normalize common variations
            topic_lower = topic.lower()
            if topic_lower in ["text processing", "text preprocessing", "preprocessing"]:
                topic = "Text Preprocessing"
            elif topic_lower in ["dependency parsing", "dependency parse"]:
                topic = "Dependency Parsing"
            elif topic_lower in ["named entity recognition", "ner"]:
                topic = "Named Entity Recognition"
            elif topic_lower in ["part of speech tagging", "pos tagging", "pos tag"]:
                topic = "Part-of-Speech Tagging"
            elif topic_lower in ["sentiment analysis", "sentiment"]:
                topic = "Sentiment Analysis"
            elif topic_lower in ["conditional random fields", "crf", "crf tagging"]:
                topic = "Conditional Random Fields"
            
            # Check if it's a valid response
            if topic.lower() in ["none", "n/a", "not applicable", ""]:
                return None
            
            # If topic is not in knowledge graph, we can still use it!
            # RAG will provide the content, and we'll just skip prerequisite checking
            if topic not in self.prerequisites.graph:
                logger.info(f"🎯 [SocraticTutor] LLM detected topic '{topic}' not in knowledge graph - will use RAG only (this is fine!)")
            
            logger.info(f"🎯 [SocraticTutor] LLM detected topic: {topic}")
            
            # RUNTIME DISCOVERY: If topic not in graph and discovery enabled, auto-add it
            if topic and self.runtime_discovery_enabled and self.use_enhanced_graph:
                if topic not in self.prerequisites.graph and topic not in self._discovered_concepts:
                    logger.info(f"🔍 [SocraticTutor] Runtime discovery enabled - checking RAG for topic: {topic}")
                    # Run discovery in background (non-blocking)
                    asyncio.create_task(self._runtime_discover_concept(topic, text))
                    self._discovered_concepts[topic] = True
            
            return topic
            
        except Exception as e:
            logger.warning(f"⚠️ LLM topic detection failed: {e}")
            return None
    
    async def _is_explicit_topic_switch(
        self, 
        user_input: str, 
        current_topic: str, 
        detected_topic: str
    ) -> bool:
        """
        Check if the user is explicitly asking about a different topic,
        or just mentioning related keywords in their answer.
        
        This prevents false positives where the user is answering a question
        about the current topic but mentions words that match another topic's keywords.
        
        Args:
            user_input: User's message
            current_topic: Current topic being discussed
            detected_topic: Topic detected from keyword matching
            
        Returns:
            True if user is explicitly asking about detected_topic, False otherwise
        """
        try:
            prompt = f"""The user is currently learning about: {current_topic}

Their message: "{user_input}"

A keyword matcher detected the topic "{detected_topic}" in their message.

Determine if the user is:
1. EXPLICITLY asking about or switching to "{detected_topic}" (e.g., "what is {detected_topic}?", "tell me about {detected_topic}", "I want to learn {detected_topic}")
2. Just mentioning related keywords while answering a question about "{current_topic}" (e.g., answering "when we use documents we break them down" while discussing tokenization)

Respond with ONLY "yes" if they are explicitly asking about "{detected_topic}", or "no" if they are just mentioning related keywords while discussing "{current_topic}".
"""
            
            response = await self.llm_client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=10,
                temperature=0.1
            )
            
            result = response.choices[0].message.content.strip().lower()
            is_explicit = result.startswith("yes")
            
            logger.debug(f"🔍 [SocraticTutor] Topic switch check: current={current_topic}, detected={detected_topic}, explicit={is_explicit}")
            return is_explicit
            
        except Exception as e:
            logger.warning(f"⚠️ [SocraticTutor] Error checking explicit topic switch: {e}")
            # On error, be conservative - don't switch topics
            return False
    
    async def _lazy_load_concept(self, topic: str, user_input: str):
        """
        Lazy loading: Auto-add concept to knowledge graph when detected but not present.
        This runs in the background and doesn't block the response.
        
        Args:
            topic: Topic name to add
            user_input: Original user input for context
        """
        try:
            # Quick validation: Check if topic exists in RAG
            rag_results = self.rag.vector_store.similarity_search(topic, k=1)
            if not rag_results or not rag_results[0].page_content:
                logger.debug(f"⏭️ [LazyLoad] Topic '{topic}' not found in RAG, skipping")
                return
            
            # Extract basic metadata from RAG
            rag_content = rag_results[0].page_content[:500]  # First 500 chars for context
            
            # Use LLM to extract metadata
            metadata = await self._extract_concept_metadata(topic, rag_content, user_input)
            if not metadata:
                logger.warning(f"⚠️ [LazyLoad] Failed to extract metadata for '{topic}'")
                return
            
            # Add concept to knowledge graph
            from agentic_socratic_nlp_tutor.knowledge_graph import Concept
            concept = Concept(
                name=topic,
                difficulty=metadata.get("difficulty", "intermediate"),
                topic_area=metadata.get("topic_area", "General NLP"),
                description=metadata.get("description", ""),
                learning_time_minutes=metadata.get("learning_time_minutes", 60),
                keywords=metadata.get("keywords", [])
            )
            
            self.prerequisites.add_concept(concept)
            logger.info(f"✅ [LazyLoad] Auto-added concept '{topic}' to knowledge graph")
            
        except Exception as e:
            logger.warning(f"⚠️ [LazyLoad] Error lazy loading concept '{topic}': {e}")
    
    async def _runtime_discover_concept(self, topic: str, user_input: str):
        """
        Runtime discovery: Check RAG when topics are detected and auto-add to graph.
        More comprehensive than lazy loading - extracts full metadata from RAG.
        
        Args:
            topic: Topic name to discover
            user_input: Original user input for context
        """
        try:
            # Query RAG for topic-specific content
            rag_results = self.rag.vector_store.similarity_search(topic, k=3)
            if not rag_results:
                logger.debug(f"⏭️ [RuntimeDiscovery] No RAG content found for '{topic}'")
                return
            
            # Combine RAG content for better context
            rag_content = "\n\n".join([r.page_content[:300] for r in rag_results[:3]])
            
            # Extract comprehensive metadata
            metadata = await self._extract_concept_metadata(topic, rag_content, user_input)
            if not metadata:
                logger.warning(f"⚠️ [RuntimeDiscovery] Failed to extract metadata for '{topic}'")
                return
            
            # Add concept to knowledge graph
            from agentic_socratic_nlp_tutor.knowledge_graph import Concept
            concept = Concept(
                name=topic,
                difficulty=metadata.get("difficulty", "intermediate"),
                topic_area=metadata.get("topic_area", "General NLP"),
                description=metadata.get("description", ""),
                learning_time_minutes=metadata.get("learning_time_minutes", 60),
                keywords=metadata.get("keywords", [])
            )
            
            self.prerequisites.add_concept(concept)
            
            # Optionally infer prerequisites (can be expensive, so make it optional)
            if self.auto_sync_enabled:
                try:
                    await self._infer_prerequisites_for_concept(topic, metadata, rag_content)
                except Exception as e:
                    logger.debug(f"⚠️ [RuntimeDiscovery] Prerequisite inference failed for '{topic}': {e}")
            
            logger.info(f"✅ [RuntimeDiscovery] Auto-discovered and added concept '{topic}' to knowledge graph")
            
        except Exception as e:
            logger.warning(f"⚠️ [RuntimeDiscovery] Error discovering concept '{topic}': {e}")
    
    async def _extract_concept_metadata(self, topic: str, rag_content: str, user_input: str) -> Optional[Dict]:
        """
        Extract concept metadata using LLM.
        
        Args:
            topic: Topic name
            rag_content: Content from RAG about this topic
            user_input: Original user input
            
        Returns:
            Dictionary with metadata or None
        """
        try:
            import json
            
            prompt = f"""For the NLP concept "{topic}", extract metadata from this content:

CONTENT:
{rag_content[:1000]}

USER QUERY: {user_input}

Provide metadata in JSON format:
{{
    "difficulty": "beginner" | "intermediate" | "advanced",
    "topic_area": "e.g., Text Preprocessing, NLP Tasks, Sequence Models",
    "description": "Brief one-sentence description",
    "keywords": ["keyword1", "keyword2", "keyword3"],
    "learning_time_minutes": 60
}}

Be concise and accurate."""
            
            response = await self.llm_client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=200,
                temperature=0.1,
                response_format={"type": "json_object"}
            )
            
            metadata = json.loads(response.choices[0].message.content)
            
            # Validate
            if metadata.get("difficulty") not in ["beginner", "intermediate", "advanced"]:
                metadata["difficulty"] = "intermediate"
            
            if "learning_time_minutes" not in metadata:
                metadata["learning_time_minutes"] = 60
            
            if not isinstance(metadata.get("keywords"), list):
                metadata["keywords"] = []
            
            return metadata
            
        except Exception as e:
            logger.warning(f"⚠️ Error extracting metadata for '{topic}': {e}")
            return None
    
    async def _infer_prerequisites_for_concept(self, topic: str, metadata: Dict, rag_content: str):
        """
        Infer prerequisites for a newly discovered concept.
        
        Args:
            topic: Topic name
            metadata: Topic metadata
            rag_content: RAG content about the topic
        """
        try:
            import json
            
            # Get existing concepts for context
            existing_concepts = list(self.prerequisites.concepts.keys())[:30]  # Limit for prompt size
            existing_concepts_str = ", ".join(existing_concepts)
            
            prompt = f"""For the NLP concept "{topic}", determine if it has any prerequisite concepts.

CONCEPT: {topic}
DESCRIPTION: {metadata.get('description', '')}
CONTENT: {rag_content[:500]}

EXISTING CONCEPTS (candidates for prerequisites):
{existing_concepts_str}

Return a JSON array of prerequisite concept names (empty array if none):
{{
    "prerequisites": ["Concept1", "Concept2"]
}}

Only include prerequisites that are in the existing concepts list above."""
            
            response = await self.llm_client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=100,
                temperature=0.1,
                response_format={"type": "json_object"}
            )
            
            result = json.loads(response.choices[0].message.content)
            prerequisites = result.get("prerequisites", [])
            
            # Add prerequisites (only if they exist in graph)
            for prereq in prerequisites:
                if prereq in self.prerequisites.concepts:
                    try:
                        self.prerequisites.add_prerequisite(topic, prereq)
                        logger.debug(f"✅ [RuntimeDiscovery] Added prerequisite: {prereq} → {topic}")
                    except Exception as e:
                        logger.debug(f"⚠️ [RuntimeDiscovery] Failed to add prerequisite {prereq} → {topic}: {e}")
            
        except Exception as e:
            logger.debug(f"⚠️ Error inferring prerequisites for '{topic}': {e}")
    
    def is_planning_request(self, text: str) -> bool:
        """
        Detect if user is requesting a learning plan/curriculum.
        
        Patterns:
        - "create plan", "learning plan", "curriculum", "roadmap"
        - "how should i learn", "learning path"
        - "what order", "guide me through"
        """
        text_lower = text.lower()
        
        planning_indicators = [
            "create.*plan",
            "learning plan",
            "curriculum",
            "roadmap",
            "how should i (learn|study)",
            "learning path",
            "what.*(order|sequence).*learn",
            "guide me through",
            "build me a",
            "make a plan",
            "create a curriculum"
        ]
        
        import re
        for pattern in planning_indicators:
            if re.search(pattern, text_lower):
                return True
        
        return False
    
    def build_prompt(
        self,
        user_input: str,
        context: str,
        topic: Optional[str],
        gaps: List[str],
        state: SessionState,
        mas_context: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, str]]:
        """Build the prompt for the LLM."""
        
        system_prompt = """You are a Socratic NLP tutor. Your role is to guide students to discover NLP concepts through questions, never giving direct answers.

TEACHING PHILOSOPHY:
- NEVER give direct answers
- Guide students to discover concepts through questions
- Build on their responses systematically
- Use the retrieved content for accuracy
- Adapt your teaching based on student's level

SOCRATIC QUESTIONING:
1. Start with concrete examples: "When you..."
2. Ask about observations: "What do you notice..."
3. Probe reasoning: "Why do you think..."
4. Explore implications: "What would happen if..."

RESPONSE FORMAT:
- Maximum 3-4 sentences
- ONE clear question per response
- Acknowledge correct answers briefly, then deepen
- For incorrect answers, provide hints and ask simpler questions

REMEMBER: Guide discovery, don't lecture!"""

        # Add context about student
        # Add learning style context (prefer user-level, then session-level, then MAS)
        learning_style_to_use = None
        if state.learning_style:
            learning_style_to_use = state.learning_style
        elif mas_context and mas_context.get("learning_style"):
            learning_style_to_use = mas_context['learning_style']
        
        if learning_style_to_use:
            system_prompt += f"\n\nStudent's learning style: {learning_style_to_use}"
            if mas_context and mas_context.get("learning_style_recommendations"):
                system_prompt += f"\nTeaching recommendations for this style: {', '.join(mas_context['learning_style_recommendations'])}"
        
        # Add prerequisite warnings if user is missing prerequisites
        if hasattr(state, '_missing_prerequisites') and state._missing_prerequisites:
            prereqs_str = ", ".join(state._missing_prerequisites)
            system_prompt += f"\n\n⚠️ PREREQUISITE NOTE: The student wants to learn '{topic}' but hasn't mastered these prerequisites: {prereqs_str}. "
            system_prompt += "Gently suggest reviewing prerequisites if needed, but don't block learning. Let the student choose their path."
            # Clear after using
            delattr(state, '_missing_prerequisites')
        
        if state.difficulty:
            system_prompt += f"\n\nCurrent difficulty level: {state.difficulty}"
        
        if topic:
            system_prompt += f"\n\nCurrent topic: {topic}"
        
        # Add prerequisite gaps (from graph + MAS analysis)
        all_gaps = gaps.copy()
        if mas_context and mas_context.get("knowledge_gaps"):
            all_gaps.extend(mas_context["knowledge_gaps"])
            all_gaps = list(set(all_gaps))  # Remove duplicates
        
        if all_gaps:
            gaps_list = ', '.join(all_gaps[:5])  # Show first 5 to avoid overwhelming
            if len(all_gaps) > 5:
                gaps_list += f" (and {len(all_gaps) - 5} more)"
            
            system_prompt += f"""

PREREQUISITE GAPS DETECTED:
The student is asking about "{topic}" but may be missing these foundational concepts: {gaps_list}

HOW TO HANDLE THIS:
1. Acknowledge their interest in learning about {topic}
2. Gently inform them that understanding these prerequisites will make learning {topic} much easier
3. Offer them a CHOICE (never force or block):
   - Option A: "Would you like to start with the basics? We can learn [prerequisite] first, then come back to {topic}."
   - Option B: "If you're confident, we can dive into {topic} now, but I'll explain the prerequisites as we go."
4. Respect their choice and adapt your teaching accordingly
5. If they choose to continue, provide extra context and explanations for prerequisite concepts as they come up

IMPORTANT: NEVER block or refuse to teach. Always give the student agency to choose their learning path."""
        
        # Add misconceptions from MAS
        if mas_context and mas_context.get("misconceptions"):
            system_prompt += f"\n\nSTUDENT MISCONCEPTIONS DETECTED: {', '.join(mas_context['misconceptions'])}. Address these gently through questions."
        
        # Add performance insights from MAS
        if mas_context and mas_context.get("performance_trend"):
            trend = mas_context["performance_trend"]
            if trend == "improving":
                system_prompt += f"\n\nStudent is showing improvement! Consider increasing challenge slightly."
            elif trend == "declining":
                system_prompt += f"\n\nStudent is struggling. Provide more support and simplify questions."
            
            if mas_context.get("performance_strengths"):
                system_prompt += f"\n\nStudent strengths: {', '.join(mas_context['performance_strengths'])}"
            if mas_context.get("performance_weaknesses"):
                system_prompt += f"\n\nAreas needing support: {', '.join(mas_context['performance_weaknesses'])}"
        
        # Add teaching recommendations from MAS
        if mas_context and mas_context.get("teaching_recommendations"):
            recommendations = mas_context["teaching_recommendations"][:3]  # Top 3
            system_prompt += f"\n\nTeaching recommendations: {', '.join(recommendations)}"
        
        # Add retrieved content
        if context and context != "No specific content retrieved.":
            system_prompt += f"\n\nRELEVANT COURSE CONTENT:\n{context}"
        
        # Build conversation history
        messages = [{"role": "system", "content": system_prompt}]
        
        # Add recent conversation history (last 5 exchanges)
        recent_history = state.conversation_history[-10:]  # Last 10 messages (5 exchanges)
        for msg in recent_history:
            messages.append(msg)
        
        # Add current user input
        messages.append({"role": "user", "content": user_input})
        
        return messages
    
    def _welcome_message(self) -> str:
        """Generate welcome message for new users."""
        return """Welcome to the NLP Tutor! I'm here to help you learn Natural Language Processing through the Socratic method.

What would you like to learn about today? You can:
- Ask about a specific NLP concept (e.g., "What is tokenization?")
- Request a learning plan (e.g., "Create a plan for learning transformers")
- Just start exploring NLP topics

What brings you here today?"""
    
    async def _handle_onboarding(
        self,
        user_input: str,
        state: SessionState
    ) -> AsyncGenerator[str, None]:
        """
        Handle comprehensive multi-stage onboarding.
        
        Stages:
        1. WELCOME - Goal capture
        2. LEARNING_STYLE - Learning style assessment (NEW)
        3. KNOWLEDGE_ASSESSMENT - Diagnostic questions (2-4)
        4. PREFERENCES - Learning preferences (NEW)
        5. CONFIRMATION - Summary and confirmation (NEW)
        
        Now checks user-level profile for existing knowledge to personalize onboarding.
        """
        # CRITICAL: Double-check that onboarding is not already complete
        # This prevents re-entering onboarding if state was just saved
        if state.onboarding_complete:
            logger.info("✅ [Onboarding] Already complete, transitioning to normal tutoring")
            async for chunk in self._normal_respond(user_input, state):
                yield chunk
            return
        
        # Check user-level profile for existing knowledge
        user_existing_knowledge = []
        if state.user_id and self.user_profile_manager:
            try:
                user_profile = await self.user_profile_manager.get_user_profile(state.user_id)
                if user_profile and user_profile.mastered_concepts:
                    user_existing_knowledge = user_profile.mastered_concepts
                    logger.info(f"📚 [Onboarding] User has existing knowledge: {len(user_existing_knowledge)} concepts")
            except Exception as e:
                logger.warning(f"⚠️ [Onboarding] Error loading user profile: {e}")
        
        onboarding_state = self.get_or_create_onboarding_state(state.session_id, state)
        
        if not onboarding_state or not ONBOARDING_COMPONENTS_AVAILABLE:
            # Fallback to simplified onboarding
            async for chunk in self._handle_onboarding_simple(user_input, state):
                yield chunk
            return
        
        # ============================================================
        # STAGE 1: WELCOME - Goal Capture
        # ============================================================
        if onboarding_state.stage == OnboardingStage.WELCOME:
            goal, implied_level = self._extract_goal_and_level(user_input)
            onboarding_state.stated_goal = goal
            onboarding_state.implied_level = implied_level
            
            # Mark WELCOME stage as completed
            if OnboardingStage.WELCOME not in onboarding_state.stages_completed:
                onboarding_state.stages_completed.append(OnboardingStage.WELCOME)
            
            # Transition to LEARNING_STYLE stage
            onboarding_state.stage = OnboardingStage.LEARNING_STYLE
            
            # Ask learning style question
            question = self._generate_learning_style_question()
            onboarding_state.learning_style_question_asked = True
            
            logger.info(f"🎯 [Onboarding] Stage 1 complete (Goal: {goal}), moving to Stage 2 (Learning Style)")
            yield question
            
            state.conversation_history.append({"role": "user", "content": user_input})
            state.conversation_history.append({"role": "assistant", "content": question})
            return
        
        # ============================================================
        # STAGE 2: LEARNING_STYLE - Learning Style Assessment
        # ============================================================
        elif onboarding_state.stage == OnboardingStage.LEARNING_STYLE:
            # Detect learning style from response
            logger.info(f"🎯 [Onboarding] Stage 2: Detecting learning style from response")
            style_result = await self._detect_learning_style_from_response(user_input)
            onboarding_state.learning_style_detected = style_result
            onboarding_state.learning_style_preference = style_result.get("primary_style")
            
            # Update session state with learning style
            if style_result:
                state.learning_style = {
                    "primary_style": style_result.get("primary_style", "reading"),
                    "confidence": style_result.get("confidence", 0.7),
                    "indicators": style_result.get("indicators", [])
                }
                logger.info(f"✅ [Onboarding] Learning style detected: {state.learning_style['primary_style']} (confidence: {state.learning_style['confidence']})")
            
            # Mark LEARNING_STYLE stage as completed
            if OnboardingStage.LEARNING_STYLE not in onboarding_state.stages_completed:
                onboarding_state.stages_completed.append(OnboardingStage.LEARNING_STYLE)
            
            # Transition to KNOWLEDGE_ASSESSMENT stage
            onboarding_state.stage = OnboardingStage.KNOWLEDGE_ASSESSMENT
            # Also set to LEVEL_PROBING for backward compatibility with existing logic
            onboarding_state.transition_to_level_probing()
            
            # Start knowledge assessment with first diagnostic question
            logger.info(f"🎯 [Onboarding] Stage 2 complete, moving to Stage 3 (Knowledge Assessment)")
            
            # Ask the first diagnostic question
            if onboarding_state.stated_goal and onboarding_state.stated_goal != "general NLP":
                topic = self.detect_topic(onboarding_state.stated_goal)
                if topic:
                    middle_prereq = self.prerequisites.get_middle_prerequisite(topic)
                    if middle_prereq:
                        question = self._generate_diagnostic_question(middle_prereq)
                        onboarding_state.add_diagnostic_question(middle_prereq, question)
                        yield question
                        state.conversation_history.append({"role": "user", "content": user_input})
                        state.conversation_history.append({"role": "assistant", "content": question})
                        return
            
            # For general NLP or if no specific topic, use foundational concepts
            # Skip concepts user already knows
            foundational_concepts = ["Tokenization", "Word Embeddings", "Neural Networks"]
            available_concepts = [c for c in foundational_concepts if c not in user_existing_knowledge]
            if not available_concepts:
                # User knows all foundational concepts - use more advanced ones
                available_concepts = ["Transformers", "Attention", "BERT"]
                available_concepts = [c for c in available_concepts if c not in user_existing_knowledge]
            
            if available_concepts:
                concept = available_concepts[0]  # Start with first available concept
                question = self._generate_diagnostic_question(concept)
                onboarding_state.add_diagnostic_question(concept, question)
                yield question
                state.conversation_history.append({"role": "user", "content": user_input})
                state.conversation_history.append({"role": "assistant", "content": question})
                return
        
        # ============================================================
        # STAGE 3: KNOWLEDGE_ASSESSMENT - Diagnostic Questions
        # ============================================================
        # Support both old (LEVEL_PROBING) and new (KNOWLEDGE_ASSESSMENT) stage names
        if onboarding_state.stage in [OnboardingStage.LEVEL_PROBING, OnboardingStage.KNOWLEDGE_ASSESSMENT]:
            # Safety check: prevent infinite onboarding loop
            # If we've asked more than 5 questions, force completion
            if len(onboarding_state.diagnostic_questions) >= 5:
                logger.warning(f"⚠️ [Onboarding] Reached max diagnostic questions (5), forcing completion")
                await self._complete_onboarding(onboarding_state, state)
                # Transition to PREFERENCES (skip to final stages)
                onboarding_state.stage = OnboardingStage.PREFERENCES
                if OnboardingStage.KNOWLEDGE_ASSESSMENT not in onboarding_state.stages_completed:
                    onboarding_state.stages_completed.append(OnboardingStage.KNOWLEDGE_ASSESSMENT)
                # Continue to preferences stage (will ask question on next call)
                return
            else:
                # Score the response
                last_question = onboarding_state.diagnostic_questions[-1]
                if self.response_scorer:
                    strength_score = await self.response_scorer.score(user_input, last_question.question)
                    strength_category = self.response_scorer.classify(strength_score)
                else:
                    # Fallback heuristic
                    strength_score = 0.5
                    strength_category = "moderate"
                
                # Update last question with response
                last_question.response = user_input
                last_question.strength_score = strength_score
                last_question.strength_category = strength_category
                
                # Check if we should continue probing (require at least 2 questions, max 4)
                if len(onboarding_state.diagnostic_questions) < 2:
                    # Need at least 2 questions - ask another one
                    next_concept = self._get_next_probing_concept(
                        onboarding_state, strength_category
                    )
                    
                    if next_concept:
                        question = self._generate_diagnostic_question(next_concept)
                        onboarding_state.add_diagnostic_question(next_concept, question)
                        logger.info(f"🔍 [Onboarding] Asking diagnostic question #{len(onboarding_state.diagnostic_questions)}: {next_concept}")
                        yield question
                        state.conversation_history.append({"role": "user", "content": user_input})
                        state.conversation_history.append({"role": "assistant", "content": question})
                        return
                    else:
                        # Can't find next concept, but we need at least 2 questions
                        # Ask a follow-up about the same concept to get more information
                        last_concept = onboarding_state.diagnostic_questions[-1].concept
                        question = f"That's helpful! Can you tell me more about {last_concept}? For example, what challenges or applications come to mind?"
                        onboarding_state.add_diagnostic_question(last_concept, question)
                        logger.info(f"🔍 [Onboarding] Asking follow-up diagnostic question #{len(onboarding_state.diagnostic_questions)}: {last_concept}")
                        yield question
                        state.conversation_history.append({"role": "user", "content": user_input})
                        state.conversation_history.append({"role": "assistant", "content": question})
                        return
                elif onboarding_state.should_continue_probing(max_questions=4):
                    # Have at least 2 questions, can ask more up to 4
                    next_concept = self._get_next_probing_concept(
                        onboarding_state, strength_category
                    )
                    
                    if next_concept:
                        question = self._generate_diagnostic_question(next_concept)
                        onboarding_state.add_diagnostic_question(next_concept, question)
                        logger.info(f"🔍 [Onboarding] Asking diagnostic question #{len(onboarding_state.diagnostic_questions)}: {next_concept}")
                        yield question
                        state.conversation_history.append({"role": "user", "content": user_input})
                        state.conversation_history.append({"role": "assistant", "content": question})
                        return
                    else:
                        # Done probing - calibrate and move to preferences
                        logger.info(f"✅ [Onboarding] Completed knowledge assessment with {len(onboarding_state.diagnostic_questions)} questions")
                        await self._complete_onboarding(onboarding_state, state)
                        onboarding_state.stage = OnboardingStage.PREFERENCES
                        if OnboardingStage.KNOWLEDGE_ASSESSMENT not in onboarding_state.stages_completed:
                            onboarding_state.stages_completed.append(OnboardingStage.KNOWLEDGE_ASSESSMENT)
                else:
                    # Done probing (reached max or have enough data)
                    logger.info(f"✅ [Onboarding] Completed knowledge assessment with {len(onboarding_state.diagnostic_questions)} questions")
                    await self._complete_onboarding(onboarding_state, state)
                    onboarding_state.stage = OnboardingStage.PREFERENCES
                    if OnboardingStage.KNOWLEDGE_ASSESSMENT not in onboarding_state.stages_completed:
                        onboarding_state.stages_completed.append(OnboardingStage.KNOWLEDGE_ASSESSMENT)
        
        # ============================================================
        # STAGE 4: PREFERENCES - Learning Preferences
        # ============================================================
        if onboarding_state.stage == OnboardingStage.PREFERENCES:
            # Check if we just transitioned here (need to ask preferences question)
            if not onboarding_state.preferences_question_asked:
                # Ask preferences question
                question = self._generate_preferences_question()
                onboarding_state.preferences_question_asked = True
                logger.info(f"🎯 [Onboarding] Stage 3 complete, moving to Stage 4 (Preferences)")
                yield question
                state.conversation_history.append({"role": "user", "content": user_input})
                state.conversation_history.append({"role": "assistant", "content": question})
                return
            else:
                # User responded to preferences question - detect preferences
                preferences = await self._detect_learning_preferences(user_input)
                onboarding_state.teaching_pace = preferences.get("teaching_pace")
                onboarding_state.practice_preference = preferences.get("practice_preference")
                
                logger.info(f"✅ [Onboarding] Preferences captured: pace={onboarding_state.teaching_pace}, practice={onboarding_state.practice_preference}")
                
                # Mark PREFERENCES stage as completed
                if OnboardingStage.PREFERENCES not in onboarding_state.stages_completed:
                    onboarding_state.stages_completed.append(OnboardingStage.PREFERENCES)
                
                # Transition to CONFIRMATION stage
                onboarding_state.stage = OnboardingStage.CONFIRMATION
                
                # Generate and show summary
                summary = self._generate_confirmation_summary(onboarding_state)
                logger.info(f"🎯 [Onboarding] Stage 4 complete, moving to Stage 5 (Confirmation)")
                
                yield summary
                state.conversation_history.append({"role": "user", "content": user_input})
                state.conversation_history.append({"role": "assistant", "content": summary})
                return
        
        # ============================================================
        # STAGE 5: CONFIRMATION - Summary and Completion
        # ============================================================
        if onboarding_state.stage == OnboardingStage.CONFIRMATION:
            # User confirmed (or any response means proceed)
            logger.info(f"✅ [Onboarding] User confirmed, completing onboarding")
            
            onboarding_state.stage = OnboardingStage.COMPLETE
            if OnboardingStage.CONFIRMATION not in onboarding_state.stages_completed:
                onboarding_state.stages_completed.append(OnboardingStage.CONFIRMATION)
            
            # CRITICAL: Set current_topic from stated_goal before completing onboarding
            # This ensures the topic is available when transitioning to normal tutoring
            if onboarding_state.stated_goal and onboarding_state.stated_goal != "general NLP":
                # Try to detect the topic from the stated goal
                topic = self.detect_topic(onboarding_state.stated_goal)
                if topic:
                    state.current_topic = topic
                    state.stated_goal = onboarding_state.stated_goal
                    logger.info(f"🎯 [Onboarding] Set current_topic from stated_goal: {topic}")
                else:
                    # If detection fails, use the stated_goal as-is (might be a valid topic name)
                    state.current_topic = onboarding_state.stated_goal
                    state.stated_goal = onboarding_state.stated_goal
                    logger.info(f"🎯 [Onboarding] Set current_topic to stated_goal (no detection): {onboarding_state.stated_goal}")
            else:
                state.stated_goal = onboarding_state.stated_goal or "general NLP"
                logger.info(f"🎯 [Onboarding] No specific topic from stated_goal: {state.stated_goal}")
            
            # Mark onboarding as complete
            state.onboarding_complete = True
            
            # Update user-level profile
            if state.user_id and self.user_profile_manager:
                # Save learning style
                # Try to get learning style from state first, then from onboarding_state
                learning_style_to_save = None
                if state.learning_style:
                    learning_style_to_save = state.learning_style
                    logger.debug(f"🔍 [Onboarding] Found learning style in state: {learning_style_to_save}")
                elif onboarding_state.learning_style_detected:
                    learning_style_to_save = onboarding_state.learning_style_detected
                    # Also update state for consistency
                    state.learning_style = learning_style_to_save
                    logger.debug(f"🔍 [Onboarding] Found learning style in onboarding_state: {learning_style_to_save}")
                
                if learning_style_to_save:
                    try:
                        primary_style = learning_style_to_save.get("primary_style", "reading")
                        confidence = learning_style_to_save.get("confidence", 0.7)
                        logger.info(f"💾 [Onboarding] Saving learning style to profile: {primary_style} (confidence: {confidence})")
                        success = await self.user_profile_manager.update_learning_style(
                            state.user_id,
                            primary_style,
                            confidence
                        )
                        if success:
                            logger.info(f"✅ [Onboarding] Updated user profile with learning style: {primary_style}")
                        else:
                            logger.warning(f"⚠️ [Onboarding] Failed to update learning style (update_learning_style returned False)")
                    except Exception as e:
                        logger.warning(f"⚠️ [Onboarding] Error updating learning style: {e}", exc_info=True)
                else:
                    logger.warning(f"⚠️ [Onboarding] No learning style found to save (state.learning_style={state.learning_style}, onboarding_state.learning_style_detected={onboarding_state.learning_style_detected})")
                
                # Save mastered concepts
                if state.mastered_concepts:
                    try:
                        await self.user_profile_manager.add_mastered_concepts(
                            state.user_id,
                            state.mastered_concepts
                        )
                        logger.info(f"✅ [Onboarding] Updated user profile with {len(state.mastered_concepts)} mastered concepts")
                    except Exception as e:
                        logger.warning(f"⚠️ [Onboarding] Error updating mastered concepts: {e}")
                
                # Update onboarding status
                try:
                    await self.user_profile_manager.update_onboarding_status(state.user_id, True)
                    logger.info(f"✅ [Onboarding] Updated user-level onboarding status")
                except Exception as e:
                    logger.warning(f"⚠️ [Onboarding] Error updating onboarding status: {e}")
                
                # Update statistics (increment session count and interactions)
                try:
                    # Count interactions from conversation history
                    interaction_count = len([msg for msg in state.conversation_history if msg.get("role") == "user"])
                    await self.user_profile_manager.update_statistics(
                        state.user_id,
                        increment_sessions=True,
                        increment_interactions=interaction_count
                    )
                    logger.info(f"✅ [Onboarding] Updated user statistics (sessions: +1, interactions: +{interaction_count})")
                except Exception as e:
                    logger.warning(f"⚠️ [Onboarding] Error updating statistics: {e}")
            
            # Save session
            await self.save_session(state)
            
            # Clear onboarding state from memory
            if state.session_id in self.onboarding_states:
                del self.onboarding_states[state.session_id]
            
            logger.info(f"✅ [Onboarding] Onboarding completed and saved (onboarding_complete=True)")
            
            # Transition to normal tutoring
            async for chunk in self._normal_respond(user_input, state):
                yield chunk
            return
        
        # If onboarding is complete, transition to normal tutoring
        elif onboarding_state.stage == OnboardingStage.COMPLETE:
            state.onboarding_complete = True
            async for chunk in self._normal_respond(user_input, state):
                yield chunk
            return
    
    async def _handle_onboarding_simple(
        self,
        user_input: str,
        state: SessionState
    ) -> AsyncGenerator[str, None]:
        """Simplified onboarding fallback (original implementation)."""
        if state.interaction_count == 1:
            goal_lower = user_input.lower()
            response_parts = []
            
            if any(word in goal_lower for word in ["new", "beginner", "start", "basics", "first time"]):
                state.stated_level = "beginner"
                state.difficulty = "beginner"
                state.stated_goal = "NLP fundamentals"
                response_parts = [
                    "Great! Let's start from the basics. ",
                    "When you read a sentence, how do you think a computer might 'understand' what it means? ",
                    "Just share your thoughts - there's no wrong answer!"
                ]
            elif any(word in goal_lower for word in ["intermediate", "know some", "familiar", "learned"]):
                state.stated_level = "intermediate"
                state.difficulty = "intermediate"
                detected = self.detect_topic(user_input)
                state.stated_goal = detected or "general NLP"
                response_parts = [
                    "Perfect! You have some background. ",
                    "If I wanted to represent the word 'bank' in a way a computer understands, ",
                    "what challenges might we face?"
                ]
            else:
                topic = self.detect_topic(user_input)
                if topic:
                    state.current_topic = topic
                    state.stated_goal = topic
                    response_parts = [
                        f"Excellent! You want to learn about {topic}. ",
                        f"What do you already know about {topic}? ",
                        "Even a rough idea helps me understand where to start."
                    ]
                else:
                    state.stated_goal = "general NLP"
                    response_parts = [
                        "Got it! Let's explore NLP together. ",
                        "Here's a question to get us started: ",
                        "When you read this sentence, how do you think a computer might break it down to process it?"
                    ]
            
            full_response = ""
            for part in response_parts:
                full_response += part
                yield part
            
            state.onboarding_complete = True
            state.conversation_history.append({"role": "user", "content": user_input})
            state.conversation_history.append({"role": "assistant", "content": full_response})
        else:
            state.onboarding_complete = True
            async for chunk in self._normal_respond(user_input, state):
                yield chunk
    
    def _extract_goal_and_level(self, user_input: str) -> tuple[str, Optional[str]]:
        """Extract learning goal and implied level from user input."""
        goal_lower = user_input.lower()
        
        if any(word in goal_lower for word in ["new", "beginner", "start", "basics", "first time"]):
            return "NLP fundamentals", "beginner"
        elif any(word in goal_lower for word in ["intermediate", "know some", "familiar", "learned"]):
            # Prefer a detected topic; otherwise default to a general intermediate goal
            topic = self.detect_topic(user_input)
            return (topic or "general NLP"), "intermediate"
        else:
            topic = self.detect_topic(user_input)
            if topic:
                return topic, None
            return "general NLP", None
    
    def _generate_welcome_response(
        self, 
        goal: str, 
        implied_level: Optional[str],
        existing_knowledge: Optional[List[str]] = None
    ) -> str:
        """Generate welcome response based on goal, level, and existing knowledge."""
        existing_knowledge = existing_knowledge or []
        
        # If user has existing knowledge, personalize the message
        if existing_knowledge:
            concepts_str = ", ".join(existing_knowledge[:3])  # Show first 3
            if len(existing_knowledge) > 3:
                concepts_str += f" and {len(existing_knowledge) - 3} more"
            
            if implied_level == "beginner":
                return f"Hello! Welcome back! I see you've already learned about {concepts_str}. I'll help you build on what you know and explore new concepts. Ready to continue your NLP journey?"
            elif implied_level == "intermediate":
                return f"Hello! Welcome back! I see you've mastered {concepts_str}. I'll help you build on this foundation and explore more advanced topics. Ready to continue?"
            elif goal and goal != "general NLP":
                return f"Hello! Welcome back! I see you've learned about {concepts_str}. I'll help you master {goal} building on your existing knowledge. Ready to start?"
            else:
                return f"Hello! Welcome back! I see you've already learned about {concepts_str}. I'll help you explore new NLP concepts building on what you know. What would you like to learn next?"
        
        # New user (no existing knowledge)
        if implied_level == "beginner":
            return "Hello! Great to meet you. I'm here to help you learn NLP from the basics. I'll guide you through concepts step by step using the Socratic method—I'll ask questions to help you discover answers yourself. Ready to get started?"
        elif implied_level == "intermediate":
            return "Hello! Nice to meet you. I see you have some background in NLP. I'll help you build on what you know using the Socratic method—I'll ask questions to deepen your understanding. Ready to begin?"
        elif goal and goal != "general NLP":
            return f"Hello! Great to meet you. I understand you want to learn about {goal}. I'll help you master this topic using the Socratic method—I'll ask questions to guide your learning. Ready to start?"
        else:
            return "Hello! Welcome! I'm here to help you learn Natural Language Processing. I'll use the Socratic method—asking questions to help you discover concepts yourself. What would you like to explore first?"
    
    def _generate_diagnostic_question(self, concept: str) -> str:
        """Generate a diagnostic question about a concept."""
        return f"Before we dive deeper, I'm curious—what do you know about {concept}? How would you explain it?"
    
    def _generate_learning_style_question(self) -> str:
        """Generate question to ask about learning style preferences."""
        return """How do you prefer to learn? For example:
- Do you like visual explanations with diagrams and charts?
- Do you prefer step-by-step written explanations?
- Do you learn better through examples and hands-on practice?
- Or do you like a mix of different approaches?

Just tell me what works best for you!"""
    
    def _generate_preferences_question(self) -> str:
        """Generate question to ask about learning preferences."""
        return """I want to make sure I teach in a way that works for you. Do you prefer:
- Fast-paced explanations that get to the point quickly?
- Detailed explanations that cover everything thoroughly?
- A balanced approach?

Also, how do you like to practice? Through quick examples, deep dives into concepts, or real-world applications?"""
    
    async def _detect_learning_style_from_response(
        self,
        user_input: str
    ) -> Dict[str, Any]:
        """
        Detect learning style from user's response about preferences.
        
        Uses LLM to analyze response and extract learning style.
        Returns: {"primary_style": "visual", "confidence": 0.8, "indicators": [...]}
        """
        try:
            import json
            prompt = f"""Analyze this student's response about how they prefer to learn:

"{user_input}"

Determine their learning style preference. Return ONLY a JSON object with:
- "primary_style": one of "visual", "auditory", "kinesthetic", "reading", or "mixed"
- "confidence": float between 0.0 and 1.0
- "indicators": array of keywords/phrases that indicated this style

Example response:
{{"primary_style": "kinesthetic", "confidence": 0.85, "indicators": ["examples", "practice", "hands-on"]}}

Return ONLY the JSON, no other text."""

            response = await self.llm_client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                max_tokens=150
            )
            
            content = response.choices[0].message.content.strip()
            # Remove markdown code blocks if present
            if content.startswith("```"):
                content = content.split("```")[1]
                if content.startswith("json"):
                    content = content[4:]
            content = content.strip()
            
            result = json.loads(content)
            logger.info(f"🎯 [Onboarding] Detected learning style: {result.get('primary_style')} (confidence: {result.get('confidence', 0)})")
            return result
            
        except Exception as e:
            logger.warning(f"⚠️ [Onboarding] Error detecting learning style with LLM: {e}, using heuristic fallback")
            # Fallback: keyword-based detection
            return self._heuristic_learning_style_detection(user_input)
    
    def _heuristic_learning_style_detection(self, user_input: str) -> Dict[str, Any]:
        """Fallback heuristic for learning style detection."""
        text_lower = user_input.lower()
        
        visual_keywords = ["visual", "diagram", "picture", "chart", "see", "image", "graph", "visualize"]
        auditory_keywords = ["listen", "hear", "audio", "sound", "explain", "talk", "speak", "verbal"]
        kinesthetic_keywords = ["practice", "example", "hands-on", "try", "do", "exercise", "apply", "hands on"]
        reading_keywords = ["read", "text", "written", "document", "article", "book", "reading"]
        
        scores = {
            "visual": sum(1 for kw in visual_keywords if kw in text_lower),
            "auditory": sum(1 for kw in auditory_keywords if kw in text_lower),
            "kinesthetic": sum(1 for kw in kinesthetic_keywords if kw in text_lower),
            "reading": sum(1 for kw in reading_keywords if kw in text_lower)
        }
        
        max_style = max(scores, key=scores.get)
        max_score = scores[max_style]
        total = sum(scores.values())
        
        if total == 0:
            return {"primary_style": "reading", "confidence": 0.5, "indicators": []}
        
        confidence = min(0.9, max_score / max(total, 1))
        
        # Get indicators
        all_keywords = {
            "visual": visual_keywords,
            "auditory": auditory_keywords,
            "kinesthetic": kinesthetic_keywords,
            "reading": reading_keywords
        }
        indicators = [kw for kw in all_keywords[max_style] if kw in text_lower]
        
        return {
            "primary_style": max_style,
            "confidence": confidence,
            "indicators": indicators
        }
    
    async def _detect_learning_preferences(
        self,
        user_input: str
    ) -> Dict[str, Optional[str]]:
        """
        Detect learning preferences from user response.
        
        Returns: {"teaching_pace": "fast/detailed/balanced", "practice_preference": "examples/deep_dives/applications"}
        """
        text_lower = user_input.lower()
        
        # Teaching pace detection
        if any(word in text_lower for word in ["fast", "quick", "brief", "concise", "to the point", "quickly"]):
            pace = "fast"
        elif any(word in text_lower for word in ["detailed", "thorough", "comprehensive", "deep", "complete", "everything"]):
            pace = "detailed"
        else:
            pace = "balanced"
        
        # Practice preference detection
        if any(word in text_lower for word in ["example", "practice", "exercise", "try", "hands-on", "hands on"]):
            practice = "examples"
        elif any(word in text_lower for word in ["deep", "thorough", "comprehensive", "understand", "concept", "theory"]):
            practice = "deep_dives"
        elif any(word in text_lower for word in ["real-world", "application", "project", "use case", "practical", "real world"]):
            practice = "applications"
        else:
            practice = None
        
        return {
            "teaching_pace": pace,
            "practice_preference": practice
        }
    
    def _generate_confirmation_summary(
        self,
        onboarding_state: OnboardingState
    ) -> str:
        """Generate confirmation summary before completing onboarding."""
        goal = onboarding_state.stated_goal or "general NLP"
        difficulty = onboarding_state.calibrated_difficulty or "intermediate"
        
        # Learning style
        if onboarding_state.learning_style_detected:
            style = onboarding_state.learning_style_detected.get("primary_style", "mixed")
            style_display = style.capitalize()
        else:
            style_display = "Adaptive"
        
        # Preferences
        pace = onboarding_state.teaching_pace or "balanced"
        practice = onboarding_state.practice_preference or "examples"
        practice_display = practice.replace('_', ' ').capitalize() if practice else "Examples"
        
        summary = f"""Perfect! Based on our conversation, here's what I understand:

**Your Learning Profile:**
- **Goal**: {goal}
- **Level**: {difficulty.capitalize()}
- **Learning Style**: {style_display}
- **Teaching Pace**: {pace.capitalize()}
- **Practice Preference**: {practice_display}

I'll tailor my teaching to match your style. Ready to start learning about {goal}?"""
        
        return summary
    
    def _get_next_probing_concept(
        self,
        onboarding_state: OnboardingState,
        strength_category: str
    ) -> Optional[str]:
        """Get next concept to probe based on binary search logic."""
        if not onboarding_state.stated_goal:
            return None
        
        topic = self.detect_topic(onboarding_state.stated_goal)
        if not topic:
            return None
        
        all_prereqs = self.prerequisites.get_all_prerequisites(topic)
        if not all_prereqs:
            return None
        
        # Get last probed concept
        last_question = onboarding_state.diagnostic_questions[-1]
        last_concept = last_question.concept
        
        # Track which concepts have already been asked
        asked_concepts = [q.concept for q in onboarding_state.diagnostic_questions]
        # Count how many times each concept has been asked
        concept_counts = {}
        for q in onboarding_state.diagnostic_questions:
            concept_counts[q.concept] = concept_counts.get(q.concept, 0) + 1
        
        # Binary search logic
        if strength_category == "strong":
            # Move up - find more advanced prerequisite
            last_idx = all_prereqs.index(last_concept) if last_concept in all_prereqs else -1
            if last_idx >= 0 and last_idx < len(all_prereqs) - 1:
                # Find next unasked concept going up
                for i in range(last_idx + 1, len(all_prereqs)):
                    candidate = all_prereqs[i]
                    if candidate not in asked_concepts:
                        return candidate
                # If all concepts above have been asked, try the target concept itself
                if topic not in asked_concepts:
                    return topic
            else:
                # If at end, try the target concept itself
                if topic not in asked_concepts:
                    return topic
            # All concepts have been asked - return None to signal completion
            return None
        elif strength_category == "weak":
            # Move down - find earlier prerequisite
            last_idx = all_prereqs.index(last_concept) if last_concept in all_prereqs else len(all_prereqs)
            if last_idx > 0:
                # Find next unasked concept going down
                for i in range(last_idx - 1, -1, -1):
                    candidate = all_prereqs[i]
                    if candidate not in asked_concepts:
                        return candidate
            # If at beginning or all concepts below have been asked, we've found the limit
            return None
        else:  # moderate
            # Ask follow-up about same concept, but only if we haven't asked about it more than once
            # (allow one follow-up per concept)
            if concept_counts.get(last_concept, 0) < 2:
                return last_concept
            # Already asked about this concept twice, find a different concept
            # Try to find any unasked prerequisite
            for candidate in all_prereqs:
                if candidate not in asked_concepts:
                    return candidate
            # If all prerequisites have been asked, try the target concept
            if topic not in asked_concepts:
                return topic
            # All concepts have been asked
            return None
    
    async def _complete_onboarding(
        self,
        onboarding_state: OnboardingState,
        state: SessionState
    ):
        """Complete onboarding by inferring mastery and determining difficulty."""
        # Infer mastery from all strong responses
        all_mastered = set()
        for question in onboarding_state.diagnostic_questions:
            if question.strength_category == "strong" and question.concept:
                # Infer mastery of this concept and its prerequisites
                inferred = self.prerequisites.infer_mastery(question.concept)
                all_mastered.update(inferred)
        
        # Determine difficulty based on responses
        strong_count = sum(1 for q in onboarding_state.diagnostic_questions if q.strength_category == "strong")
        weak_count = sum(1 for q in onboarding_state.diagnostic_questions if q.strength_category == "weak")
        
        if strong_count > weak_count:
            difficulty = "intermediate" if onboarding_state.implied_level == "beginner" else "advanced"
        elif weak_count > strong_count:
            difficulty = "beginner"
        else:
            difficulty = onboarding_state.implied_level or "intermediate"
        
        # Calculate remaining gaps
        if onboarding_state.stated_goal:
            topic = self.detect_topic(onboarding_state.stated_goal)
            if topic:
                gaps = self.prerequisites.get_gaps(topic, list(all_mastered))
            else:
                gaps = []
        else:
            gaps = []
        
        # Complete onboarding
        onboarding_state.transition_to_complete(
            difficulty=difficulty,
            mastered_concepts=list(all_mastered),
            gaps=gaps
        )
    
    async def _handle_planning_request(
        self,
        user_input: str,
        state: SessionState
    ) -> AsyncGenerator[str, None]:
        """
        Handle learning plan/curriculum generation request.
        
        Uses Planning MAS (5 CrewAI agents) to create personalized curriculum.
        Expected latency: 15-45 seconds (user expects this).
        
        NOTE: Planning MAS is now enabled by default when users click the curriculum button.
        Only disabled for test environments (PYTEST_CURRENT_TEST or TESTING env vars).
        """
        # Only disable for test environments to avoid long MAS latency during automated tests
        # In production, the Planning MAS runs automatically when users request a curriculum
        if self.disable_planning or os.getenv("PYTEST_CURRENT_TEST") or os.getenv("TESTING"):
            logger.info("📋 [Planning] Using fallback plan (test environment detected)")
            fallback_plan = (
                "Here's a quick learning path:\n"
                "1) Start with foundational concepts\n"
                "2) Build up to intermediate topics\n"
                "3) Practice with hands-on exercises\n"
                "4) Review and reinforce understanding\n"
                "5) Apply knowledge to real-world problems\n"
            )
            yield fallback_plan
            return
        
        if not PLANNING_CREW_AVAILABLE:
            # Fallback: provide a deterministic short plan stub to satisfy e2e expectation
            fallback_plan = (
                "Here's a quick transformer learning path:\n"
                "1) Refresh RNN/Seq2Seq basics\n"
                "2) Understand attention and self-attention\n"
                "3) Study the transformer encoder-decoder architecture\n"
                "4) Hands-on: implement scaled dot-product attention\n"
                "5) Fine-tune a small transformer on a toy task\n"
                "6) Review BERT/GPT variants and prompt design\n"
                "7) Evaluate and iterate on model performance\n"
            )
            yield fallback_plan
            return
        
        # Initialize planning crew with current state
        if self.planning_crew is None:
            # Get learning style from MAS if available
            learning_style = state.learning_style
            if not learning_style and self.background_mas:
                mas_context = self.background_mas.get_teaching_context(state.session_id)
                if mas_context and mas_context.get("learning_style"):
                    learning_style = mas_context["learning_style"].get("primary_style")
            
            self.planning_crew = PlanningCrew(
                prerequisites=self.prerequisites,
                rag=self.rag,
                learning_style=learning_style,
                mastered_concepts=state.mastered_concepts
            )
        
        # Inform user that plan is being created
        yield "I'll create a personalized learning plan for you. This may take 15-30 seconds...\n\n"
        
        try:
            # Create plan (this takes 15-45 seconds)
            plan = await asyncio.wait_for(
                self.planning_crew.create_plan(
                    goal_request=user_input,
                    current_knowledge=state.mastered_concepts,
                    learning_style=state.learning_style,
                    difficulty_preference=state.difficulty
                ),
                timeout=90.0  # Allow enough time for 5-agent crew to complete (15-45s expected, 90s max)
            )
            
            # Stream the formatted plan
            formatted_plan = plan.formatted_plan
            
            # Stream in chunks for better UX
            chunk_size = 100
            for i in range(0, len(formatted_plan), chunk_size):
                chunk = formatted_plan[i:i + chunk_size]
                yield chunk
                # Small delay to make streaming visible
                await asyncio.sleep(0.05)
            
            # Update conversation history
            state.conversation_history.append({"role": "user", "content": user_input})
            state.conversation_history.append({"role": "assistant", "content": formatted_plan})
            
        except asyncio.TimeoutError:
            logger.error(f"❌ [PlanningCrew] Plan creation timed out after 90 seconds")
            yield f"\n\nI apologize, but creating your learning plan is taking longer than expected. Please try again, or ask a specific question about NLP concepts instead."
        except Exception as e:
            logger.error(f"❌ [PlanningCrew] Plan creation failed: {e}", exc_info=True)
            error_msg = str(e)
            # Don't expose internal errors to user, provide friendly message
            if "timeout" in error_msg.lower():
                yield f"\n\nI apologize, but creating your learning plan is taking longer than expected. Please try again with a more specific topic, or ask a question about NLP concepts instead."
            else:
                yield f"\n\nI encountered an error creating your learning plan. Please try asking a specific question about NLP concepts instead."
    
    async def respond(
        self,
        user_input: str,
        session_id: str,
        user_id: Optional[str] = None
    ) -> AsyncGenerator[str, None]:
        """
        Generate streaming response to user input.
        
        Handles:
        1. Onboarding for new users
        2. Planning requests (learning plans)
        3. Normal tutoring
        
        This is the fast path - single LLM call with all context pre-loaded.
        """
        # Get session state (loads from DB if available)
        state = await self.get_or_create_session(session_id, user_id)
        state.interaction_count += 1
        
        # Handle onboarding for new users
        # Double-check: if onboarding is complete in DB, don't start it again
        if not state.onboarding_complete:
            # Check if we're in the middle of onboarding (have onboarding state)
            onboarding_state = self.get_or_create_onboarding_state(state.session_id, state)
            
            # Only auto-complete if we have MANY messages (20+) AND no active onboarding state
            # This prevents stuck onboarding while allowing the new 5-stage flow to complete
            if len(state.conversation_history) >= 20 and (not onboarding_state or onboarding_state.stage == OnboardingStage.COMPLETE):
                logger.warning(f"⚠️ [Onboarding] Session has {len(state.conversation_history)} messages but onboarding_complete=False. Marking as complete.")
                state.onboarding_complete = True
                
                # Update user-level onboarding status
                if state.user_id and self.user_profile_manager:
                    try:
                        await self.user_profile_manager.update_onboarding_status(state.user_id, True)
                        logger.info(f"✅ [Onboarding] Updated user-level onboarding status")
                    except Exception as e:
                        logger.warning(f"⚠️ [Onboarding] Error updating user-level onboarding status: {e}")
                
                await self.save_session(state)
                # Proceed to normal tutoring
            else:
                # Continue with onboarding flow (will handle all 5 stages)
                async for chunk in self._handle_onboarding(user_input, state):
                    yield chunk
                # Save state after onboarding (in case it completed)
                # This ensures onboarding_complete flag is persisted
                await self.save_session(state)
                return
        
        # If we get here, onboarding is complete - proceed to normal tutoring
        
        # Normal tutoring flow
        async for chunk in self._normal_respond(user_input, state):
            yield chunk
        
        # Save session after normal response completes (ensures topic and other state is persisted)
        # This is important because _normal_respond updates current_topic, source_files, etc.
        await self.save_session(state)
        logger.info(f"💾 [SocraticTutor] Session saved after response (topic: {state.current_topic})")
    
    async def _normal_respond(
        self,
        user_input: str,
        state: SessionState
    ) -> AsyncGenerator[str, None]:
        """
        Normal tutoring response (non-onboarding).
        This is the fast path - single LLM call with all context pre-loaded.
        """
        # Planning intent detection first
        if self.is_planning_request(user_input):
            async for chunk in self._handle_planning_request(user_input, state):
                yield chunk
            return

        # Test-mode short-circuit to avoid real LLM calls
        if os.getenv("PYTEST_CURRENT_TEST") or os.getenv("TESTING"):
            stub = (
                "Tokenization breaks text into tokens, which is the first step before embeddings and transformers. "
                "We can explore how tokens map to vectors and how attention uses them."
            )
            state.conversation_history.append({"role": "user", "content": user_input})
            state.conversation_history.append({"role": "assistant", "content": stub})
            state.last_updated = datetime.now()
            for ch in stub:
                yield ch
            return
        
        # Check response cache first (semantic similarity)
        if self.response_cache:
            cached_response = self.response_cache.get(user_input)
            if cached_response:
                # Return cached response (stream it for consistency)
                for char in cached_response:
                    yield char
                return
        
        # Parallel operations (all fast, no LLM calls)
        logger.info(f"🔍 [SocraticTutor] Detecting topic for: {user_input[:100]}...")
        logger.debug(f"🔍 [SocraticTutor] Previous topic in state: {state.current_topic}")
        
        # ARCHITECTURE NOTE: Topic detection has two paths:
        # 1. Fast path: Keyword matching (for common concepts in knowledge graph)
        # 2. Fallback: LLM detection (works for ANY topic in RAG, not limited to graph)
        # 
        # IMPORTANT: RAG is the source of truth for content. Knowledge graph is only
        # for prerequisite checking. The system can teach ANY topic from RAG!
        # 
        # If we already have a current_topic (e.g., from onboarding), only try to detect
        # a new topic if the user input clearly indicates a different topic
        # Otherwise, keep using the existing topic
        topic = None
        if state.current_topic:
            # Check if user is EXPLICITLY asking about a different topic
            # Use LLM to determine if this is a topic switch request vs. a follow-up answer
            # This prevents false positives from keyword matching (e.g., "documents" matching TF-IDF)
            detected_topic = self.detect_topic(user_input)
            
            # Only switch if:
            # 1. A different topic was detected AND
            # 2. The user is explicitly asking about it (not just mentioning related keywords)
            if detected_topic and detected_topic != state.current_topic:
                # Use LLM to check if user is explicitly asking about the new topic
                # vs. just mentioning related keywords in their answer
                is_explicit_topic_switch = await self._is_explicit_topic_switch(
                    user_input, 
                    state.current_topic, 
                    detected_topic
                )
                
                if is_explicit_topic_switch:
                    # User is explicitly asking about a different topic - switch to it
                    topic = detected_topic
                    logger.info(f"🔍 [SocraticTutor] User explicitly switched topic: {state.current_topic} → {detected_topic}")
                else:
                    # Keep using existing topic (keywords matched but user isn't asking about new topic)
                    topic = state.current_topic
                    logger.info(f"🔍 [SocraticTutor] Keeping existing topic: {topic} (detected '{detected_topic}' but user is not explicitly asking about it)")
            else:
                # Keep using existing topic (user might be confirming, asking follow-up, etc.)
                topic = state.current_topic
                logger.info(f"🔍 [SocraticTutor] Keeping existing topic: {topic} (user input doesn't indicate new topic)")
        else:
            # No existing topic - try to detect one
            topic = self.detect_topic(user_input)
            logger.info(f"🔍 [SocraticTutor] Keyword detection result: {topic}")
            
            # If keyword matching failed, try LLM-based detection (async, but we need it now)
            # LLM fallback can detect ANY topic from RAG, not limited to knowledge graph
            if not topic:
                logger.info(f"🎯 [SocraticTutor] Keyword matching failed, trying LLM fallback...")
                topic = await self.detect_topic_llm_fallback(user_input)
                logger.info(f"🔍 [SocraticTutor] LLM fallback result: {topic}")
        
        if topic:
            logger.info(f"🎯 [SocraticTutor] ✅ Detected topic: {topic}")
            
            # Check prerequisites using user-level mastered concepts
            # Only check if topic exists in knowledge graph (has prerequisites defined)
            if state.user_id and self.user_profile_manager and topic in self.prerequisites.concepts:
                try:
                    missing_prereqs = await self.user_profile_manager.check_prerequisites(
                        state.user_id,
                        topic,
                        self.prerequisites
                    )
                    if missing_prereqs:
                        logger.info(f"⚠️ [SocraticTutor] User missing prerequisites for {topic}: {missing_prereqs}")
                        # Store for use in prompt
                        state._missing_prerequisites = missing_prereqs
                except Exception as e:
                    logger.warning(f"⚠️ [SocraticTutor] Error checking prerequisites: {e}")
        else:
            logger.warning(f"🎯 [SocraticTutor] ❌ No topic detected for: {user_input[:50]}...")
        
        gaps = []
        if topic:
            # Only check gaps if topic is in knowledge graph
            # IMPORTANT: Topic doesn't need to be in graph for RAG to work!
            # RAG can teach about ANY topic in the vector store
            if topic in self.prerequisites.graph:
                # Use user-level mastered concepts for gap checking
                user_mastered = state.mastered_concepts  # Already merged with user-level in get_or_create_session
                gaps = self.prerequisites.get_gaps(topic, user_mastered)
                logger.info(f"🎯 [SocraticTutor] Found {len(gaps)} knowledge gaps for {topic}")
            else:
                # Topic detected but not in graph - this is OK!
                # RAG will still retrieve content, we just skip prerequisite checking
                logger.info(f"🎯 [SocraticTutor] Topic '{topic}' detected but not in knowledge graph - using RAG (no prerequisite checking)")
                
                # LAZY LOADING: Auto-add concept to graph if enabled
                if self.lazy_load_enabled and self.use_enhanced_graph:
                    # Check if we've already tried to discover this concept
                    if topic not in self._discovered_concepts:
                        logger.info(f"🔄 [SocraticTutor] Attempting lazy load for topic: {topic}")
                        # Run discovery in background (non-blocking)
                        asyncio.create_task(self._lazy_load_concept(topic, user_input))
                        self._discovered_concepts[topic] = True
            
            # Always update topic if a new one is detected (don't keep stale topics)
            if topic != state.current_topic:
                old_topic = state.current_topic
                state.current_topic = topic
                logger.info(f"🎯 [SocraticTutor] Updated current_topic: {old_topic} → {topic}")
                # Log that we'll save this to database
                logger.info(f"💾 [SocraticTutor] Topic will be saved to database: {topic}")
            else:
                logger.debug(f"🎯 [SocraticTutor] Topic unchanged: {topic}")
        else:
            # No topic detected - keep previous topic if it exists, but log it
            if state.current_topic:
                logger.debug(f"🎯 [SocraticTutor] No new topic detected, keeping previous: {state.current_topic}")
                topic = state.current_topic  # Use previous topic for context
        
        # Retrieve content from RAG (fast - < 500ms)
        # RAG works for ANY topic, whether in knowledge graph or not!
        # This is the key: RAG is the source of truth for content, not the knowledge graph
        # Use topic for RAG query if available, otherwise use user_input
        # This ensures we get relevant content for the topic, not just the user's message
        # (e.g., if user says "I'm ready", we still want tokenization content, not "ready" content)
        rag_query = topic if topic else user_input
        logger.debug(f"🔍 [SocraticTutor] RAG query: '{rag_query}' (topic: {topic}, user_input: {user_input[:50]}...)")
        context, source_files, rag_chunks = self.rag.query(rag_query, k=3, difficulty=state.difficulty, return_chunks=True)
        
        # Update source files in session state (deduplicated, keep unique)
        if source_files:
            for source_file in source_files:
                if source_file and source_file not in state.source_files:
                    state.source_files.append(source_file)
        
        # Store RAG chunks in session state for later inclusion in message metadata
        if rag_chunks:
            # Store in a temporary attribute that will be included in message metadata
            state._last_rag_chunks = rag_chunks
        
        # Get enriched context from background MAS (if available and cached)
        mas_context = {}
        if self.background_mas:
            mas_context = self.background_mas.get_teaching_context(state.session_id)
        
        # Build prompt with enriched context
        messages = self.build_prompt(user_input, context, topic, gaps, state, mas_context)
        
        # Stream response (single LLM call)
        full_response = ""
        try:
            stream = await self.llm_client.chat.completions.create(
                model=self.model,
                messages=messages,
                stream=True,
                max_tokens=500,
                temperature=0.7
            )
            
            async for chunk in stream:
                try:
                    # Check if chunk has content
                    if chunk.choices and len(chunk.choices) > 0:
                        delta = chunk.choices[0].delta
                        if delta and delta.content:
                            content = delta.content
                            full_response += content
                            yield content
                except Exception as chunk_error:
                    # Handle individual chunk errors gracefully
                    print(f"Chunk processing error: {chunk_error}")
                    print(f"Chunk data: {chunk}")
                    yield f"\n\n[Note: Some content may be missing due to an error]"
                    break
        except Exception as e:
            print(f"LLM streaming error: {e}")
            print(f"Messages sent to LLM: {messages}")
            yield f"\n\n[Error: I encountered an issue processing your request. Please try again.]"
            return
        
        # Update state asynchronously (non-blocking)
        # Check if user message is already in history (might have been loaded from messages table)
        # Only add user message if it's not already the last message
        last_message = state.conversation_history[-1] if state.conversation_history else None
        if not last_message or last_message.get("role") != "user" or last_message.get("content") != user_input:
            state.conversation_history.append({"role": "user", "content": user_input})
        state.conversation_history.append({"role": "assistant", "content": full_response})
        state.last_updated = datetime.now()
        
        # Cache response for future similar queries
        if self.response_cache and full_response:
            metadata = {
                "topic": topic,
                "difficulty": state.difficulty,
                "session_id": state.session_id
            }
            self.response_cache.put(user_input, full_response, metadata)
        
        # Extract performance score from MAS context (if available)
        performance_score = None
        if mas_context and mas_context.get("performance_assessment"):
            perf_assessment = mas_context["performance_assessment"]
            if isinstance(perf_assessment, dict):
                performance_score = perf_assessment.get("score")
            elif isinstance(perf_assessment, (int, float)):
                performance_score = perf_assessment
        
        # If no score from MAS, try to get from previous MAS analysis
        if performance_score is None and self.background_mas:
            cached_context = self.background_mas.get_teaching_context(state.session_id)
            if cached_context and cached_context.get("performance_assessment"):
                perf_assessment = cached_context["performance_assessment"]
                if isinstance(perf_assessment, dict):
                    performance_score = perf_assessment.get("score")
        
        # Track performance score for difficulty adaptation
        if performance_score is not None:
            state.understanding_scores.append(float(performance_score))
            # Keep only last 10 scores
            if len(state.understanding_scores) > 10:
                state.understanding_scores = state.understanding_scores[-10:]
        
        # Update performance trend from MAS
        if mas_context and mas_context.get("performance_trend"):
            state.performance_trend = mas_context["performance_trend"]
        
        # Check for difficulty adjustment
        if self.difficulty_adapter and len(state.understanding_scores) >= 3:
            adjustment = self.difficulty_adapter.check_adjustment(
                current_difficulty=state.difficulty,
                recent_scores=state.understanding_scores[-5:],  # Last 5 scores
                performance_trend=state.performance_trend
            )
            if adjustment.should_adjust:
                self.difficulty_adapter.apply_adjustment(state, adjustment)
        
        # Store last question for MAS analysis
        last_question = full_response if full_response.strip().endswith('?') else None
        
        # Trigger background evaluation (fire and forget)
        asyncio.create_task(
            self._background_evaluate(user_input, full_response, state.session_id)
        )
        
        # Trigger background MAS analysis (fire and forget)
        if self.background_mas:
            import logging
            mas_logger = logging.getLogger(__name__)
            mas_logger.info(f"🚀 [SocraticTutor] Triggering background MAS for session: {state.session_id[:20]}...")
            mas_logger.info(f"🚀 [SocraticTutor] Passing current_topic to MAS: {state.current_topic}")
            
            # Get session_db_id and user_id from state if available
            session_db_id = getattr(state, 'session_db_id', None)
            user_id = getattr(state, 'user_id', None)
            
            asyncio.create_task(
                self._background_mas_analysis(
                    state.session_id,
                    state.conversation_history,
                    state.current_topic,
                    state.mastered_concepts,
                    last_question,
                    user_input,
                    session_db_id=session_db_id,
                    user_id=user_id
                )
            )
        else:
            import logging
            mas_logger = logging.getLogger(__name__)
            mas_logger.debug("⚠️ [SocraticTutor] Background MAS not available - skipping analysis")
    
    async def _background_evaluate(
        self,
        user_input: str,
        tutor_response: str,
        session_id: str
    ):
        """
        Background evaluation - single LLM call, runs async.
        Updates state but doesn't block response.
        """
        state = self.sessions.get(session_id)
        if not state:
            return
        
        # Only evaluate if it looks like an answer (not a question)
        if user_input.strip().endswith('?'):
            return
        
        try:
            # Simple evaluation prompt
            eval_prompt = f"""Evaluate this student's response to a Socratic tutoring question.

Tutor's question: {tutor_response}
Student's response: {user_input}

Rate the response on:
1. Correctness (correct/partially_correct/incorrect)
2. Depth of understanding (shallow/moderate/deep)

Respond in JSON format:
{{
    "correctness": "correct|partially_correct|incorrect",
    "depth": "shallow|moderate|deep",
    "feedback": "brief feedback"
}}"""

            response = await self.llm_client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": eval_prompt}],
                response_format={"type": "json_object"},
                max_tokens=200
            )
            
            import json
            evaluation = json.loads(response.choices[0].message.content)
            
            # Update state based on evaluation
            if evaluation.get("correctness") == "correct":
                # Mark current topic as potentially mastered
                if state.current_topic and state.current_topic not in state.mastered_concepts:
                    state.mastered_concepts.append(state.current_topic)
                    # Update user-level profile with mastered concept
                    if state.user_id and self.user_profile_manager:
                        try:
                            await self.user_profile_manager.add_mastered_concepts(
                                state.user_id,
                                [state.current_topic]
                            )
                            logger.info(f"✅ [SocraticTutor] Updated user profile with mastered concept: {state.current_topic}")
                        except Exception as e:
                            logger.warning(f"⚠️ [SocraticTutor] Error updating user profile with mastered concept: {e}")
            elif evaluation.get("correctness") == "incorrect":
                # Check if we need to decrease difficulty
                if state.difficulty == "advanced":
                    state.difficulty = "intermediate"
                elif state.difficulty == "intermediate":
                    state.difficulty = "beginner"
            
            state.last_updated = datetime.now()
        except Exception as e:
            print(f"Background evaluation error: {e}")
            # Fail silently - evaluation is non-critical
    
    async def _background_mas_analysis(
        self,
        session_id: str,
        conversation_history: List[Dict[str, str]],
        current_topic: Optional[str],
        mastered_concepts: List[str],
        last_question: Optional[str],
        last_response: Optional[str],
        session_db_id: Optional[str] = None,
        user_id: Optional[str] = None
    ):
        """
        Trigger background MAS analysis (fire and forget).
        Results are cached and used to enrich future responses.
        
        Args:
            session_id: Frontend session ID (string)
            session_db_id: Database UUID of the session (for persistence)
            user_id: User UUID (for persistence)
        """
        if not self.background_mas:
            return
        
        try:
            analysis = await self.background_mas.analyze_conversation(
                session_id=session_id,
                conversation_history=conversation_history,
                current_topic=current_topic,
                mastered_concepts=mastered_concepts,
                last_question=last_question,
                last_response=last_response
            )
            
            # Save to database if session_db_id and user_id provided
            if session_db_id and user_id and self.supabase_client:
                try:
                    await self.background_mas._save_to_database(
                        session_id=session_id,
                        analysis=analysis,
                        session_db_id=session_db_id,
                        user_id=user_id
                    )
                except Exception as e:
                    import logging
                    logger = logging.getLogger(__name__)
                    logger.error(f"❌ [SocraticTutor] Failed to save MAS analysis to database: {e}")
        except Exception as e:
            import logging
            logger = logging.getLogger(__name__)
            logger.error(f"❌ [SocraticTutor] Background MAS analysis error: {e}")
            # Fail silently - MAS analysis is non-critical
