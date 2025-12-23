"""
Session Manager for State Persistence

Manages SessionState persistence using Supabase.
Replaces in-memory storage with database-backed storage.
"""

import json
from typing import Optional, Dict, Any
from datetime import datetime
from agentic_socratic_nlp_tutor.session_state import SessionState


class SessionManager:
    """
    Manages session state persistence in Supabase.
    
    Stores and retrieves SessionState objects from database,
    enabling persistence across server restarts and multi-instance deployments.
    """
    
    def __init__(self, supabase_client=None):
        """
        Initialize SessionManager.
        
        Args:
            supabase_client: Supabase client instance (optional)
        """
        self.supabase = supabase_client
        self.use_supabase = supabase_client is not None
        
        # Always initialize in-memory fallback (used in error cases)
        self._in_memory_sessions: Dict[str, SessionState] = {}
    
    def session_to_dict(self, session: SessionState) -> Dict[str, Any]:
        """
        Convert SessionState to dictionary for storage.
        
        Args:
            session: SessionState object
            
        Returns:
            Dictionary representation
        """
        return {
            "session_id": session.session_id,
            "current_topic": session.current_topic,
            # mastered_concepts removed - profiles table is source of truth (user-level)
            # learning_style removed - profiles table is source of truth (user-level)
            # difficulty removed - profiles.overall_difficulty is source of truth (user-level)
            # conversation_history removed - messages table is source of truth
            "source_files": json.dumps(session.source_files) if session.source_files else None,
            # onboarding_complete removed - profiles table is source of truth for user-level onboarding
            "interaction_count": session.interaction_count,
            "stated_goal": session.stated_goal,
            "stated_level": session.stated_level,
            "understanding_scores": json.dumps(session.understanding_scores) if session.understanding_scores else None,
            "performance_trend": session.performance_trend,
            "session_db_id": session.session_db_id,
            "user_id": session.user_id,
            "created_at": session.created_at.isoformat() if session.created_at else None,
            "last_updated": session.last_updated.isoformat() if session.last_updated else None,
        }
    
    def dict_to_session(self, data: Dict[str, Any]) -> SessionState:
        """
        Convert dictionary to SessionState object.
        
        Args:
            data: Dictionary from database
            
        Returns:
            SessionState object
        """
        # Parse JSON fields
        # mastered_concepts removed - will be loaded from user profile in get_or_create_session()
        source_files = json.loads(data.get("source_files") or "[]")
        # conversation_history is loaded from messages table in get_session(), not from sessions table
        understanding_scores = json.loads(data.get("understanding_scores") or "[]")
        
        # Parse datetime fields
        created_at = datetime.fromisoformat(data["created_at"]) if data.get("created_at") else datetime.now()
        last_updated = datetime.fromisoformat(data["last_updated"]) if data.get("last_updated") else datetime.now()
        
        return SessionState(
            session_id=data["session_id"],
            current_topic=data.get("current_topic"),
            # mastered_concepts removed - will be loaded from user profile in get_or_create_session()
            mastered_concepts=[],  # Default, will be overridden by user profile check
            source_files=source_files,
            # learning_style removed - will be loaded from user profile in get_or_create_session()
            learning_style=None,  # Default, will be overridden by user profile check
            # difficulty removed - will be loaded from user profile (overall_difficulty) in get_or_create_session()
            difficulty="intermediate",  # Default, will be overridden by user profile check
            conversation_history=[],  # Will be loaded from messages table in get_session()
            # onboarding_complete removed - will be set from user profile in get_or_create_session()
            onboarding_complete=False,  # Default, will be overridden by user profile check
            interaction_count=data.get("interaction_count", 0),
            stated_goal=data.get("stated_goal"),
            stated_level=data.get("stated_level"),
            understanding_scores=understanding_scores,
            performance_trend=data.get("performance_trend"),
            session_db_id=data.get("session_db_id"),
            user_id=data.get("user_id"),
            created_at=created_at,
            last_updated=last_updated
        )
    
    async def get_session(
        self,
        session_id: str,
        user_id: Optional[str] = None
    ) -> Optional[SessionState]:
        """
        Load session state from database.
        
        Also loads conversation history from messages table to ensure
        conversation_history is populated with actual messages.
        
        Args:
            session_id: Session identifier
            user_id: User ID (optional, for Supabase)
            
        Returns:
            SessionState object or None if not found
        """
        if not self.use_supabase:
            # Fallback to in-memory
            return self._in_memory_sessions.get(session_id)
        
        try:
            # Query Supabase sessions table
            query = self.supabase.table('sessions').select('*').eq('session_id', session_id)
            
            if user_id:
                query = query.eq('user_id', user_id)
            
            result = query.execute()
            
            if result.data and len(result.data) > 0:
                session_data = result.data[0]
                
                # Convert to SessionState
                session = self.dict_to_session(session_data)
                session.session_db_id = session_data.get("id")  # Database ID
                session.user_id = session_data.get("user_id")
                
                # CRITICAL FIX: Load conversation history from messages table
                # This ensures conversation_history is populated with actual messages
                # rather than relying on the (potentially stale) conversation_history in sessions table
                try:
                    if session.session_db_id:
                        messages_result = self.supabase.table('messages') \
                            .select('role, content') \
                            .eq('session_id', session.session_db_id) \
                            .order('created_at', desc=False) \
                            .execute()
                        
                        if messages_result.data:
                            # Convert messages to conversation_history format
                            # Format: [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]
                            conversation_history = [
                                {"role": msg["role"], "content": msg["content"]}
                                for msg in messages_result.data
                            ]
                            # Only update if we got messages (messages table is source of truth)
                            if conversation_history:
                                session.conversation_history = conversation_history
                                print(f"✅ [SessionManager] Loaded {len(conversation_history)} messages from messages table")
                except Exception as msg_error:
                    print(f"⚠️ Error loading messages for conversation_history: {msg_error}")
                    # Continue with session loaded from sessions table (may have stale history)
                
                return session
            
            return None
            
        except Exception as e:
            print(f"⚠️ Error loading session from database: {e}")
            # Fallback to in-memory
            return self._in_memory_sessions.get(session_id)
    
    async def save_session(self, session: SessionState) -> bool:
        """
        Save session state to database.
        
        Args:
            session: SessionState object to save
            
        Returns:
            True if saved successfully, False otherwise
        """
        if not self.use_supabase:
            # Fallback to in-memory
            self._in_memory_sessions[session.session_id] = session
            return True
        
        try:
            session_dict = self.session_to_dict(session)
            
            # Update last_updated
            session.last_updated = datetime.now()
            session_dict["last_updated"] = session.last_updated.isoformat()
            
            if session.session_db_id:
                # Update existing session
                # Only update session-specific fields (user-level fields are in profiles table)
                update_data = {
                    "current_topic": session_dict["current_topic"],
                    # mastered_concepts removed - stored in profiles table
                    # learning_style removed - stored in profiles table
                    # difficulty removed - stored in profiles.overall_difficulty
                    # conversation_history removed - messages table is source of truth
                    # onboarding_complete removed - stored in profiles table
                    "interaction_count": session_dict["interaction_count"],
                    "stated_goal": session_dict["stated_goal"],
                    "stated_level": session_dict["stated_level"],
                    "understanding_scores": session_dict["understanding_scores"],
                    "performance_trend": session_dict["performance_trend"],
                    "source_files": session_dict["source_files"],
                }
                # Only add last_updated if it exists in schema (will be added in migration)
                # For now, skip it to avoid errors
                # update_data["last_updated"] = session_dict["last_updated"]
                
                result = self.supabase.table('sessions').update(update_data).eq('id', session.session_db_id).execute()
            else:
                # Create new session
                # Remove None values and exclude last_updated (will be set by database default)
                insert_data = {k: v for k, v in session_dict.items() if v is not None and k != "last_updated"}
                
                result = self.supabase.table('sessions').insert(insert_data).execute()
                
                if result.data and len(result.data) > 0:
                    session.session_db_id = result.data[0].get("id")
                    session.user_id = result.data[0].get("user_id")
            
            return True
            
        except Exception as e:
            print(f"⚠️ Error saving session to database: {e}")
            # Fallback to in-memory
            self._in_memory_sessions[session.session_id] = session
            return False
    
    async def delete_session(self, session_id: str) -> bool:
        """
        Delete session from database.
        
        Args:
            session_id: Session identifier
            
        Returns:
            True if deleted, False otherwise
        """
        if not self.use_supabase:
            # Fallback to in-memory
            if session_id in self._in_memory_sessions:
                del self._in_memory_sessions[session_id]
                return True
            return False
        
        try:
            # Find session first to get database ID
            session = await self.get_session(session_id)
            if session and session.session_db_id:
                self.supabase.table('sessions').delete().eq('id', session.session_db_id).execute()
                return True
            return False
            
        except Exception as e:
            print(f"⚠️ Error deleting session: {e}")
            return False

