"""
User Profile Manager

Manages user-level learning data aggregated across all sessions.
Enables personalized learning that persists across sessions.
"""

import json
import logging
from typing import Optional, List, Set, Dict, Any
from datetime import datetime
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class UserProfile:
    """User-level learning profile."""
    user_id: str
    learning_style: Optional[str] = None
    learning_style_confidence: float = 0.0
    learning_style_updated_at: Optional[datetime] = None
    mastered_concepts: List[str] = None
    mastered_concepts_updated_at: Optional[datetime] = None
    overall_difficulty: str = "intermediate"
    difficulty_updated_at: Optional[datetime] = None
    total_sessions: int = 0
    total_interactions: int = 0
    last_activity: Optional[datetime] = None
    onboarding_complete: bool = False
    onboarding_completed_at: Optional[datetime] = None
    
    def __post_init__(self):
        if self.mastered_concepts is None:
            self.mastered_concepts = []


class UserProfileManager:
    """
    Manages user-level learning data.
    
    Aggregates data from all sessions to provide personalized learning
    that persists across sessions.
    """
    
    def __init__(self, supabase_client=None):
        """
        Initialize UserProfileManager.
        
        Args:
            supabase_client: Supabase client instance (optional)
        """
        self.supabase = supabase_client
        self.use_supabase = supabase_client is not None
        
        if not self.use_supabase:
            logger.warning("⚠️ [UserProfileManager] Supabase not available, using in-memory fallback")
    
    async def get_user_profile(self, user_id: str) -> Optional[UserProfile]:
        """
        Get user's learning profile.
        
        Args:
            user_id: User UUID
            
        Returns:
            UserProfile object or None if not found
        """
        if not self.use_supabase:
            return None
        
        try:
            result = self.supabase.table('profiles') \
                .select('*') \
                .eq('id', user_id) \
                .single() \
                .execute()
            
            if result.data:
                data = result.data
                return self._dict_to_profile(data)
            
            return None
        except Exception as e:
            logger.error(f"❌ [UserProfileManager] Error loading user profile: {e}")
            return None
    
    async def update_learning_style(
        self,
        user_id: str,
        style: str,
        confidence: float = 0.0
    ) -> bool:
        """
        Update user's learning style.
        
        Args:
            user_id: User UUID
            style: Learning style (visual, auditory, kinesthetic, reading)
            confidence: Confidence score (0.0-1.0)
            
        Returns:
            True if successful
        """
        if not self.use_supabase:
            return False
        
        try:
            # First, check if profile exists
            profile_check = self.supabase.table('profiles') \
                .select('id') \
                .eq('id', user_id) \
                .execute()
            
            if not profile_check.data:
                logger.warning(f"⚠️ [UserProfileManager] Profile not found for user {user_id[:20]}..., cannot update learning style")
                return False
            
            update_data = {
                'learning_style': style,
                'learning_style_confidence': confidence,
                'learning_style_updated_at': datetime.now().isoformat()
            }
            
            result = self.supabase.table('profiles') \
                .update(update_data) \
                .eq('id', user_id) \
                .execute()
            
            if result.data:
                logger.info(f"✅ [UserProfileManager] Updated learning style for user {user_id[:20]}...: {style} (confidence: {confidence})")
                return True
            else:
                logger.warning(f"⚠️ [UserProfileManager] Update returned no data for user {user_id[:20]}...")
                return False
        except Exception as e:
            logger.error(f"❌ [UserProfileManager] Error updating learning style: {e}", exc_info=True)
            return False
    
    async def add_mastered_concepts(
        self,
        user_id: str,
        new_concepts: List[str]
    ) -> bool:
        """
        Add new concepts to user's mastered list.
        
        Merges with existing concepts (no duplicates).
        
        Args:
            user_id: User UUID
            new_concepts: List of concept names to add
            
        Returns:
            True if successful
        """
        if not self.use_supabase or not new_concepts:
            return False
        
        try:
            # Get current profile
            profile = await self.get_user_profile(user_id)
            if not profile:
                # Create profile entry if it doesn't exist
                current_concepts = []
            else:
                current_concepts = profile.mastered_concepts or []
            
            # Merge: union of current and new concepts
            current_set = set(current_concepts)
            new_set = set(new_concepts)
            updated_concepts = sorted(list(current_set | new_set))
            
            # Update profile
            update_data = {
                'mastered_concepts': json.dumps(updated_concepts),
                'mastered_concepts_updated_at': datetime.now().isoformat()
            }
            
            self.supabase.table('profiles') \
                .update(update_data) \
                .eq('id', user_id) \
                .execute()
            
            logger.info(f"✅ [UserProfileManager] Added {len(new_concepts)} concepts to user {user_id[:20]}... (total: {len(updated_concepts)})")
            return True
        except Exception as e:
            logger.error(f"❌ [UserProfileManager] Error adding mastered concepts: {e}")
            return False
    
    async def get_user_mastered_concepts(self, user_id: str) -> List[str]:
        """
        Get all concepts user has mastered across all sessions.
        
        Args:
            user_id: User UUID
            
        Returns:
            List of concept names
        """
        profile = await self.get_user_profile(user_id)
        if profile:
            return profile.mastered_concepts or []
        return []
    
    async def check_prerequisites(
        self,
        user_id: str,
        topic: str,
        prerequisites_graph
    ) -> List[str]:
        """
        Check if user has prerequisites for a topic.
        
        Args:
            user_id: User UUID
            topic: Topic name to check prerequisites for
            prerequisites_graph: EnhancedPrerequisiteGraph instance
            
        Returns:
            List of missing prerequisite concept names
        """
        user_mastered = set(await self.get_user_mastered_concepts(user_id))
        
        # Get prerequisites for topic from graph using the proper method
        topic_prerequisites = prerequisites_graph.get_prerequisites(topic)
        if not topic_prerequisites:
            return []
        
        # Check which prerequisites are missing
        missing_prereqs = [
            prereq for prereq in topic_prerequisites
            if prereq not in user_mastered
        ]
        
        return missing_prereqs
    
    async def update_onboarding_status(
        self,
        user_id: str,
        complete: bool = True
    ) -> bool:
        """
        Update user's onboarding completion status.
        
        Args:
            user_id: User UUID
            complete: Whether onboarding is complete
            
        Returns:
            True if successful
        """
        if not self.use_supabase:
            return False
        
        try:
            # Check if profile exists
            profile_check = self.supabase.table('profiles') \
                .select('id') \
                .eq('id', user_id) \
                .execute()
            
            if not profile_check.data:
                logger.warning(f"⚠️ [UserProfileManager] Profile not found for user {user_id[:20]}..., cannot update onboarding status")
                return False
            
            update_data = {
                'onboarding_complete': complete
            }
            
            if complete:
                update_data['onboarding_completed_at'] = datetime.now().isoformat()
            
            result = self.supabase.table('profiles') \
                .update(update_data) \
                .eq('id', user_id) \
                .execute()
            
            if result.data:
                logger.info(f"✅ [UserProfileManager] Updated onboarding status for user {user_id[:20]}...: {complete}")
                return True
            else:
                logger.warning(f"⚠️ [UserProfileManager] Update returned no data for user {user_id[:20]}...")
                return False
        except Exception as e:
            logger.error(f"❌ [UserProfileManager] Error updating onboarding status: {e}", exc_info=True)
            return False
    
    async def update_statistics(
        self,
        user_id: str,
        increment_sessions: bool = False,
        increment_interactions: int = 0
    ) -> bool:
        """
        Update user statistics (total_sessions, total_interactions, last_activity).
        
        Args:
            user_id: User UUID
            increment_sessions: Whether to increment total_sessions by 1
            increment_interactions: Number of interactions to add
            
        Returns:
            True if successful
        """
        if not self.use_supabase:
            return False
        
        try:
            # Check if profile exists
            profile_check = self.supabase.table('profiles') \
                .select('id, total_sessions, total_interactions') \
                .eq('id', user_id) \
                .execute()
            
            if not profile_check.data:
                logger.warning(f"⚠️ [UserProfileManager] Profile not found for user {user_id[:20]}..., cannot update statistics")
                return False
            
            # Get current values
            current_data = profile_check.data[0]
            current_sessions = current_data.get('total_sessions', 0) or 0
            current_interactions = current_data.get('total_interactions', 0) or 0
            
            # Calculate new values
            update_data = {
                'last_activity': datetime.now().isoformat()
            }
            
            if increment_sessions:
                update_data['total_sessions'] = current_sessions + 1
            
            if increment_interactions > 0:
                update_data['total_interactions'] = current_interactions + increment_interactions
            
            result = self.supabase.table('profiles') \
                .update(update_data) \
                .eq('id', user_id) \
                .execute()
            
            if result.data:
                logger.info(f"✅ [UserProfileManager] Updated statistics for user {user_id[:20]}... (sessions: {update_data.get('total_sessions', current_sessions)}, interactions: {update_data.get('total_interactions', current_interactions)})")
                return True
            else:
                logger.warning(f"⚠️ [UserProfileManager] Statistics update returned no data for user {user_id[:20]}...")
                return False
        except Exception as e:
            logger.error(f"❌ [UserProfileManager] Error updating statistics: {e}", exc_info=True)
            return False
    
    async def aggregate_from_sessions(self, user_id: str) -> bool:
        """
        Aggregate data from all user's sessions to update user profile.
        
        This can be run periodically to sync user profile with session data.
        
        Args:
            user_id: User UUID
            
        Returns:
            True if successful
        """
        if not self.use_supabase:
            return False
        
        try:
            # Get all sessions for user
            sessions_result = self.supabase.table('sessions') \
                .select('learning_style, mastered_concepts, difficulty, onboarding_complete, interaction_count') \
                .eq('user_id', user_id) \
                .execute()
            
            if not sessions_result.data:
                return False
            
            sessions = sessions_result.data
            
            # Aggregate learning styles (most common)
            learning_styles = [
                s.get('learning_style') for s in sessions
                if s.get('learning_style')
            ]
            if learning_styles:
                most_common_style = max(set(learning_styles), key=learning_styles.count)
                await self.update_learning_style(user_id, most_common_style, 0.8)
            
            # Aggregate mastered concepts (union of all)
            all_mastered = set()
            for session in sessions:
                mastered = session.get('mastered_concepts')
                if mastered:
                    try:
                        concepts = json.loads(mastered) if isinstance(mastered, str) else mastered
                        all_mastered.update(concepts)
                    except:
                        pass
            
            if all_mastered:
                await self.add_mastered_concepts(user_id, list(all_mastered))
            
            # Check if any session has onboarding complete
            any_onboarding_complete = any(
                s.get('onboarding_complete', False) for s in sessions
            )
            if any_onboarding_complete:
                await self.update_onboarding_status(user_id, True)
            
            # Update statistics
            total_sessions = len(sessions)
            total_interactions = sum(s.get('interaction_count', 0) for s in sessions)
            
            self.supabase.table('profiles') \
                .update({
                    'total_sessions': total_sessions,
                    'total_interactions': total_interactions,
                    'last_activity': datetime.now().isoformat()
                }) \
                .eq('id', user_id) \
                .execute()
            
            logger.info(f"✅ [UserProfileManager] Aggregated data from {total_sessions} sessions for user {user_id[:20]}...")
            return True
        except Exception as e:
            logger.error(f"❌ [UserProfileManager] Error aggregating from sessions: {e}")
            return False
    
    def _dict_to_profile(self, data: Dict[str, Any]) -> UserProfile:
        """Convert dictionary to UserProfile object."""
        # Parse JSON fields
        mastered_concepts = []
        if data.get('mastered_concepts'):
            try:
                mastered_concepts = json.loads(data['mastered_concepts']) if isinstance(data['mastered_concepts'], str) else data['mastered_concepts']
            except:
                mastered_concepts = []
        
        # Parse datetime fields
        learning_style_updated_at = None
        if data.get('learning_style_updated_at'):
            try:
                learning_style_updated_at = datetime.fromisoformat(data['learning_style_updated_at'].replace('Z', '+00:00'))
            except:
                pass
        
        mastered_concepts_updated_at = None
        if data.get('mastered_concepts_updated_at'):
            try:
                mastered_concepts_updated_at = datetime.fromisoformat(data['mastered_concepts_updated_at'].replace('Z', '+00:00'))
            except:
                pass
        
        last_activity = None
        if data.get('last_activity'):
            try:
                last_activity = datetime.fromisoformat(data['last_activity'].replace('Z', '+00:00'))
            except:
                pass
        
        onboarding_completed_at = None
        if data.get('onboarding_completed_at'):
            try:
                onboarding_completed_at = datetime.fromisoformat(data['onboarding_completed_at'].replace('Z', '+00:00'))
            except:
                pass
        
        return UserProfile(
            user_id=data['id'],
            learning_style=data.get('learning_style'),
            learning_style_confidence=data.get('learning_style_confidence', 0.0),
            learning_style_updated_at=learning_style_updated_at,
            mastered_concepts=mastered_concepts,
            mastered_concepts_updated_at=mastered_concepts_updated_at,
            overall_difficulty=data.get('overall_difficulty', 'intermediate'),
            total_sessions=data.get('total_sessions', 0),
            total_interactions=data.get('total_interactions', 0),
            last_activity=last_activity,
            onboarding_complete=data.get('onboarding_complete', False),
            onboarding_completed_at=onboarding_completed_at
        )

