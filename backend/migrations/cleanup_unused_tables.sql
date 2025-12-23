-- =====================================================
-- CLEANUP UNUSED TABLES AND COLUMNS
-- Safe removal of unused database objects
-- =====================================================

-- =====================================================
-- STEP 1: Remove unused tables
-- =====================================================

-- These tables were created but never used in the codebase
-- No queries or inserts found for these tables

DROP TABLE IF EXISTS public.progress CASCADE;
DROP TABLE IF EXISTS public.learning_styles CASCADE;
DROP TABLE IF EXISTS public.documents CASCADE;

-- =====================================================
-- STEP 2: Remove conversation_history column from sessions
-- =====================================================

-- This column is redundant - conversation history is loaded from messages table
-- SessionManager.get_session() already loads from messages table and ignores this column
-- Safe to remove as it's not used anywhere in the codebase

ALTER TABLE public.sessions DROP COLUMN IF EXISTS conversation_history;

-- =====================================================
-- VERIFICATION
-- =====================================================

-- After running this migration, verify:
-- 1. Sessions table still has all required columns:
--    - session_id, user_id, current_topic, difficulty, interaction_count
--    - mastered_concepts, learning_style, onboarding_complete
--    - stated_goal, stated_level, understanding_scores, performance_trend
--    - source_files, mas_analysis, last_analysis_at
--    - created_at, last_activity, last_updated
--
-- 2. Messages table is the source of truth for conversation history
--
-- 3. mas_analysis table exists for full history
--
-- 4. profiles table exists for user data

