-- =====================================================
-- REMOVE USER-LEVEL FIELDS FROM SESSIONS TABLE
-- These fields are now stored in profiles table (user-level)
-- =====================================================

-- Remove mastered_concepts (user-level, aggregated across all sessions)
ALTER TABLE public.sessions DROP COLUMN IF EXISTS mastered_concepts;

-- Remove learning_style (user-level, detected across all sessions)
ALTER TABLE public.sessions DROP COLUMN IF EXISTS learning_style;

-- Remove difficulty (user-level, stored as overall_difficulty in profiles)
ALTER TABLE public.sessions DROP COLUMN IF EXISTS difficulty;

-- Comments for documentation
COMMENT ON TABLE public.sessions IS 'Session-specific state only. User-level data (mastered_concepts, learning_style, overall_difficulty, onboarding_complete) is stored in profiles table.';

