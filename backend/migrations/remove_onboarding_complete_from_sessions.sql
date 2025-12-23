-- =====================================================
-- REMOVE ONBOARDING_COMPLETE FROM SESSIONS TABLE
-- Onboarding status is now stored at user level in profiles table
-- =====================================================

-- Remove onboarding_complete column from sessions table
ALTER TABLE public.sessions DROP COLUMN IF EXISTS onboarding_complete;

-- Comment for documentation
COMMENT ON TABLE public.sessions IS 'Session state is now user-agnostic. User-level data (onboarding, learning style, mastered concepts) is stored in profiles table.';

