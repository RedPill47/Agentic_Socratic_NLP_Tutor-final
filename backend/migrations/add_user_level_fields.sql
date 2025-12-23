-- =====================================================
-- ADD USER-LEVEL LEARNING DATA TO PROFILES
-- Enables personalized learning across sessions
-- =====================================================

-- Learning Style (most common/recent across all sessions)
ALTER TABLE public.profiles ADD COLUMN IF NOT EXISTS learning_style TEXT;
ALTER TABLE public.profiles ADD COLUMN IF NOT EXISTS learning_style_confidence FLOAT DEFAULT 0.0 CHECK (learning_style_confidence >= 0 AND learning_style_confidence <= 1);
ALTER TABLE public.profiles ADD COLUMN IF NOT EXISTS learning_style_updated_at TIMESTAMPTZ;

-- Mastered Concepts (aggregated from all sessions)
-- Stores JSON array of concept names: ["Tokenization", "Word Embeddings", "Transformers"]
ALTER TABLE public.profiles ADD COLUMN IF NOT EXISTS mastered_concepts TEXT DEFAULT '[]';
ALTER TABLE public.profiles ADD COLUMN IF NOT EXISTS mastered_concepts_updated_at TIMESTAMPTZ;

-- Overall Difficulty Level (aggregated from sessions)
ALTER TABLE public.profiles ADD COLUMN IF NOT EXISTS overall_difficulty TEXT DEFAULT 'intermediate' CHECK (overall_difficulty IN ('beginner', 'intermediate', 'advanced'));
ALTER TABLE public.profiles ADD COLUMN IF NOT EXISTS difficulty_updated_at TIMESTAMPTZ;

-- Learning Statistics
ALTER TABLE public.profiles ADD COLUMN IF NOT EXISTS total_sessions INTEGER DEFAULT 0;
ALTER TABLE public.profiles ADD COLUMN IF NOT EXISTS total_interactions INTEGER DEFAULT 0;
ALTER TABLE public.profiles ADD COLUMN IF NOT EXISTS last_activity TIMESTAMPTZ;

-- Onboarding Status (user-level, not session-level)
-- Once a user completes onboarding in any session, they're marked as complete
ALTER TABLE public.profiles ADD COLUMN IF NOT EXISTS onboarding_complete BOOLEAN DEFAULT FALSE;
ALTER TABLE public.profiles ADD COLUMN IF NOT EXISTS onboarding_completed_at TIMESTAMPTZ;

-- Indexes for performance
CREATE INDEX IF NOT EXISTS idx_profiles_learning_style ON public.profiles(learning_style);
CREATE INDEX IF NOT EXISTS idx_profiles_onboarding ON public.profiles(onboarding_complete);
CREATE INDEX IF NOT EXISTS idx_profiles_last_activity ON public.profiles(last_activity DESC);

-- Comments for documentation
COMMENT ON COLUMN public.profiles.learning_style IS 'User''s primary learning style detected across all sessions';
COMMENT ON COLUMN public.profiles.mastered_concepts IS 'JSON array of all concepts user has mastered across all sessions';
COMMENT ON COLUMN public.profiles.overall_difficulty IS 'User''s overall difficulty level based on all sessions';
COMMENT ON COLUMN public.profiles.onboarding_complete IS 'Whether user has completed onboarding in any session';

