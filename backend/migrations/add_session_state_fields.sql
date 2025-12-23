-- =====================================================
-- ADD SESSION STATE FIELDS TO SESSIONS TABLE
-- Adds fields needed for full SessionState persistence
-- =====================================================

-- Add fields for SessionState persistence
ALTER TABLE public.sessions ADD COLUMN IF NOT EXISTS mastered_concepts TEXT DEFAULT '[]';
ALTER TABLE public.sessions ADD COLUMN IF NOT EXISTS learning_style TEXT;
ALTER TABLE public.sessions ADD COLUMN IF NOT EXISTS conversation_history TEXT DEFAULT '[]';
ALTER TABLE public.sessions ADD COLUMN IF NOT EXISTS onboarding_complete BOOLEAN DEFAULT FALSE;
ALTER TABLE public.sessions ADD COLUMN IF NOT EXISTS stated_goal TEXT;
ALTER TABLE public.sessions ADD COLUMN IF NOT EXISTS stated_level TEXT;
ALTER TABLE public.sessions ADD COLUMN IF NOT EXISTS understanding_scores TEXT DEFAULT '[]';
ALTER TABLE public.sessions ADD COLUMN IF NOT EXISTS performance_trend TEXT;

-- Note: These fields store JSON arrays/objects as TEXT
-- SessionManager handles JSON serialization/deserialization

-- Index for faster lookups
CREATE INDEX IF NOT EXISTS idx_sessions_user_id ON public.sessions(user_id);
CREATE INDEX IF NOT EXISTS idx_sessions_session_id ON public.sessions(session_id);

