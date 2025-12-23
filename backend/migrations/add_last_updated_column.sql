-- =====================================================
-- ADD last_updated COLUMN TO SESSIONS TABLE
-- Adds timestamp tracking for session updates
-- =====================================================

-- Add last_updated column if it doesn't exist
ALTER TABLE public.sessions ADD COLUMN IF NOT EXISTS last_updated TIMESTAMPTZ DEFAULT NOW();

-- Update existing rows to have current timestamp
UPDATE public.sessions SET last_updated = NOW() WHERE last_updated IS NULL;

-- Create index for faster queries
CREATE INDEX IF NOT EXISTS idx_sessions_last_updated ON public.sessions(last_updated);

