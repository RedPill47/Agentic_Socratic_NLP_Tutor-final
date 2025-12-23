-- =====================================================
-- ADD source_files COLUMN TO SESSIONS TABLE
-- Stores list of RAG source files used in the session
-- =====================================================

-- Add source_files column to store JSON array of source file names
ALTER TABLE public.sessions ADD COLUMN IF NOT EXISTS source_files TEXT DEFAULT '[]';

-- Comment for documentation
COMMENT ON COLUMN public.sessions.source_files IS 'JSON array of RAG source file names used in this session (e.g., ["lecture_1.pdf", "tutorial_2.pdf"])';

