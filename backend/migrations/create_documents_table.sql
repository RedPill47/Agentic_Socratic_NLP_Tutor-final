-- =====================================================
-- CREATE DOCUMENTS TABLE
-- =====================================================
-- This table stores knowledge base documents uploaded by teachers
-- Used by the teacher dashboard to track document ingestion

CREATE TABLE IF NOT EXISTS public.documents (
  id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
  uploaded_by UUID NOT NULL REFERENCES public.profiles(id) ON DELETE CASCADE,
  filename TEXT NOT NULL,
  file_path TEXT NOT NULL,
  status TEXT DEFAULT 'pending' CHECK (status IN ('pending', 'processing', 'completed', 'failed')),
  chunks_created INTEGER DEFAULT 0,
  error_message TEXT,
  uploaded_at TIMESTAMPTZ DEFAULT NOW(),
  processed_at TIMESTAMPTZ
);

-- =====================================================
-- INDEXES for Performance
-- =====================================================
CREATE INDEX IF NOT EXISTS idx_documents_uploaded_by ON public.documents(uploaded_by);
CREATE INDEX IF NOT EXISTS idx_documents_uploaded_at ON public.documents(uploaded_at DESC);
CREATE INDEX IF NOT EXISTS idx_documents_status ON public.documents(status);

-- =====================================================
-- ENABLE ROW LEVEL SECURITY
-- =====================================================
ALTER TABLE public.documents ENABLE ROW LEVEL SECURITY;

-- =====================================================
-- RLS POLICIES
-- =====================================================
-- Users can view their own documents
DROP POLICY IF EXISTS "Users can view own documents" ON public.documents;
CREATE POLICY "Users can view own documents"
  ON public.documents FOR SELECT
  USING (auth.uid() = uploaded_by);

-- Users can insert their own documents
DROP POLICY IF EXISTS "Users can insert own documents" ON public.documents;
CREATE POLICY "Users can insert own documents"
  ON public.documents FOR INSERT
  WITH CHECK (auth.uid() = uploaded_by);

-- Users can update their own documents
DROP POLICY IF EXISTS "Users can update own documents" ON public.documents;
CREATE POLICY "Users can update own documents"
  ON public.documents FOR UPDATE
  USING (auth.uid() = uploaded_by);

-- Note: Teacher access policies are handled in add_teacher_access_policies.sql
-- This migration creates the table, and the teacher policies migration adds teacher access

