-- =====================================================
-- ADD TEACHER ACCESS POLICIES
-- Allows teachers to view all student data for dashboard
-- =====================================================

-- Helper function to check if user is a teacher
CREATE OR REPLACE FUNCTION public.is_teacher(user_id UUID)
RETURNS BOOLEAN AS $$
BEGIN
  RETURN EXISTS (
    SELECT 1 FROM public.profiles
    WHERE id = user_id AND role = 'teacher'
  );
END;
$$ LANGUAGE plpgsql SECURITY DEFINER;

-- =====================================================
-- PROFILES: Teachers can view all student profiles
-- =====================================================
-- Drop existing policy if it exists, then create new one
DROP POLICY IF EXISTS "Teachers can view all profiles" ON public.profiles;
CREATE POLICY "Teachers can view all profiles"
  ON public.profiles FOR SELECT
  USING (
    auth.uid() = id OR -- Own profile
    public.is_teacher(auth.uid()) -- Or is a teacher
  );

-- =====================================================
-- SESSIONS: Teachers can view all student sessions
-- =====================================================
-- Drop existing policy if it exists, then create new one
DROP POLICY IF EXISTS "Teachers can view all sessions" ON public.sessions;
CREATE POLICY "Teachers can view all sessions"
  ON public.sessions FOR SELECT
  USING (
    auth.uid() = user_id OR -- Own sessions
    public.is_teacher(auth.uid()) -- Or is a teacher
  );

-- =====================================================
-- MAS_ANALYSIS: Teachers can view all student analyses
-- =====================================================
-- Drop existing policy if it exists, then create new one
DROP POLICY IF EXISTS "Teachers can view all mas_analysis" ON public.mas_analysis;
CREATE POLICY "Teachers can view all mas_analysis"
  ON public.mas_analysis FOR SELECT
  USING (
    auth.uid() = user_id OR -- Own analysis
    public.is_teacher(auth.uid()) -- Or is a teacher
  );

-- =====================================================
-- DOCUMENTS: Teachers can view all documents
-- =====================================================
-- Note: Documents table might not have RLS enabled yet
-- If it does, add this policy
DO $$
BEGIN
  IF EXISTS (
    SELECT 1 FROM pg_tables 
    WHERE schemaname = 'public' 
    AND tablename = 'documents'
  ) THEN
    -- Check if RLS is enabled
    IF EXISTS (
      SELECT 1 FROM pg_tables t
      JOIN pg_class c ON c.relname = t.tablename
      WHERE t.schemaname = 'public' 
      AND t.tablename = 'documents'
      AND c.relrowsecurity = true
    ) THEN
      -- Drop policy if it exists, then create it
      DROP POLICY IF EXISTS "Teachers can view all documents" ON public.documents;
      CREATE POLICY "Teachers can view all documents"
        ON public.documents FOR SELECT
        USING (
          auth.uid() = uploaded_by OR -- Own documents
          public.is_teacher(auth.uid()) -- Or is a teacher
        );
    END IF;
  END IF;
END $$;

