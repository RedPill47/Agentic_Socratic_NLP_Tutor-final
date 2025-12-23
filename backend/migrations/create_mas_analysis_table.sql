-- =====================================================
-- MAS ANALYSIS TABLE
-- Stores background Multi-Agent System analysis results
-- =====================================================

CREATE TABLE IF NOT EXISTS public.mas_analysis (
  id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
  session_id UUID NOT NULL REFERENCES public.sessions(id) ON DELETE CASCADE,
  user_id UUID NOT NULL REFERENCES public.profiles(id) ON DELETE CASCADE,
  
  -- Learning Style Analysis
  learning_style JSONB,
  
  -- Performance Assessment
  performance_assessment JSONB,
  
  -- Knowledge Gaps
  knowledge_gaps TEXT[],
  misconceptions TEXT[],
  
  -- Teaching Recommendations
  teaching_recommendations TEXT[],
  
  -- Metadata
  confidence_score FLOAT DEFAULT 0.0 CHECK (confidence_score >= 0 AND confidence_score <= 1),
  created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Indexes for performance
CREATE INDEX IF NOT EXISTS idx_mas_analysis_session ON public.mas_analysis(session_id);
CREATE INDEX IF NOT EXISTS idx_mas_analysis_user ON public.mas_analysis(user_id);
CREATE INDEX IF NOT EXISTS idx_mas_analysis_created ON public.mas_analysis(created_at DESC);

-- Add latest analysis cache to sessions table
ALTER TABLE public.sessions ADD COLUMN IF NOT EXISTS mas_analysis JSONB DEFAULT '{}'::jsonb;
ALTER TABLE public.sessions ADD COLUMN IF NOT EXISTS last_analysis_at TIMESTAMPTZ;

-- Row Level Security (RLS)
ALTER TABLE public.mas_analysis ENABLE ROW LEVEL SECURITY;

-- Policy: Users can only see their own analysis
CREATE POLICY "Users can view own mas_analysis" ON public.mas_analysis
  FOR SELECT USING (auth.uid() = user_id);

-- Policy: Users can insert their own analysis
CREATE POLICY "Users can insert own mas_analysis" ON public.mas_analysis
  FOR INSERT WITH CHECK (auth.uid() = user_id);

-- Policy: Users can update their own analysis
CREATE POLICY "Users can update own mas_analysis" ON public.mas_analysis
  FOR UPDATE USING (auth.uid() = user_id);

