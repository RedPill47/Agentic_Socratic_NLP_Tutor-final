# Supabase Setup Guide

Complete guide to set up Supabase for the Agentic Socratic NLP Tutor.

**Last Updated:** December 2024  
**Status:** Current with latest schema changes

---

## Architecture Overview

**Frontend (Next.js):**
- Handles authentication (login/signup)
- Reads/writes data directly to Supabase
- Role-based UI (student vs teacher views)

**Backend (FastAPI):**
- Uses Supabase for session persistence
- Validates requests via Supabase JWT tokens
- Uses service role key for admin operations

**Supabase:**
- Authentication (email/password)
- User profiles with roles (student/teacher)
- Conversation history & learning data
- Background MAS analysis storage
- Teacher document management

**ChromaDB:**
- Remains local on backend filesystem
- Teachers can upload PDFs via UI → triggers backend ingestion

---

## Phase 1: Supabase Project Setup

### Step 1.1: Create Supabase Project

1. Go to https://supabase.com
2. Sign in and click "New Project"
3. Fill in:
   - **Project Name:** `nlp-tutor` (or your choice)
   - **Database Password:** (save this securely!)
   - **Region:** Choose closest to you
4. Wait for project to finish provisioning (~2 minutes)

### Step 1.2: Get Your Credentials

From your Supabase project dashboard:

1. Go to **Settings** → **API**
2. Copy these values:
   - **Project URL** (e.g., `https://xxxxx.supabase.co`)
   - **anon/public key** (starts with `eyJ...`)
   - **service_role key** (starts with `eyJ...`) - **Keep this secret!**

### Step 1.3: Create Environment Files

**Frontend** (`frontend/.env.local`):
```bash
NEXT_PUBLIC_SUPABASE_URL=https://xxxxx.supabase.co
NEXT_PUBLIC_SUPABASE_ANON_KEY=eyJhbGc...your-anon-key
NEXT_PUBLIC_API_URL=http://localhost:8000
```

**Backend** (`backend/.env` or root `.env`):
```bash
OPENAI_API_KEY=your_openai_key_here
OPENAI_MODEL=gpt-4o-mini
SUPABASE_URL=https://xxxxx.supabase.co
SUPABASE_SERVICE_KEY=eyJhbGc...your-service-role-key
```

> **Note:** Service role key bypasses Row Level Security - use carefully! Only use in backend.

---

## Phase 2: Database Schema

### Step 2.1: Run Migrations in Order

**IMPORTANT:** Run migrations in the order listed below. They are in `backend/migrations/`.

#### Migration Order:

1. **Base Schema** (run this first):
   ```sql
   -- Enable UUID extension
   CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

   -- =====================================================
   -- PROFILES TABLE (extends auth.users)
   -- =====================================================
   CREATE TABLE IF NOT EXISTS public.profiles (
     id UUID PRIMARY KEY REFERENCES auth.users(id) ON DELETE CASCADE,
     email TEXT UNIQUE NOT NULL,
     full_name TEXT,
     role TEXT NOT NULL DEFAULT 'student' CHECK (role IN ('student', 'teacher')),
     created_at TIMESTAMPTZ DEFAULT NOW(),
     updated_at TIMESTAMPTZ DEFAULT NOW()
   );

   -- =====================================================
   -- SESSIONS TABLE
   -- =====================================================
   CREATE TABLE IF NOT EXISTS public.sessions (
     id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
     user_id UUID NOT NULL REFERENCES public.profiles(id) ON DELETE CASCADE,
     session_id TEXT UNIQUE NOT NULL,
     current_topic TEXT,
     interaction_count INTEGER DEFAULT 0,
     created_at TIMESTAMPTZ DEFAULT NOW(),
     last_activity TIMESTAMPTZ DEFAULT NOW()
   );

   -- =====================================================
   -- MESSAGES TABLE
   -- =====================================================
   CREATE TABLE IF NOT EXISTS public.messages (
     id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
     session_id UUID NOT NULL REFERENCES public.sessions(id) ON DELETE CASCADE,
     role TEXT NOT NULL CHECK (role IN ('user', 'assistant', 'system')),
     content TEXT NOT NULL,
     metadata JSONB DEFAULT '{}'::jsonb,
     created_at TIMESTAMPTZ DEFAULT NOW()
   );

   -- Indexes
   CREATE INDEX IF NOT EXISTS idx_sessions_user_id ON public.sessions(user_id);
   CREATE INDEX IF NOT EXISTS idx_sessions_session_id ON public.sessions(session_id);
   CREATE INDEX IF NOT EXISTS idx_sessions_last_activity ON public.sessions(last_activity DESC);
   CREATE INDEX IF NOT EXISTS idx_messages_session_id ON public.messages(session_id);
   CREATE INDEX IF NOT EXISTS idx_messages_created_at ON public.messages(created_at DESC);

   -- Auto-update updated_at timestamp
   CREATE OR REPLACE FUNCTION update_updated_at()
   RETURNS TRIGGER AS $$
   BEGIN
     NEW.updated_at = NOW();
     RETURN NEW;
   END;
   $$ LANGUAGE plpgsql;

   CREATE TRIGGER update_profiles_updated_at
     BEFORE UPDATE ON public.profiles
     FOR EACH ROW
     EXECUTE FUNCTION update_updated_at();

   -- Auto-create profile on user signup
   CREATE OR REPLACE FUNCTION public.handle_new_user()
   RETURNS TRIGGER AS $$
   BEGIN
     INSERT INTO public.profiles (id, email, full_name, role)
     VALUES (
       NEW.id,
       NEW.email,
       NEW.raw_user_meta_data->>'full_name',
       COALESCE(NEW.raw_user_meta_data->>'role', 'student')
     );
     RETURN NEW;
   END;
   $$ LANGUAGE plpgsql SECURITY DEFINER;

   CREATE TRIGGER on_auth_user_created
     AFTER INSERT ON auth.users
     FOR EACH ROW
     EXECUTE FUNCTION public.handle_new_user();
   ```

2. **Add Session State Fields** (`add_session_state_fields.sql`):
   - Adds: `stated_goal`, `stated_level`, `understanding_scores`, `performance_trend`
   - Note: Some fields added here were later moved to profiles (see migration 4)

3. **Add Source Files Column** (`add_source_files_column.sql`):
   - Adds: `source_files` (JSON array of RAG source files)

4. **Add User-Level Fields to Profiles** (`add_user_level_fields.sql`):
   - Adds to `profiles`: `learning_style`, `mastered_concepts`, `overall_difficulty`, `onboarding_complete`, statistics

5. **Create MAS Analysis Table** (`create_mas_analysis_table.sql`):
   - Creates `mas_analysis` table for background analysis results
   - Adds `mas_analysis` and `last_analysis_at` to `sessions` table

6. **Create Documents Table** (`create_documents_table.sql`):
   - Creates `documents` table for teacher-uploaded files

7. **Remove User-Level Fields from Sessions** (`remove_user_level_fields_from_sessions.sql`):
   - Removes: `mastered_concepts`, `learning_style`, `difficulty` from `sessions`
   - These are now in `profiles` table (user-level)

8. **Remove Onboarding Complete from Sessions** (`remove_onboarding_complete_from_sessions.sql`):
   - Removes: `onboarding_complete` from `sessions`
   - Now in `profiles` table (user-level)

9. **Add Teacher Access Policies** (`add_teacher_access_policies.sql`):
   - Adds RLS policies for teachers to view all student data

10. **Add Last Updated Column** (`add_last_updated_column.sql`):
    - Adds: `last_updated` timestamp to `sessions`

### Step 2.2: Current Database Schema

#### `profiles` Table

**Purpose:** User-level data (persists across all sessions)

```sql
CREATE TABLE public.profiles (
  id UUID PRIMARY KEY REFERENCES auth.users(id) ON DELETE CASCADE,
  email TEXT UNIQUE NOT NULL,
  full_name TEXT,
  role TEXT NOT NULL DEFAULT 'student' CHECK (role IN ('student', 'teacher')),
  
  -- User-Level Learning Data
  learning_style TEXT,  -- 'visual', 'auditory', 'kinesthetic', 'reading'
  learning_style_confidence FLOAT DEFAULT 0.0,
  learning_style_updated_at TIMESTAMPTZ,
  
  mastered_concepts TEXT DEFAULT '[]',  -- JSON array
  mastered_concepts_updated_at TIMESTAMPTZ,
  
  overall_difficulty TEXT DEFAULT 'intermediate' CHECK (overall_difficulty IN ('beginner', 'intermediate', 'advanced')),
  difficulty_updated_at TIMESTAMPTZ,
  
  -- Statistics
  total_sessions INTEGER DEFAULT 0,
  total_interactions INTEGER DEFAULT 0,
  last_activity TIMESTAMPTZ,
  
  -- Onboarding
  onboarding_complete BOOLEAN DEFAULT FALSE,
  onboarding_completed_at TIMESTAMPTZ,
  
  created_at TIMESTAMPTZ DEFAULT NOW(),
  updated_at TIMESTAMPTZ DEFAULT NOW()
);
```

#### `sessions` Table

**Purpose:** Session-specific state (one per chat session)

```sql
CREATE TABLE public.sessions (
  id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
  user_id UUID NOT NULL REFERENCES public.profiles(id) ON DELETE CASCADE,
  session_id TEXT UNIQUE NOT NULL,  -- Frontend-generated
  
  -- Session State
  current_topic TEXT,
  interaction_count INTEGER DEFAULT 0,
  
  -- Session-Specific Learning Data
  stated_goal TEXT,
  stated_level TEXT,
  understanding_scores TEXT DEFAULT '[]',  -- JSON array of floats
  performance_trend TEXT,
  
  -- RAG Traceability
  source_files TEXT DEFAULT '[]',  -- JSON array of file names
  
  -- Background MAS Cache
  mas_analysis JSONB DEFAULT '{}'::jsonb,
  last_analysis_at TIMESTAMPTZ,
  
  -- Timestamps
  created_at TIMESTAMPTZ DEFAULT NOW(),
  last_activity TIMESTAMPTZ DEFAULT NOW(),
  last_updated TIMESTAMPTZ DEFAULT NOW()
);
```

**Note:** User-level fields (`learning_style`, `mastered_concepts`, `difficulty`, `onboarding_complete`) are NOT in this table - they're in `profiles`.

#### `messages` Table

**Purpose:** Conversation history (source of truth)

```sql
CREATE TABLE public.messages (
  id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
  session_id UUID NOT NULL REFERENCES public.sessions(id) ON DELETE CASCADE,
  role TEXT NOT NULL CHECK (role IN ('user', 'assistant', 'system')),
  content TEXT NOT NULL,
  metadata JSONB DEFAULT '{}'::jsonb,  -- RAG sources, learning style, etc.
  created_at TIMESTAMPTZ DEFAULT NOW()
);
```

#### `mas_analysis` Table

**Purpose:** Background Multi-Agent System analysis results

```sql
CREATE TABLE public.mas_analysis (
  id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
  session_id UUID NOT NULL REFERENCES public.sessions(id) ON DELETE CASCADE,
  user_id UUID NOT NULL REFERENCES public.profiles(id) ON DELETE CASCADE,
  
  -- Analysis Results
  learning_style JSONB,
  performance_assessment JSONB,
  knowledge_gaps TEXT[],
  misconceptions TEXT[],
  teaching_recommendations TEXT[],
  confidence_score FLOAT DEFAULT 0.0,
  
  created_at TIMESTAMPTZ DEFAULT NOW()
);
```

#### `documents` Table

**Purpose:** Teacher-uploaded knowledge base documents

```sql
CREATE TABLE public.documents (
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
```

### Step 2.3: Set Up Row Level Security (RLS)

Run all RLS policies from `backend/migrations/add_teacher_access_policies.sql`:

```sql
-- Enable RLS on all tables
ALTER TABLE public.profiles ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.sessions ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.messages ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.mas_analysis ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.documents ENABLE ROW LEVEL SECURITY;

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

-- PROFILES: Users can view own profile, teachers can view all
CREATE POLICY "Users can view own profile"
  ON public.profiles FOR SELECT
  USING (auth.uid() = id);

CREATE POLICY "Users can update own profile"
  ON public.profiles FOR UPDATE
  USING (auth.uid() = id);

-- Teachers can view all profiles (see add_teacher_access_policies.sql)
-- This is handled in the migration file

-- SESSIONS: Users can only access their own sessions
CREATE POLICY "Users can view own sessions"
  ON public.sessions FOR SELECT
  USING (auth.uid() = user_id);

CREATE POLICY "Users can create own sessions"
  ON public.sessions FOR INSERT
  WITH CHECK (auth.uid() = user_id);

CREATE POLICY "Users can update own sessions"
  ON public.sessions FOR UPDATE
  USING (auth.uid() = user_id);

-- MESSAGES: Users can only access messages in their sessions
CREATE POLICY "Users can view own messages"
  ON public.messages FOR SELECT
  USING (
    EXISTS (
      SELECT 1 FROM public.sessions
      WHERE sessions.id = messages.session_id
      AND sessions.user_id = auth.uid()
    )
  );

CREATE POLICY "Users can create messages in own sessions"
  ON public.messages FOR INSERT
  WITH CHECK (
    EXISTS (
      SELECT 1 FROM public.sessions
      WHERE sessions.id = messages.session_id
      AND sessions.user_id = auth.uid()
    )
  );

-- MAS_ANALYSIS: Users can only access their own analysis
CREATE POLICY "Users can view own mas_analysis"
  ON public.mas_analysis FOR SELECT
  USING (auth.uid() = user_id);

CREATE POLICY "Users can insert own mas_analysis"
  ON public.mas_analysis FOR INSERT
  WITH CHECK (auth.uid() = user_id);

-- DOCUMENTS: Users can manage their own documents
CREATE POLICY "Users can view own documents"
  ON public.documents FOR SELECT
  USING (auth.uid() = uploaded_by);

CREATE POLICY "Users can insert own documents"
  ON public.documents FOR INSERT
  WITH CHECK (auth.uid() = uploaded_by);

-- Teacher access policies are in add_teacher_access_policies.sql
```

---

## Phase 3: Running Migrations

### Option 1: Run Migrations Manually

1. Go to Supabase Dashboard → **SQL Editor**
2. Run each migration file in order (from `backend/migrations/`)
3. Check for errors

### Option 2: Use Supabase CLI (Recommended)

```bash
# Install Supabase CLI
npm install -g supabase

# Link to your project
supabase link --project-ref your-project-ref

# Run migrations
supabase db push
```

---

## Phase 4: Verification

### Test Connection

Use the diagnostic scripts:

```bash
# Test Supabase URL and connectivity
python scripts/verify_supabase_url.py

# Or minimal connectivity test
python scripts/supabase_connect_minimal.py
```

### Verify Tables

In Supabase SQL Editor, run:

```sql
-- Check all tables exist
SELECT table_name 
FROM information_schema.tables 
WHERE table_schema = 'public'
ORDER BY table_name;

-- Should show:
-- documents
-- mas_analysis
-- messages
-- profiles
-- sessions
```

### Verify RLS

```sql
-- Check RLS is enabled
SELECT tablename, rowsecurity 
FROM pg_tables 
WHERE schemaname = 'public';

-- All should show rowsecurity = true
```

---

## Phase 5: Frontend Setup

### Step 5.1: Install Supabase Packages

```bash
cd frontend
npm install @supabase/supabase-js @supabase/ssr
```

### Step 5.2: Supabase Client

The frontend already has Supabase clients set up in:
- `frontend/lib/supabase/client.ts` - Browser client
- `frontend/lib/supabase/server.ts` - Server component client
- `frontend/middleware.ts` - Auth middleware

---

## Phase 6: Backend Setup

### Step 6.1: Supabase Client

The backend uses:
- `backend/lib/supabase_client.py` - Service role client for admin operations

### Step 6.2: Environment Variables

Ensure `backend/.env` has:
```bash
SUPABASE_URL=https://xxxxx.supabase.co
SUPABASE_SERVICE_KEY=eyJhbGc...your-service-role-key
```

---

## Important Notes

### User-Level vs Session-Level Data

**User-Level (in `profiles` table):**
- `learning_style` - Detected across all sessions
- `mastered_concepts` - Aggregated from all sessions
- `overall_difficulty` - Based on all sessions
- `onboarding_complete` - Once complete, stays complete
- `total_sessions`, `total_interactions`, `last_activity` - Statistics

**Session-Level (in `sessions` table):**
- `current_topic` - Current learning topic
- `interaction_count` - Messages in this session
- `stated_goal`, `stated_level` - From onboarding
- `understanding_scores` - This session's scores
- `performance_trend` - This session's trend
- `source_files` - RAG files used in this session
- `mas_analysis` - Cached latest analysis

### Tables That DON'T Exist

These tables were removed (unused):
- ❌ `progress` - Dropped (unused)
- ❌ `learning_styles` - Dropped (merged into `profiles`)

### Migration Order Matters

Run migrations in this order:
1. Base schema (profiles, sessions, messages)
2. Session state fields
3. Source files
4. User-level fields
5. MAS analysis table
6. Documents table
7. Remove user-level from sessions
8. Remove onboarding from sessions
9. Teacher access policies
10. Last updated column

---

## Troubleshooting

### Connection Issues

1. **DNS Resolution Failed:**
   - Check `SUPABASE_URL` is correct
   - Verify project exists in Supabase dashboard

2. **Authentication Failed:**
   - Check API keys are correct
   - Verify service role key for backend
   - Verify anon key for frontend

3. **Table Not Found:**
   - Run migrations in order
   - Check migration files exist in `backend/migrations/`

### RLS Issues

1. **Can't Read Own Data:**
   - Check RLS policies are created
   - Verify `auth.uid()` matches `user_id`

2. **Teachers Can't Access Student Data:**
   - Run `add_teacher_access_policies.sql`
   - Verify `is_teacher()` function exists

---

## Next Steps

1. ✅ Run all migrations
2. ✅ Verify tables exist
3. ✅ Test connection with diagnostic scripts
4. ✅ Set up frontend environment variables
5. ✅ Set up backend environment variables
6. ✅ Test authentication flow
7. ✅ Test chat functionality

---

## Reference

- **Migration Files:** `backend/migrations/`
- **Backend Client:** `backend/lib/supabase_client.py`
- **Frontend Clients:** `frontend/lib/supabase/`
- **Diagnostic Scripts:** `scripts/verify_supabase_url.py`, `scripts/supabase_connect_minimal.py`
- **Documentation:** `DOCUMENTATION.md` (Database Schema section)
