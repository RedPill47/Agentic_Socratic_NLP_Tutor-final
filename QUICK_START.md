# Quick Start Guide

**Get the Agentic Socratic NLP Tutor running in 5 minutes!**

> **Note:** Everything is pre-configured! Supabase credentials, ChromaDB embeddings, and knowledge graph are already included in the repository.

---

## Prerequisites

- Python 3.10-3.13
- Node.js 18+
- OpenAI API key

---

## Step 1: Environment Variables (1 minute)

Copy the example file and add your OpenAI API key:

```bash
cp .env.example .env
```

Then edit `.env` and set your OpenAI API key:
```env
OPENAI_API_KEY=your_actual_openai_api_key_here
```

> **Note:** 
> - Both backend and frontend read from this single root `.env` file
> - The frontend automatically syncs `NEXT_PUBLIC_*` variables to `frontend/.env.local` (runs automatically before `npm run dev`)
> - Supabase credentials are pre-configured (no need to set them)
> - See `.env.example` for all available options
> - If you need to use your own Supabase instance, see the [Optional Setup](#optional-setup) section below

---

## Step 2: Install Dependencies (3 minutes)

**Backend:**
```bash
cd backend
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

**Frontend:**
```bash
cd frontend
npm install
```

---

## Step 3: Start Services (2 minutes)

**Terminal 1 - Backend:**
```bash
cd backend
source venv/bin/activate  # Windows: venv\Scripts\activate
python main.py
```

**Terminal 2 - Frontend:**
```bash
cd frontend
npm run dev
```

---

## Step 4: Test (1 minute)

1. Open: `http://localhost:3000`
2. Start a conversation
3. Verify: Response in 2-4 seconds ✅

---

## ✅ You're Done!

The system is running with:
- ✅ Pre-configured Supabase database
- ✅ Pre-computed ChromaDB embeddings
- ✅ Knowledge graph with NLP concepts

---

## Optional Setup

If you want to set up your own Supabase instance or rebuild components from scratch:

### Set Up Your Own Supabase

1. Create a new Supabase project at https://supabase.com
2. Run migrations from `backend/migrations/` (see `SUPABASE_SETUP.md` for details)
3. Update the root `.env` file with your Supabase credentials:
   ```env
   # Backend
   SUPABASE_URL=your_supabase_url
   SUPABASE_SERVICE_KEY=your_supabase_service_role_key
   SUPABASE_JWT_SECRET=your_jwt_secret
   
   # Frontend (Next.js requires NEXT_PUBLIC_ prefix)
   NEXT_PUBLIC_SUPABASE_URL=your_supabase_url
   NEXT_PUBLIC_SUPABASE_ANON_KEY=your_supabase_anon_key
   NEXT_PUBLIC_API_URL=http://localhost:8000
   ```

### Rebuild ChromaDB Embeddings

```bash
python src/ingest_pdfs_enhanced.py
```

### Rebuild Knowledge Graph

```bash
python scripts/extract_topics_from_rag.py
```

---

**Need Help?** See `README.md` or `SUPABASE_SETUP.md` for detailed instructions.
