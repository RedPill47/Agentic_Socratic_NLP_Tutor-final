# Agentic Socratic NLP Tutor - Complete Documentation

**Version:** 1.0  
**Date:** December 2024  
**Status:** Production Ready (95% Complete)

---

## Table of Contents

1. [Executive Summary](#executive-summary)
2. [System Architecture](#system-architecture)
3. [RAG-First Architecture & Auto-Sync](#rag-first-architecture--auto-sync)
4. [Core Features](#core-features)
5. [Technical Components](#technical-components)
6. [Implementation Details](#implementation-details)
7. [API Reference](#api-reference)
8. [Database Schema](#database-schema)
9. [Performance Metrics](#performance-metrics)
10. [RAGAS Evaluation](#ragas-evaluation)
11. [Configuration](#configuration)
12. [Troubleshooting](#troubleshooting)

---

## Executive Summary

The **Agentic Socratic NLP Tutor** is a hybrid AI tutoring system that combines:

- **Fast single-LLM responses** (2-4 seconds) for real-time tutoring
- **Background multi-agent intelligence** (async analysis, 5-10 seconds)
- **On-demand curriculum planning** (5-agent MAS, 15-45 seconds)
- **Automatic adaptation** (difficulty, learning style, prerequisites)
- **Persistent state** (database-backed sessions)

### Key Achievement

**Transformed a 15-30 second multi-agent system into a 2-4 second single-LLM system with background intelligence, achieving 5-10x latency reduction and 5-10x cost reduction while maintaining educational quality.**

### Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                                                                  │
│   REAL-TIME TUTORING        Single LLM + Functions   2-4s      │
│   (Every message)           No agents in critical path          │
│                                                                  │
│   BACKGROUND INTELLIGENCE   Multi-Agent System        10-30s    │
│   (After each response)     3 specialized analyzers  (async)   │
│                                                                  │
│   ON-DEMAND FEATURES        Multi-Agent System        15-45s    │
│   (When requested)          Curriculum planning                 │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

---

## System Architecture

### High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                         USER INTERFACE                           │
│                      (Next.js/TypeScript)                        │
│                                                                  │
│   ┌──────────────┐  ┌──────────────┐  ┌──────────────┐          │
│   │    Chat      │  │   Progress   │  │  Curriculum  │          │
│   │  Interface   │  │  Dashboard   │  │    Viewer    │          │
│   └──────────────┘  └──────────────┘  └──────────────┘          │
│                                                                  │
│   ┌──────────────┐                                              │
│   │   Teacher    │  (Role-based access)                          │
│   │  Dashboard   │                                              │
│   └──────────────┘                                              │
└─────────────────────────────────────────────────────────────────┘
                              │
                              │ HTTP/SSE (Streaming)
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                         API LAYER (FastAPI)                      │
│                                                                  │
│   POST /api/chat/stream    GET /api/session                     │
│   POST /api/curriculum     GET /api/progress/{session_id}        │
└─────────────────────────────────────────────────────────────────┘
                              │
            ┌─────────────────┼─────────────────┐
            │                 │                 │
            ▼                 ▼                 ▼
┌───────────────────┐ ┌───────────────────┐ ┌───────────────────┐
│                   │ │                   │ │                   │
│  SINGLE LLM TUTOR │ │  BACKGROUND MAS   │ │  ON-DEMAND MAS    │
│                   │ │                   │ │                   │
│  • Real-time chat │ │  • Learning style │ │  • Curriculum     │
│  • Streaming      │ │  • Performance    │ │    generation     │
│  • Socratic       │ │  • Knowledge gaps │ │  • Learning paths │
│                   │ │                   │ │                   │
│  Latency: 2-4s    │ │  Latency: async   │ │  Latency: 15-45s  │
└───────────────────┘ └───────────────────┘ └───────────────────┘
            │                 │                 │
            └─────────────────┼─────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                      SHARED INFRASTRUCTURE                       │
│                                                                  │
│   ┌──────────────┐  ┌──────────────┐  ┌──────────────┐         │
│   │  Vector DB   │  │  Knowledge   │  │    State     │         │
│   │  (ChromaDB)  │  │    Graph     │  │ (Supabase)   │         │
│   └──────────────┘  └──────────────┘  └──────────────┘         │
└─────────────────────────────────────────────────────────────────┘
```

### Request Flow

```
User Message
    │
    ▼
┌─────────────────────────────────────────────────────────────────┐
│                        INTENT DETECTION                          │
│                                                                  │
│   "Create a curriculum for transformers" → ON-DEMAND MAS         │
│   "What is tokenization?" → SINGLE LLM TUTOR                     │
│   (First message, new session) → ONBOARDING FLOW               │
└─────────────────────────────────────────────────────────────────┘
    │
    ├─── Curriculum Request ──────────────────────┐
    │                                             │
    ├─── New Session ─────────────────┐           │
    │                                 │           │
    ▼                                 ▼           ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│  SINGLE LLM     │    │   ONBOARDING    │    │  PLANNING MAS   │
│    TUTOR        │    │   FLOW          │    │                 │
│                 │    │                 │    │  15-45 seconds  │
│  2-4 seconds    │    │  2-3 questions  │    │                 │
└────────┬────────┘    └────────┬────────┘    └─────────────────┘
         │                      │
         │                      ▼
         │             Session Initialized
         │                      │
         │                      ▼
         │             Background MAS (async)
         │                      │
         ▼                      ▼
    Response Streamed      Analysis Saved
```

---

## Core Features

### 1. Real-Time Socratic Tutoring ✅

**Component:** `SocraticTutor` class

**Features:**
- Single LLM call per response (GPT-4o-mini)
- Streaming responses for real-time feedback
- Socratic method (questions, not answers)
- Topic detection from user input
- Prerequisite gap detection
- Response-level semantic caching (20-30% cost reduction)

**Performance:**
- Response time: 2-4 seconds ✅
- Time to first token: < 1 second ✅
- RAG query: < 500ms ✅

**Key Methods:**
- `respond()` - Main entry point for user messages
- `detect_topic()` - Keyword + LLM-based topic detection
- `build_prompt()` - Constructs Socratic prompts with context
- `_normal_respond()` - Handles normal tutoring flow
- `_handle_onboarding()` - Multi-stage onboarding process

---

### 2. Background Multi-Agent System ✅

**Component:** `BackgroundAnalysisMAS` class

**Agents:**
1. **Learning Style Analyzer** - Detects visual, auditory, kinesthetic, or reading preferences
2. **Performance Evaluator** - Assesses correctness, depth, and trends
3. **Knowledge Gap Detector** - Identifies missing prerequisites and misconceptions

**Features:**
- CrewAI agents (structured, reliable)
- Async execution (non-blocking, user doesn't wait)
- Database persistence (Supabase)
- Comprehensive logging
- Automatic triggering after each response

**Performance:**
- Execution: 5-10 seconds (async)
- User impact: None (non-blocking)
- Database save: < 500ms

**Key Methods:**
- `analyze_conversation()` - Main analysis entry point
- `_save_to_database()` - Persists results to Supabase
- `load_from_database()` - Retrieves latest analysis

---

### 3. On-Demand Planning MAS ✅

**Component:** `PlanningCrew` class

**Agents:**
1. **Goal Analyzer** - Parses learning objectives
2. **Gap Assessor** - Identifies knowledge gaps
3. **Path Builder** - Creates learning sequence
4. **Resource Matcher** - Matches RAG content to steps
5. **Plan Formatter** - Formats final curriculum

**Features:**
- Automatic intent detection ("create a plan", "curriculum", etc.)
- Personalized based on current knowledge
- UI button integration
- Structured output (JSON → formatted markdown)

**Performance:**
- Execution: 15-45 seconds
- User expectation: Explicit wait (informed via UI)

**Trigger:**
- User clicks "Create Curriculum" button
- User types planning request ("create a learning plan for...")
- Automatic detection via `is_planning_request()`

---

### 4. 5-Stage Onboarding Flow ✅ (Updated December 2024)

**Component:** `OnboardingState`, `ResponseStrengthScorer`, `OnboardingSummaryGenerator`, `UserProfileManager`

**Stages:**
1. **WELCOME** - Extract goal and implied level
2. **LEARNING_STYLE** - Assess learning style preference (visual, auditory, kinesthetic, reading)
3. **KNOWLEDGE_ASSESSMENT** - Diagnostic questions (2-4 questions using binary search)
4. **PREFERENCES** - Capture teaching pace and practice preferences
5. **CONFIRMATION** - Generate summary and save to user profile

**Features:**
- Binary search algorithm for efficient knowledge calibration
- Hybrid response scoring (heuristic + LLM)
- Mastery inference from demonstrated knowledge
- Learning style detection from user responses
- User-level profile updates (persists across sessions)
- Statistics tracking (sessions, interactions, last activity)

**Flow:**
```
Welcome → Learning Style → Knowledge Assessment → Preferences → Confirmation → Complete
```

**Key Components:**
- `OnboardingState` - Manages stage transitions and state
- `ResponseStrengthScorer` - Evaluates student responses
- `OnboardingSummaryGenerator` - Creates personalized summary
- `UserProfileManager` - Saves data to user-level profile
- `_detect_learning_style_from_response()` - LLM-based style detection
- `_detect_learning_preferences()` - Pace and practice detection

**User-Level Storage:**
- `learning_style` - Primary learning style (visual, auditory, kinesthetic, reading)
- `learning_style_confidence` - Confidence score (0.0-1.0)
- `mastered_concepts` - JSON array of concepts user has mastered
- `overall_difficulty` - User's overall difficulty level
- `onboarding_complete` - User-level onboarding status
- `total_sessions` - Total number of sessions
- `total_interactions` - Total number of interactions
- `last_activity` - Last activity timestamp

---

### 6. Teacher Dashboard ✅ (December 2024)

**Component:** `frontend/app/teacher/page.tsx`

**Features:**
- **Role-based access control** - Teachers redirected to dashboard, students to chat
- **Student progress monitoring** - View all students' learning progress
- **Analytics dashboard** - Total students, sessions, documents, chunks
- **Document management** - Upload and track knowledge base documents
- **Performance insights** - Mastery scores, knowledge gaps, learning styles
- **Real-time data** - Fetches from `sessions`, `profiles`, `mas_analysis`, `documents` tables

**Access Control:**
- Middleware redirects teachers from `/chat` to `/teacher`
- Teachers cannot access chat interface
- Students cannot access teacher dashboard
- RLS policies allow teachers to view all student data

**Data Sources:**
- `profiles` - Student information, learning styles, statistics
- `sessions` - Current topics, understanding scores, activity
- `mas_analysis` - Performance assessments, knowledge gaps
- `documents` - Uploaded knowledge base documents (processing planned)

**Implementation:**
- Direct Supabase queries (no backend required for viewing)
- PDF upload to Supabase Storage
- Document processing pipeline planned (see `docs/DOCUMENT_UPLOAD_SOLUTIONS.md`)

---

### 7. Automatic Difficulty Adaptation ✅

**Component:** `DifficultyAdapter` class

**Features:**
- Performance-based adjustment
- Trend analysis (improving/declining/stable)
- Automatic transitions (beginner ↔ intermediate ↔ advanced)
- Minimum 3 interactions before adjustment

**Algorithm:**
```
1. Calculate average of last 3 understanding scores
2. Determine performance trend
3. If avg >= 0.8 AND trend == "improving" → Increase difficulty
4. If avg <= 0.4 AND trend == "declining" → Decrease difficulty
5. Otherwise → Maintain current difficulty
```

**Integration:**
- Automatically called after each response
- Updates `SessionState.difficulty`
- Logs adaptation events

---

### 6. Enhanced Knowledge Graph ✅

**Component:** `EnhancedPrerequisiteGraph` class

**Features:**
- `Concept` dataclass with rich metadata:
  - Name, difficulty, topic area
  - Description, learning time
  - Keywords for topic detection
- 30+ NLP concepts with full metadata
- Learning path generation
- Topic/difficulty filtering
- Prerequisite gap detection
- LLM-based topic extraction from RAG

**Key Methods:**
- `add_concept()` - Add concept with metadata
- `add_prerequisite()` - Add prerequisite relationship
- `get_gaps()` - Find missing prerequisites
- `get_learning_path()` - Generate path to target concept
- `detect_topic()` - Sophisticated keyword matching

**Auto-Expansion:**
- Script: `scripts/extract_topics_from_rag.py`
- Extracts topics from RAG chunks
- Enriches with metadata using LLM
- Infers prerequisites
- Generates code for knowledge graph

---

### 7. User-Level Profile Storage ✅ (Updated December 2024)

**Component:** `UserProfileManager` class

**Purpose:**
- Store user-level learning data that persists across ALL sessions
- Enable true personalized learning that builds on previous knowledge
- Aggregate learning style, mastered concepts, and difficulty across sessions

**Features:**
- Learning style persistence (visual, auditory, kinesthetic, reading)
- Mastered concepts aggregation (all concepts learned across sessions)
- Overall difficulty tracking (user's general level)
- Statistics tracking (total_sessions, total_interactions, last_activity)
- Onboarding status (user-level, not session-level)
- Prerequisite checking uses user-level knowledge

**Database Schema (`profiles` table):**
- `learning_style` - Primary learning style
- `learning_style_confidence` - Confidence score (0.0-1.0)
- `learning_style_updated_at` - Last update timestamp
- `mastered_concepts` - JSON array of all mastered concepts
- `mastered_concepts_updated_at` - Last update timestamp
- `overall_difficulty` - User's overall difficulty level
- `difficulty_updated_at` - Last update timestamp
- `total_sessions` - Total number of sessions
- `total_interactions` - Total number of interactions
- `last_activity` - Last activity timestamp
- `onboarding_complete` - User-level onboarding status
- `onboarding_completed_at` - Onboarding completion timestamp

**Key Methods:**
- `get_user_profile()` - Load user profile from database
- `update_learning_style()` - Update learning style and confidence
- `add_mastered_concepts()` - Add concepts (with deduplication)
- `check_prerequisites()` - Check if user has prerequisites (uses user-level knowledge)
- `update_onboarding_status()` - Mark onboarding complete
- `update_statistics()` - Update session/interaction counts and last activity

**Integration:**
- Loaded on session start in `SocraticTutor.get_or_create_session()`
- Updated during onboarding completion
- Used for prerequisite checking across all sessions
- Background MAS updates learning style periodically

### 8. State Persistence ✅

**Component:** `SessionManager` class

**Features:**
- Database-backed sessions (Supabase PostgreSQL)
- Survives server restarts
- Multi-instance support
- Automatic save after each interaction
- Load on session creation
- **Session-specific data only** (user-level data in `profiles` table)

**Schema:**
- `sessions` table - Session-specific state (current_topic, interaction_count, etc.)
- `messages` table - Conversation history (source of truth)
- `mas_analysis` table - Background MAS results
- `profiles` table - User-level data (learning style, mastered concepts, etc.)

**Key Methods:**
- `get_session()` - Load from database
- `save_session()` - Persist to database
- In-memory cache for performance

---

### 8. Response-Level Semantic Caching ✅

**Component:** `ResponseCache` class

**Features:**
- Semantic similarity matching (SentenceTransformer)
- Cache key based on: user input, difficulty, topic, gaps
- Similarity threshold: 0.85
- TTL: 24 hours
- Max size: 100 entries

**Performance:**
- Cache hit: < 10ms
- Cost reduction: 20-30%
- Automatic cache eviction

---

### 9. RAG (Retrieval-Augmented Generation) ✅

**Component:** `FastRAG` class

**Features:**
- ChromaDB vector store
- HuggingFace embeddings (all-MiniLM-L6-v2)
- MMR (Maximal Marginal Relevance) search
- Difficulty-based filtering
- Fast queries (< 500ms)

**Integration:**
- Used by `SocraticTutor` for content retrieval
- Used by `PlanningCrew` for resource matching
- Automatic topic extraction from chunks

**Key Insight:**
- **RAG is the source of truth** - can retrieve content about ANY topic in your materials
- Works independently of knowledge graph
- System can teach any topic from RAG, even if not in knowledge graph

---

### 10. RAGAS Evaluation ✅

**Component:** RAGAS (Retrieval-Augmented Generation Assessment) framework

**Purpose:**
- Evaluate RAG system quality and performance
- Measure retrieval accuracy and response faithfulness
- Benchmark system improvements

**Location:**
- Evaluation scripts: `scripts/evaluate_ragas_eurlex.py`
- Evaluation data: `data/eval/`
- Results: `data/eval/eurlex_ragas_results.json` and `.csv`

**Metrics Evaluated:**

1. **Context Precision** (0.0 - 1.0)
   - Measures how many retrieved contexts are relevant to the question
   - Higher = better retrieval quality
   - Current average: ~0.674 (std: 0.363)

2. **Context Recall** (0.0 - 1.0)
   - Measures how many relevant contexts were retrieved
   - Higher = better coverage of relevant information
   - Current average: ~0.750 (std: 0.433)

3. **Faithfulness** (0.0 - 1.0)
   - Measures if the generated answer is grounded in retrieved contexts
   - Higher = less hallucination, more factual accuracy
   - Current average: ~0.752 (std: 0.365)

**Evaluation Setup:**

- **Dataset**: 12 GDPR questions with manually authored reference contexts
- **Collection**: `eurlex_gdpr` (ChromaDB)
- **Embeddings**: `all-MiniLM-L6-v2` (HuggingFace)
- **LLM**: `gpt-4o-mini` (temperature: 0.0)
- **Retriever**: Similarity search (k=5) for evaluation
- **Source**: EUR-Lex GDPR regulation (CELEX: 32016R0679)

**Configuration Files:**

- `data/eval/eurlex_eval_config.json` - Evaluation configuration
- `data/eval/eurlex_eval.json` - Evaluation dataset (questions + reference contexts)
- `data/eval/eurlex_ingest_config.json` - Data ingestion configuration
- `data/eval/eurlex_runs.jsonl` - Detailed run logs (retrieved contexts, answers)
- `data/eval/eurlex_ragas_results.json` - Evaluation results (JSON)
- `data/eval/eurlex_ragas_results.csv` - Evaluation results (CSV)

**Running Evaluations:**

```bash
# From project root
cd scripts
python evaluate_ragas_eurlex.py
```

**Requirements:**
- `OPENAI_API_KEY` environment variable set
- ChromaDB collection `eurlex_gdpr` populated
- RAGAS library installed: `pip install ragas`

**Important Notes:**

1. **Evaluation vs Production Retrieval:**
   - Evaluation uses similarity search (k=5)
   - Production uses MMR (k=3, fetch_k=9, lambda=0.7)
   - Results are most applicable to similarity-based retrieval
   - For production MMR evaluation, re-run with matching retriever settings

2. **Reference Contexts:**
   - Manually authored and treated as authoritative
   - Used for calculating context_recall and context_precision
   - Logged in `eurlex_runs.jsonl` for traceability

3. **Robustness:**
   - Results assume stable LLM behavior (gpt-4o-mini, temperature 0)
   - Evaluation framework can be reused with different retriever configurations
   - All inputs/outputs are logged for reproducibility

**Interpreting Results:**

- **High Context Precision + High Context Recall**: Excellent retrieval quality
- **High Faithfulness**: Answers are well-grounded in retrieved contexts
- **Low scores**: Indicates need for retrieval/embedding improvements

**See Also:**
- `docs/evaluation_robustness.md` - Detailed evaluation methodology
- `data/eval/eurlex_runs.jsonl` - Full evaluation traces

---

### 11. Automatic Concept Syncing ✅

**Components:** 
- `_lazy_load_concept()` - Immediate concept addition
- `_runtime_discover_concept()` - Comprehensive discovery
- `BackgroundSync` - Periodic batch sync

**Features:**
- **Lazy Loading**: Auto-adds concepts when detected but not in graph (< 1s, non-blocking)
- **Runtime Discovery**: Comprehensive metadata extraction (2-3s, non-blocking)
- **Background Sync**: Periodic batch extraction (every 24h, 5-10min)

**How it works:**
1. User asks about topic not in knowledge graph
2. Topic detected via LLM fallback
3. RAG validates topic exists
4. Metadata extracted automatically
5. Concept added to knowledge graph in background
6. Future requests can use prerequisite checking

**Configuration:**
```bash
AUTO_SYNC_CONCEPTS=true          # Master switch
LAZY_LOAD_CONCEPTS=true          # Fast per-topic sync
RUNTIME_DISCOVERY=true           # Comprehensive per-topic sync
BACKGROUND_SYNC_ENABLED=true     # Periodic batch sync
BACKGROUND_SYNC_INTERVAL_HOURS=24
```

**Benefits:**
- No manual maintenance required
- Knowledge graph stays up-to-date automatically
- System works with any topic from RAG
- All sync operations are non-blocking

---

## RAG-First Architecture & Auto-Sync

### Core Principle: RAG is Source of Truth

The system has **two independent pillars**:

1. **RAG (Retrieval-Augmented Generation)**
   - **Purpose**: Source of truth for content
   - **Contains**: All your course PDFs, chunked and vectorized
   - **Capability**: Can retrieve content about ANY topic in your materials
   - **Location**: ChromaDB vector store (`data/chroma_db/`)

2. **Knowledge Graph**
   - **Purpose**: Prerequisite checking and learning paths
   - **Contains**: Structured concepts with relationships
   - **Capability**: Tracks what students need to learn first
   - **Location**: `EnhancedPrerequisiteGraph` class

### Key Insight

**RAG and Knowledge Graph are independent!**
- RAG can teach any topic, even if not in the graph
- Knowledge graph is only needed for prerequisite checking
- The system works perfectly fine with topics not in the graph

### Complete Flow: User Input → Response

#### Step 1: User Sends Message
```
User: "I want to learn about text processing"
```

#### Step 2: Session Management
- Load session state from Supabase (or create new)
- Check onboarding status
- Load conversation history

#### Step 3: Topic Detection (3-Tier System with Explicit Switch Check)

**Tier 1: Fast Keyword Matching** (< 10ms)
- Checks knowledge graph concepts
- Manual keyword mappings
- Enhanced graph keyword matching

**Tier 2: LLM Fallback** (~1-2s, if Tier 1 fails)
- Queries RAG to see what topics are relevant
- Uses LLM to extract topic from user input + RAG context
- Normalizes topic name
- **Key Feature**: Can detect ANY topic, not just those in knowledge graph!

**Tier 3: Explicit Topic Switch Check** (NEW - December 2024)
- If current topic exists and different topic detected:
  - Uses LLM to check if user is EXPLICITLY asking about new topic
  - Prevents false positives from keyword matching
  - Only switches if user explicitly requests different topic
  - Example: User answering about tokenization won't switch to TF-IDF just because "documents" is mentioned

**Tier 4: Keep Previous Topic** (if all fail or not explicit switch)
- Uses previous topic for context
- Topic persists after onboarding completion

#### Step 4: Prerequisite Checking
- If topic is in knowledge graph: Check for missing prerequisites
- If topic is NOT in graph: Skip prerequisite checking (but RAG still works!)

#### Step 5: RAG Content Retrieval
- Semantic search in ChromaDB vector store
- Retrieves 3 most relevant chunks
- **Key Point**: This works for ANY topic in RAG, whether in knowledge graph or not!

#### Step 6: Auto-Sync (Background, Non-Blocking)

**A. Lazy Loading** (Immediate, < 1s)
- Validates topic exists in RAG
- Extracts basic metadata (difficulty, description, keywords)
- Adds concept to knowledge graph
- Runs in background (doesn't block response)

**B. Runtime Discovery** (Comprehensive, 2-3s)
- Queries multiple RAG chunks for better context
- Extracts comprehensive metadata
- Optionally infers prerequisite relationships
- Adds concept to knowledge graph

#### Step 7: Build Prompt
- User's question
- Relevant RAG content (the actual course material)
- Current topic
- Knowledge gaps (if topic in graph)
- Student's difficulty level
- Conversation history
- Learning style insights (from background MAS)

#### Step 8: Generate Response
- LLM generates Socratic response
- Uses RAG content for explanations
- Uses knowledge graph info for prerequisite awareness
- Streams response token by token

#### Step 9: Save Session
- Updates conversation history in Supabase
- Saves current topic and state

### Auto-Sync Features (3 Layers)

#### Layer 1: Lazy Loading (Immediate)
**When**: Topic detected but not in graph

**What it does:**
- Quick RAG validation
- Basic metadata extraction
- Auto-add to graph

**Speed**: < 1s, non-blocking

**Example:**
```
User: "What is CRF tagging?"
→ Topic detected: "Conditional Random Fields"
→ Not in graph → Lazy load triggers
→ Concept added to graph in background
```

#### Layer 2: Runtime Discovery (Comprehensive)
**When**: Topic detected via LLM fallback

**What it does:**
- Multiple RAG chunk queries
- Full metadata extraction
- Prerequisite inference
- Auto-add to graph

**Speed**: ~2-3s, non-blocking

#### Layer 3: Background Sync (Periodic)
**When**: Every 24 hours (configurable)

**What it does:**
- Runs `extract_topics_from_rag.py` script
- Batch discovers ALL topics from RAG
- Adds them to knowledge graph
- Generates code for graph updates

**Speed**: 5-10 minutes, runs in background

**How it works:**
```python
# In backend/main.py startup
@app.on_event("startup")
async def startup_event():
    background_sync = get_background_sync()
    if background_sync:
        await background_sync.start()  # Starts periodic sync loop
```

### Data Flow Diagram

```
User Input
    ↓
Session Load (Supabase)
    ↓
Topic Detection
    ├─→ Fast Keyword Match (10ms)
    ├─→ LLM Fallback (1-2s) ← Uses RAG context!
    └─→ Keep Previous Topic
    ↓
Check Knowledge Graph
    ├─→ In Graph → Check Prerequisites
    └─→ Not in Graph → Skip (but RAG still works!)
    ↓
RAG Query (ChromaDB)
    └─→ Retrieve 3 most relevant chunks
    ↓
Auto-Sync (Background, Non-Blocking)
    ├─→ Lazy Load (if not in graph)
    └─→ Runtime Discovery (comprehensive)
    ↓
Build Prompt
    ├─→ User input
    ├─→ RAG content
    ├─→ Topic
    ├─→ Gaps (if in graph)
    ├─→ Session state
    └─→ Background MAS context
    ↓
LLM Generation
    └─→ Stream response
    ↓
Save Session (Supabase)
    └─→ Update conversation history
```

### Teaching Flow Examples

#### Example 1: Topic in Knowledge Graph
**User**: "I want to learn about transformers"

**Flow:**
1. Topic detected: "Transformer" ✅ (in graph)
2. Prerequisites checked: ["Self-Attention", "Neural Networks"]
3. Gaps found: ["Self-Attention"] (student hasn't learned it)
4. RAG retrieves: Content about transformers
5. LLM response: "Before we dive into transformers, let's make sure you understand self-attention..."

**Result**: Prerequisite-aware teaching

#### Example 2: Topic NOT in Knowledge Graph
**User**: "I want to learn about text processing"

**Flow:**
1. Topic detected: "Text Preprocessing" ❌ (not in graph initially)
2. Prerequisites: Skipped (no graph entry)
3. RAG retrieves: Content about text processing ✅
4. LLM response: "Great! Text processing involves several steps..."
5. Background: Lazy load adds "Text Preprocessing" to graph

**Result**: Still teaches perfectly! Graph gets updated automatically

#### Example 3: Completely New Topic
**User**: "What is BERT fine-tuning?"

**Flow:**
1. Topic detected: "BERT Fine-tuning" ❌ (not in graph)
2. LLM fallback: Uses RAG to validate topic exists
3. RAG retrieves: Content about BERT fine-tuning ✅
4. LLM response: "BERT fine-tuning is..."
5. Background: Runtime discovery adds concept with full metadata

**Result**: System adapts automatically!

### Key Benefits

1. **Self-Updating**: Knowledge graph automatically syncs with RAG
2. **No Manual Maintenance**: Concepts discover themselves
3. **Works with Any Topic**: RAG is source of truth, graph is optional
4. **Non-Blocking**: All sync operations run in background
5. **Robust**: System works even if topic not in graph
6. **Scalable**: Handles new topics automatically

---

## Technical Components

### Backend Stack

- **Framework:** FastAPI
- **LLM:** OpenAI GPT-4o-mini (configurable)
- **Embeddings:** HuggingFace SentenceTransformer (all-MiniLM-L6-v2)
- **Vector DB:** ChromaDB
- **Database:** Supabase (PostgreSQL)
- **Agent Framework:** CrewAI
- **Async:** asyncio, AsyncOpenAI

### Frontend Stack

- **Framework:** Next.js 14
- **Language:** TypeScript
- **UI:** React, Tailwind CSS
- **HTTP Client:** fetch API
- **State Management:** React hooks

### Key Files

**Backend:**
- `backend/main.py` - FastAPI application
- `agentic_socratic_nlp_tutor/src/agentic_socratic_nlp_tutor/socratic_tutor.py` - Core tutor
- `agentic_socratic_nlp_tutor/src/agentic_socratic_nlp_tutor/background_analysis.py` - Background MAS
- `agentic_socratic_nlp_tutor/src/agentic_socratic_nlp_tutor/planning_crew.py` - Planning MAS
- `agentic_socratic_nlp_tutor/src/agentic_socratic_nlp_tutor/session_manager.py` - State persistence
- `agentic_socratic_nlp_tutor/src/agentic_socratic_nlp_tutor/knowledge_graph.py` - Knowledge graph

**Frontend:**
- `frontend/app/chat/page.tsx` - Chat page
- `frontend/components/ChatInterface.tsx` - Chat component
- `frontend/components/MessageList.tsx` - Message display

---

## Implementation Details

### Onboarding Flow (Updated December 2024 - 5 Stages)

**Stage 1: WELCOME**
- Extract `stated_goal` and `implied_level` from user input
- Generate welcome message
- Transition to `LEARNING_STYLE`

**Stage 2: LEARNING_STYLE**
- Ask learning style question
- Detect learning style from user response (LLM-based)
- Store in `onboarding_state.learning_style_detected`
- Update `state.learning_style`
- Transition to `KNOWLEDGE_ASSESSMENT`

**Stage 3: KNOWLEDGE_ASSESSMENT**
- Binary search algorithm:
  - Start with middle prerequisite
  - Ask diagnostic question
  - Score response (strong/weak/moderate)
  - Adjust bounds based on score
  - Repeat 2-4 times (minimum 2 questions)
- Infer mastery from demonstrated knowledge
- Transition to `PREFERENCES`

**Stage 4: PREFERENCES**
- Ask about teaching pace (fast/detailed/balanced)
- Ask about practice preference (examples/deep_dives/applications)
- Detect preferences from user response
- Store in `onboarding_state`
- Transition to `CONFIRMATION`

**Stage 5: CONFIRMATION**
- Generate personalized summary
- Set `current_topic` from `stated_goal` (NEW)
- Update user-level profile:
  - `learning_style` and `learning_style_confidence`
  - `mastered_concepts` (aggregated)
  - `onboarding_complete = True`
  - `total_sessions` (increment)
  - `total_interactions` (count)
  - `last_activity` (timestamp)
- Update `SessionState`:
  - `onboarding_complete = True`
  - `current_topic` (from stated_goal)
- Transition to normal tutoring

### Background MAS Flow

**Trigger:** After each user response (async)

**Process:**
1. Extract conversation context
2. Run 3 agents in parallel:
   - Learning Style Analyzer
   - Performance Evaluator
   - Knowledge Gap Detector
3. Combine results into `AnalysisResult`
4. Save to database (`mas_analysis` table)
5. Update `sessions` table with latest analysis

**Output:**
- Learning style (primary + confidence)
- Performance assessment (correctness, depth, trend)
- Knowledge gaps (missing prerequisites)
- Misconceptions (incorrect beliefs)
- Teaching recommendations

### Planning MAS Flow

**Trigger:** User requests curriculum (button or text)

**Process:**
1. Detect planning intent
2. Initialize `PlanningCrew` (if not exists)
3. Run 5 agents sequentially:
   - Goal Analyzer → Goal structure
   - Gap Assessor → Missing prerequisites
   - Path Builder → Learning sequence
   - Resource Matcher → RAG content per step
   - Plan Formatter → Final curriculum
4. Stream formatted plan to user

**Output:**
- Structured learning plan (markdown)
- Step-by-step curriculum
- Resource links (RAG citations)
- Estimated time per step

### Difficulty Adaptation Flow

**Trigger:** After each response (before MAS)

**Process:**
1. Check minimum interactions (>= 3)
2. Calculate average of last 3 understanding scores
3. Determine performance trend
4. Apply adjustment logic:
   - Increase if excelling
   - Decrease if struggling
   - Maintain otherwise
5. Update `SessionState.difficulty`
6. Log adaptation event

---

## API Reference

### POST `/api/chat/stream`

**Purpose:** Stream chat responses

**Request:**
```json
{
  "content": "What is tokenization?",
  "session_id": "session_123",
  "user_id": "user_456"
}
```

**Response:** Server-Sent Events (SSE) stream

**Headers:**
- `Content-Type: text/event-stream`
- `Cache-Control: no-cache`

---

### GET `/api/session/{session_id}`

**Purpose:** Get session state

**Response:**
```json
{
  "session_id": "session_123",
  "current_topic": "Tokenization",
  "difficulty": "intermediate",
  "mastered_concepts": ["Tokenization", "Word Embeddings"],
  "onboarding_complete": true,
  ...
}
```

---

### GET `/api/progress/{session_id}`

**Purpose:** Get progress metrics

**Response:**
```json
{
  "total_interactions": 15,
  "mastered_concepts_count": 5,
  "current_difficulty": "intermediate",
  "performance_trend": "improving",
  "average_understanding_score": 0.75,
  ...
}
```

---

### POST `/api/curriculum`

**Purpose:** Generate learning curriculum (triggered automatically via chat)

**Request:** Same as `/api/chat/stream` with planning intent

**Response:** Streaming curriculum (15-45 seconds)

---

## Database Schema

### `sessions` Table

```sql
CREATE TABLE sessions (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    session_id TEXT UNIQUE NOT NULL,
    user_id TEXT NOT NULL,
    current_topic TEXT,
    mastered_concepts TEXT[],
    difficulty TEXT DEFAULT 'intermediate',
    learning_style TEXT,
    stated_goal TEXT,
    stated_level TEXT,
    understanding_scores FLOAT[],
    performance_trend TEXT,
    conversation_history JSONB,
    onboarding_complete BOOLEAN DEFAULT FALSE,
    mas_analysis JSONB,
    last_analysis_at TIMESTAMPTZ,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    last_updated TIMESTAMPTZ DEFAULT NOW()
);
```

### `mas_analysis` Table

```sql
CREATE TABLE mas_analysis (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    session_id UUID REFERENCES sessions(id),
    user_id TEXT NOT NULL,
    learning_style TEXT,
    performance_assessment JSONB,
    knowledge_gaps TEXT[],
    misconceptions TEXT[],
    teaching_recommendations TEXT[],
    confidence_score FLOAT,
    created_at TIMESTAMPTZ DEFAULT NOW()
);
```

---

## Performance Metrics

### Latency Targets vs Achieved

| Component | Target | Achieved | Status |
|-----------|--------|----------|--------|
| Tutoring (TTFT) | < 1s | < 1s | ✅ |
| Tutoring (total) | 2-4s | 2-4s | ✅ |
| RAG query | < 500ms | < 500ms | ✅ |
| Background MAS | < 10s | 5-10s | ✅ |
| Planning MAS | 15-45s | 15-45s | ✅ |
| Onboarding | 2-3 questions | 2-3 questions | ✅ |
| Cache hit | < 10ms | < 10ms | ✅ |

### Cost Optimization

- **Single LLM call** per response (vs 4-6 before)
- **Simplified RAG** (no expansion/re-ranking)
- **Response caching** (20-30% reduction)
- **Background MAS** (async, doesn't block)

**Estimated cost reduction:** 5-10x

---

## Configuration

### Environment Variables

#### Auto-Sync Configuration
```bash
# Auto-sync master switch
AUTO_SYNC_CONCEPTS=true

# Lazy loading (fast, per-topic)
LAZY_LOAD_CONCEPTS=true

# Runtime discovery (comprehensive, per-topic)
RUNTIME_DISCOVERY=true

# Background sync (periodic batch)
BACKGROUND_SYNC_ENABLED=true
BACKGROUND_SYNC_INTERVAL_HOURS=24
```

#### Core Configuration
```bash
# OpenAI API
OPENAI_API_KEY=your_key_here
OPENAI_MODEL=gpt-4o-mini

# Supabase
SUPABASE_URL=your_url
SUPABASE_KEY=your_key

# Feature Flags
USE_LEGACY_MANAGER=false  # Use SocraticTutor (new) vs ConversationManager (legacy)
```

### What Gets Synced

**Lazy Loading:**
- Basic metadata (difficulty, description, keywords)
- Single RAG chunk validation

**Runtime Discovery:**
- Full metadata (difficulty, topic_area, description, keywords, learning_time)
- Multiple RAG chunks for context
- Prerequisite inference

**Background Sync:**
- All topics from RAG
- Batch processing
- Code generation for knowledge graph

### Manual Sync

You can also manually trigger a sync:

```bash
# Run extraction script manually
python scripts/extract_topics_from_rag.py

# Or with auto-confirm (for automation)
python scripts/extract_topics_from_rag.py --auto-confirm
```

### Monitoring

Check logs for sync activity:
- `🔄 [LazyLoad]` - Lazy loading events
- `🔍 [RuntimeDiscovery]` - Runtime discovery events
- `🔄 [BackgroundSync]` - Background sync events

---

## Troubleshooting

### Common Issues

**1. Topic Not Detected**
- **Symptom:** `Current topic: None` in logs
- **Solution:** 
  - Check if topic exists in knowledge graph
  - Run `scripts/extract_topics_from_rag.py` to add topics
  - LLM fallback should catch most cases

**2. Background MAS Not Running**
- **Symptom:** No MAS logs in console
- **Solution:**
  - Check `background_mas` is initialized in `SocraticTutor`
  - Verify async task is created: `asyncio.create_task()`
  - Check database connection (Supabase)

**3. Planning MAS Not Triggering**
- **Symptom:** Button click doesn't generate curriculum
- **Solution:**
  - Check `is_planning_request()` detection
  - Verify `PlanningCrew` initialization
  - Check CrewAI availability

**4. Teacher Dashboard Not Loading Data**
- **Symptom:** Dashboard shows "No data" or errors
- **Solution:**
  - Verify `documents` table exists (run migration)
  - Check RLS policies are applied (run `add_teacher_access_policies.sql`)
  - Verify Supabase Storage bucket `documents` exists
  - Check browser console for specific errors

**5. Documents Table Not Found**
- **Symptom:** Error: "Could not find the table 'public.documents'"
- **Solution:**
  - Run `backend/migrations/create_documents_table.sql` in Supabase SQL Editor
  - Refresh dashboard (errors are non-blocking, dashboard will load)

**4. Session State Not Persisting**
- **Symptom:** State lost on restart
- **Solution:**
  - Verify Supabase connection
  - Check `SessionManager.save_session()` is called
  - Check database migrations are applied

**7. Import Errors**
- **Symptom:** `ModuleNotFoundError`
- **Solution:**
  - Ensure virtual environment is activated
  - Run `pip install -r requirements.txt`
  - Check Python path includes project root

**6. Concepts Not Being Auto-Added**
- **Symptom:** Topics detected but not added to knowledge graph
- **Solution:**
  - Check that `AUTO_SYNC_CONCEPTS=true`
  - Verify `LAZY_LOAD_CONCEPTS=true` or `RUNTIME_DISCOVERY=true`
  - Verify RAG has content about the topic
  - Check logs for errors (`🔄 [LazyLoad]` or `🔍 [RuntimeDiscovery]`)

**7. Background Sync Not Running**
- **Symptom:** No periodic sync happening
- **Solution:**
  - Verify `BACKGROUND_SYNC_ENABLED=true`
  - Check that `extract_topics_from_rag.py` exists
  - Look for startup logs (`✅ Background sync started`)
  - Check sync interval: `BACKGROUND_SYNC_INTERVAL_HOURS=24`

**8. Topic Detection Failing**
- **Symptom:** `Current topic: None` in logs
- **Solution:**
  - Check if topic exists in knowledge graph (for fast detection)
  - LLM fallback should catch most cases (check logs for `🎯 [SocraticTutor] LLM fallback`)
  - Verify RAG has content about the topic
  - Check that topic name is normalized correctly

**9. Topic Switching Too Aggressively** (Fixed December 2024)
- **Symptom:** Topic switches when user is answering about current topic
- **Cause:** Keyword matching detects related topics in user's answer
- **Solution:** Added explicit topic switch check using LLM
- **Fix:** `_is_explicit_topic_switch()` method checks if user is explicitly asking about different topic
- **Result:** Topic only switches when user explicitly requests it

**10. Profile Not Updating** (Fixed December 2024)
- **Symptom:** Learning style, statistics not saved to profiles table
- **Cause:** Profile existence not checked before update
- **Solution:** Added profile existence checks and better error handling
- **Fix:** `update_learning_style()`, `update_statistics()` now check profile exists
- **Result:** All user-level data properly saved to database

**11. Too Many Concepts Being Added**
- **Symptom:** Knowledge graph growing too fast
- **Solution:**
  - Disable lazy loading: `LAZY_LOAD_CONCEPTS=false`
  - Disable runtime discovery: `RUNTIME_DISCOVERY=false`
  - Increase background sync interval
  - Review extracted topics manually before adding

---

## Additional Resources

### Setup Guides
- **Quick Start:** See `QUICK_START.md` in project root
- **Supabase Setup:** See `SUPABASE_SETUP.md` in project root
- **Backend Setup:** See `backend/SETUP_INSTRUCTIONS.md`

### Architecture & Implementation
- **Architecture Diagram:** See `docs/nlp_tutor_complete_system_architecture.md`
- **Archived Documentation:** See `docs/archive/` for historical implementation notes and reports

---

---

## Summary

### System Architecture Summary

**The system works in 3 layers:**

1. **RAG**: Always works, can teach any topic
2. **Knowledge Graph**: Optional, for prerequisite checking
3. **Auto-Sync**: Keeps graph updated automatically

**The flow:**
- User asks question
- Topic detected (fast keyword → LLM fallback)
- RAG retrieves content (always works)
- Prerequisites checked (if in graph)
- Response generated (uses RAG + graph info)
- Auto-sync updates graph (background)

**The result:**
- System can teach ANY topic from your course materials
- Knowledge graph stays up-to-date automatically
- No manual maintenance required!

### Key Achievements

1. **5-10x latency reduction**: From 15-30s to 2-4s for tutoring
2. **5-10x cost reduction**: Single LLM call vs multi-agent system
3. **Self-updating knowledge graph**: Auto-sync from RAG
4. **RAG-first architecture**: Works with any topic, graph optional
5. **Non-blocking intelligence**: Background MAS + auto-sync

---

**Last Updated:** December 2024  
**Version:** 2.0 (with Auto-Sync)

