# Agentic Socratic NLP Tutor

**Final Demo Version**

An intelligent, adaptive AI tutoring system that teaches Natural Language Processing (NLP) concepts through Socratic dialogue. Built with FastAPI, Next.js, and CrewAI.

## Features

- **5-Stage Onboarding Flow** - Comprehensive learning style and knowledge assessment
- **User-Level Profile Storage** - Personalized learning that persists across sessions
- **Smart Topic Detection** - Explicit topic switching prevents false positives
- **Real-Time Tutoring** - 2-4 second response times with streaming
- **Background Intelligence** - Multi-agent system for learning analysis
- **Automatic Adaptation** - Difficulty, learning style, and prerequisite checking

## Prerequisites

- **Python** >= 3.10, < 3.13
- **Node.js** >= 18
- **OpenAI API key**

## Installation

1. **Set up environment variables**

   Create a `.env` file in the project root:
   ```env
   OPENAI_API_KEY=your_actual_openai_api_key_here
   ```
   
   > **Note:** Both backend and frontend read from this single root `.env` file. Supabase credentials are pre-configured in the repository.

2. **Install backend dependencies**
   ```bash
   cd backend
   python -m venv venv
   
   # Windows:
   venv\Scripts\activate
   # Linux/Mac:
   source venv/bin/activate
   
   pip install -r requirements.txt
   ```

3. **Install frontend dependencies**
   ```bash
   cd frontend
   npm install
   ```

## Running the Application

### Start Backend

```bash
cd backend
source venv/bin/activate  # Windows: venv\Scripts\activate
python main.py
```

Backend runs on `http://localhost:8000`

### Start Frontend

```bash
cd frontend
npm run dev
```

Frontend runs on `http://localhost:3000`

### Access the Application

Open `http://localhost:3000` in your browser.

The app is ready to use with:
- Pre-configured Supabase database
- Pre-computed ChromaDB embeddings
- Knowledge graph with NLP concepts

## Project Structure

```
Agentic_Socratic_NLP_Tutor/
├── backend/                    # FastAPI backend
│   ├── main.py                 # API entry point
│   ├── requirements.txt        # Python dependencies
│   └── migrations/             # Database migrations
├── frontend/                    # Next.js frontend
│   ├── app/                    # Next.js app directory
│   ├── components/             # React components
│   └── package.json           # Node dependencies
├── agentic_socratic_nlp_tutor/ # Core package
│   └── src/
│       └── agentic_socratic_nlp_tutor/
│           ├── socratic_tutor.py      # Core tutor
│           ├── background_analysis.py # Background MAS
│           ├── planning_crew.py      # Planning MAS
│           └── ...
├── data/
│   ├── chroma_db/              # Vector database
│   └── slides/                 # PDF course materials
└── tests/                       # Test suite
```

## Documentation

- **Quick Start:** See `QUICK_START.md`
- **Supabase Setup:** See `SUPABASE_SETUP.md`
- **Full Documentation:** See `DOCUMENTATION.md`

## Testing

```bash
# Unit tests
pytest tests/unit/

# End-to-end tests
pytest tests/e2e/
```

## License

[Your License Here]

---

**This is the final demo version prepared for academic submission.**

