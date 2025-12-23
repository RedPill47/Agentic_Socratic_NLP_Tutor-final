# FastAPI Backend Setup

## Installation

### Option 1: Use Existing Virtual Environment (Recommended)

If you already have a virtual environment with CrewAI installed:

```bash
# Activate your existing venv (the one that works with crewai run)
# Then install FastAPI dependencies
cd backend
pip install -r requirements.txt
```

### Option 2: Create New Virtual Environment

```bash
cd backend

# Create virtual environment
python -m venv venv

# Activate it
# Windows:
venv\Scripts\activate
# Linux/Mac:
source venv/bin/activate

# Install all dependencies
pip install -r requirements.txt
```

## Environment Variables

Create a `.env` file in the `backend/` directory:

```env
OPENAI_API_KEY=your_openai_api_key_here
```

Or use the `.env` file from the project root.

## Running the Backend

```bash
# Make sure you're in the backend directory
cd backend

# Activate virtual environment (if using one)
# Windows:
venv\Scripts\activate
# Linux/Mac:
source venv/bin/activate

# Run the server (auto-reload enabled by default)
python main.py

# Or with uvicorn directly (same as above)
uvicorn main:app --reload --port 8000
```

The backend will run on `http://localhost:8000`

## Troubleshooting

### Error: "No module named 'crewai'"

**Solution:** Make sure CrewAI is installed in your Python environment:

```bash
pip install crewai[tools]>=0.100.0
```

Or use the same virtual environment where `crewai run` works.

### Error: "No module named 'agentic_socratic_nlp_tutor'"

**Solution:** The backend needs to find the package. Make sure:
1. You're running from the `backend/` directory
2. The `agentic_socratic_nlp_tutor/` directory exists at the project root
3. The path setup in `main.py` is correct

### Error: "Flow system not available"

This is expected if CrewAI Flows aren't available (Windows compatibility issue).
The backend will still work with the fallback mode.

## Testing

Test the API:

```bash
# Health check
curl http://localhost:8000/

# Chat endpoint
curl -X POST http://localhost:8000/api/chat \
  -H "Content-Type: application/json" \
  -d '{"content": "Hello", "student_id": "test_student"}'
```

