# Backend Setup - Quick Fix

## The Problem

You're getting `No module named 'crewai'` because:
1. You're using Python 3.14, but CrewAI requires Python 3.10-3.13
2. The backend needs to use the **same Python environment** where `crewai run` works

## Solution: Use Your Existing Environment

Since `crewai run` works, you already have the right environment set up. Just use it for the backend too!

### Step-by-Step:

1. **Find where `crewai run` works:**
   ```bash
   # This should work:
   cd agentic_socratic_nlp_tutor
   crewai run
   ```

2. **Activate that same environment:**
   ```bash
   # If using UV (CrewAI's package manager):
   cd agentic_socratic_nlp_tutor
   # Environment should already be active
   
   # If using venv:
   source .venv/bin/activate  # Linux/Mac
   # or
   .venv\Scripts\activate  # Windows
   ```

3. **Install FastAPI in that environment:**
   ```bash
   pip install fastapi uvicorn[standard] pydantic python-dotenv
   ```

4. **Run the backend:**
   ```bash
   cd ../backend
   python main.py
   ```

## Why This Works

- ✅ You already have CrewAI installed in that environment
- ✅ You already have Python 3.10-3.13 in that environment
- ✅ You just need to add FastAPI (which works with any Python 3.7+)

## Alternative: Check Your Python Version

If you want to use a different environment, make sure it has Python 3.10-3.13:

```bash
python --version  # Should show 3.10, 3.11, 3.12, or 3.13
```

If it shows 3.14+, you need to use a different Python version or the existing environment.

## Quick Test

After setup, test that it works:

```bash
# In the environment where crewai run works:
python -c "import crewai; print('CrewAI OK')"
python -c "import fastapi; print('FastAPI OK')"

# Both should work without errors
```

