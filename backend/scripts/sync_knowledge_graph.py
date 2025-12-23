"""
Backend wrapper for knowledge graph sync.

This script runs in the backend environment and properly sets up paths
and environment variables before calling the main extraction script.

This avoids .env and venv conflicts by running in the backend's context.
"""

import os
import sys
from pathlib import Path

# Set UTF-8 encoding for Windows compatibility
# Only wrap if not already wrapped and not being captured by subprocess
if sys.platform == "win32":
    try:
        import io
        # Check if stdout is a TextIOWrapper (already wrapped) or if it's being captured
        if hasattr(sys.stdout, 'buffer') and not isinstance(sys.stdout, io.TextIOWrapper):
            try:
                # Test if we can write to it
                sys.stdout.write('')
                sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
            except (ValueError, OSError):
                # File is closed or being captured, don't wrap
                pass
        if hasattr(sys.stderr, 'buffer') and not isinstance(sys.stderr, io.TextIOWrapper):
            try:
                # Test if we can write to it
                sys.stderr.write('')
                sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')
            except (ValueError, OSError):
                # File is closed or being captured, don't wrap
                pass
    except Exception:
        pass

# Get project root (backend's parent)
backend_dir = Path(__file__).parent.parent
project_root = backend_dir.parent

# Add project root to Python path
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "agentic_socratic_nlp_tutor" / "src"))

# Set working directory to project root
os.chdir(str(project_root))

# Load environment variables from project root .env
from dotenv import load_dotenv
env_path = project_root / ".env"
if env_path.exists():
    load_dotenv(env_path)
else:
    # Try backend .env as fallback
    backend_env = backend_dir / ".env"
    if backend_env.exists():
        load_dotenv(backend_env)
    else:
        load_dotenv()  # Default behavior

# Now import and run the actual extraction
if __name__ == "__main__":
    # Import the main function from the extraction script
    sys.path.insert(0, str(project_root / "scripts"))
    
    try:
        from extract_topics_from_rag import main
        import asyncio
        import argparse
        
        parser = argparse.ArgumentParser(description="Sync knowledge graph from RAG")
        parser.add_argument(
            "--auto-confirm",
            action="store_true",
            help="Automatically confirm adding topics"
        )
        
        args = parser.parse_args()
        asyncio.run(main(auto_confirm=args.auto_confirm))
    
    except ImportError as e:
        try:
            print(f"ERROR: Failed to import extraction script: {e}", file=sys.stderr)
            print(f"   Project root: {project_root}", file=sys.stderr)
            print(f"   Scripts path: {project_root / 'scripts'}", file=sys.stderr)
        except (ValueError, OSError):
            # Fallback if stdout/stderr are closed
            import traceback
            traceback.print_exception(type(e), e, e.__traceback__)
        sys.exit(1)
    except Exception as e:
        try:
            print(f"ERROR: {e}", file=sys.stderr)
            import traceback
            traceback.print_exc(file=sys.stderr)
        except (ValueError, OSError):
            # Fallback if stdout/stderr are closed
            import traceback
            traceback.print_exception(type(e), e, e.__traceback__)
        sys.exit(1)

