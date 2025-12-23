"""
Supabase client for backend operations
"""
import os
from supabase import create_client, Client
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
load_dotenv('../.env')  # Also try parent directory

_supabase_client: Client = None

def get_supabase_client() -> Client:
    """Get or create Supabase client singleton"""
    global _supabase_client

    if _supabase_client is None:
        url = os.getenv("SUPABASE_URL")
        # Use service role key in backend for admin operations
        key = os.getenv("SUPABASE_SERVICE_KEY")

        if not url or not key:
            raise ValueError("SUPABASE_URL and SUPABASE_SERVICE_KEY must be set in environment")

        _supabase_client = create_client(url, key)

    return _supabase_client
