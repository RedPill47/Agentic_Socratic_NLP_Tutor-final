"""Backend utilities"""
from .supabase_client import get_supabase_client
from .auth import get_current_user, require_teacher

__all__ = ["get_supabase_client", "get_current_user", "require_teacher"]
