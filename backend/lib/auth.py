"""
Authentication utilities for JWT validation
"""
import os
from typing import Optional
from fastapi import HTTPException, Header
from jose import JWTError, jwt
from dotenv import load_dotenv
from .supabase_client import get_supabase_client

load_dotenv()
load_dotenv('../.env')

# Get JWT secret from Supabase (this is the same secret Supabase uses)
JWT_SECRET = os.getenv("SUPABASE_JWT_SECRET")
JWT_ALGORITHM = "HS256"

async def get_current_user(authorization: Optional[str] = Header(None)):
    """
    Validate JWT token and return user info

    Args:
        authorization: Bearer token from Authorization header

    Returns:
        dict: User information including id, email, role

    Raises:
        HTTPException: If token is invalid or user not found
    """
    if not authorization:
        raise HTTPException(status_code=401, detail="Authorization header required")

    if not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Invalid authorization header format")

    token = authorization.replace("Bearer ", "")

    try:
        # Validate token with Supabase
        supabase = get_supabase_client()

        # Get user from Supabase using the token
        user_response = supabase.auth.get_user(token)

        if not user_response or not user_response.user:
            raise HTTPException(status_code=401, detail="Invalid or expired token")

        user = user_response.user

        # Get profile with role
        profile_response = supabase.table('profiles').select('*').eq('id', user.id).single().execute()

        if not profile_response.data:
            raise HTTPException(status_code=404, detail="User profile not found")

        profile = profile_response.data

        return {
            "id": user.id,
            "email": user.email,
            "role": profile.get("role", "student"),
            "full_name": profile.get("full_name"),
            "profile": profile
        }

    except HTTPException:
        raise
    except Exception as e:
        print(f"Auth error: {e}")
        raise HTTPException(status_code=401, detail="Could not validate credentials")


def require_teacher(user: dict):
    """
    Check if user has teacher role

    Args:
        user: User dict from get_current_user

    Raises:
        HTTPException: If user is not a teacher
    """
    if user.get("role") != "teacher":
        raise HTTPException(status_code=403, detail="Teacher access required")
