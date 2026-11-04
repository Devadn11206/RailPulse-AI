"""Authentication and User Data Models for RailPulse AI RBAC."""

from __future__ import annotations

from typing import List, Optional
from pydantic import BaseModel, Field


class User(BaseModel):
    """Internal user model representation."""

    username: str
    email: str
    role: str = Field(..., description="Role: 'admin', 'analyst', or 'developer'")
    disabled: bool = False
    hashed_password: str


class UserInDB(User):
    """User stored in memory/database with hashed credentials."""
    pass


class Token(BaseModel):
    """JWT Token response schema."""

    access_token: str
    token_type: str = "bearer"
    role: str
    expires_in_minutes: int


class TokenData(BaseModel):
    """Payload decoded from JWT token."""

    username: Optional[str] = None
    role: Optional[str] = None


class LoginRequest(BaseModel):
    """User credentials request."""

    username: str
    password: str


class UserResponse(BaseModel):
    """Public user profile response."""

    username: str
    email: str
    role: str
    disabled: bool = False

