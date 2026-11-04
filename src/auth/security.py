"""Security, JWT Token Management, and Role-Based Access Control (RBAC).

Implements PRD Module 08:
- Direct bcrypt password hashing and verification
- JWT creation and decoding via PyJWT
- Role-based authorization dependencies for FastAPI endpoints
- Developer API key validation via X-API-Key header
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from typing import Callable, Dict, List, Optional

import bcrypt
import jwt
from fastapi import Depends, Header, HTTPException, Security, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from src.auth.models import TokenData, User, UserInDB

logger = logging.getLogger(__name__)

# Security configuration
SECRET_KEY = "railpulse-ai-secure-jwt-secret-key-production-grade"
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 60

# Developer API Keys registry
REGISTERED_API_KEYS: Dict[str, str] = {
    "railpulse-dev-key-9f8a3c2b1e": "developer",
    "railpulse-admin-key-7d4e1a0b": "admin",
}

security_bearer = HTTPBearer(auto_error=False)


def hash_password(password: str) -> str:
    """Hash a plaintext password using bcrypt."""
    salt = bcrypt.gensalt(rounds=12)
    hashed = bcrypt.hashpw(password.encode("utf-8"), salt)
    return hashed.decode("utf-8")


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verify password against bcrypt hash."""
    try:
        return bcrypt.checkpw(plain_password.encode("utf-8"), hashed_password.encode("utf-8"))
    except Exception as exc:
        logger.warning("Bcrypt check failed: %s", exc)
        return False


# In-memory mock database of system users for RBAC
# Passwords:
# admin / admin123
# analyst / analyst123
# developer / dev123
USERS_DB: Dict[str, UserInDB] = {
    "admin": UserInDB(
        username="admin",
        email="admin@railpulse.ai",
        role="admin",
        disabled=False,
        hashed_password=hash_password("admin123"),
    ),
    "analyst": UserInDB(
        username="analyst",
        email="analyst@railpulse.ai",
        role="analyst",
        disabled=False,
        hashed_password=hash_password("analyst123"),
    ),
    "developer": UserInDB(
        username="developer",
        email="dev@railpulse.ai",
        role="developer",
        disabled=False,
        hashed_password=hash_password("dev123"),
    ),
}


def authenticate_user(username: str, password: str) -> Optional[UserInDB]:
    """Validate username and password against users database."""
    user = USERS_DB.get(username)
    if not user:
        return None
    if not verify_password(password, user.hashed_password):
        return None
    return user


def create_access_token(data: dict, expires_delta: Optional[timedelta] = None) -> str:
    """Create a signed JWT access token."""
    to_encode = data.copy()
    expire = datetime.now(timezone.utc) + (expires_delta or timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES))
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt


def decode_access_token(token: str) -> TokenData:
    """Decode and validate a JWT access token."""
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        role: str = payload.get("role")
        if username is None:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Token payload missing subject identifier",
                headers={"WWW-Authenticate": "Bearer"},
            )
        return TokenData(username=username, role=role)
    except jwt.PyJWTError as exc:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=f"Invalid or expired token: {str(exc)}",
            headers={"WWW-Authenticate": "Bearer"},
        ) from exc


async def get_current_user(
    credentials: Optional[HTTPAuthorizationCredentials] = Security(security_bearer),
) -> UserInDB:
    """FastAPI dependency to extract and verify the current authenticated user."""
    if credentials is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing Authorization Bearer token header",
            headers={"WWW-Authenticate": "Bearer"},
        )

    token_data = decode_access_token(credentials.credentials)
    user = USERS_DB.get(token_data.username)
    if user is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="User not found in system records",
            headers={"WWW-Authenticate": "Bearer"},
        )
    if user.disabled:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Inactive user account")
    return user


def require_role(allowed_roles: List[str]) -> Callable:
    """Factory creating an authorization dependency enforcing allowed roles."""

    async def role_checker(current_user: UserInDB = Depends(get_current_user)) -> UserInDB:
        if current_user.role not in allowed_roles:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=(
                    f"Role '{current_user.role}' is not authorized. "
                    f"Required roles: {allowed_roles}"
                ),
            )
        return current_user

    return role_checker


async def verify_api_key(x_api_key: Optional[str] = Header(None)) -> str:
    """Validate X-API-Key header for external developer integrations."""
    if not x_api_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing X-API-Key header",
        )
    role = REGISTERED_API_KEYS.get(x_api_key)
    if not role:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Invalid or revoked API key",
        )
    return role

