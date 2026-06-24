"""Pydantic models for authentication — request/response schemas."""
from datetime import datetime

from pydantic import BaseModel, EmailStr, Field


class UserCreate(BaseModel):
    """Request body for registering a new user with email/password."""

    email: EmailStr = Field(..., description="User email address")
    password: str = Field(..., min_length=8, max_length=128, description="Password (min 8 characters)")
    full_name: str | None = Field(None, description="Optional display name")


class UserLogin(BaseModel):
    """Request body for email/password login."""

    email: EmailStr = Field(..., description="User email address")
    password: str = Field(..., max_length=128, description="Password")


class GoogleAuthRequest(BaseModel):
    """Request body for Google OAuth sign-in."""

    credential: str = Field(..., description="Google ID token from the Sign-In button")


class TokenResponse(BaseModel):
    """JWT token response returned after successful authentication."""

    access_token: str = Field(..., description="JWT bearer token")
    token_type: str = Field(default="bearer")
    user: "UserPublic"


class OTPVerifyRequest(BaseModel):
    """Request body for verifying an OTP code."""
    email: EmailStr = Field(..., description="User email address")
    otp_code: str = Field(..., description="6-digit OTP code")


class OTPResendRequest(BaseModel):
    """Request body for resending an OTP code."""
    email: EmailStr = Field(..., description="User email address")


class ForgotPasswordRequest(BaseModel):
    """Request body for forgot password."""
    email: EmailStr = Field(..., description="User email address")


class ResetPasswordRequest(BaseModel):
    """Request body for resetting password with OTP."""
    email: EmailStr = Field(..., description="User email address")
    otp_code: str = Field(..., description="6-digit OTP code")
    new_password: str = Field(..., min_length=8, max_length=128, description="New password (min 8 characters)")


class UserPublic(BaseModel):
    """Public user representation — safe to expose in API responses."""

    id: str
    email: str
    full_name: str | None
    auth_provider: str
    is_verified: bool
    created_at: datetime

    model_config = {"from_attributes": True}


class TokenData(BaseModel):
    """Decoded JWT payload contents."""

    user_id: str
    email: str
    session_id: str | None = None

