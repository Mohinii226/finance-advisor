"""Pydantic request models (FastAPI). Validation limits mirror api/handlers.py and core/auth.py."""
from typing import Optional

from pydantic import BaseModel, ConfigDict, Field


class RegisterRequest(BaseModel):
    email: str = Field(min_length=3, max_length=254)
    password: str = Field(min_length=1, max_length=128)      # strength rules (min 10) are enforced in core/auth.py
    name: str = Field(min_length=1, max_length=80)
    monthly_income: float = Field(default=0, ge=0, le=10_000_000, description="Rs per month (optional)")
    accept_privacy_notice: bool = Field(description="Must be true: the user has read and accepted the privacy notice")
    confirm_adult: bool = Field(description="Must be true: the user confirms they are 18 or older")


class LoginRequest(BaseModel):
    email: str = Field(min_length=1, max_length=254)
    password: str = Field(min_length=1, max_length=128)


class ChatRequest(BaseModel):
    # No customer_id on purpose: the customer is whoever the bearer token says. Clients that still send one get a 422
    # instead of silently being ignored, so nobody believes they can pick another customer.
    model_config = ConfigDict(extra="forbid")
    message: str = Field(min_length=1, max_length=1000)
    session_id: Optional[str] = Field(default=None, max_length=64)


class SavingsRequest(BaseModel):
    target_amount: float = Field(gt=0, le=10_000_000, description="Rupees to save per month")


class EmailRequest(BaseModel):
    email: str = Field(min_length=1, max_length=254)


class VerifyEmailRequest(BaseModel):
    token: str = Field(min_length=10, max_length=200)


class ResetPasswordRequest(BaseModel):
    token: str = Field(min_length=10, max_length=200)
    new_password: str = Field(min_length=1, max_length=128)


class DeleteAccountRequest(BaseModel):
    password: str = Field(min_length=1, max_length=128)
