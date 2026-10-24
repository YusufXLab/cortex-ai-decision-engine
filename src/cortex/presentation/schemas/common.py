"""Common API response and error envelopes."""

from datetime import datetime, timezone
from typing import Any, Generic, TypeVar
from pydantic import BaseModel, Field

DataT = TypeVar("DataT")


class APIResponse(BaseModel, Generic[DataT]):
    """Standard unified response wrapper for all API endpoints."""

    success: bool = Field(default=True, description="Indicates whether the request succeeded.")
    data: DataT = Field(..., description="Payload data returned by the endpoint.")
    correlation_id: str | None = Field(default=None, description="Unique trace identifier for debugging.")
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class ErrorDetail(BaseModel):
    """Detailed error description."""

    code: str = Field(..., description="Machine-readable error classification.")
    message: str = Field(..., description="Human-readable error explanation.")
    details: dict[str, Any] = Field(default_factory=dict, description="Contextual debugging information.")


class APIErrorResponse(BaseModel):
    """Standard unified error response structure."""

    success: bool = Field(default=False)
    error: ErrorDetail
    correlation_id: str | None = None
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
