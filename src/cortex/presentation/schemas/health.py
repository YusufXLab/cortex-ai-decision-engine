"""Health check response models."""

from typing import Literal
from pydantic import BaseModel, Field


class ComponentStatus(BaseModel):
    """Status of an individual upstream or subsystem."""

    name: str = Field(..., description="Name of the component.")
    status: Literal["healthy", "degraded", "unhealthy", "mocked"] = Field(...)
    details: str | None = Field(default=None, description="Optional diagnostic notes.")


class HealthResponse(BaseModel):
    """Aggregate service health status."""

    status: Literal["healthy", "degraded", "unhealthy"] = Field(...)
    version: str = Field(..., description="Application version.")
    environment: str = Field(..., description="Runtime environment.")
    components: list[ComponentStatus] = Field(default_factory=list)
