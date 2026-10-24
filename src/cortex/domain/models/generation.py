"""Domain models representing generation requests sent to Amazon Bedrock.

Strict Architecture Boundary:
Amazon Bedrock ONLY generates output responses based on instructions/context
provided by the Strands Decider 2B decision plane.
"""

from datetime import datetime, timezone
from typing import Any
from pydantic import BaseModel, Field


class GenerationInput(BaseModel):
    """Payload sent to Amazon Bedrock for text/multimodal generation."""

    prompt: str = Field(..., description="Prompt prepared after decider evaluation.")
    model_id: str = Field(..., description="Bedrock foundation model identifier.")
    temperature: float = Field(default=0.7, ge=0.0, le=1.0)
    max_tokens: int = Field(default=2048, ge=1)
    system_prompt: str | None = Field(default=None, description="System instructions.")
    metadata: dict[str, Any] = Field(default_factory=dict, description="Request tracking metadata.")


class GenerationOutcome(BaseModel):
    """Response returned by Amazon Bedrock."""

    model_id: str = Field(..., description="Foundation model that generated the response.")
    content: str = Field(..., description="Generated text content.")
    prompt_tokens: int = Field(default=0, ge=0)
    completion_tokens: int = Field(default=0, ge=0)
    total_tokens: int = Field(default=0, ge=0)
    latency_ms: float = Field(..., ge=0.0, description="Generation latency in milliseconds.")
    metadata: dict[str, Any] = Field(default_factory=dict)
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
