"""Demo endpoint DTOs (scaffolded for upcoming logic)."""

from typing import Any
from pydantic import BaseModel, Field
from cortex.domain.models.decision import DecisionOutcome
from cortex.domain.models.generation import GenerationOutcome


class DemoPipelineRequest(BaseModel):
    """Demo input payload for the decision -> generation pipeline."""

    query: str = Field(..., min_length=1, max_length=4000, description="User query to process.")
    context: dict[str, Any] = Field(default_factory=dict, description="Execution context.")
    enforce_guardrail: bool = Field(default=True, description="Whether to apply guardrail decider check.")


class DemoPipelineResponse(BaseModel):
    """Demo response returning decisions made and subsequent Bedrock generation."""

    decisions: list[DecisionOutcome] = Field(default_factory=list, description="Decisions from Strands Decider 2B.")
    generation: GenerationOutcome | None = Field(default=None, description="Response from Amazon Bedrock.")
    execution_notes: str = Field(default="Foundation scaffold ready for AI logic implementation.")
