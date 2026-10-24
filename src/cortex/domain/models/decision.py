"""Domain models representing decisions produced by Strands Decider 2B.

Strict Architecture Boundary:
Strands Decider 2B ONLY makes decisions (routing, classification, gating, policy checks).
It NEVER generates final responses for the user.
"""

from datetime import datetime, timezone
from enum import Enum
from typing import Any
from pydantic import BaseModel, Field


class DecisionType(str, Enum):
    """Categorization of decisions made by Strands Decider 2B."""

    ROUTING = "routing"              # Route to specific expert agent / model
    GUARDRAIL = "guardrail"          # Safety and content policy gate
    INTENT = "intent"                # Query intent classification
    COMPLEXITY = "complexity"        # Task complexity tier (low, medium, high)
    TOOL_SELECTION = "tool_selection"# Decision on which external tool is required
    CONFIDENCE = "confidence"        # Confidence check
    CACHE = "cache"                  # Cache eligibility check
    PRIORITY = "priority"            # Request priority classification
    CONTEXT_PRUNING = "context"      # Context reduction decision
    VERIFICATION = "verification"    # Verification / sanity check
    CUSTOM = "custom"                # Custom domain decision module


class DecisionInput(BaseModel):
    """Payload passed into a Strands Decider 2B module."""

    query: str = Field(..., description="User prompt or agent state to evaluate.")
    context: dict[str, Any] = Field(default_factory=dict, description="Metadata and execution context.")
    parameters: dict[str, Any] = Field(default_factory=dict, description="Optional inference parameters.")


class DecisionOutcome(BaseModel):
    """Immutable result emitted by a Strands Decider 2B decision module."""

    module_id: str = Field(..., description="Identifier of the decider module (1 of 11).")
    decision_type: DecisionType = Field(..., description="Category of decision made.")
    action: str = Field(..., description="Actionable decision instruction (e.g. 'ROUTE_BEDROCK_CLAUDE', 'REJECT').")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence score [0.0 - 1.0].")
    reasoning: str | None = Field(default=None, description="Decider reasoning trace.")
    metadata: dict[str, Any] = Field(default_factory=dict, description="Additional decider telemetry/flags.")
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
