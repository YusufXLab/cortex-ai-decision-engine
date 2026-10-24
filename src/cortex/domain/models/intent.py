"""Intent domain models and enumeration for IntentDecider."""

from enum import Enum
from pydantic import BaseModel, Field


class IntentCategory(str, Enum):
    """The 7 strictly allowed intent categories recognized by Strands Decider 2B."""

    CHAT = "CHAT"
    WRITING = "WRITING"
    CODING = "CODING"
    RESEARCH = "RESEARCH"
    ANALYSIS = "ANALYSIS"
    PLANNING = "PLANNING"
    VISION = "VISION"


class IntentDecisionResult(BaseModel):
    """Output structure produced by the IntentDecider module."""

    selected_intent: IntentCategory = Field(
        ...,
        description="The single best intent category determined by Strands Decider 2B.",
    )
    confidence: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Confidence score [0.0 - 1.0] from Strands Decider 2B.",
    )
    reasoning: str = Field(
        ...,
        description="Reasoning trace justifying why this intent was selected.",
    )
    latency_ms: float = Field(
        ...,
        ge=0.0,
        description="Inference latency of the Strands Decider 2B call in milliseconds.",
    )
