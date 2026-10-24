"""Complexity domain models and output representations for ComplexityDecider."""

from enum import Enum
from typing import Any
from pydantic import BaseModel, Field


class ComplexityLevel(str, Enum):
    """The 4 approved task complexity classifications evaluated by Strands Decider 2B."""

    TRIVIAL = "TRIVIAL"
    SIMPLE = "SIMPLE"
    COMPLEX = "COMPLEX"
    EXPERT = "EXPERT"
    MODERATE = "MODERATE"  # Backward compatibility alias for legacy tests

    @classmethod
    def _missing_(cls, value: object) -> Any:
        """Case-insensitive and legacy value fallback handler."""
        if isinstance(value, str):
            val_upper = value.upper()
            for member in cls:
                if member.value == val_upper or member.name == val_upper:
                    return member
        return None


class ComplexityDecisionResult(BaseModel):
    """Output structure produced by the ComplexityDecider module."""

    complexity_level: ComplexityLevel = Field(
        ...,
        description="The strictly evaluated complexity classification from Strands Decider 2B.",
    )
    confidence: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Confidence score [0.0 - 1.0] from Strands Decider 2B.",
    )
    reasoning: str = Field(
        ...,
        description="Reasoning trace explaining why this complexity level was assigned.",
    )
    latency_ms: float = Field(
        ...,
        ge=0.0,
        description="Inference latency of the Strands Decider 2B call in milliseconds.",
    )
