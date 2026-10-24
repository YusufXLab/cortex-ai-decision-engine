"""Privacy domain models and output representations for PrivacyDecider."""

from pydantic import BaseModel, Field
from cortex.domain.models.state import PrivacyLevel


class PrivacyDecisionResult(BaseModel):
    """Output structure produced by the PrivacyDecider module."""

    privacy_level: PrivacyLevel = Field(
        ...,
        description="The strictly evaluated privacy classification from Strands Decider 2B.",
    )
    confidence: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Confidence score [0.0 - 1.0] from Strands Decider 2B.",
    )
    reasoning: str = Field(
        ...,
        description="Reasoning trace explaining why this privacy classification was assigned.",
    )
    latency_ms: float = Field(
        ...,
        ge=0.0,
        description="Inference latency of the Strands Decider 2B call in milliseconds.",
    )
