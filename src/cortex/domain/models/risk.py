"""Risk domain models and output representations for RiskDecider."""

from pydantic import BaseModel, Field
from cortex.domain.models.state import RiskLevel


class RiskDecisionResult(BaseModel):
    """Output structure produced by the RiskDecider module."""

    risk_level: RiskLevel = Field(
        ...,
        description="The operational and safety risk classification from Strands Decider 2B.",
    )
    confidence: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Confidence score [0.0 - 1.0] from Strands Decider 2B.",
    )
    reasoning: str = Field(
        ...,
        description="Reasoning trace explaining why this risk level was assigned.",
    )
    latency_ms: float = Field(
        ...,
        ge=0.0,
        description="Inference latency of the Strands Decider 2B call in milliseconds.",
    )
    mitigation_action: str | None = Field(
        default=None,
        description="Recommended operational mitigation (e.g. 'REQUIRE_STEP_VERIFICATION', 'HALT').",
    )
