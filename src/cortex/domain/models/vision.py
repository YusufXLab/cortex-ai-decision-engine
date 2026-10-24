"""Vision domain models and enumeration for VisionRequirementDecider."""

from enum import Enum
from pydantic import BaseModel, Field


class VisionRequirementType(str, Enum):
    """The 6 approved visual processing classifications evaluated by Strands Decider 2B."""

    NONE = "NONE"
    OCR = "OCR"
    IMAGE_UNDERSTANDING = "IMAGE_UNDERSTANDING"
    DIAGRAM_ANALYSIS = "DIAGRAM_ANALYSIS"
    UI_ANALYSIS = "UI_ANALYSIS"
    MULTIMODAL = "MULTIMODAL"


class VisionDecisionResult(BaseModel):
    """Output structure produced by the VisionRequirementDecider module."""

    vision_requirement: VisionRequirementType = Field(
        ...,
        description="The specific visual understanding category required by the request.",
    )
    confidence: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Confidence score [0.0 - 1.0] from Strands Decider 2B.",
    )
    reasoning: str = Field(
        ...,
        description="Reasoning trace explaining why this visual processing type was selected.",
    )
    latency_ms: float = Field(
        ...,
        ge=0.0,
        description="Inference latency of the Strands Decider 2B call in milliseconds.",
    )
