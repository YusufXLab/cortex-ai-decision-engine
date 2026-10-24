"""Tool domain models and enumeration for ToolSelectionDecider."""

from enum import Enum
from pydantic import BaseModel, Field


class ToolType(str, Enum):
    """The 9 approved tools and capabilities supported by the Cortex AI Decision Engine."""

    NONE = "NONE"
    WEB_SEARCH = "WEB_SEARCH"
    CALCULATOR = "CALCULATOR"
    FILE_READER = "FILE_READER"
    VISION = "VISION"
    DATABASE = "DATABASE"
    CODE_EXECUTION = "CODE_EXECUTION"
    MEMORY = "MEMORY"
    RAG = "RAG"


class ToolDecisionResult(BaseModel):
    """Output structure produced by the ToolSelectionDecider module."""

    selected_tools: list[ToolType] = Field(
        ...,
        description="List of one or more approved tools required to fulfill the request.",
    )
    confidence: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Confidence score [0.0 - 1.0] from Strands Decider 2B.",
    )
    reasoning: str = Field(
        ...,
        description="Reasoning trace explaining why these specific capabilities are required.",
    )
    latency_ms: float = Field(
        ...,
        ge=0.0,
        description="Inference latency of the Strands Decider 2B call in milliseconds.",
    )
