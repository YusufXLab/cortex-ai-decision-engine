"""Context domain models and enumeration for ContextSelectionDecider."""

from enum import Enum
from pydantic import BaseModel, Field


class ContextSourceType(str, Enum):
    """The 8 approved context sources eligible for injection into the model prompt."""

    NONE = "NONE"
    CHAT_HISTORY = "CHAT_HISTORY"
    USER_MEMORY = "USER_MEMORY"
    UPLOADED_FILES = "UPLOADED_FILES"
    RAG_KNOWLEDGE_BASE = "RAG_KNOWLEDGE_BASE"
    WEB_RESULTS = "WEB_RESULTS"
    DATABASE_RESULTS = "DATABASE_RESULTS"
    SYSTEM_PROMPT_ONLY = "SYSTEM_PROMPT_ONLY"


class ContextDecisionResult(BaseModel):
    """Output structure produced by the ContextSelectionDecider module."""

    selected_context_sources: list[ContextSourceType] = Field(
        ...,
        description="List of one or more approved context sources to retrieve and inject.",
    )
    confidence: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Confidence score [0.0 - 1.0] from Strands Decider 2B.",
    )
    reasoning: str = Field(
        ...,
        description="Reasoning trace explaining why these specific context sources are required.",
    )
    latency_ms: float = Field(
        ...,
        ge=0.0,
        description="Inference latency of the Strands Decider 2B call in milliseconds.",
    )
