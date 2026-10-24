"""Agent domain models and metadata for specialized agent routing."""

from typing import Any
from pydantic import BaseModel, Field
from cortex.domain.models.intent import IntentCategory


class AgentProfile(BaseModel):
    """Specification of a specialized internal agent."""

    name: str = Field(
        ...,
        description="Unique name of the agent (e.g. 'SoftwareEngineerAgent').",
    )
    intents: list[IntentCategory] = Field(
        ...,
        description="List of IntentCategory values for which this agent is eligible.",
    )
    description: str = Field(
        ...,
        description="Detailed description of the agent's specialization and domain expertise.",
    )
    capabilities: list[str] = Field(
        default_factory=list,
        description="Specific capabilities, tools, or skills this agent possesses.",
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Additional configuration or routing metadata.",
    )


class AgentDecisionResult(BaseModel):
    """Result emitted by the AgentSelectionDecider."""

    selected_agent: str = Field(
        ...,
        description="The specialized agent selected by Strands Decider 2B.",
    )
    intent_context: str = Field(
        ...,
        description="The preceding intent that constrained candidate agents.",
    )
    candidate_agents: list[str] = Field(
        ...,
        description="List of candidate agents evaluated by Strands Decider 2B.",
    )
    confidence: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Confidence score [0.0 - 1.0] from Strands Decider 2B.",
    )
    reasoning: str = Field(
        ...,
        description="Reasoning trace explaining why this specialized agent was selected.",
    )
    latency_ms: float = Field(
        ...,
        ge=0.0,
        description="Inference latency in milliseconds.",
    )
