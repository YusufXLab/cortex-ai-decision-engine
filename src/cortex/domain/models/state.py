"""CortexState: The central, strongly typed execution state for the Cortex AI Decision Engine.

Represents the complete lifecycle of a user request as it transitions through
the 11 Strands Decider 2B decision modules and into Amazon Bedrock response generation.
"""

from datetime import datetime, timezone
from enum import Enum
from typing import Any
import uuid
from pydantic import BaseModel, Field


class PipelineStatus(str, Enum):
    """Execution status of the pipeline."""

    INITIALIZED = "initialized"
    DECIDING = "deciding"
    DECIDED = "decided"
    GENERATING = "generating"
    COMPLETED = "completed"
    FAILED = "failed"
    BLOCKED = "blocked"  # Stopped due to risk/policy violations


class PrivacyLevel(str, Enum):
    """Privacy classification determined by the Privacy Decider."""

    PUBLIC = "PUBLIC"
    PERSONAL = "PERSONAL"
    FINANCIAL = "FINANCIAL"
    MEDICAL = "MEDICAL"
    LEGAL = "LEGAL"
    CONFIDENTIAL = "CONFIDENTIAL"


class RiskLevel(str, Enum):
    """Safety and operational risk level determined by RiskDecider."""

    LOW = "LOW"
    MODERATE = "MODERATE"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"


class ComplexityLevel(str, Enum):
    """Task complexity determined by the Complexity Decider."""

    SIMPLE = "simple"
    MODERATE = "moderate"
    COMPLEX = "complex"
    EXPERT = "expert"


class CostStrategy(str, Enum):
    """Cost-performance tradeoff strategy determined by the Cost Decider."""

    MIN_COST = "min_cost"
    BALANCED = "balanced"
    MAX_PERFORMANCE = "max_performance"


class TraceStep(BaseModel):
    """Individual record in the execution trace capturing each decision module's action."""

    step_number: int = Field(..., ge=1, description="Sequential index of the decision step (1-11).")
    module_id: str = Field(..., description="Unique identifier of the decision module.")
    decision_name: str = Field(..., description="Human-readable decision name.")
    action: str = Field(..., description="Action or output produced by the decider.")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence score [0.0 - 1.0].")
    reasoning: str | None = Field(default=None, description="Decider reasoning or rationale.")
    duration_ms: float = Field(default=0.0, ge=0.0, description="Module execution duration in milliseconds.")
    success: bool = Field(default=True, description="Whether the decision module executed successfully.")
    error_message: str | None = Field(default=None, description="Error detail if step failed.")
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class CortexState(BaseModel):
    """Complete strongly typed execution state of a user request.

    Updated sequentially after each of the 11 Strands Decider 2B modules.
    Adheres to the strict boundary: Decider decisions populate this state;
    Bedrock consumes this state to generate the response.
    """

    # Identifiers & Status
    request_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Unique request trace identifier.",
    )
    status: PipelineStatus = Field(
        default=PipelineStatus.INITIALIZED,
        description="Current pipeline lifecycle status.",
    )

    # Core Input
    user_input: str = Field(
        ...,
        min_length=1,
        description="Original user prompt or task specification.",
    )

    # The 11 Decisions (Populated by the 11 Strands Decider 2B modules)
    intent: str | None = Field(
        default=None,
        description="1. Detected user intent / task classification.",
    )
    selected_agent: str | None = Field(
        default=None,
        description="2. Specialized agent chosen to handle the request.",
    )
    privacy_level: PrivacyLevel | None = Field(
        default=None,
        description="3. Data privacy classification.",
    )
    risk_level: RiskLevel | None = Field(
        default=None,
        description="4. Content and safety risk level.",
    )
    selected_tools: list[str] = Field(
        default_factory=list,
        description="5. External tools or APIs required to fulfill the request.",
    )
    selected_context: list[str] = Field(
        default_factory=list,
        description="6. Filtered and pruned context documents/memories.",
    )
    vision_required: bool = Field(
        default=False,
        description="7. Whether multimodal / image analysis is required.",
    )
    complexity: ComplexityLevel | None = Field(
        default=None,
        description="8. Assessed task complexity tier.",
    )
    selected_model: str | None = Field(
        default=None,
        description="9. Specific Bedrock foundation model ID selected based on complexity & cost.",
    )
    cost_strategy: CostStrategy | None = Field(
        default=None,
        description="10. Cost vs. latency/quality optimization profile.",
    )
    confidence: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="11. Overall aggregate confidence score across all decisions.",
    )

    # Telemetry, Tracing & Generation Response
    execution_trace: list[TraceStep] = Field(
        default_factory=list,
        description="Chronological log of all decision steps and their latencies.",
    )
    generation_output: str | None = Field(
        default=None,
        description="Final response generated by Amazon Bedrock (if permitted).",
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Arbitrary execution metadata, headers, and tags.",
    )
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Timestamp when request was received.",
    )
    completed_at: datetime | None = Field(
        default=None,
        description="Timestamp when pipeline completed execution.",
    )

    # State update helper methods
    def add_trace_step(self, step: TraceStep) -> None:
        """Appends an execution step to the chronological trace."""
        self.execution_trace.append(step)

    def mark_failed(self, reason: str) -> None:
        """Transitions state to FAILED and records timestamp."""
        self.status = PipelineStatus.FAILED
        self.completed_at = datetime.now(timezone.utc)
        self.metadata["failure_reason"] = reason

    def mark_blocked(self, reason: str) -> None:
        """Transitions state to BLOCKED (e.g. policy/risk rejection) and records timestamp."""
        self.status = PipelineStatus.BLOCKED
        self.completed_at = datetime.now(timezone.utc)
        self.metadata["blocked_reason"] = reason

    def mark_completed(self, output: str) -> None:
        """Transitions state to COMPLETED with Bedrock response output."""
        self.status = PipelineStatus.COMPLETED
        self.generation_output = output
        self.completed_at = datetime.now(timezone.utc)

    @property
    def total_decision_latency_ms(self) -> float:
        """Computes aggregate latency across all decision steps."""
        return sum(step.duration_ms for step in self.execution_trace)
