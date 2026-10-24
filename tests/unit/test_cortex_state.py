"""Unit tests for the CortexState model and state transitions."""

import pytest
from pydantic import ValidationError
from cortex.domain.models.state import (
    ComplexityLevel,
    CortexState,
    CostStrategy,
    PipelineStatus,
    PrivacyLevel,
    RiskLevel,
    TraceStep,
)


def test_cortex_state_initialization() -> None:
    """Test standard initialization of CortexState."""
    state = CortexState(user_input="Analyze financial transaction fraud")

    assert state.user_input == "Analyze financial transaction fraud"
    assert state.status == PipelineStatus.INITIALIZED
    assert state.intent is None
    assert state.selected_agent is None
    assert state.privacy_level is None
    assert state.risk_level is None
    assert state.selected_tools == []
    assert state.selected_context == []
    assert state.vision_requirement is None
    assert state.vision_required is False
    assert state.complexity_level is None
    assert state.complexity is None
    assert state.selected_model is None
    assert state.cost_strategy is None
    assert state.confidence is None
    assert state.execution_trace == []
    assert state.generation_output is None
    assert state.completed_at is None


def test_cortex_state_validation() -> None:
    """Test that empty user_input or invalid confidence raises validation error."""
    with pytest.raises(ValidationError):
        CortexState(user_input="")  # min_length is 1

    with pytest.raises(ValidationError):
        CortexState(user_input="valid input", confidence=1.5)  # > 1.0


def test_trace_step_addition_and_latency() -> None:
    """Test appending trace steps and calculating aggregate latency."""
    state = CortexState(user_input="Optimize database index")

    step1 = TraceStep(
        step_number=1,
        module_id="decider_01_intent",
        decision_name="Intent Decider",
        action="DATABASE_OPTIMIZATION",
        confidence=0.95,
        duration_ms=12.4,
    )
    step2 = TraceStep(
        step_number=2,
        module_id="decider_02_agent",
        decision_name="Agent Selection Decider",
        action="dba_specialist_agent",
        confidence=0.90,
        duration_ms=8.6,
    )

    state.add_trace_step(step1)
    state.add_trace_step(step2)

    assert len(state.execution_trace) == 2
    assert state.total_decision_latency_ms == pytest.approx(21.0, 0.01)


def test_state_lifecycle_transitions() -> None:
    """Test mark_failed, mark_blocked, and mark_completed transitions."""
    # Failed
    state_failed = CortexState(user_input="test input")
    state_failed.mark_failed("Module crash")
    assert state_failed.status == PipelineStatus.FAILED
    assert state_failed.completed_at is not None
    assert state_failed.metadata["failure_reason"] == "Module crash"

    # Blocked
    state_blocked = CortexState(user_input="malicious input")
    state_blocked.mark_blocked("Risk level high")
    assert state_blocked.status == PipelineStatus.BLOCKED
    assert state_blocked.completed_at is not None
    assert state_blocked.metadata["blocked_reason"] == "Risk level high"

    # Completed
    state_completed = CortexState(user_input="normal input")
    state_completed.mark_completed("Final generation text")
    assert state_completed.status == PipelineStatus.COMPLETED
    assert state_completed.generation_output == "Final generation text"
    assert state_completed.completed_at is not None
