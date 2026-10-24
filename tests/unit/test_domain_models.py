"""Unit tests for pure domain models."""

import pytest
from pydantic import ValidationError
from cortex.domain.models.decision import DecisionInput, DecisionOutcome, DecisionType
from cortex.domain.models.generation import GenerationInput, GenerationOutcome


def test_decision_outcome_creation() -> None:
    """Validate DecisionOutcome instantiation and constraints."""
    outcome = DecisionOutcome(
        module_id="intent_decider_01",
        decision_type=DecisionType.INTENT,
        action="ROUTE_TO_BEDROCK_CLAUDE",
        confidence=0.92,
        reasoning="Query requires analytical coding response.",
    )
    assert outcome.module_id == "intent_decider_01"
    assert outcome.decision_type == DecisionType.INTENT
    assert outcome.confidence == 0.92


def test_decision_outcome_confidence_bounds() -> None:
    """Validate that confidence must stay between 0.0 and 1.0."""
    with pytest.raises(ValidationError):
        DecisionOutcome(
            module_id="guardrail_01",
            decision_type=DecisionType.GUARDRAIL,
            action="ALLOW",
            confidence=1.5,  # Exceeds 1.0
        )


def test_generation_models() -> None:
    """Validate GenerationInput and GenerationOutcome entities."""
    req = GenerationInput(
        prompt="Explain quantum entanglement",
        model_id="anthropic.claude-3-5-sonnet-20241022-v2:0",
    )
    assert req.temperature == 0.7

    res = GenerationOutcome(
        model_id=req.model_id,
        content="Quantum entanglement is a physical phenomenon...",
        latency_ms=142.5,
    )
    assert res.latency_ms == 142.5
