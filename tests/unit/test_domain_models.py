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


def test_vision_domain_models() -> None:
    """Validate VisionRequirementType enum and VisionDecisionResult model."""
    from cortex.domain.models.vision import VisionDecisionResult, VisionRequirementType

    assert len(VisionRequirementType) == 6
    assert set(v.value for v in VisionRequirementType) == {
        "NONE",
        "OCR",
        "IMAGE_UNDERSTANDING",
        "DIAGRAM_ANALYSIS",
        "UI_ANALYSIS",
        "MULTIMODAL",
    }

    result = VisionDecisionResult(
        vision_requirement=VisionRequirementType.DIAGRAM_ANALYSIS,
        confidence=0.96,
        reasoning="Architecture diagram requires topological analysis.",
        latency_ms=15.4,
    )
    assert result.vision_requirement == VisionRequirementType.DIAGRAM_ANALYSIS
    assert result.confidence == 0.96
    assert result.latency_ms == 15.4

    with pytest.raises(ValidationError):
        VisionDecisionResult(
            vision_requirement="INVALID_TYPE",  # type: ignore[arg-type]
            confidence=0.9,
            reasoning="Invalid",
            latency_ms=10.0,
        )

    with pytest.raises(ValidationError):
        VisionDecisionResult(
            vision_requirement=VisionRequirementType.OCR,
            confidence=1.2,  # > 1.0
            reasoning="Too high",
            latency_ms=10.0,
        )


def test_complexity_domain_models() -> None:
    """Validate ComplexityLevel enum and ComplexityDecisionResult model."""
    from cortex.domain.models.complexity import ComplexityDecisionResult, ComplexityLevel

    assert set(ComplexityLevel) >= {
        ComplexityLevel.TRIVIAL,
        ComplexityLevel.SIMPLE,
        ComplexityLevel.COMPLEX,
        ComplexityLevel.EXPERT,
    }
    assert ComplexityLevel("TRIVIAL") == ComplexityLevel.TRIVIAL
    assert ComplexityLevel("simple") == ComplexityLevel.SIMPLE

    result = ComplexityDecisionResult(
        complexity_level=ComplexityLevel.COMPLEX,
        confidence=0.97,
        reasoning="Multi-tool query requiring data reconciliation.",
        latency_ms=18.2,
    )
    assert result.complexity_level == ComplexityLevel.COMPLEX
    assert result.confidence == 0.97
    assert result.latency_ms == 18.2

    with pytest.raises(ValidationError):
        ComplexityDecisionResult(
            complexity_level="IMPOSSIBLE",  # type: ignore[arg-type]
            confidence=0.9,
            reasoning="Invalid level",
            latency_ms=12.0,
        )

    with pytest.raises(ValidationError):
        ComplexityDecisionResult(
            complexity_level=ComplexityLevel.SIMPLE,
            confidence=-0.1,  # < 0.0
            reasoning="Invalid",
            latency_ms=12.0,
        )


