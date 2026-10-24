"""Unit tests for PrivacyDecider powered by Strands Decider 2B.

Validates:
- All 6 allowed privacy levels: PUBLIC, PERSONAL, FINANCIAL, MEDICAL, LEGAL, CONFIDENTIAL
- Rejection of unknown/invalid classifications
- Malformed confidence handling
- Missing required fields in model response
- API timeout handling without silent fallback
- State mutation (state.privacy_level)
- Execution trace recording (TraceStep step 3)
- Contextual prompt composition (Intent, Selected Agent, Metadata)
"""

from unittest.mock import AsyncMock
import pytest
from cortex.core.exceptions import (
    DeciderInferenceError,
    DeciderTimeoutError,
)
from cortex.domain.models.intent import IntentCategory
from cortex.domain.models.state import CortexState, PrivacyLevel
from cortex.domain.protocols.strands_client import StrandsDeciderClientProtocol
from cortex.infrastructure.deciders.privacy_decider import PrivacyDecider


@pytest.fixture
def mock_strands_client() -> AsyncMock:
    """Mock Strands Decider 2B client."""
    client = AsyncMock(spec=StrandsDeciderClientProtocol)
    return client


# ==============================================================================
# 1. Tests for the 6 Allowed Privacy Classifications
# ==============================================================================
@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("prompt", "expected_level", "reasoning"),
    [
        (
            "How do I sort a list in Python using timsort?",
            PrivacyLevel.PUBLIC,
            "Public open-source algorithm inquiry with zero sensitive data.",
        ),
        (
            "Here is my personal home address and social security number for verification.",
            PrivacyLevel.PERSONAL,
            "Contains personally identifiable information (PII) of a private citizen.",
        ),
        (
            "Audit our corporate quarterly balance sheet, EBITDA margins, and wire transfer logs.",
            PrivacyLevel.FINANCIAL,
            "Contains sensitive banking transactions, payroll figures, and corporate accounting.",
        ),
        (
            "Patient lab results indicate elevated liver enzymes and MRI brain scan abnormalities.",
            PrivacyLevel.MEDICAL,
            "Protected Health Information (PHI) requiring strict clinical privacy handling.",
        ),
        (
            "Review section 4.2 of this mutual Non-Disclosure Agreement regarding patent liability.",
            PrivacyLevel.LEGAL,
            "Confidential attorney-client and contractual legal document.",
        ),
        (
            "Here are our internal AWS production master keys and unreleased M&A roadmap.",
            PrivacyLevel.CONFIDENTIAL,
            "Proprietary corporate secrets, API tokens, and confidential strategic plans.",
        ),
    ],
)
async def test_all_six_privacy_levels(
    mock_strands_client: AsyncMock,
    prompt: str,
    expected_level: PrivacyLevel,
    reasoning: str,
) -> None:
    """Tests that Strands Decider 2B accurately classifies all 6 privacy categories."""
    mock_strands_client.decide.return_value = {
        "decision": expected_level.value,
        "confidence": 0.97,
        "reasoning": reasoning,
    }

    decider = PrivacyDecider(client=mock_strands_client)
    state = CortexState(user_input=prompt)
    state.intent = IntentCategory.RESEARCH.value
    state.selected_agent = "TechnicalResearchAgent"

    outcome = await decider.decide(state)

    # Verify decision outcome
    assert outcome.action == expected_level.value
    assert outcome.confidence == 0.97
    assert outcome.reasoning == reasoning

    # Verify state mutation
    assert state.privacy_level == expected_level

    # Verify trace recording
    assert len(state.execution_trace) == 1
    step = state.execution_trace[0]
    assert step.step_number == 3
    assert step.module_id == "decider_03_privacy"
    assert step.decision_name == "Privacy Decider"
    assert step.action == expected_level.value
    assert step.success is True
    assert step.duration_ms >= 0.0


# ==============================================================================
# 2. Contextual Prompt Composition
# ==============================================================================
@pytest.mark.asyncio
async def test_contextual_prompt_incorporates_upstream_state(
    mock_strands_client: AsyncMock,
) -> None:
    """Ensures that upstream intent, selected_agent, and metadata are fed to the prompt."""
    mock_strands_client.decide.return_value = {
        "decision": "CONFIDENTIAL",
        "confidence": 0.95,
        "reasoning": "Proprietary code.",
    }

    decider = PrivacyDecider(client=mock_strands_client)
    state = CortexState(
        user_input="Optimize this internal database query",
        metadata={"tenant_id": "enterprise-corp", "client_env": "production"},
    )
    state.intent = "CODING"
    state.selected_agent = "SoftwareEngineerAgent"

    await decider.decide(state)

    # Inspect the prompt passed to Strands Decider 2B
    call_args = mock_strands_client.decide.call_args[1]
    user_input_sent = call_args["user_input"]

    assert "Intent: CODING" in user_input_sent
    assert "Target Agent: SoftwareEngineerAgent" in user_input_sent
    assert "tenant_id=enterprise-corp" in user_input_sent
    assert "Optimize this internal database query" in user_input_sent


# ==============================================================================
# 3. Validation: Reject Unknown Classifications & Malformed Values
# ==============================================================================
@pytest.mark.asyncio
async def test_reject_unknown_privacy_level(mock_strands_client: AsyncMock) -> None:
    """If Strands Decider 2B returns an unapproved category, reject with DeciderInferenceError."""
    mock_strands_client.decide.return_value = {
        "decision": "TOP_SECRET_MILITARY_CLASSIFIED",
        "confidence": 0.99,
        "reasoning": "Made up category.",
    }

    decider = PrivacyDecider(client=mock_strands_client)
    state = CortexState(user_input="Secret file")

    with pytest.raises(DeciderInferenceError) as exc_info:
        await decider.decide(state)

    assert "invalid privacy classification" in str(exc_info.value).lower()
    # Ensure failure is traced
    assert len(state.execution_trace) == 1
    assert state.execution_trace[0].success is False
    assert state.execution_trace[0].action == "ERROR"


@pytest.mark.asyncio
@pytest.mark.parametrize("invalid_conf", ["not_a_float", 1.5, -0.1, None])
async def test_reject_malformed_confidence(
    mock_strands_client: AsyncMock,
    invalid_conf: object,
) -> None:
    """Confidence scores outside [0.0, 1.0] or non-numeric must raise DeciderInferenceError."""
    mock_strands_client.decide.return_value = {
        "decision": "PUBLIC",
        "confidence": invalid_conf,
        "reasoning": "Invalid confidence.",
    }

    decider = PrivacyDecider(client=mock_strands_client)
    state = CortexState(user_input="Test")

    with pytest.raises(DeciderInferenceError) as exc_info:
        await decider.decide(state)

    assert "invalid confidence" in str(exc_info.value).lower()


@pytest.mark.asyncio
async def test_missing_decision_key_raises_error(mock_strands_client: AsyncMock) -> None:
    """Missing 'decision' key in model output must raise DeciderInferenceError."""
    mock_strands_client.decide.return_value = {
        "confidence": 0.90,
        "reasoning": "Missing decision field.",
    }

    decider = PrivacyDecider(client=mock_strands_client)
    state = CortexState(user_input="Test")

    with pytest.raises(DeciderInferenceError) as exc_info:
        await decider.decide(state)

    assert "missing required 'decision' key" in str(exc_info.value).lower()


# ==============================================================================
# 4. Timeout & Network Failures (No Silent Fallback)
# ==============================================================================
@pytest.mark.asyncio
async def test_timeout_raises_domain_exception(mock_strands_client: AsyncMock) -> None:
    """Inference timeout raises DeciderTimeoutError without silent fallback."""
    mock_strands_client.decide.side_effect = DeciderTimeoutError(
        "Strands Decider 2B timed out after 5.0s"
    )

    decider = PrivacyDecider(client=mock_strands_client)
    state = CortexState(user_input="Test timeout")

    with pytest.raises(DeciderTimeoutError) as exc_info:
        await decider.decide(state)

    assert "timed out" in str(exc_info.value).lower()
    assert len(state.execution_trace) == 1
    assert state.execution_trace[0].success is False


@pytest.mark.asyncio
async def test_connection_error_raises_domain_exception(mock_strands_client: AsyncMock) -> None:
    """Connection errors raise DeciderInferenceError without silent fallback."""
    mock_strands_client.decide.side_effect = DeciderInferenceError(
        "Failed to reach Strands Decider 2B daemon"
    )

    decider = PrivacyDecider(client=mock_strands_client)
    state = CortexState(user_input="Test connection error")

    with pytest.raises(DeciderInferenceError) as exc_info:
        await decider.decide(state)

    assert "failed to reach" in str(exc_info.value).lower()
    assert len(state.execution_trace) == 1
    assert state.execution_trace[0].success is False
