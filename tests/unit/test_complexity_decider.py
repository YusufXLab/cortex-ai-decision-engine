"""Unit tests for ComplexityDecider powered by Strands Decider 2B.

Validates:
- All 4 complexity classifications:
  - TRIVIAL (Basic arithmetic, greeting, single-fact question)
  - SIMPLE (Email writing, small code snippet, short explanation)
  - COMPLEX (Multi-step reasoning, multiple tools, file analysis, web search + summarization)
  - EXPERT (Large multi-document workflows, heavy orchestration, multiple context sources, long chains)
- Rejection of unknown/invalid complexity classifications
- Malformed confidence scores (negative, >1.0, non-numeric)
- Missing decision key in model response
- Timeout and inference failure without silent fallback
- State mutation: state.complexity_level and state.complexity
- Execution trace recording: TraceStep step 8 with module_id "decider_08_complexity"
- Contextual prompt composition (preceding decisions 1-7 & metadata)
"""

from unittest.mock import AsyncMock
import pytest
from cortex.core.exceptions import (
    DeciderInferenceError,
    DeciderTimeoutError,
)
from cortex.domain.models.complexity import ComplexityLevel
from cortex.domain.models.state import CortexState, PrivacyLevel, RiskLevel
from cortex.domain.models.vision import VisionRequirementType
from cortex.domain.protocols.strands_client import StrandsDeciderClientProtocol
from cortex.infrastructure.deciders.complexity_decider import ComplexityDecider


@pytest.fixture
def mock_strands_client() -> AsyncMock:
    """Mock Strands Decider 2B client."""
    client = AsyncMock(spec=StrandsDeciderClientProtocol)
    return client


# ==============================================================================
# 1. Tests for All 4 Approved Complexity Levels
# ==============================================================================
@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("decision_returned", "expected_level", "prompt", "reasoning"),
    [
        (
            "TRIVIAL",
            ComplexityLevel.TRIVIAL,
            "What is 2 + 2?",
            "Elementary arithmetic requiring single-step evaluation.",
        ),
        (
            "SIMPLE",
            ComplexityLevel.SIMPLE,
            "Write a short thank you email to my mentor.",
            "Single-turn text drafting with straightforward structure.",
        ),
        (
            "COMPLEX",
            ComplexityLevel.COMPLEX,
            "Search the web for the latest AWS Bedrock models and summarize their pricing differences.",
            "Multi-step reasoning coordinating web search and comparative analysis.",
        ),
        (
            "EXPERT",
            ComplexityLevel.EXPERT,
            "Synthesize these 5 research papers on distributed consensus, reconcile differing Raft variants, and prove formal safety invariants.",
            "Deep multi-document reasoning chain with rigorous mathematical synthesis.",
        ),
    ],
)
async def test_all_complexity_levels(
    mock_strands_client: AsyncMock,
    decision_returned: str,
    expected_level: ComplexityLevel,
    prompt: str,
    reasoning: str,
) -> None:
    """Tests that all 4 approved complexity levels are correctly evaluated and applied to state."""
    mock_strands_client.decide.return_value = {
        "decision": decision_returned,
        "confidence": 0.98,
        "reasoning": reasoning,
    }

    decider = ComplexityDecider(client=mock_strands_client)
    state = CortexState(user_input=prompt)
    state.intent = "RESEARCH" if expected_level == ComplexityLevel.EXPERT else "CHAT"
    state.selected_agent = "ResearchAgent" if expected_level == ComplexityLevel.EXPERT else "GeneralAgent"
    state.privacy_level = PrivacyLevel.PUBLIC
    state.risk_level = RiskLevel.LOW
    state.selected_tools = ["WEB_SEARCH", "FILE_READER"] if expected_level in (ComplexityLevel.COMPLEX, ComplexityLevel.EXPERT) else ["NONE"]
    state.selected_context = ["UPLOADED_FILES", "WEB_RESULTS"] if expected_level in (ComplexityLevel.COMPLEX, ComplexityLevel.EXPERT) else ["SYSTEM_PROMPT_ONLY"]
    state.vision_requirement = VisionRequirementType.NONE

    outcome = await decider.decide(state)

    # Verify decision outcome
    assert outcome.action == decision_returned
    assert outcome.confidence == 0.98
    assert outcome.reasoning == reasoning

    # Verify state updates (both complexity_level and legacy alias complexity)
    assert state.complexity_level == expected_level
    assert state.complexity == expected_level

    # Verify trace step
    assert len(state.execution_trace) == 1
    step = state.execution_trace[0]
    assert step.step_number == 8
    assert step.module_id == "decider_08_complexity"
    assert step.decision_name == "Complexity Decider"
    assert step.action == decision_returned
    assert step.confidence == 0.98
    assert step.reasoning == reasoning
    assert step.success is True
    assert step.duration_ms >= 0.0


# ==============================================================================
# 2. Validation & Error Handling: Invalid Responses
# ==============================================================================
@pytest.mark.asyncio
async def test_rejects_unknown_complexity_level(mock_strands_client: AsyncMock) -> None:
    """Tests that unapproved or hallucinated complexity tiers raise DeciderInferenceError."""
    mock_strands_client.decide.return_value = {
        "decision": "ULTRA_DIFFICULT",  # Invalid complexity level
        "confidence": 0.95,
        "reasoning": "Unprecedented difficulty level.",
    }

    decider = ComplexityDecider(client=mock_strands_client)
    state = CortexState(user_input="Solve P vs NP")

    with pytest.raises(DeciderInferenceError) as exc_info:
        await decider.decide(state)

    assert "unknown complexity level" in str(exc_info.value).lower()
    # State must not be updated
    assert state.complexity_level is None
    assert state.complexity is None
    # Failed trace step must be appended
    assert len(state.execution_trace) == 1
    assert state.execution_trace[0].success is False
    assert state.execution_trace[0].step_number == 8


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "invalid_confidence",
    [-0.2, 1.3, "very_high", None, {}],
)
async def test_rejects_malformed_confidence(
    mock_strands_client: AsyncMock,
    invalid_confidence: object,
) -> None:
    """Tests rejection of invalid confidence scores."""
    mock_strands_client.decide.return_value = {
        "decision": "SIMPLE",
        "confidence": invalid_confidence,
        "reasoning": "Standard simple explanation.",
    }

    decider = ComplexityDecider(client=mock_strands_client)
    state = CortexState(user_input="Explain photosynthesis")

    with pytest.raises(DeciderInferenceError) as exc_info:
        await decider.decide(state)

    assert "invalid confidence" in str(exc_info.value).lower()
    assert state.complexity_level is None
    assert len(state.execution_trace) == 1
    assert state.execution_trace[0].success is False


@pytest.mark.asyncio
async def test_missing_decision_key_raises_error(mock_strands_client: AsyncMock) -> None:
    """Tests rejection when response lacks a 'decision' key."""
    mock_strands_client.decide.return_value = {
        "confidence": 0.94,
        "reasoning": "Missing decision attribute.",
    }

    decider = ComplexityDecider(client=mock_strands_client)
    state = CortexState(user_input="Summarize document")

    with pytest.raises(DeciderInferenceError) as exc_info:
        await decider.decide(state)

    assert "missing required 'decision' key" in str(exc_info.value).lower()
    assert state.complexity_level is None
    assert len(state.execution_trace) == 1
    assert state.execution_trace[0].success is False


# ==============================================================================
# 3. Timeouts and Network Failures
# ==============================================================================
@pytest.mark.asyncio
async def test_timeout_propagates_without_silent_fallback(mock_strands_client: AsyncMock) -> None:
    """Tests that timeout exceptions are strictly raised and recorded."""
    mock_strands_client.decide.side_effect = DeciderTimeoutError(
        "Strands Decider 2B timed out after 5000ms"
    )

    decider = ComplexityDecider(client=mock_strands_client)
    state = CortexState(user_input="Estimate complexity")

    with pytest.raises(DeciderTimeoutError):
        await decider.decide(state)

    assert state.complexity_level is None
    assert state.complexity is None
    assert len(state.execution_trace) == 1
    step = state.execution_trace[0]
    assert step.success is False
    assert step.step_number == 8
    assert step.action == "ERROR"


@pytest.mark.asyncio
async def test_unexpected_inference_exception_wrapped_in_domain_error(
    mock_strands_client: AsyncMock,
) -> None:
    """Tests that unexpected client errors are wrapped in DeciderInferenceError."""
    mock_strands_client.decide.side_effect = ConnectionResetError("Socket aborted")

    decider = ComplexityDecider(client=mock_strands_client)
    state = CortexState(user_input="Analyze complexity")

    with pytest.raises(DeciderInferenceError) as exc_info:
        await decider.decide(state)

    assert "complexity evaluation failed" in str(exc_info.value).lower()
    assert len(state.execution_trace) == 1
    assert state.execution_trace[0].success is False


# ==============================================================================
# 4. Contextual Prompt Composition
# ==============================================================================
@pytest.mark.asyncio
async def test_contextual_prompt_incorporates_upstream_state(
    mock_strands_client: AsyncMock,
) -> None:
    """Tests that decisions 1-7 and metadata are included in the prompt to Strands Decider 2B."""
    mock_strands_client.decide.return_value = {
        "decision": "COMPLEX",
        "confidence": 0.95,
        "reasoning": "Multi-tool file analysis request.",
    }

    decider = ComplexityDecider(client=mock_strands_client)
    state = CortexState(user_input="Reconcile balance sheet against database")
    state.intent = "ANALYSIS"
    state.selected_agent = "FinancialAnalysisAgent"
    state.privacy_level = PrivacyLevel.FINANCIAL
    state.risk_level = RiskLevel.LOW
    state.selected_tools = ["FILE_READER", "DATABASE"]
    state.selected_context = ["UPLOADED_FILES", "DATABASE_RESULTS"]
    state.vision_requirement = VisionRequirementType.OCR
    state.metadata["session_source"] = "enterprise_portal"

    await decider.decide(state)

    mock_strands_client.decide.assert_called_once()
    _, kwargs = mock_strands_client.decide.call_args
    prompt_sent = kwargs["user_input"]

    assert "Intent: ANALYSIS" in prompt_sent
    assert "Assigned Agent: FinancialAnalysisAgent" in prompt_sent
    assert "Privacy Level: FINANCIAL" in prompt_sent
    assert "Risk Level: LOW" in prompt_sent
    assert "Selected Tools: ['FILE_READER', 'DATABASE']" in prompt_sent
    assert "Selected Context: ['UPLOADED_FILES', 'DATABASE_RESULTS']" in prompt_sent
    assert "Vision Requirement: OCR" in prompt_sent
    assert "session_source=enterprise_portal" in prompt_sent
    assert "Reconcile balance sheet against database" in prompt_sent


# ==============================================================================
# 5. Default Client Offline Simulation Tests
# ==============================================================================
@pytest.mark.asyncio
async def test_offline_simulation_all_complexity_examples() -> None:
    """Tests the real ComplexityDecider with default offline StrandsDeciderClient."""
    decider = ComplexityDecider()

    test_cases = [
        ("What is 2 + 2?", ComplexityLevel.TRIVIAL),
        ("Hello, good morning!", ComplexityLevel.TRIVIAL),
        ("Write a thank you email to my team", ComplexityLevel.SIMPLE),
        ("Write a function to reverse a string in Python", ComplexityLevel.SIMPLE),
        ("Search the web and summarize the latest AI model announcements", ComplexityLevel.COMPLEX),
        ("Analyze this uploaded financial balance sheet PDF", ComplexityLevel.COMPLEX),
        ("Design distributed consensus with formal verification across 50 nodes", ComplexityLevel.EXPERT),
        ("Large multi-document synthesis across 10 annual reports", ComplexityLevel.EXPERT),
    ]

    for user_prompt, expected_level in test_cases:
        state = CortexState(user_input=user_prompt)
        outcome = await decider.decide(state)

        assert outcome.action == expected_level.value
        assert state.complexity_level == expected_level
        assert state.complexity == expected_level
        assert outcome.confidence >= 0.9
        assert len(state.execution_trace) == 1
        assert state.execution_trace[0].step_number == 8
        assert state.execution_trace[0].success is True
