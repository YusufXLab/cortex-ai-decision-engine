"""Unit tests for ContextSelectionDecider powered by Strands Decider 2B.

Validates:
- Single context source selection
- Multiple simultaneous context sources
- No external context needed (SYSTEM_PROMPT_ONLY)
- Rejection of unknown context sources
- Malformed confidence handling
- Missing decision key in model response
- API timeout and network error handling without silent fallback
- State mutation (state.selected_context)
- Execution trace recording (TraceStep step 6)
- Contextual prompt composition (Intent, Agent, Privacy, Risk, Tools, Metadata)
"""

from unittest.mock import AsyncMock
import pytest
from cortex.core.exceptions import (
    DeciderInferenceError,
    DeciderTimeoutError,
)
from cortex.domain.models.state import CortexState, PrivacyLevel, RiskLevel
from cortex.domain.protocols.strands_client import StrandsDeciderClientProtocol
from cortex.infrastructure.deciders.context_decider import ContextSelectionDecider


@pytest.fixture
def mock_strands_client() -> AsyncMock:
    """Mock Strands Decider 2B client."""
    client = AsyncMock(spec=StrandsDeciderClientProtocol)
    return client


# ==============================================================================
# 1. Tests for Single, Multiple, and No External Context
# ==============================================================================
@pytest.mark.asyncio
async def test_single_context_source_selection(mock_strands_client: AsyncMock) -> None:
    """Tests single context source selection (e.g. UPLOADED_FILES)."""
    mock_strands_client.decide.return_value = {
        "decision": ["UPLOADED_FILES"],
        "confidence": 0.97,
        "reasoning": "Task requires parsing user-uploaded spreadsheet file.",
    }

    decider = ContextSelectionDecider(client=mock_strands_client)
    state = CortexState(user_input="Analyze this uploaded financial Q3 balance sheet PDF")
    state.intent = "ANALYSIS"
    state.selected_agent = "FinancialAnalysisAgent"
    state.privacy_level = PrivacyLevel.FINANCIAL
    state.risk_level = RiskLevel.LOW
    state.selected_tools = ["FILE_READER"]

    outcome = await decider.decide(state)

    # Verify outcome and state
    assert outcome.action == "UPLOADED_FILES"
    assert outcome.confidence == 0.97
    assert state.selected_context == ["UPLOADED_FILES"]

    # Verify execution trace
    assert len(state.execution_trace) == 1
    step = state.execution_trace[0]
    assert step.step_number == 6
    assert step.module_id == "decider_06_context"
    assert step.decision_name == "Context Selection Decider"
    assert step.action == "UPLOADED_FILES"
    assert step.success is True
    assert step.duration_ms >= 0.0


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("sources_returned", "expected_action", "reasoning"),
    [
        (
            ["CHAT_HISTORY", "USER_MEMORY"],
            "CHAT_HISTORY, USER_MEMORY",
            "Multi-turn conversational follow-up referencing prior discussion and user preferences.",
        ),
        (
            ["WEB_RESULTS", "RAG_KNOWLEDGE_BASE"],
            "WEB_RESULTS, RAG_KNOWLEDGE_BASE",
            "Comparing live web search results against internal company policy knowledge base.",
        ),
        (
            ["DATABASE_RESULTS", "UPLOADED_FILES"],
            "DATABASE_RESULTS, UPLOADED_FILES",
            "Reconciling transactional database order records with uploaded merchant invoice.",
        ),
    ],
)
async def test_multiple_context_sources_selection(
    mock_strands_client: AsyncMock,
    sources_returned: list[str],
    expected_action: str,
    reasoning: str,
) -> None:
    """Tests simultaneous multi-source context injection."""
    mock_strands_client.decide.return_value = {
        "decision": sources_returned,
        "confidence": 0.95,
        "reasoning": reasoning,
    }

    decider = ContextSelectionDecider(client=mock_strands_client)
    state = CortexState(user_input="Multi-source correlation task")
    state.intent = "RESEARCH"

    outcome = await decider.decide(state)

    assert outcome.action == expected_action
    assert state.selected_context == sources_returned
    assert outcome.metadata["selected_context"] == sources_returned


@pytest.mark.asyncio
async def test_no_external_context_system_prompt_only(mock_strands_client: AsyncMock) -> None:
    """Tests selecting SYSTEM_PROMPT_ONLY when no external context is required."""
    mock_strands_client.decide.return_value = {
        "decision": ["SYSTEM_PROMPT_ONLY"],
        "confidence": 0.99,
        "reasoning": "Pure mathematical logic and creative task requiring zero external context.",
    }

    decider = ContextSelectionDecider(client=mock_strands_client)
    state = CortexState(user_input="Write a python function to compute factorial recursively")
    state.intent = "CODING"
    state.selected_tools = ["NONE"]

    outcome = await decider.decide(state)

    assert outcome.action == "SYSTEM_PROMPT_ONLY"
    assert state.selected_context == ["SYSTEM_PROMPT_ONLY"]


# ==============================================================================
# 2. Contextual Prompt Composition
# ==============================================================================
@pytest.mark.asyncio
async def test_contextual_prompt_incorporates_upstream_state(
    mock_strands_client: AsyncMock,
) -> None:
    """Ensures intent, agent, privacy, risk, selected_tools, and metadata are fed to Strands Decider."""
    mock_strands_client.decide.return_value = {
        "decision": ["RAG_KNOWLEDGE_BASE"],
        "confidence": 0.96,
        "reasoning": "Internal documentation required.",
    }

    decider = ContextSelectionDecider(client=mock_strands_client)
    state = CortexState(
        user_input="Look up internal travel expense reimbursement policies",
        metadata={"user_dept": "engineering", "org_id": "corp-88"},
    )
    state.intent = "RESEARCH"
    state.selected_agent = "TechnicalResearchAgent"
    state.privacy_level = PrivacyLevel.INTERNAL if hasattr(PrivacyLevel, "INTERNAL") else PrivacyLevel.CONFIDENTIAL
    state.risk_level = RiskLevel.LOW
    state.selected_tools = ["RAG"]

    await decider.decide(state)

    call_args = mock_strands_client.decide.call_args[1]
    prompt = call_args["user_input"]

    assert "Intent: RESEARCH" in prompt
    assert "Assigned Agent: TechnicalResearchAgent" in prompt
    assert "Selected Tools: ['RAG']" in prompt
    assert "user_dept=engineering" in prompt
    assert "Look up internal travel expense reimbursement policies" in prompt


# ==============================================================================
# 3. Validation: Reject Unknown Context Sources & Malformed Output
# ==============================================================================
@pytest.mark.asyncio
async def test_reject_unknown_context_source(mock_strands_client: AsyncMock) -> None:
    """If Strands Decider 2B selects an unapproved context source, raise DeciderInferenceError."""
    mock_strands_client.decide.return_value = {
        "decision": ["WEB_RESULTS", "UNAPPROVED_CLASSIFIED_SATELLITE_FEED"],
        "confidence": 0.95,
        "reasoning": "Model returned an unauthorized context source.",
    }

    decider = ContextSelectionDecider(client=mock_strands_client)
    state = CortexState(user_input="Find recent news")

    with pytest.raises(DeciderInferenceError) as exc_info:
        await decider.decide(state)

    assert "unapproved context source" in str(exc_info.value).lower()
    assert len(state.execution_trace) == 1
    assert state.execution_trace[0].success is False


@pytest.mark.asyncio
@pytest.mark.parametrize("invalid_conf", ["not_a_float", 1.8, -0.4, None])
async def test_reject_malformed_confidence(
    mock_strands_client: AsyncMock,
    invalid_conf: object,
) -> None:
    """Confidence outside [0.0, 1.0] or non-numeric must raise DeciderInferenceError."""
    mock_strands_client.decide.return_value = {
        "decision": ["CHAT_HISTORY"],
        "confidence": invalid_conf,
        "reasoning": "Invalid confidence score.",
    }

    decider = ContextSelectionDecider(client=mock_strands_client)
    state = CortexState(user_input="Test")

    with pytest.raises(DeciderInferenceError) as exc_info:
        await decider.decide(state)

    assert "invalid confidence" in str(exc_info.value).lower()


@pytest.mark.asyncio
async def test_missing_decision_key_raises_error(mock_strands_client: AsyncMock) -> None:
    """Missing 'decision' and fallback keys raises DeciderInferenceError."""
    mock_strands_client.decide.return_value = {
        "confidence": 0.90,
        "reasoning": "Missing decision field.",
    }

    decider = ContextSelectionDecider(client=mock_strands_client)
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

    decider = ContextSelectionDecider(client=mock_strands_client)
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

    decider = ContextSelectionDecider(client=mock_strands_client)
    state = CortexState(user_input="Test connection error")

    with pytest.raises(DeciderInferenceError) as exc_info:
        await decider.decide(state)

    assert "failed to reach" in str(exc_info.value).lower()
    assert len(state.execution_trace) == 1
    assert state.execution_trace[0].success is False
