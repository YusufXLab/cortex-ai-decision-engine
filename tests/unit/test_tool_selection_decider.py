"""Unit tests for ToolSelectionDecider powered by Strands Decider 2B.

Validates:
- Single tool selection
- Multiple simultaneous tools selection
- No tools needed (NONE)
- Rejection of unknown tool names
- Malformed confidence handling
- Missing decision key in model response
- API timeout and network error handling without silent fallback
- State mutation (state.selected_tools)
- Execution trace recording (TraceStep step 5)
- Contextual prompt composition (Intent, Agent, Privacy, Risk, Metadata)
"""

from unittest.mock import AsyncMock
import pytest
from cortex.core.exceptions import (
    DeciderInferenceError,
    DeciderTimeoutError,
)
from cortex.domain.models.state import CortexState, PrivacyLevel, RiskLevel
from cortex.domain.protocols.strands_client import StrandsDeciderClientProtocol
from cortex.infrastructure.deciders.tool_decider import ToolSelectionDecider


@pytest.fixture
def mock_strands_client() -> AsyncMock:
    """Mock Strands Decider 2B client."""
    client = AsyncMock(spec=StrandsDeciderClientProtocol)
    return client


# ==============================================================================
# 1. Tests for Single, Multiple, and No Tools
# ==============================================================================
@pytest.mark.asyncio
async def test_single_tool_selection(mock_strands_client: AsyncMock) -> None:
    """Tests single tool selection (e.g. WEB_SEARCH)."""
    mock_strands_client.decide.return_value = {
        "decision": ["WEB_SEARCH"],
        "confidence": 0.96,
        "reasoning": "Real-time query requires live web search capability.",
    }

    decider = ToolSelectionDecider(client=mock_strands_client)
    state = CortexState(user_input="What is the latest stock price of AWS today?")
    state.intent = "RESEARCH"
    state.selected_agent = "WebResearchAgent"
    state.privacy_level = PrivacyLevel.PUBLIC
    state.risk_level = RiskLevel.LOW

    outcome = await decider.decide(state)

    # Verify outcome and state
    assert outcome.action == "WEB_SEARCH"
    assert outcome.confidence == 0.96
    assert state.selected_tools == ["WEB_SEARCH"]

    # Verify execution trace
    assert len(state.execution_trace) == 1
    step = state.execution_trace[0]
    assert step.step_number == 5
    assert step.module_id == "decider_05_tools"
    assert step.decision_name == "Tool Selection Decider"
    assert step.action == "WEB_SEARCH"
    assert step.success is True
    assert step.duration_ms >= 0.0


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("tools_returned", "expected_action", "reasoning"),
    [
        (
            ["FILE_READER", "CALCULATOR"],
            "FILE_READER, CALCULATOR",
            "Reading uploaded CSV spreadsheet and performing financial math.",
        ),
        (
            ["VISION", "FILE_READER"],
            "VISION, FILE_READER",
            "Extracting text from uploaded PDF architecture diagram image.",
        ),
        (
            ["DATABASE", "CODE_EXECUTION"],
            "DATABASE, CODE_EXECUTION",
            "Querying customer table and executing statistical Python script.",
        ),
        (
            ["RAG", "WEB_SEARCH", "MEMORY"],
            "RAG, WEB_SEARCH, MEMORY",
            "Retrieving corporate docs, querying web, and referencing user history.",
        ),
    ],
)
async def test_multiple_tools_selection(
    mock_strands_client: AsyncMock,
    tools_returned: list[str],
    expected_action: str,
    reasoning: str,
) -> None:
    """Tests simultaneous multi-tool selection."""
    mock_strands_client.decide.return_value = {
        "decision": tools_returned,
        "confidence": 0.95,
        "reasoning": reasoning,
    }

    decider = ToolSelectionDecider(client=mock_strands_client)
    state = CortexState(user_input="Complex multi-capability task")
    state.intent = "ANALYSIS"

    outcome = await decider.decide(state)

    assert outcome.action == expected_action
    assert state.selected_tools == tools_returned
    assert outcome.metadata["tools"] == tools_returned


@pytest.mark.asyncio
async def test_no_tools_selection(mock_strands_client: AsyncMock) -> None:
    """Tests selecting NONE when no tools are required."""
    mock_strands_client.decide.return_value = {
        "decision": ["NONE"],
        "confidence": 0.98,
        "reasoning": "Pure conceptual language reasoning without external tools.",
    }

    decider = ToolSelectionDecider(client=mock_strands_client)
    state = CortexState(user_input="Write a haiku about autumn leaves")
    state.intent = "WRITING"

    outcome = await decider.decide(state)

    assert outcome.action == "NONE"
    assert state.selected_tools == ["NONE"]


# ==============================================================================
# 2. Contextual Prompt Composition
# ==============================================================================
@pytest.mark.asyncio
async def test_contextual_prompt_incorporates_upstream_state(
    mock_strands_client: AsyncMock,
) -> None:
    """Ensures intent, agent, privacy, risk, and metadata are fed to Strands Decider."""
    mock_strands_client.decide.return_value = {
        "decision": ["DATABASE"],
        "confidence": 0.95,
        "reasoning": "Requires database access.",
    }

    decider = ToolSelectionDecider(client=mock_strands_client)
    state = CortexState(
        user_input="Query customer orders table",
        metadata={"session_id": "sess-99", "env": "prod"},
    )
    state.intent = "CODING"
    state.selected_agent = "DevOpsAgent"
    state.privacy_level = PrivacyLevel.CONFIDENTIAL
    state.risk_level = RiskLevel.MODERATE

    await decider.decide(state)

    call_args = mock_strands_client.decide.call_args[1]
    prompt = call_args["user_input"]

    assert "Intent: CODING" in prompt
    assert "Assigned Agent: DevOpsAgent" in prompt
    assert "Privacy Level: CONFIDENTIAL" in prompt
    assert "Risk Level: MODERATE" in prompt
    assert "session_id=sess-99" in prompt
    assert "Query customer orders table" in prompt


# ==============================================================================
# 3. Validation: Reject Unknown Tools & Malformed Output
# ==============================================================================
@pytest.mark.asyncio
async def test_reject_unknown_tool_name(mock_strands_client: AsyncMock) -> None:
    """If Strands Decider 2B selects an unapproved tool, raise DeciderInferenceError."""
    mock_strands_client.decide.return_value = {
        "decision": ["WEB_SEARCH", "UNAUTHORIZED_SATELLITE_HACKER_TOOL"],
        "confidence": 0.95,
        "reasoning": "Model returned an invalid tool.",
    }

    decider = ToolSelectionDecider(client=mock_strands_client)
    state = CortexState(user_input="Run task")

    with pytest.raises(DeciderInferenceError) as exc_info:
        await decider.decide(state)

    assert "unapproved tool" in str(exc_info.value).lower()
    assert len(state.execution_trace) == 1
    assert state.execution_trace[0].success is False


@pytest.mark.asyncio
@pytest.mark.parametrize("invalid_conf", ["invalid", 1.8, -0.3, None])
async def test_reject_malformed_confidence(
    mock_strands_client: AsyncMock,
    invalid_conf: object,
) -> None:
    """Confidence outside [0.0, 1.0] or non-numeric must raise DeciderInferenceError."""
    mock_strands_client.decide.return_value = {
        "decision": ["WEB_SEARCH"],
        "confidence": invalid_conf,
        "reasoning": "Invalid confidence.",
    }

    decider = ToolSelectionDecider(client=mock_strands_client)
    state = CortexState(user_input="Test")

    with pytest.raises(DeciderInferenceError) as exc_info:
        await decider.decide(state)

    assert "invalid confidence" in str(exc_info.value).lower()


@pytest.mark.asyncio
async def test_missing_decision_and_tools_key_raises_error(
    mock_strands_client: AsyncMock,
) -> None:
    """Missing both 'decision' and 'tools' keys raises DeciderInferenceError."""
    mock_strands_client.decide.return_value = {
        "confidence": 0.90,
        "reasoning": "Missing keys.",
    }

    decider = ToolSelectionDecider(client=mock_strands_client)
    state = CortexState(user_input="Test")

    with pytest.raises(DeciderInferenceError) as exc_info:
        await decider.decide(state)

    assert "missing required 'decision' or 'tools' key" in str(exc_info.value).lower()


# ==============================================================================
# 4. Timeout & Network Failures (No Silent Fallback)
# ==============================================================================
@pytest.mark.asyncio
async def test_timeout_raises_domain_exception(mock_strands_client: AsyncMock) -> None:
    """Inference timeout raises DeciderTimeoutError without silent fallback."""
    mock_strands_client.decide.side_effect = DeciderTimeoutError(
        "Strands Decider 2B timed out after 5.0s"
    )

    decider = ToolSelectionDecider(client=mock_strands_client)
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

    decider = ToolSelectionDecider(client=mock_strands_client)
    state = CortexState(user_input="Test connection error")

    with pytest.raises(DeciderInferenceError) as exc_info:
        await decider.decide(state)

    assert "failed to reach" in str(exc_info.value).lower()
    assert len(state.execution_trace) == 1
    assert state.execution_trace[0].success is False
