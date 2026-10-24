"""Unit tests for IntentDecider powered by Strands Decider 2B.

Validates:
- Successful intent detection for all 7 allowed categories
- Low confidence propagation
- Invalid / unallowed intent detection and error raising
- Malformed model responses (missing keys, non-numeric confidence)
- API failures (network error, timeout)
- State mutation (state.intent)
- Execution trace recording (TraceStep)
- Orchestrator integration on failure
"""

from unittest.mock import AsyncMock
import pytest
from cortex.application.orchestrators.decision_orchestrator import DecisionOrchestrator
from cortex.core.exceptions import (
    DeciderInferenceError,
    DeciderTimeoutError,
)
from cortex.domain.models.intent import IntentCategory
from cortex.domain.models.state import CortexState, PipelineStatus
from cortex.domain.protocols.strands_client import StrandsDeciderClientProtocol
from cortex.infrastructure.bedrock.client import MockBedrockClient
from cortex.infrastructure.deciders.intent_decider import IntentDecider


@pytest.fixture
def mock_strands_client() -> AsyncMock:
    """Mock Strands Decider 2B client."""
    client = AsyncMock(spec=StrandsDeciderClientProtocol)
    return client


# ==============================================================================
# 1. Successful Intent Detection & State Update & Trace Recording
# ==============================================================================
@pytest.mark.asyncio
async def test_successful_intent_detection(mock_strands_client: AsyncMock) -> None:
    """Test successful classification of CODING intent with trace recording."""
    mock_strands_client.decide.return_value = {
        "decision": "CODING",
        "confidence": 0.98,
        "reasoning": "User prompt requests building an asynchronous Python microservice.",
    }

    decider = IntentDecider(client=mock_strands_client)
    state = CortexState(user_input="Build an async Python FastAPI service with uv")

    outcome = await decider.decide(state)

    # Verify decision outcome
    assert outcome.module_id == "decider_01_intent"
    assert outcome.action == IntentCategory.CODING.value
    assert outcome.confidence == 0.98
    assert "Python FastAPI" in outcome.reasoning
    assert outcome.metadata["latency_ms"] >= 0.0

    # Verify state mutation
    assert state.intent == "CODING"

    # Verify execution trace step
    assert len(state.execution_trace) == 1
    step = state.execution_trace[0]
    assert step.step_number == 1
    assert step.module_id == "decider_01_intent"
    assert step.decision_name == "Intent Decider"
    assert step.action == "CODING"
    assert step.confidence == 0.98
    assert step.success is True
    assert step.duration_ms >= 0.0
    assert step.error_message is None


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("intent_enum", "reasoning_text"),
    [
        (IntentCategory.CHAT, "Casual conversational greeting."),
        (IntentCategory.WRITING, "Drafting executive summary report."),
        (IntentCategory.CODING, "Writing SQL queries and database indexes."),
        (IntentCategory.RESEARCH, "Investigating quantum computing literature."),
        (IntentCategory.ANALYSIS, "Evaluating financial portfolio volatility."),
        (IntentCategory.PLANNING, "Creating a 2-week agile sprint backlog."),
        (IntentCategory.VISION, "Extracting text and tables from uploaded diagram."),
    ],
)
async def test_all_seven_allowed_intents(
    mock_strands_client: AsyncMock,
    intent_enum: IntentCategory,
    reasoning_text: str,
) -> None:
    """Ensure all 7 allowed intent categories are supported and correctly applied."""
    mock_strands_client.decide.return_value = {
        "decision": intent_enum.value,
        "confidence": 0.95,
        "reasoning": reasoning_text,
    }

    decider = IntentDecider(client=mock_strands_client)
    state = CortexState(user_input="Arbitrary user input")

    outcome = await decider.decide(state)

    assert outcome.action == intent_enum.value
    assert state.intent == intent_enum.value
    assert state.execution_trace[0].action == intent_enum.value


# ==============================================================================
# 2. Low Confidence Handling
# ==============================================================================
@pytest.mark.asyncio
async def test_low_confidence_intent_detection(mock_strands_client: AsyncMock) -> None:
    """Test that low confidence decisions are returned as-is and trigger orchestrator gating."""
    mock_strands_client.decide.return_value = {
        "decision": "RESEARCH",
        "confidence": 0.42,  # Low confidence below threshold 0.75
        "reasoning": "Ambiguous input with multiple conflicting interpretations.",
    }

    decider = IntentDecider(client=mock_strands_client)
    state = CortexState(user_input="Something vague")

    outcome = await decider.decide(state)

    assert outcome.action == "RESEARCH"
    assert outcome.confidence == 0.42
    assert state.intent == "RESEARCH"


# ==============================================================================
# 3. Invalid Response Handling
# ==============================================================================
@pytest.mark.asyncio
async def test_invalid_response_unrecognized_intent(mock_strands_client: AsyncMock) -> None:
    """Test that returning an intent outside the allowed 7 categories raises DeciderInferenceError."""
    mock_strands_client.decide.return_value = {
        "decision": "UNSUPPORTED_RANDOM_INTENT",
        "confidence": 0.95,
        "reasoning": "Model made up an unauthorized category.",
    }

    decider = IntentDecider(client=mock_strands_client)
    state = CortexState(user_input="Hello world")

    with pytest.raises(DeciderInferenceError) as exc_info:
        await decider.decide(state)

    assert "invalid intent" in str(exc_info.value).lower()
    # Verify failed trace recording
    assert len(state.execution_trace) == 1
    assert state.execution_trace[0].success is False
    assert state.execution_trace[0].action == "ERROR"


@pytest.mark.asyncio
async def test_invalid_response_missing_decision_key(mock_strands_client: AsyncMock) -> None:
    """Test that missing 'decision' key in Strands response raises DeciderInferenceError."""
    mock_strands_client.decide.return_value = {
        "confidence": 0.90,
        "reasoning": "Missing decision field.",
    }

    decider = IntentDecider(client=mock_strands_client)
    state = CortexState(user_input="Hello world")

    with pytest.raises(DeciderInferenceError) as exc_info:
        await decider.decide(state)

    assert "missing required 'decision' key" in str(exc_info.value).lower()


@pytest.mark.asyncio
@pytest.mark.parametrize("invalid_confidence", ["not_a_float", 1.8, -0.2, None])
async def test_invalid_response_malformed_confidence(
    mock_strands_client: AsyncMock,
    invalid_confidence: object,
) -> None:
    """Test that non-float or out-of-bounds confidence values raise DeciderInferenceError."""
    mock_strands_client.decide.return_value = {
        "decision": "CODING",
        "confidence": invalid_confidence,
        "reasoning": "Invalid confidence score.",
    }

    decider = IntentDecider(client=mock_strands_client)
    state = CortexState(user_input="Code a binary tree")

    with pytest.raises(DeciderInferenceError) as exc_info:
        await decider.decide(state)

    assert "invalid confidence" in str(exc_info.value).lower()


# ==============================================================================
# 4. API Failure Handling (Network Error & Timeout)
# ==============================================================================
@pytest.mark.asyncio
async def test_api_failure_network_error(mock_strands_client: AsyncMock) -> None:
    """Test that network connection errors raise DeciderInferenceError without silent fallback."""
    mock_strands_client.decide.side_effect = DeciderInferenceError(
        "Connection refused at http://localhost:8080/v1/decide"
    )

    decider = IntentDecider(client=mock_strands_client)
    state = CortexState(user_input="Check network")

    with pytest.raises(DeciderInferenceError) as exc_info:
        await decider.decide(state)

    assert "Connection refused" in str(exc_info.value)
    # Check trace step recorded error
    assert len(state.execution_trace) == 1
    assert state.execution_trace[0].success is False


@pytest.mark.asyncio
async def test_api_failure_timeout(mock_strands_client: AsyncMock) -> None:
    """Test that inference timeout raises DeciderTimeoutError without silent fallback."""
    mock_strands_client.decide.side_effect = DeciderTimeoutError(
        "Strands Decider 2B request timed out after 5.0s"
    )

    decider = IntentDecider(client=mock_strands_client)
    state = CortexState(user_input="Heavy computation prompt")

    with pytest.raises(DeciderTimeoutError) as exc_info:
        await decider.decide(state)

    assert "timed out" in str(exc_info.value)
    assert len(state.execution_trace) == 1
    assert state.execution_trace[0].success is False


# ==============================================================================
# 5. Decision Orchestrator Integration with Intent Failure
# ==============================================================================
@pytest.mark.asyncio
async def test_orchestrator_halts_on_intent_decider_failure(
    mock_strands_client: AsyncMock,
) -> None:
    """Ensure DecisionOrchestrator immediately stops and never calls Bedrock when IntentDecider fails."""
    mock_strands_client.decide.side_effect = DeciderInferenceError(
        "Strands Decider 2B daemon unresponsive"
    )

    failing_intent_decider = IntentDecider(client=mock_strands_client)
    bedrock_mock = MockBedrockClient()

    orchestrator = DecisionOrchestrator(
        modules=[failing_intent_decider],
        bedrock_client=bedrock_mock,
    )

    state = await orchestrator.execute("Hello")

    # Pipeline must be marked FAILED
    assert state.status == PipelineStatus.FAILED
    assert "Strands Decider 2B daemon unresponsive" in state.metadata["failure_reason"]

    # Invariant: Bedrock must NEVER be invoked
    assert bedrock_mock.invocation_count == 0
    assert state.generation_output is None
