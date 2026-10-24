"""Unit tests for VisionRequirementDecider powered by Strands Decider 2B.

Validates:
- All 6 vision requirement classifications:
  - NONE (No visual processing)
  - OCR (Text extraction from receipts/invoices/forms)
  - IMAGE_UNDERSTANDING (Photos and scenes)
  - DIAGRAM_ANALYSIS (Architecture, flowcharts, schemas)
  - UI_ANALYSIS (Screenshots, app interfaces, wireframes)
  - MULTIMODAL (Mixed PDFs with charts, images, and text)
- Rejection of unknown/invalid vision classifications
- Malformed confidence scores (negative, >1.0, non-numeric)
- Missing decision key in model response
- Timeout and inference failure without silent fallback
- State mutation: state.vision_requirement and state.vision_required
- Execution trace recording: TraceStep step 7 with module_id "decider_07_vision"
- Contextual prompt composition (preceding decisions & attachment metadata)
"""

from unittest.mock import AsyncMock
import pytest
from cortex.core.exceptions import (
    DeciderInferenceError,
    DeciderTimeoutError,
)
from cortex.domain.models.state import CortexState, PrivacyLevel, RiskLevel
from cortex.domain.models.vision import VisionRequirementType
from cortex.domain.protocols.strands_client import StrandsDeciderClientProtocol
from cortex.infrastructure.deciders.vision_decider import VisionRequirementDecider


@pytest.fixture
def mock_strands_client() -> AsyncMock:
    """Mock Strands Decider 2B client."""
    client = AsyncMock(spec=StrandsDeciderClientProtocol)
    return client


# ==============================================================================
# 1. Tests for All 6 Approved Vision Requirement Types
# ==============================================================================
@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("decision_returned", "expected_type", "expected_vision_required", "prompt", "reasoning"),
    [
        (
            "NONE",
            VisionRequirementType.NONE,
            False,
            "Write an essay about the Roman Empire.",
            "Text-only prompt with no visual content.",
        ),
        (
            "OCR",
            VisionRequirementType.OCR,
            True,
            "Extract the line items, merchant name, and total amount from this receipt image.",
            "Optical Character Recognition required for tabular receipt text.",
        ),
        (
            "IMAGE_UNDERSTANDING",
            VisionRequirementType.IMAGE_UNDERSTANDING,
            True,
            "Describe the scene and colors in this photo of a sunset.",
            "Natural scene and aesthetic image comprehension required.",
        ),
        (
            "DIAGRAM_ANALYSIS",
            VisionRequirementType.DIAGRAM_ANALYSIS,
            True,
            "Analyze this AWS cloud architecture diagram and identify single points of failure.",
            "Structural and topological analysis of architectural schema required.",
        ),
        (
            "UI_ANALYSIS",
            VisionRequirementType.UI_ANALYSIS,
            True,
            "Look at this screenshot of our web app and explain why the submit button is misaligned.",
            "Visual inspection of user interface layout and styling elements.",
        ),
        (
            "MULTIMODAL",
            VisionRequirementType.MULTIMODAL,
            True,
            "Review this 20-page annual report PDF containing financial tables, trend charts, and photos.",
            "Mixed multimodal reasoning across interleaved text, figures, and charts.",
        ),
    ],
)
async def test_all_vision_requirement_types(
    mock_strands_client: AsyncMock,
    decision_returned: str,
    expected_type: VisionRequirementType,
    expected_vision_required: bool,
    prompt: str,
    reasoning: str,
) -> None:
    """Tests that all 6 approved vision requirements are correctly classified and applied to state."""
    mock_strands_client.decide.return_value = {
        "decision": decision_returned,
        "confidence": 0.98,
        "reasoning": reasoning,
    }

    decider = VisionRequirementDecider(client=mock_strands_client)
    state = CortexState(user_input=prompt)
    state.intent = "VISION" if expected_vision_required else "WRITING"
    state.selected_agent = "VisionSpecialistAgent" if expected_vision_required else "GeneralAgent"
    state.privacy_level = PrivacyLevel.PUBLIC
    state.risk_level = RiskLevel.LOW
    state.selected_tools = ["VISION"] if expected_vision_required else ["NONE"]
    state.selected_context = ["UPLOADED_FILES"] if expected_vision_required else ["SYSTEM_PROMPT_ONLY"]

    outcome = await decider.decide(state)

    # Verify decision outcome
    assert outcome.action == decision_returned
    assert outcome.confidence == 0.98
    assert outcome.reasoning == reasoning

    # Verify state updates
    assert state.vision_requirement == expected_type
    assert state.vision_required is expected_vision_required

    # Verify trace step
    assert len(state.execution_trace) == 1
    step = state.execution_trace[0]
    assert step.step_number == 7
    assert step.module_id == "decider_07_vision"
    assert step.decision_name == "Vision Requirement Decider"
    assert step.action == decision_returned
    assert step.confidence == 0.98
    assert step.reasoning == reasoning
    assert step.success is True
    assert step.duration_ms >= 0.0


# ==============================================================================
# 2. Validation & Error Handling: Invalid Responses
# ==============================================================================
@pytest.mark.asyncio
async def test_rejects_unknown_vision_requirement(mock_strands_client: AsyncMock) -> None:
    """Tests that unapproved or hallucinated vision requirements raise DeciderInferenceError."""
    mock_strands_client.decide.return_value = {
        "decision": "VIDEO_GENERATION",  # Invalid classification
        "confidence": 0.95,
        "reasoning": "Generating a video file.",
    }

    decider = VisionRequirementDecider(client=mock_strands_client)
    state = CortexState(user_input="Create a 3D animation")

    with pytest.raises(DeciderInferenceError) as exc_info:
        await decider.decide(state)

    assert "unknown vision requirement" in str(exc_info.value).lower()
    # State must not be updated
    assert state.vision_requirement is None
    # Failed trace step must be appended
    assert len(state.execution_trace) == 1
    assert state.execution_trace[0].success is False
    assert state.execution_trace[0].step_number == 7


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "invalid_confidence",
    [-0.5, 1.5, "high", None, []],
)
async def test_rejects_malformed_confidence(
    mock_strands_client: AsyncMock,
    invalid_confidence: object,
) -> None:
    """Tests rejection of invalid confidence scores."""
    mock_strands_client.decide.return_value = {
        "decision": "IMAGE_UNDERSTANDING",
        "confidence": invalid_confidence,
        "reasoning": "Valid reasoning",
    }

    decider = VisionRequirementDecider(client=mock_strands_client)
    state = CortexState(user_input="Describe this picture")

    with pytest.raises(DeciderInferenceError) as exc_info:
        await decider.decide(state)

    assert "invalid confidence" in str(exc_info.value).lower()
    assert state.vision_requirement is None
    assert len(state.execution_trace) == 1
    assert state.execution_trace[0].success is False


@pytest.mark.asyncio
async def test_missing_decision_key_raises_error(mock_strands_client: AsyncMock) -> None:
    """Tests rejection when the response lacks a 'decision' key."""
    mock_strands_client.decide.return_value = {
        "confidence": 0.95,
        "reasoning": "Missing decision property.",
    }

    decider = VisionRequirementDecider(client=mock_strands_client)
    state = CortexState(user_input="Scan this document")

    with pytest.raises(DeciderInferenceError) as exc_info:
        await decider.decide(state)

    assert "missing required 'decision' key" in str(exc_info.value).lower()
    assert state.vision_requirement is None
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

    decider = VisionRequirementDecider(client=mock_strands_client)
    state = CortexState(user_input="Analyze diagram")

    with pytest.raises(DeciderTimeoutError):
        await decider.decide(state)

    assert state.vision_requirement is None
    assert state.vision_required is False
    assert len(state.execution_trace) == 1
    step = state.execution_trace[0]
    assert step.success is False
    assert step.step_number == 7
    assert step.action == "ERROR"


@pytest.mark.asyncio
async def test_unexpected_inference_exception_wrapped_in_domain_error(
    mock_strands_client: AsyncMock,
) -> None:
    """Tests that unexpected client errors are wrapped in DeciderInferenceError."""
    mock_strands_client.decide.side_effect = RuntimeError("Connection closed by peer")

    decider = VisionRequirementDecider(client=mock_strands_client)
    state = CortexState(user_input="Analyze image")

    with pytest.raises(DeciderInferenceError) as exc_info:
        await decider.decide(state)

    assert "vision requirement evaluation failed" in str(exc_info.value).lower()
    assert len(state.execution_trace) == 1
    assert state.execution_trace[0].success is False


# ==============================================================================
# 4. Contextual Prompt & Attachment Metadata
# ==============================================================================
@pytest.mark.asyncio
async def test_attachment_metadata_included_in_contextual_prompt(
    mock_strands_client: AsyncMock,
) -> None:
    """Tests that attachment metadata and upstream pipeline states are passed to Strands Decider 2B."""
    mock_strands_client.decide.return_value = {
        "decision": "OCR",
        "confidence": 0.99,
        "reasoning": "Receipt attachment detected with scanned table.",
    }

    decider = VisionRequirementDecider(client=mock_strands_client)
    state = CortexState(user_input="Read the receipt total")
    state.intent = "VISION"
    state.selected_agent = "OCRAgent"
    state.privacy_level = PrivacyLevel.FINANCIAL
    state.risk_level = RiskLevel.LOW
    state.selected_tools = ["VISION", "FILE_READER"]
    state.selected_context = ["UPLOADED_FILES"]
    state.metadata["attachments"] = [{"filename": "receipt_nov2026.png", "mime": "image/png"}]

    await decider.decide(state)

    # Inspect call args sent to client.decide
    mock_strands_client.decide.assert_called_once()
    _, kwargs = mock_strands_client.decide.call_args
    prompt_sent = kwargs["user_input"]

    assert "Intent: VISION" in prompt_sent
    assert "Assigned Agent: OCRAgent" in prompt_sent
    assert "Privacy Level: FINANCIAL" in prompt_sent
    assert "Risk Level: LOW" in prompt_sent
    assert "Selected Tools: ['VISION', 'FILE_READER']" in prompt_sent
    assert "Selected Context: ['UPLOADED_FILES']" in prompt_sent
    assert "receipt_nov2026.png" in prompt_sent
    assert "Read the receipt total" in prompt_sent


# ==============================================================================
# 5. Default Client Offline Simulation Tests
# ==============================================================================
@pytest.mark.asyncio
async def test_offline_simulation_all_examples() -> None:
    """Tests the real VisionRequirementDecider with default offline StrandsDeciderClient."""
    decider = VisionRequirementDecider()

    test_cases = [
        ("No uploaded images, just tell me a story", VisionRequirementType.NONE, False),
        ("Extract total and items from this receipt image", VisionRequirementType.OCR, True),
        ("Describe this photo of a mountain sunset", VisionRequirementType.IMAGE_UNDERSTANDING, True),
        ("Analyze this cloud architecture diagram for bottlenecks", VisionRequirementType.DIAGRAM_ANALYSIS, True),
        ("Check this software screenshot to see if the navbar is responsive", VisionRequirementType.UI_ANALYSIS, True),
        ("Analyze this rich PDF containing text, charts and images", VisionRequirementType.MULTIMODAL, True),
    ]

    for user_prompt, expected_vision_type, expected_bool in test_cases:
        state = CortexState(user_input=user_prompt)
        outcome = await decider.decide(state)

        assert outcome.action == expected_vision_type.value
        assert state.vision_requirement == expected_vision_type
        assert state.vision_required is expected_bool
        assert outcome.confidence >= 0.9
        assert len(state.execution_trace) == 1
        assert state.execution_trace[0].step_number == 7
        assert state.execution_trace[0].success is True
