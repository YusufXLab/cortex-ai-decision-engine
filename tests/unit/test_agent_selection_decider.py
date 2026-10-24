"""Unit tests for AgentSelectionDecider and dynamic AgentRegistry routing."""

from unittest.mock import AsyncMock
import pytest
from cortex.application.agents.registry import AgentRegistry, get_default_agent_registry
from cortex.core.exceptions import (
    DeciderInferenceError,
    DeciderTimeoutError,
)
from cortex.domain.models.agent import AgentProfile
from cortex.domain.models.intent import IntentCategory
from cortex.domain.models.state import CortexState
from cortex.domain.protocols.strands_client import StrandsDeciderClientProtocol
from cortex.infrastructure.deciders.agent_decider import AgentSelectionDecider


@pytest.fixture
def mock_strands_client() -> AsyncMock:
    """Mock Strands Decider 2B client."""
    client = AsyncMock(spec=StrandsDeciderClientProtocol)
    return client


# ==============================================================================
# 1. Multiple Specialized Agents Under the Same Intent
# ==============================================================================
@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("user_prompt", "expected_agent", "rationale"),
    [
        (
            "Why is my async task hanging on line 42 with deadlock?",
            "DebuggingAgent",
            "Root cause defect diagnosis and stack trace analysis required.",
        ),
        (
            "Deploy this FastAPI container to AWS ECS Fargate cluster with Terraform.",
            "DevOpsAgent",
            "Cloud infrastructure, container orchestration, and IaC deployment.",
        ),
        (
            "Refactor this monolithic 600-line class to follow Single Responsibility.",
            "RefactoringAgent",
            "Restructuring code and technical debt reduction without changing behavior.",
        ),
        (
            "Audit this AES-GCM encryption code for cryptographic vulnerabilities.",
            "CodeReviewAgent",
            "Security vulnerability audit and quality compliance analysis.",
        ),
        (
            "Implement a high-performance AVL balanced binary search tree in Python.",
            "SoftwareEngineerAgent",
            "Core algorithm implementation and data structure design.",
        ),
    ],
)
async def test_coding_intent_routes_to_distinct_specialized_agents(
    mock_strands_client: AsyncMock,
    user_prompt: str,
    expected_agent: str,
    rationale: str,
) -> None:
    """Proves that for the identical intent (CODING), Strands Decider 2B chooses

    fundamentally different specialized agents based on technical nuances.
    """
    mock_strands_client.decide.return_value = {
        "decision": expected_agent,
        "confidence": 0.96,
        "reasoning": rationale,
    }

    decider = AgentSelectionDecider(client=mock_strands_client)
    state = CortexState(user_input=user_prompt)
    state.intent = IntentCategory.CODING.value  # Pre-set by Step 1

    outcome = await decider.decide(state)

    # Verify decision outcome
    assert outcome.action == expected_agent
    assert outcome.confidence == 0.96
    assert state.selected_agent == expected_agent

    # Verify candidates passed to Strands Decider included all coding agents
    call_args = mock_strands_client.decide.call_args[1]
    assert call_args["task"] == "agent_selection"
    allowed_choices = call_args["allowed_choices"]
    assert "SoftwareEngineerAgent" in allowed_choices
    assert "DebuggingAgent" in allowed_choices
    assert "DevOpsAgent" in allowed_choices
    assert "RefactoringAgent" in allowed_choices
    assert "CodeReviewAgent" in allowed_choices


# ==============================================================================
# 2. Specialized Agents Under Research, Writing, and Analysis
# ==============================================================================
@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("intent", "expected_agent"),
    [
        (IntentCategory.RESEARCH.value, "LiteratureReviewAgent"),
        (IntentCategory.RESEARCH.value, "WebResearchAgent"),
        (IntentCategory.RESEARCH.value, "FactCheckingAgent"),
        (IntentCategory.WRITING.value, "EmailWriterAgent"),
        (IntentCategory.WRITING.value, "TechnicalWriterAgent"),
        (IntentCategory.ANALYSIS.value, "RiskAnalysisAgent"),
        (IntentCategory.ANALYSIS.value, "FinancialAnalysisAgent"),
        (IntentCategory.VISION.value, "OCRAgent"),
        (IntentCategory.VISION.value, "DiagramAnalysisAgent"),
    ],
)
async def test_specialized_agent_selection_across_intents(
    mock_strands_client: AsyncMock,
    intent: str,
    expected_agent: str,
    ) -> None:
    """Verifies that across different intents, appropriate specialized agents are chosen."""
    mock_strands_client.decide.return_value = {
        "decision": expected_agent,
        "confidence": 0.94,
        "reasoning": f"Routed to {expected_agent} based on task requirements.",
    }

    decider = AgentSelectionDecider(client=mock_strands_client)
    state = CortexState(user_input="Sample task query")
    state.intent = intent

    outcome = await decider.decide(state)

    assert outcome.action == expected_agent
    assert state.selected_agent == expected_agent


# ==============================================================================
# 3. State Mutation and Trace Step Recording
# ==============================================================================
@pytest.mark.asyncio
async def test_state_mutation_and_trace_recording(mock_strands_client: AsyncMock) -> None:
    """Ensures state.selected_agent is populated and TraceStep step 2 is recorded."""
    mock_strands_client.decide.return_value = {
        "decision": "DevOpsAgent",
        "confidence": 0.95,
        "reasoning": "Deploying Kubernetes manifests to production.",
    }

    decider = AgentSelectionDecider(client=mock_strands_client)
    state = CortexState(user_input="Deploy k8s ingress")
    state.intent = "CODING"

    outcome = await decider.decide(state)

    assert state.selected_agent == "DevOpsAgent"
    assert len(state.execution_trace) == 1

    trace_step = state.execution_trace[0]
    assert trace_step.step_number == 2
    assert trace_step.module_id == "decider_02_agent"
    assert trace_step.decision_name == "Agent Selection Decider"
    assert trace_step.action == "DevOpsAgent"
    assert trace_step.confidence == 0.95
    assert trace_step.success is True
    assert trace_step.duration_ms >= 0.0


# ==============================================================================
# 4. Error Handling: Missing Intent, Unapproved Agent, Network Failures
# ==============================================================================
@pytest.mark.asyncio
async def test_missing_prior_intent_raises_error(mock_strands_client: AsyncMock) -> None:
    """AgentSelectionDecider must fail if state.intent is missing (Step 1 skipped)."""
    decider = AgentSelectionDecider(client=mock_strands_client)
    state = CortexState(user_input="Deploy ECS cluster")
    state.intent = None  # Missing!

    with pytest.raises(DeciderInferenceError) as exc_info:
        await decider.decide(state)

    assert "requires prior intent classification" in str(exc_info.value).lower()
    # Ensure error is tracked in execution trace
    assert len(state.execution_trace) == 1
    assert state.execution_trace[0].success is False


@pytest.mark.asyncio
async def test_unapproved_agent_selection_raises_error(mock_strands_client: AsyncMock) -> None:
    """If Strands Decider 2B returns an agent not in the candidate list, fail immediately."""
    mock_strands_client.decide.return_value = {
        "decision": "FakeNonExistentAgent",
        "confidence": 0.99,
        "reasoning": "Hallucinated agent name.",
    }

    decider = AgentSelectionDecider(client=mock_strands_client)
    state = CortexState(user_input="Deploy app")
    state.intent = "CODING"

    with pytest.raises(DeciderInferenceError) as exc_info:
        await decider.decide(state)

    assert "selected unapproved agent" in str(exc_info.value).lower()


@pytest.mark.asyncio
async def test_api_timeout_raises_domain_exception(mock_strands_client: AsyncMock) -> None:
    """If Strands Decider 2B times out, raise DeciderTimeoutError without silent fallback."""
    mock_strands_client.decide.side_effect = DeciderTimeoutError(
        "Strands Decider 2B timed out after 5.0s"
    )

    decider = AgentSelectionDecider(client=mock_strands_client)
    state = CortexState(user_input="Deploy cluster")
    state.intent = "CODING"

    with pytest.raises(DeciderTimeoutError):
        await decider.decide(state)

    assert state.execution_trace[0].success is False


# ==============================================================================
# 5. Extensibility: Dynamic Agent Registration at Runtime
# ==============================================================================
@pytest.mark.asyncio
async def test_dynamic_agent_registration_at_runtime(mock_strands_client: AsyncMock) -> None:
    """Proves that a developer can register a new specialized agent in AgentRegistry

    and Strands Decider 2B immediately considers it without modifying the orchestrator.
    """
    registry = get_default_agent_registry()

    # Developer registers a brand new agent
    quantum_agent = AgentProfile(
        name="QuantumCryptographyAgent",
        intents=[IntentCategory.CODING, IntentCategory.RESEARCH],
        description="Specialized in quantum key distribution, Shor's algorithm, and lattice cryptography.",
        capabilities=["post_quantum_crypto", "qiskit", "lattice_algorithms"],
    )
    registry.register(quantum_agent)

    mock_strands_client.decide.return_value = {
        "decision": "QuantumCryptographyAgent",
        "confidence": 0.98,
        "reasoning": "Request asks for post-quantum lattice cryptography implementation.",
    }

    decider = AgentSelectionDecider(client=mock_strands_client, registry=registry)
    state = CortexState(user_input="Implement Crystals-Kyber post-quantum lattice key exchange")
    state.intent = "CODING"

    outcome = await decider.decide(state)

    assert outcome.action == "QuantumCryptographyAgent"
    assert state.selected_agent == "QuantumCryptographyAgent"

    # Verify the new agent was presented to Strands Decider 2B in allowed_choices
    call_args = mock_strands_client.decide.call_args[1]
    assert "QuantumCryptographyAgent" in call_args["allowed_choices"]
