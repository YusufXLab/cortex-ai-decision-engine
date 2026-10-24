"""Demo endpoint demonstrating the full Cortex pipeline."""

from fastapi import APIRouter, Depends, status
from cortex.application.orchestrators.decision_orchestrator import DecisionOrchestrator
from cortex.domain.models.state import CortexState
from cortex.presentation.api.deps import get_orchestrator
from cortex.presentation.schemas.common import APIResponse
from cortex.presentation.schemas.demo import DemoPipelineRequest

router = APIRouter(prefix="/demo", tags=["Demo & Scaffolding"])


@router.post(
    "/decide-and-generate",
    response_model=APIResponse[CortexState],
    status_code=status.HTTP_200_OK,
    summary="Run Decision and Generation Pipeline",
    description=(
        "Executes the full pipeline: initializes CortexState, passes through all "
        "11 Strands Decider 2B decision modules, applies safety/confidence gating, "
        "and if approved, delegates response generation to Amazon Bedrock."
    ),
)
async def demo_decide_and_generate(
    payload: DemoPipelineRequest,
    orchestrator: DecisionOrchestrator = Depends(get_orchestrator),
) -> APIResponse[CortexState]:
    """Runs the request through the Decision Orchestrator."""
    state = await orchestrator.execute(
        user_input=payload.query,
        metadata={"enforce_guardrail": payload.enforce_guardrail, **payload.context},
    )
    return APIResponse(data=state)
