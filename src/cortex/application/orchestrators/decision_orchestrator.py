"""Decision Orchestrator: Central coordinator for the Cortex AI Decision Engine.

Coordinates the complete pipeline:
1. Accepts user request and initializes CortexState.
2. Sequentially executes the 11 Strands Decider 2B decision modules.
3. Updates CortexState after each decision.
4. Gating: Stops immediately if a decision fails or violates safety/risk policy.
5. Once all decisions are approved, hands off to Amazon Bedrock for response generation.
6. Returns the completed CortexState and final response.
"""

from typing import Any
import uuid
from cortex.core.config import Settings, get_settings
from cortex.core.exceptions import (
    BedrockException,
    DeciderException,
    OrchestrationException,
)
from cortex.core.logging import get_logger
from cortex.domain.models.generation import GenerationInput
from cortex.domain.models.state import (
    CortexState,
    PipelineStatus,
    RiskLevel,
)
from cortex.domain.protocols.bedrock import BedrockClientProtocol
from cortex.domain.protocols.decider import DecisionModuleProtocol

logger = get_logger("cortex.orchestrator")


class DecisionOrchestrator:
    """Coordinates Strands Decider 2B decision modules and Amazon Bedrock generation.

    Employs Dependency Injection for all collaborators:
    - modules: List of 11 decision modules implementing DecisionModuleProtocol.
    - bedrock_client: Amazon Bedrock client adapter implementing BedrockClientProtocol.
    - settings: Application configuration.
    """

    def __init__(
        self,
        modules: list[DecisionModuleProtocol],
        bedrock_client: BedrockClientProtocol,
        settings: Settings | None = None,
    ) -> None:
        self.modules = sorted(modules, key=lambda m: m.order)
        self.bedrock_client = bedrock_client
        self.settings = settings or get_settings()

    async def execute(
        self,
        user_input: str,
        request_id: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> CortexState:
        """Executes the complete decision and generation pipeline.

        Args:
            user_input: Raw user prompt or query.
            request_id: Optional correlation / request ID (generates UUID if omitted).
            metadata: Optional execution metadata, headers, or client attributes.

        Returns:
            The complete, finalized CortexState.
        """
        # Step 1: Initialize CortexState
        state = CortexState(
            request_id=request_id or str(uuid.uuid4()),
            user_input=user_input,
            metadata=metadata or {},
        )
        state.status = PipelineStatus.DECIDING

        logger.info(
            "Starting decision pipeline for request_id=%s [input_len=%d]",
            state.request_id,
            len(user_input),
        )

        # Step 2: Execute decision modules sequentially
        for module in self.modules:
            try:
                outcome = await module.decide(state)
            except Exception as exc:
                error_msg = f"Decision module '{module.module_id}' failed: {exc}"
                logger.error(error_msg, exc_info=True)
                state.mark_failed(reason=error_msg)
                return state

            # Step 3: Gating Checks after each decision
            # 3a. Risk/Guardrail Gate (Halt pipeline on CRITICAL risk or explicit REJECT)
            if state.risk_level == RiskLevel.CRITICAL or outcome.action == "REJECT":
                reason = f"Request blocked by {module.name}: action='{outcome.action}', risk='{state.risk_level}'"
                logger.warning("Pipeline halted at step %d: %s", module.order, reason)
                state.mark_blocked(reason=reason)
                return state

            # 3b. Confidence Gate
            confidence_threshold = self.settings.orchestrator.decision_confidence_threshold
            if outcome.confidence < confidence_threshold:
                reason = (
                    f"Decision confidence {outcome.confidence:.2f} in module '{module.module_id}' "
                    f"fell below required threshold of {confidence_threshold:.2f}"
                )
                logger.warning(reason)
                state.mark_failed(reason=reason)
                return state

        # Step 4: Decisions completed successfully
        state.status = PipelineStatus.DECIDED
        logger.info(
            "All %d decision modules completed successfully. Total decision latency: %.2f ms",
            len(self.modules),
            state.total_decision_latency_ms,
        )

        # Step 5: Amazon Bedrock Response Generation
        state.status = PipelineStatus.GENERATING
        selected_model = state.selected_model or self.settings.bedrock.default_model_id

        generation_input = GenerationInput(
            prompt=state.user_input,
            model_id=selected_model,
            metadata={
                "request_id": state.request_id,
                "intent": state.intent,
                "selected_agent": state.selected_agent,
                "tools": state.selected_tools,
            },
        )

        try:
            logger.info(
                "Handing off to Bedrock model='%s' for request_id=%s",
                selected_model,
                state.request_id,
            )
            generation_outcome = await self.bedrock_client.generate(generation_input)
            state.mark_completed(output=generation_outcome.content)
            logger.info("Pipeline completed successfully for request_id=%s", state.request_id)
            return state

        except Exception as exc:
            error_msg = f"Bedrock generation failed: {exc}"
            logger.error(error_msg, exc_info=True)
            state.mark_failed(reason=error_msg)
            return state
