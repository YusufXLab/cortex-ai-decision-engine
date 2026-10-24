"""IntentDecider: The first production-ready AI-powered Strands Decider 2B module.

================================================================================
ARCHITECTURE & DESIGN RATIONALE
================================================================================
1. Why Strands Decider 2B?
   - In production AI pipelines, using frontier LLMs (e.g. Claude 3.5 Sonnet, GPT-4)
     for early classification incurs massive latency (800ms-2s) and high token costs.
   - Traditional rule-based/regex approaches fail on semantic nuances and context.
   - Strands Decider 2B is a specialized 2B parameter decision model engineered
     exclusively for deterministic, sub-50ms classification, routing, and gating.
   - Strict Invariant: Strands Decider 2B ONLY makes the decision; it NEVER
     generates conversational or user-facing response tokens.

2. How This Module Works:
   - Receives the current CortexState.
   - Formulates a structured decision task for Strands Decider 2B with strictly
     constrained output choices: CHAT, WRITING, CODING, RESEARCH, ANALYSIS, PLANNING, VISION.
   - Calls the asynchronous StrandsDeciderClient.
   - Validates that the returned intent is strictly one of the allowed categories
     and that the confidence score is within [0.0, 1.0].
   - If the decider fails or returns an invalid response, raises a domain exception
     (DeciderInferenceError / DeciderTimeoutError). Never falls back silently.
   - Updates `state.intent` and appends a `TraceStep` to `state.execution_trace`.

3. Reusability Pattern for Developers:
   To implement any of the remaining 10 decision modules:
   a. Inherit from `BaseDecisionModule`.
   b. Set `module_id`, `name`, `decision_type`, and `order`.
   c. Inject `StrandsDeciderClientProtocol` via `__init__`.
   d. In `_compute_decision(state)`, call `self.client.decide(...)` with task-specific
      instructions and allowed choices.
   e. Validate output strictly; raise `DeciderInferenceError` on invalid payloads.
   f. In `apply_to_state(state, outcome)`, mutate the specific field on `CortexState`.
================================================================================
"""

import time
from typing import Any
from cortex.core.config import Settings, get_settings
from cortex.core.exceptions import (
    DeciderInferenceError,
    DeciderTimeoutError,
)
from cortex.core.logging import get_logger
from cortex.domain.models.decision import DecisionOutcome, DecisionType
from cortex.domain.models.intent import IntentCategory, IntentDecisionResult
from cortex.domain.models.state import CortexState
from cortex.domain.protocols.strands_client import StrandsDeciderClientProtocol
from cortex.infrastructure.deciders.base import BaseDecisionModule
from cortex.infrastructure.deciders.client import StrandsDeciderClient

logger = get_logger("cortex.deciders.intent")

# System instruction guiding Strands Decider 2B to pick exactly one category
INTENT_SYSTEM_INSTRUCTION = """You are Strands Decider 2B, an ultra-fast, deterministic decision engine.
Your sole job is to classify the user's request into EXACTLY ONE of the following intent categories:
- CHAT: Casual conversation, greetings, humor, or general chit-chat.
- WRITING: Drafting essays, emails, prose, letters, poems, or document editing.
- CODING: Software engineering, programming, debugging, scripts, algorithms, or infrastructure code.
- RESEARCH: Deep factual inquiries, literature search, synthesis of external documentation, or academic topics.
- ANALYSIS: Data analysis, logical deductions, risk assessments, financial evaluations, or comparative critiques.
- PLANNING: Creating roadmaps, schedules, step-by-step action plans, or itineraries.
- VISION: Requests involving visual assets, diagrams, images, photos, OCR, or multimodal perception.

Decision Rules:
1. Select the single best matching category.
2. Return a confidence score between 0.0 and 1.0.
3. Provide a concise 1-sentence reasoning statement.
4. Output valid JSON in the format:
   {"decision": "<CATEGORY>", "confidence": <float>, "reasoning": "<string>"}
5. Do NOT generate conversational responses. ONLY output the decision.
"""


class IntentDecider(BaseDecisionModule):
    """Production-ready Intent Decider powered by Strands Decider 2B.

    Classifies the user input into exactly one of the 7 allowed IntentCategory values.
    """

    ALLOWED_INTENTS: list[str] = [intent.value for intent in IntentCategory]

    def __init__(
        self,
        client: StrandsDeciderClientProtocol | None = None,
        settings: Settings | None = None,
    ) -> None:
        super().__init__(
            module_id="decider_01_intent",
            name="Intent Decider",
            decision_type=DecisionType.INTENT,
            order=1,
        )
        self.settings = settings or get_settings()
        self.client = client or StrandsDeciderClient(settings=self.settings)

    async def _compute_decision(self, state: CortexState) -> DecisionOutcome:
        """Invokes Strands Decider 2B to classify query intent.

        Args:
            state: Current CortexState with user_input and context.

        Returns:
            DecisionOutcome containing selected intent, confidence, and reasoning.

        Raises:
            DeciderInferenceError: If model response is invalid or missing required keys.
            DeciderTimeoutError: If inference times out.
        """
        logger.info(
            "Evaluating intent with Strands Decider 2B for request_id=%s",
            state.request_id,
        )

        start_time = time.perf_counter()

        try:
            raw_response = await self.client.decide(
                task="intent_classification",
                user_input=state.user_input,
                allowed_choices=self.ALLOWED_INTENTS,
                system_instruction=INTENT_SYSTEM_INSTRUCTION,
            )
        except (DeciderTimeoutError, DeciderInferenceError):
            # Re-raise domain exceptions directly for orchestrator
            raise
        except Exception as exc:
            logger.error("Unexpected error calling Strands Decider 2B: %s", exc)
            raise DeciderInferenceError(
                f"Strands Decider 2B call failed for intent classification: {exc}"
            ) from exc

        latency_ms = (time.perf_counter() - start_time) * 1000.0

        # Validate structure of the response
        decision_raw = raw_response.get("decision")
        confidence_raw = raw_response.get("confidence")
        reasoning_raw = raw_response.get("reasoning", "No reasoning provided by Strands Decider 2B.")

        if not decision_raw:
            raise DeciderInferenceError(
                "Strands Decider 2B output missing required 'decision' key."
            )

        # Validate that the selected intent is strictly one of the 7 allowed categories
        decision_str = str(decision_raw).strip().upper()
        if decision_str not in self.ALLOWED_INTENTS:
            raise DeciderInferenceError(
                f"Strands Decider 2B returned invalid intent '{decision_str}'. "
                f"Must be one of: {self.ALLOWED_INTENTS}"
            )

        # Validate confidence score
        try:
            confidence = float(confidence_raw)
            if not (0.0 <= confidence <= 1.0):
                raise ValueError()
        except (TypeError, ValueError) as err:
            raise DeciderInferenceError(
                f"Strands Decider 2B returned invalid confidence '{confidence_raw}'. "
                "Must be a float between 0.0 and 1.0."
            ) from err

        selected_intent = IntentCategory(decision_str)

        # Construct strongly typed result
        result = IntentDecisionResult(
            selected_intent=selected_intent,
            confidence=confidence,
            reasoning=str(reasoning_raw),
            latency_ms=round(latency_ms, 2),
        )

        logger.info(
            "Strands Decider 2B classified intent='%s' confidence=%.2f in %.2fms",
            result.selected_intent.value,
            result.confidence,
            result.latency_ms,
        )

        return DecisionOutcome(
            module_id=self.module_id,
            decision_type=self.decision_type,
            action=result.selected_intent.value,
            confidence=result.confidence,
            reasoning=result.reasoning,
            metadata={
                "latency_ms": result.latency_ms,
                "model_name": self.settings.strands.model_name,
            },
        )

    def apply_to_state(self, state: CortexState, outcome: DecisionOutcome) -> None:
        """Applies the classified intent to CortexState."""
        state.intent = outcome.action
