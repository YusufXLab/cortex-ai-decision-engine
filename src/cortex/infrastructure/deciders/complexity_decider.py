"""ComplexityDecider: Estimates cognitive and computational complexity using Strands Decider 2B.

================================================================================
ARCHITECTURE & DESIGN RATIONALE: COMPLEXITY VS TOOL SELECTION & MODEL ROUTING
================================================================================
1. The Core Conceptual Separation:
   - Tool Selection (Module 5) answers:
     "What external capabilities or I/O tools does the system need?"
     (e.g., TOOL_SELECTION selects ["WEB_SEARCH", "CALCULATOR"]).
   - ComplexityDecider (Module 8) answers:
     "How computationally and cognitively complex is this request?"
     (e.g., TRIVIAL, SIMPLE, COMPLEX, EXPERT).
   - Model Routing (Module 9) answers:
     "Which foundation model (e.g., Claude 3.5 Sonnet, Claude 3 Haiku, Titan Express)
      is best suited to generate the response?"

   Complexity is a Workload Metric, NOT a Model Choice:
   - A task might need zero tools (`["NONE"]`) but be cognitively `EXPERT` (e.g., designing
     a formal mathematical proof of Byzantine Fault Tolerance).
   - Conversely, a task might invoke a tool (`CALCULATOR` for "2 + 2") yet be `TRIVIAL`.
   - Model Routing (Module 9) uses Complexity alongside Privacy, Risk, and Cost Strategy
     to pick the optimal Bedrock model. Decoupling Complexity from Model Routing prevents
     hardcoding model families and allows dynamic model swapping without re-architecting decisions.

2. Influence on Latency, Token Budget, Compute Allocation & Downstream Routing:
   - TRIVIAL:
     * Cognitive Workload: Basic arithmetic, casual greetings, single-fact recall.
     * Token Budget: Minimal (< 256 tokens).
     * Compute Allocation: Lowest compute; zero chain-of-thought needed.
     * Downstream Impact: Routes to ultra-fast, low-cost models (Haiku, Titan Lite) with < 200ms TTFT.
   - SIMPLE:
     * Cognitive Workload: Single-turn drafting, standard email, concise code snippet, short explanation.
     * Token Budget: Moderate (< 1,024 tokens).
     * Compute Allocation: Standard inference; direct completion.
     * Downstream Impact: Balanced routing (Claude 3 Haiku or balanced Bedrock model).
   - COMPLEX:
     * Cognitive Workload: Multi-step reasoning, multiple tool coordination, deep file parsing,
       comparative synthesis across web results and RAG documents.
     * Token Budget: High (2,048 - 4,096 tokens).
     * Compute Allocation: Elevated compute; activates step decomposition and structured JSON tool calls.
     * Downstream Impact: Routes to high-reasoning models (Claude 3.5 Sonnet).
   - EXPERT:
     * Cognitive Workload: Large multi-document synthesis, cross-modal rich PDF reasoning,
       distributed system architecture redesign, security vulnerability threat modeling.
     * Token Budget: Maximum (4,096 - 8,192+ tokens).
     * Compute Allocation: Maximum compute; extended reasoning context, rigorous self-consistency.
     * Downstream Impact: Routes exclusively to top-tier frontier models (Claude 3.5 Sonnet / Opus).
================================================================================
"""

import time
from cortex.core.config import Settings, get_settings
from cortex.core.exceptions import (
    DeciderInferenceError,
    DeciderTimeoutError,
)
from cortex.core.logging import get_logger
from cortex.domain.models.complexity import ComplexityDecisionResult, ComplexityLevel
from cortex.domain.models.decision import DecisionOutcome, DecisionType
from cortex.domain.models.state import CortexState
from cortex.domain.protocols.strands_client import StrandsDeciderClientProtocol
from cortex.infrastructure.deciders.base import BaseDecisionModule
from cortex.infrastructure.deciders.client import StrandsDeciderClient

logger = get_logger("cortex.deciders.complexity")

COMPLEXITY_SYSTEM_INSTRUCTION = """You are Strands Decider 2B, an ultra-fast, deterministic decision engine.
Your sole job is to estimate how much reasoning, orchestration, and computational workload the request requires.

Do NOT estimate privacy, risk, or model selection. Those decisions belong to other modules.

Allowed Complexity Levels:
- TRIVIAL: Basic arithmetic, greeting, single factual question, low cognitive effort.
- SIMPLE: Email writing, small code snippet, short explanation, direct single-turn response.
- COMPLEX: Multi-step reasoning, multiple tools, file analysis, web search + summarization.
- EXPERT: Large multi-document workflows, heavy orchestration, multiple context sources, long reasoning chains, mixed multimodal workflows.

Critical Invariants:
1. Return exactly ONE complexity level from the allowed list above.
2. Do NOT choose the AI foundation model.
3. Do NOT evaluate safety or privacy.
4. Do NOT generate conversational responses. ONLY output the decision.
5. Output valid JSON in the format:
   {"decision": "<EXACT_COMPLEXITY_LEVEL>", "confidence": <float>, "reasoning": "<string>"}
"""


class ComplexityDecider(BaseDecisionModule):
    """Production-ready Complexity Decider powered by Strands Decider 2B.

    Runs after VisionRequirementDecider (Order 8).
    Estimates the computational and cognitive reasoning workload of the user request.
    """

    ALLOWED_COMPLEXITY_LEVELS: list[str] = [
        ComplexityLevel.TRIVIAL.value,
        ComplexityLevel.SIMPLE.value,
        ComplexityLevel.COMPLEX.value,
        ComplexityLevel.EXPERT.value,
    ]

    def __init__(
        self,
        client: StrandsDeciderClientProtocol | None = None,
        settings: Settings | None = None,
    ) -> None:
        super().__init__(
            module_id="decider_08_complexity",
            name="Complexity Decider",
            decision_type=DecisionType.COMPLEXITY,
            order=8,
        )
        self.settings = settings or get_settings()
        self.client = client or StrandsDeciderClient(settings=self.settings)

    def _build_contextual_prompt(self, state: CortexState) -> str:
        """Assembles rich context from preceding decisions (1 through 7) and metadata."""
        intent_info = f"Intent: {state.intent}" if state.intent else "Intent: Unspecified"
        agent_info = f"Assigned Agent: {state.selected_agent}" if state.selected_agent else "Assigned Agent: Unspecified"
        privacy_info = f"Privacy Level: {state.privacy_level.value if state.privacy_level else 'Unspecified'}"
        risk_info = f"Risk Level: {state.risk_level.value if state.risk_level else 'Unspecified'}"
        tools_info = f"Selected Tools: {state.selected_tools}" if state.selected_tools else "Selected Tools: []"
        context_info = f"Selected Context: {state.selected_context}" if state.selected_context else "Selected Context: []"
        vision_info = (
            f"Vision Requirement: {state.vision_requirement.value if state.vision_requirement else 'NONE'}"
        )

        metadata_str = ""
        if state.metadata:
            meta_items = [f"{k}={v}" for k, v in state.metadata.items() if not k.startswith("_")]
            if meta_items:
                metadata_str = f"Execution Metadata: [{', '.join(meta_items)}]\n"

        return (
            f"Preceding Pipeline Decisions (Steps 1-7):\n"
            f"- {intent_info}\n"
            f"- {agent_info}\n"
            f"- {privacy_info}\n"
            f"- {risk_info}\n"
            f"- {tools_info}\n"
            f"- {context_info}\n"
            f"- {vision_info}\n"
            f"{metadata_str}\n"
            f"User Prompt to Evaluate for Computational & Cognitive Complexity:\n"
            f"{state.user_input}"
        )

    async def _compute_decision(self, state: CortexState) -> DecisionOutcome:
        """Invokes Strands Decider 2B to evaluate required reasoning complexity.

        Args:
            state: Current CortexState with upstream decisions.

        Returns:
            DecisionOutcome specifying the exact complexity level.

        Raises:
            DeciderInferenceError: On malformed output, unapproved classification, or missing fields.
            DeciderTimeoutError: On inference timeout.
        """
        logger.info(
            "Evaluating complexity with Strands Decider 2B for request_id=%s [intent='%s', tools=%s, vision=%s]",
            state.request_id,
            state.intent,
            state.selected_tools,
            state.vision_requirement,
        )

        contextual_input = self._build_contextual_prompt(state)
        start_time = time.perf_counter()

        try:
            raw_response = await self.client.decide(
                task="complexity",
                user_input=contextual_input,
                allowed_choices=self.ALLOWED_COMPLEXITY_LEVELS,
                system_instruction=COMPLEXITY_SYSTEM_INSTRUCTION,
            )
        except (DeciderTimeoutError, DeciderInferenceError):
            raise
        except Exception as exc:
            logger.error("Unexpected error in Strands Decider 2B ComplexityDecider: %s", exc)
            raise DeciderInferenceError(
                f"Strands Decider 2B complexity evaluation failed: {exc}"
            ) from exc

        latency_ms = (time.perf_counter() - start_time) * 1000.0

        decision_raw = raw_response.get("decision")
        if decision_raw is None:
            # Fallback check for "complexity_level" or "complexity" key
            decision_raw = raw_response.get("complexity_level") or raw_response.get("complexity")

        confidence_raw = raw_response.get("confidence")
        reasoning_raw = raw_response.get("reasoning", "No reasoning provided by Strands Decider 2B.")

        if decision_raw is None:
            raise DeciderInferenceError(
                "Strands Decider 2B output missing required 'decision' key in complexity evaluation."
            )

        decision_str = str(decision_raw).strip().upper()

        if decision_str not in self.ALLOWED_COMPLEXITY_LEVELS:
            raise DeciderInferenceError(
                f"Strands Decider 2B returned unknown complexity level '{decision_raw}'. "
                f"Must be exactly one of: {self.ALLOWED_COMPLEXITY_LEVELS}"
            )

        try:
            confidence = float(confidence_raw)
            if not (0.0 <= confidence <= 1.0):
                raise ValueError()
        except (TypeError, ValueError) as err:
            raise DeciderInferenceError(
                f"Strands Decider 2B returned invalid confidence '{confidence_raw}'. "
                "Must be a float between 0.0 and 1.0."
            ) from err

        complexity_tier = ComplexityLevel(decision_str)

        result = ComplexityDecisionResult(
            complexity_level=complexity_tier,
            confidence=confidence,
            reasoning=str(reasoning_raw),
            latency_ms=round(latency_ms, 2),
        )

        logger.info(
            "Strands Decider 2B determined complexity=%s (confidence=%.2f) in %.2fms",
            result.complexity_level.value,
            result.confidence,
            result.latency_ms,
        )

        return DecisionOutcome(
            module_id=self.module_id,
            decision_type=self.decision_type,
            action=result.complexity_level.value,
            confidence=result.confidence,
            reasoning=result.reasoning,
            metadata={
                "complexity_level": result.complexity_level.value,
                "latency_ms": result.latency_ms,
                "model_name": self.settings.strands.model_name,
            },
        )

    def apply_to_state(self, state: CortexState, outcome: DecisionOutcome) -> None:
        """Updates state.complexity_level and state.complexity."""
        complexity_val = outcome.metadata.get("complexity_level") or outcome.action
        state.complexity_level = ComplexityLevel(complexity_val)
        state.complexity = state.complexity_level


# Alias to maintain backwards compatibility
ComplexityAssessmentDecider = ComplexityDecider
