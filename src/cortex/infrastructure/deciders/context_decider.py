"""ContextSelectionDecider: AI-powered Context Source Selection using Strands Decider 2B.

================================================================================
ARCHITECTURE & DESIGN RATIONALE: TOOL SELECTION VS CONTEXT SELECTION
================================================================================
1. The Core Conceptual Separation:
   - Tool Selection (Module 5) answers:
     "What external capabilities or I/O interfaces are required?"
     (e.g. WEB_SEARCH, FILE_READER, RAG, DATABASE).
   - Context Selection (Module 6) answers:
     "What specific information sources should be retrieved and injected into the prompt context?"
     (e.g. UPLOADED_FILES, CHAT_HISTORY, USER_MEMORY, RAG_KNOWLEDGE_BASE, SYSTEM_PROMPT_ONLY).

   A tool is an execution interface; a context source is a payload of knowledge:
   - Even if `WEB_SEARCH` was selected as a tool, Context Selection determines whether
     `WEB_RESULTS` need to be synthesized into context, or if cached `USER_MEMORY` is sufficient.
   - If the user asks: "Write a poem about relativity in Shakespearean English", no tools
     are needed (`["NONE"]`), and context selection chooses `["SYSTEM_PROMPT_ONLY"]` - saving
     thousands of tokens by suppressing vector search, file ingestion, and chat history.

2. Economic & Latency Optimizations of Context Selection:
   - Token Window Reduction: In frontier models (Bedrock Claude 3.5 Sonnet), injecting 50k
     tokens of unpruned context costs money and degrades time-to-first-token (TTFT).
     Selecting strictly necessary context sources avoids prompt bloat.
   - Latency Optimization: Bypasses expensive vector DB queries, network calls to web APIs,
     or disk I/O when `SYSTEM_PROMPT_ONLY` is selected.
   - Hallucination Prevention: Suppressing irrelevant background context prevents the model
     from suffering from "lost-in-the-middle" attention degradation.

3. Developer Reusability Pattern:
   Inherit from BaseDecisionModule, build rich contextual prompt incorporating upstream state
   (intent, agent, privacy, risk, selected_tools), call Strands Decider 2B with strictly allowed
   context sources, validate list output, and raise domain exceptions without silent fallbacks.
================================================================================
"""

import json
import time
from typing import Any
from cortex.core.config import Settings, get_settings
from cortex.core.exceptions import (
    DeciderInferenceError,
    DeciderTimeoutError,
)
from cortex.core.logging import get_logger
from cortex.domain.models.context import ContextDecisionResult, ContextSourceType
from cortex.domain.models.decision import DecisionOutcome, DecisionType
from cortex.domain.models.state import CortexState
from cortex.domain.protocols.strands_client import StrandsDeciderClientProtocol
from cortex.infrastructure.deciders.base import BaseDecisionModule
from cortex.infrastructure.deciders.client import StrandsDeciderClient

logger = get_logger("cortex.deciders.context")

CONTEXT_SELECTION_SYSTEM_INSTRUCTION = """You are Strands Decider 2B, an ultra-fast, deterministic decision engine.
Your sole job is to determine which information sources should be retrieved and injected into the foundation model prompt context.

Allowed Context Sources:
- NONE: No external context needed. Equivalent to SYSTEM_PROMPT_ONLY.
- CHAT_HISTORY: Conversational turns from preceding interactions in the current session.
- USER_MEMORY: User-specific preferences, profile parameters, or cross-session persistent facts.
- UPLOADED_FILES: Local files, documents, PDFs, or CSV spreadsheets provided by the user.
- RAG_KNOWLEDGE_BASE: Indexed vector embeddings and documentation chunks from enterprise knowledge bases.
- WEB_RESULTS: External real-time search snippets or web page text retrieved from internet search.
- DATABASE_RESULTS: Records, schemas, or query results retrieved from transactional or analytical databases.
- SYSTEM_PROMPT_ONLY: Pure standalone prompt evaluation with zero injected external knowledge or history.

Critical Invariants:
1. Do NOT choose tools.
2. Do NOT choose the AI foundation model.
3. Do NOT choose the agent persona. Those decisions have already been made.
4. Support selecting ONE OR MORE context sources as a list (e.g. ["UPLOADED_FILES"], ["CHAT_HISTORY", "USER_MEMORY"], ["SYSTEM_PROMPT_ONLY"]).
5. If no external context is required, select ["SYSTEM_PROMPT_ONLY"].

Decision Rules:
1. Return a JSON object with:
   - "decision": A JSON array of approved context source strings (e.g. ["UPLOADED_FILES", "RAG_KNOWLEDGE_BASE"]).
   - "confidence": Float between 0.0 and 1.0.
   - "reasoning": 1-sentence explanation of why these context sources are needed.
2. Every item in the list MUST be one of the approved context sources above.
3. Output valid JSON in the format:
   {"decision": ["SOURCE_1", "SOURCE_2"], "confidence": <float>, "reasoning": "<string>"}
4. Do NOT generate conversational responses. ONLY output the decision.
"""


class ContextSelectionDecider(BaseDecisionModule):
    """Production-ready Context Selection Decider powered by Strands Decider 2B.

    Determines one or more context sources to load before generation.
    """

    ALLOWED_CONTEXT_SOURCES: list[str] = [source.value for source in ContextSourceType]

    def __init__(
        self,
        client: StrandsDeciderClientProtocol | None = None,
        settings: Settings | None = None,
    ) -> None:
        super().__init__(
            module_id="decider_06_context",
            name="Context Selection Decider",
            decision_type=DecisionType.CONTEXT_PRUNING,
            order=6,
        )
        self.settings = settings or get_settings()
        self.client = client or StrandsDeciderClient(settings=self.settings)

    def _build_contextual_prompt(self, state: CortexState) -> str:
        """Assembles rich context from preceding decisions (Intent, Agent, Privacy, Risk, Tools, Metadata)."""
        intent_info = f"Intent: {state.intent}" if state.intent else "Intent: Unspecified"
        agent_info = f"Assigned Agent: {state.selected_agent}" if state.selected_agent else "Assigned Agent: Unspecified"
        privacy_info = f"Privacy Level: {state.privacy_level.value if state.privacy_level else 'Unspecified'}"
        risk_info = f"Risk Level: {state.risk_level.value if state.risk_level else 'Unspecified'}"
        tools_info = f"Selected Tools: {state.selected_tools}" if state.selected_tools else "Selected Tools: []"

        metadata_str = ""
        if state.metadata:
            meta_items = [f"{k}={v}" for k, v in state.metadata.items() if not k.startswith("_")]
            if meta_items:
                metadata_str = f"Execution Metadata: [{', '.join(meta_items)}]\n"

        return (
            f"Preceding Pipeline Decisions:\n"
            f"- {intent_info}\n"
            f"- {agent_info}\n"
            f"- {privacy_info}\n"
            f"- {risk_info}\n"
            f"- {tools_info}\n"
            f"{metadata_str}\n"
            f"User Prompt to Evaluate for Context Ingestion:\n"
            f"{state.user_input}"
        )

    def _parse_context_list(self, raw_decision: Any) -> list[str]:
        """Safely parses and normalizes the context decision into a list of strings."""
        if isinstance(raw_decision, list):
            sources = [str(s).strip().upper() for s in raw_decision]
        elif isinstance(raw_decision, str):
            cleaned = raw_decision.strip()
            if cleaned.startswith("[") and cleaned.endswith("]"):
                try:
                    parsed = json.loads(cleaned)
                    if isinstance(parsed, list):
                        sources = [str(s).strip().upper() for s in parsed]
                    else:
                        sources = [cleaned.upper()]
                except Exception:
                    sources = [s.strip().upper() for s in cleaned.strip("[]").split(",") if s.strip()]
            else:
                sources = [cleaned.upper()]
        else:
            raise DeciderInferenceError(
                f"Strands Decider 2B returned non-list 'decision' payload: {type(raw_decision).__name__}"
            )

        if not sources:
            return ["SYSTEM_PROMPT_ONLY"]

        # If list contains SYSTEM_PROMPT_ONLY or NONE alongside other sources, normalize
        if len(sources) > 1:
            sources = [s for s in sources if s not in ("SYSTEM_PROMPT_ONLY", "NONE")]

        return sources

    async def _compute_decision(self, state: CortexState) -> DecisionOutcome:
        """Invokes Strands Decider 2B to evaluate required context sources.

        Args:
            state: Current CortexState with upstream decisions.

        Returns:
            DecisionOutcome specifying the list of selected context sources.

        Raises:
            DeciderInferenceError: On malformed output, unapproved context names, or missing fields.
            DeciderTimeoutError: On inference timeout.
        """
        logger.info(
            "Evaluating context sources with Strands Decider 2B for request_id=%s [intent='%s', tools=%s]",
            state.request_id,
            state.intent,
            state.selected_tools,
        )

        contextual_input = self._build_contextual_prompt(state)
        start_time = time.perf_counter()

        try:
            raw_response = await self.client.decide(
                task="context_selection",
                user_input=contextual_input,
                allowed_choices=self.ALLOWED_CONTEXT_SOURCES,
                system_instruction=CONTEXT_SELECTION_SYSTEM_INSTRUCTION,
            )
        except (DeciderTimeoutError, DeciderInferenceError):
            raise
        except Exception as exc:
            logger.error("Unexpected error in Strands Decider 2B ContextSelectionDecider: %s", exc)
            raise DeciderInferenceError(
                f"Strands Decider 2B context selection failed: {exc}"
            ) from exc

        latency_ms = (time.perf_counter() - start_time) * 1000.0

        decision_raw = raw_response.get("decision")
        if decision_raw is None:
            # Fallback check for "context_sources" or "sources" key
            decision_raw = raw_response.get("context_sources") or raw_response.get("sources")

        confidence_raw = raw_response.get("confidence")
        reasoning_raw = raw_response.get("reasoning", "No reasoning provided by Strands Decider 2B.")

        if decision_raw is None:
            raise DeciderInferenceError(
                "Strands Decider 2B output missing required 'decision' key in context selection."
            )

        sources_list = self._parse_context_list(decision_raw)

        # Validate that every single returned source belongs to the approved list
        for source in sources_list:
            if source not in self.ALLOWED_CONTEXT_SOURCES:
                raise DeciderInferenceError(
                    f"Strands Decider 2B selected unapproved context source '{source}'. "
                    f"All context sources must belong to: {self.ALLOWED_CONTEXT_SOURCES}"
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

        selected_enum_sources = [ContextSourceType(s) for s in sources_list]

        result = ContextDecisionResult(
            selected_context_sources=selected_enum_sources,
            confidence=confidence,
            reasoning=str(reasoning_raw),
            latency_ms=round(latency_ms, 2),
        )

        logger.info(
            "Strands Decider 2B selected context sources=%s (confidence=%.2f) in %.2fms",
            sources_list,
            result.confidence,
            result.latency_ms,
        )

        return DecisionOutcome(
            module_id=self.module_id,
            decision_type=self.decision_type,
            action=", ".join(sources_list),
            confidence=result.confidence,
            reasoning=result.reasoning,
            metadata={
                "selected_context": sources_list,
                "latency_ms": result.latency_ms,
                "model_name": self.settings.strands.model_name,
            },
        )

    def apply_to_state(self, state: CortexState, outcome: DecisionOutcome) -> None:
        """Updates state.selected_context with the list of chosen context sources."""
        sources = outcome.metadata.get("selected_context")
        if isinstance(sources, list):
            state.selected_context = sources
        else:
            state.selected_context = [s.strip() for s in outcome.action.split(",") if s.strip()]


# Alias to maintain backwards compatibility with existing imports
ContextDecider = ContextSelectionDecider
