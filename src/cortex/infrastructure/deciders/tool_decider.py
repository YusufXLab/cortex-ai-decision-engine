"""ToolSelectionDecider: AI-powered Tool & Capability Selection using Strands Decider 2B.

================================================================================
ARCHITECTURE & DESIGN RATIONALE: AGENT SELECTION VS TOOL SELECTION
================================================================================
1. The Core Conceptual Separation:
   - Agent Selection (Module 2) chooses:
     WHO will solve the request (The specialized persona, cognitive style, and domain expert).
     Examples: SoftwareEngineerAgent, DevOpsAgent, RiskAnalysisAgent, LiteratureReviewAgent.
   - Tool Selection (Module 5) chooses:
     WHAT external technical capabilities, I/O interfaces, or tools are required to fulfill it.
     Examples: WEB_SEARCH, CALCULATOR, FILE_READER, DATABASE, VISION, CODE_EXECUTION, RAG, MEMORY.

   An agent is NOT a tool:
   - A `DevOpsAgent` might require `["DATABASE", "CODE_EXECUTION"]` to inspect a cluster database.
   - A `LiteratureReviewAgent` might require `["WEB_SEARCH", "FILE_READER", "RAG"]`.
   - A `FinancialAnalysisAgent` might require `["FILE_READER", "CALCULATOR"]`.
   - Conversely, a request might need tools without requiring a specialized agent, or vice-versa.

2. Downstream Pipeline Impact of selected_tools:
   The list of tools selected here directly drives later decision modules:
   - Context Decider (Module 6): If `RAG` or `FILE_READER` is selected, fetches relevant
     vector chunks or document snippets to inject into prompt context. If `MEMORY` is selected,
     retrieves user conversation history.
   - Vision Requirement Decider (Module 7): Automatically activates vision pathways when
     `VISION` is present in `selected_tools`.
   - Complexity Decider (Module 8): Multi-tool pipelines (e.g. `["WEB_SEARCH", "CALCULATOR"]`)
     elevate task complexity tier from SIMPLE to MODERATE or COMPLEX.
   - Model Selection Decider (Module 9): Ensures the chosen foundation model natively supports
     function calling / tool use schemas (e.g. Claude 3.5 Sonnet Tool Use API).
   - Execution Optimization: Prevents allocating compute or initializing expensive tool
     sandboxes when `["NONE"]` is selected.

3. Developer Reusability Pattern:
   Inherit from BaseDecisionModule, build rich contextual prompt with upstream state
   (intent, agent, privacy, risk), invoke Strands Decider 2B with strictly allowed tools,
   validate list output, and raise domain exceptions on failure.
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
from cortex.domain.models.decision import DecisionOutcome, DecisionType
from cortex.domain.models.state import CortexState
from cortex.domain.models.tool import ToolDecisionResult, ToolType
from cortex.domain.protocols.strands_client import StrandsDeciderClientProtocol
from cortex.infrastructure.deciders.base import BaseDecisionModule
from cortex.infrastructure.deciders.client import StrandsDeciderClient

logger = get_logger("cortex.deciders.tools")

TOOL_SELECTION_SYSTEM_INSTRUCTION = """You are Strands Decider 2B, an ultra-fast, deterministic decision engine.
Your sole job is to determine which external technical tools or capabilities are required to solve the user's request.

Allowed Tools:
- NONE: No external tools required; pure language reasoning, standard conversation, or static synthesis is sufficient.
- WEB_SEARCH: Accessing current web information, real-time facts, recent news, or live internet data.
- CALCULATOR: Exact mathematical calculations, statistical computations, arithmetic formulas, or financial math.
- FILE_READER: Ingesting, parsing, or reading uploaded local documents, PDFs, CSVs, or text files.
- VISION: Inspecting and interpreting visual assets, diagrams, images, OCR scans, or flowcharts.
- DATABASE: Querying SQL databases, executing data queries, or accessing persistent record stores.
- CODE_EXECUTION: Running Python scripts, compiling code, or running commands in a secure execution sandbox.
- MEMORY: Recalling user preferences, past cross-session context, or historical conversation memory.
- RAG: Retrieving relevant domain knowledge chunks from indexed vector databases or corporate knowledge bases.

Critical Invariants:
1. Do NOT select the AI model.
2. Do NOT select the agent persona.
3. Do NOT evaluate privacy or risk levels. Those decisions have already been made.
4. Support selecting ONE OR MORE tools as a list (e.g. ["WEB_SEARCH"], ["FILE_READER", "CALCULATOR"], ["NONE"]).
5. If no tools are required, return ["NONE"].

Decision Rules:
1. Return a JSON object with:
   - "decision": A JSON array of approved tool strings (e.g. ["WEB_SEARCH", "CALCULATOR"]).
   - "confidence": Float between 0.0 and 1.0.
   - "reasoning": 1-sentence explanation of why these tools are needed.
2. Every tool string MUST be one of the approved tools listed above.
3. Output valid JSON in the format:
   {"decision": ["TOOL_1", "TOOL_2"], "confidence": <float>, "reasoning": "<string>"}
4. Do NOT generate conversational responses. ONLY output the JSON decision.
"""


class ToolSelectionDecider(BaseDecisionModule):
    """Production-ready Tool Selection Decider powered by Strands Decider 2B.

    Determines one or more external tools/capabilities required to fulfill the user request.
    """

    ALLOWED_TOOLS: list[str] = [t.value for t in ToolType]

    def __init__(
        self,
        client: StrandsDeciderClientProtocol | None = None,
        settings: Settings | None = None,
    ) -> None:
        super().__init__(
            module_id="decider_05_tools",
            name="Tool Selection Decider",
            decision_type=DecisionType.TOOL_SELECTION,
            order=5,
        )
        self.settings = settings or get_settings()
        self.client = client or StrandsDeciderClient(settings=self.settings)

    def _build_contextual_prompt(self, state: CortexState) -> str:
        """Assembles context from preceding decisions (Intent, Agent, Privacy, Risk, Metadata)."""
        intent_info = f"Intent: {state.intent}" if state.intent else "Intent: Unspecified"
        agent_info = f"Assigned Agent: {state.selected_agent}" if state.selected_agent else "Assigned Agent: Unspecified"
        privacy_info = f"Privacy Level: {state.privacy_level.value if state.privacy_level else 'Unspecified'}"
        risk_info = f"Risk Level: {state.risk_level.value if state.risk_level else 'Unspecified'}"

        metadata_str = ""
        if state.metadata:
            meta_items = [f"{k}={v}" for k, v in state.metadata.items() if not k.startswith("_")]
            if meta_items:
                metadata_str = f"Execution Metadata: [{', '.join(meta_items)}]\n"

        return (
            f"Preceding Decisions:\n"
            f"- {intent_info}\n"
            f"- {agent_info}\n"
            f"- {privacy_info}\n"
            f"- {risk_info}\n"
            f"{metadata_str}\n"
            f"User Prompt to Evaluate for Tool Requirements:\n"
            f"{state.user_input}"
        )

    def _parse_tools_list(self, raw_decision: Any) -> list[str]:
        """Safely parses and normalizes the tools decision into a list of strings."""
        if isinstance(raw_decision, list):
            tools = [str(t).strip().upper() for t in raw_decision]
        elif isinstance(raw_decision, str):
            cleaned = raw_decision.strip()
            if cleaned.startswith("[") and cleaned.endswith("]"):
                try:
                    parsed = json.loads(cleaned)
                    if isinstance(parsed, list):
                        tools = [str(t).strip().upper() for t in parsed]
                    else:
                        tools = [cleaned.upper()]
                except Exception:
                    tools = [t.strip().upper() for t in cleaned.strip("[]").split(",") if t.strip()]
            else:
                tools = [cleaned.upper()]
        else:
            raise DeciderInferenceError(
                f"Strands Decider 2B returned non-list 'decision' payload: {type(raw_decision).__name__}"
            )

        if not tools:
            return ["NONE"]

        # If list contains NONE alongside other tools, normalize by removing NONE
        if len(tools) > 1 and "NONE" in tools:
            tools = [t for t in tools if t != "NONE"]

        return tools

    async def _compute_decision(self, state: CortexState) -> DecisionOutcome:
        """Invokes Strands Decider 2B to evaluate required tools.

        Args:
            state: Current CortexState with upstream decisions.

        Returns:
            DecisionOutcome specifying the list of selected tools.

        Raises:
            DeciderInferenceError: On malformed output, unapproved tool names, or missing fields.
            DeciderTimeoutError: On inference timeout.
        """
        logger.info(
            "Evaluating tool requirements with Strands Decider 2B for request_id=%s [intent='%s', agent='%s']",
            state.request_id,
            state.intent,
            state.selected_agent,
        )

        contextual_input = self._build_contextual_prompt(state)
        start_time = time.perf_counter()

        try:
            raw_response = await self.client.decide(
                task="tool_selection",
                user_input=contextual_input,
                allowed_choices=self.ALLOWED_TOOLS,
                system_instruction=TOOL_SELECTION_SYSTEM_INSTRUCTION,
            )
        except (DeciderTimeoutError, DeciderInferenceError):
            raise
        except Exception as exc:
            logger.error("Unexpected error in Strands Decider 2B ToolSelectionDecider: %s", exc)
            raise DeciderInferenceError(
                f"Strands Decider 2B tool selection failed: {exc}"
            ) from exc

        latency_ms = (time.perf_counter() - start_time) * 1000.0

        decision_raw = raw_response.get("decision")
        if decision_raw is None:
            # Fallback check for "tools" key if model returned {"tools": [...]}
            decision_raw = raw_response.get("tools")

        confidence_raw = raw_response.get("confidence")
        reasoning_raw = raw_response.get("reasoning", "No reasoning provided by Strands Decider 2B.")

        if decision_raw is None:
            raise DeciderInferenceError(
                "Strands Decider 2B output missing required 'decision' or 'tools' key."
            )

        tools_list = self._parse_tools_list(decision_raw)

        # Validate that every single returned tool belongs to the approved list
        for tool in tools_list:
            if tool not in self.ALLOWED_TOOLS:
                raise DeciderInferenceError(
                    f"Strands Decider 2B selected unapproved tool '{tool}'. "
                    f"All tools must belong to: {self.ALLOWED_TOOLS}"
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

        selected_enum_tools = [ToolType(t) for t in tools_list]

        result = ToolDecisionResult(
            selected_tools=selected_enum_tools,
            confidence=confidence,
            reasoning=str(reasoning_raw),
            latency_ms=round(latency_ms, 2),
        )

        logger.info(
            "Strands Decider 2B selected tools=%s (confidence=%.2f) in %.2fms",
            tools_list,
            result.confidence,
            result.latency_ms,
        )

        return DecisionOutcome(
            module_id=self.module_id,
            decision_type=self.decision_type,
            action=", ".join(tools_list),
            confidence=result.confidence,
            reasoning=result.reasoning,
            metadata={
                "tools": tools_list,
                "latency_ms": result.latency_ms,
                "model_name": self.settings.strands.model_name,
            },
        )

    def apply_to_state(self, state: CortexState, outcome: DecisionOutcome) -> None:
        """Updates state.selected_tools with the list of chosen tool names."""
        tools = outcome.metadata.get("tools")
        if isinstance(tools, list):
            state.selected_tools = tools
        else:
            state.selected_tools = [t.strip() for t in outcome.action.split(",") if t.strip()]
