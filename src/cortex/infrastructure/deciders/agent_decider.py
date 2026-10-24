"""AgentSelectionDecider: AI-powered specialized agent selection using Strands Decider 2B.

================================================================================
ARCHITECTURE & DESIGN RATIONALE: WHY AGENT SELECTION != INTENT CLASSIFICATION
================================================================================
1. The Core Conceptual Distinction:
   - Intent Classification (Decider 1) answers:
     "What is the user trying to do at a high level?" (e.g. CODING, RESEARCH, WRITING).
   - Agent Selection (Decider 2) answers:
     "Which specialized internal agent is best suited to solve this specific request?"
   - A 1-to-1 mapping between Intent and Agent renders multi-agent architecture redundant.
     In a production AI system, multiple specialized agents exist under the SAME intent.

   Example:
   - Intent = CODING
     - User asks: "Why is my asyncio task hanging at line 42?"
       -> DebuggingAgent (needs root cause & stack trace diagnosis)
     - User asks: "Deploy this FastAPI container to AWS ECS Fargate."
       -> DevOpsAgent (needs Terraform, Docker, ECS infrastructure expertise)
     - User asks: "Refactor this 500-line class to follow the Single Responsibility Principle."
       -> RefactoringAgent (needs design pattern restructuring expertise)
     - User asks: "Audit this cryptographic hash function for vulnerabilities."
       -> CodeReviewAgent (needs static security audit expertise)
     - User asks: "Implement a Red-Black Tree in Python."
       -> SoftwareEngineerAgent (needs algorithmic implementation expertise)

2. How This Module Works:
   - Inspects `CortexState.intent` determined by IntentDecider in Step 1.
   - Inspects the original `CortexState.user_input` and any execution metadata.
   - Dynamically resolves eligible candidate agents from `AgentRegistry`.
   - Passes the candidate profiles + user prompt to Strands Decider 2B.
   - Strands Decider 2B semantically evaluates which specialized agent is the optimal match.
   - Returns selected_agent, confidence, reasoning, and latency.
   - Mutates `state.selected_agent` and appends a `TraceStep` to `state.execution_trace`.

3. Extensibility via AgentRegistry:
   - Future developers can register new agents in `AgentRegistry` at any time.
   - Strands Decider 2B automatically incorporates newly registered agents into its
     candidate set without requiring modifications to DecisionOrchestrator or this decider!
================================================================================
"""

import time
from typing import Any
from cortex.application.agents.registry import AgentRegistry, get_default_agent_registry
from cortex.core.config import Settings, get_settings
from cortex.core.exceptions import (
    DeciderInferenceError,
    DeciderTimeoutError,
)
from cortex.core.logging import get_logger
from cortex.domain.models.agent import AgentDecisionResult
from cortex.domain.models.decision import DecisionOutcome, DecisionType
from cortex.domain.models.state import CortexState
from cortex.domain.protocols.strands_client import StrandsDeciderClientProtocol
from cortex.infrastructure.deciders.base import BaseDecisionModule
from cortex.infrastructure.deciders.client import StrandsDeciderClient

logger = get_logger("cortex.deciders.agent")


class AgentSelectionDecider(BaseDecisionModule):
    """Production-ready Agent Selection Decider powered by Strands Decider 2B.

    Evaluates the user's request against candidate agents registered for the classified intent.
    """

    def __init__(
        self,
        client: StrandsDeciderClientProtocol | None = None,
        registry: AgentRegistry | None = None,
        settings: Settings | None = None,
    ) -> None:
        super().__init__(
            module_id="decider_02_agent",
            name="Agent Selection Decider",
            decision_type=DecisionType.ROUTING,
            order=2,
        )
        self.settings = settings or get_settings()
        self.client = client or StrandsDeciderClient(settings=self.settings)
        self.registry = registry or get_default_agent_registry()

    def _build_agent_selection_prompt(self, candidates: list[Any], intent: str) -> str:
        """Constructs system instructions detailing the profiles of candidate agents."""
        candidate_profiles = []
        for agent in candidates:
            capabilities_str = ", ".join(agent.capabilities) if agent.capabilities else "general"
            candidate_profiles.append(
                f"- {agent.name}: {agent.description} [Capabilities: {capabilities_str}]"
            )

        profiles_formatted = "\n".join(candidate_profiles)

        return (
            "You are Strands Decider 2B, an ultra-fast, deterministic decision engine.\n"
            f"The user's query has been classified under the intent: '{intent}'.\n"
            "Your task is to select the SINGLE BEST specialized internal agent to handle this request.\n\n"
            "Eligible Candidate Agents:\n"
            f"{profiles_formatted}\n\n"
            "Decision Rules:\n"
            "1. Select the agent whose specialization, domain expertise, and capabilities most precisely fit the query.\n"
            "2. Do NOT choose randomly. Evaluate the technical domain nuances of the user's prompt.\n"
            "3. Return valid JSON in the format:\n"
            '   {"decision": "<EXACT_AGENT_NAME>", "confidence": <float>, "reasoning": "<1-sentence rationale>"}\n'
            "4. The 'decision' value MUST be one of the exact candidate agent names listed above.\n"
            "5. Do NOT generate conversational responses. ONLY output the decision."
        )

    async def _compute_decision(self, state: CortexState) -> DecisionOutcome:
        """Invokes Strands Decider 2B to pick the best specialized agent.

        Args:
            state: Current CortexState containing intent and user_input.

        Returns:
            DecisionOutcome specifying the selected specialized agent.

        Raises:
            DeciderInferenceError: If intent is missing, or model output is invalid.
            DeciderTimeoutError: If inference times out.
        """
        # Gating check: Intent classification MUST precede Agent Selection
        if not state.intent:
            raise DeciderInferenceError(
                "AgentSelectionDecider requires prior intent classification in state.intent. "
                "Step 1 (IntentDecider) must execute first."
            )

        # Retrieve candidate agents for this specific intent from the registry
        candidates = self.registry.get_candidates(state.intent)
        if not candidates:
            raise DeciderInferenceError(
                f"No candidate agents registered for intent '{state.intent}' in AgentRegistry."
            )

        candidate_names = [agent.name for agent in candidates]

        logger.info(
            "Evaluating agent selection with Strands Decider 2B for request_id=%s [intent='%s', candidates=%s]",
            state.request_id,
            state.intent,
            candidate_names,
        )

        system_instruction = self._build_agent_selection_prompt(candidates, state.intent)
        start_time = time.perf_counter()

        try:
            raw_response = await self.client.decide(
                task="agent_selection",
                user_input=state.user_input,
                allowed_choices=candidate_names,
                system_instruction=system_instruction,
            )
        except (DeciderTimeoutError, DeciderInferenceError):
            raise
        except Exception as exc:
            logger.error("Unexpected error in Strands Decider 2B agent selection: %s", exc)
            raise DeciderInferenceError(
                f"Strands Decider 2B agent selection call failed: {exc}"
            ) from exc

        latency_ms = (time.perf_counter() - start_time) * 1000.0

        decision_raw = raw_response.get("decision")
        confidence_raw = raw_response.get("confidence")
        reasoning_raw = raw_response.get("reasoning", "No reasoning provided by Strands Decider 2B.")

        if not decision_raw:
            raise DeciderInferenceError(
                "Strands Decider 2B output missing required 'decision' key in agent selection."
            )

        selected_agent = str(decision_raw).strip()
        if selected_agent not in candidate_names:
            raise DeciderInferenceError(
                f"Strands Decider 2B selected unapproved agent '{selected_agent}'. "
                f"Must be one of candidates: {candidate_names}"
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

        result = AgentDecisionResult(
            selected_agent=selected_agent,
            intent_context=state.intent,
            candidate_agents=candidate_names,
            confidence=confidence,
            reasoning=str(reasoning_raw),
            latency_ms=round(latency_ms, 2),
        )

        logger.info(
            "Strands Decider 2B selected agent='%s' (confidence=%.2f) for intent='%s' in %.2fms",
            result.selected_agent,
            result.confidence,
            result.intent_context,
            result.latency_ms,
        )

        return DecisionOutcome(
            module_id=self.module_id,
            decision_type=self.decision_type,
            action=result.selected_agent,
            confidence=result.confidence,
            reasoning=result.reasoning,
            metadata={
                "candidate_agents": result.candidate_agents,
                "intent_context": result.intent_context,
                "latency_ms": result.latency_ms,
                "model_name": self.settings.strands.model_name,
            },
        )

    def apply_to_state(self, state: CortexState, outcome: DecisionOutcome) -> None:
        """Updates state.selected_agent with the chosen specialized agent."""
        state.selected_agent = outcome.action
