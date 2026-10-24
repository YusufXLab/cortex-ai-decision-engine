"""Orchestrator Protocol defining how decisions are coordinated before Bedrock generation."""

from typing import Protocol, runtime_checkable
from cortex.domain.models.decision import DecisionInput, DecisionOutcome
from cortex.domain.models.generation import GenerationOutcome


@runtime_checkable
class OrchestratorProtocol(Protocol):
    """Protocol for the Decision Orchestrator.

    Coordinates:
    1. Running Strands Decider 2B module(s) to make decisions.
    2. Evaluating decision outcomes (confidence, routing, guardrails).
    3. Handing off to Bedrock for response generation only if permitted.
    """

    async def execute_decisions(
        self,
        decision_input: DecisionInput,
        module_ids: list[str] | None = None,
    ) -> list[DecisionOutcome]:
        """Run decision modules through Strands Decider 2B."""
        ...

    async def orchestrate(
        self,
        prompt: str,
        context: dict[str, object] | None = None,
    ) -> tuple[list[DecisionOutcome], GenerationOutcome | None]:
        """Full pipeline: Decider -> Decision Evaluation -> Bedrock Generation."""
        ...
