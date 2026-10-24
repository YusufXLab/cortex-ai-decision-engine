"""Base class for all Strands Decider 2B decision modules.

Follows the Template Method pattern and Single Responsibility Principle.
"""

from abc import ABC, abstractmethod
import time
from cortex.core.logging import get_logger
from cortex.domain.models.decision import DecisionOutcome, DecisionType
from cortex.domain.models.state import CortexState, TraceStep
from cortex.domain.protocols.decider import DecisionModuleProtocol

logger = get_logger("cortex.deciders")


class BaseDecisionModule(ABC, DecisionModuleProtocol):
    """Abstract base class for all 11 Strands Decider 2B decision modules.

    Strict Invariant:
    Modules MUST only make decisions and NEVER generate responses.
    """

    def __init__(self, module_id: str, name: str, decision_type: DecisionType, order: int) -> None:
        self._module_id = module_id
        self._name = name
        self._decision_type = decision_type
        self._order = order

    @property
    def module_id(self) -> str:
        return self._module_id

    @property
    def name(self) -> str:
        return self._name

    @property
    def decision_type(self) -> DecisionType:
        return self._decision_type

    @property
    def order(self) -> int:
        return self._order

    async def decide(self, state: CortexState) -> DecisionOutcome:
        """Executes the decision logic, measures duration, and returns the decision outcome."""
        start_time = time.perf_counter()
        logger.debug("Executing decision step %d: %s (%s)", self.order, self.name, self.module_id)

        try:
            outcome = await self._compute_decision(state)
            duration_ms = (time.perf_counter() - start_time) * 1000.0

            # Record in execution trace
            trace_step = TraceStep(
                step_number=self.order,
                module_id=self.module_id,
                decision_name=self.name,
                action=outcome.action,
                confidence=outcome.confidence,
                reasoning=outcome.reasoning,
                duration_ms=round(duration_ms, 2),
                success=True,
            )
            state.add_trace_step(trace_step)

            # Apply the specific field update on the state
            self.apply_to_state(state, outcome)
            return outcome

        except Exception as exc:
            duration_ms = (time.perf_counter() - start_time) * 1000.0
            trace_step = TraceStep(
                step_number=self.order,
                module_id=self.module_id,
                decision_name=self.name,
                action="ERROR",
                confidence=0.0,
                duration_ms=round(duration_ms, 2),
                success=False,
                error_message=str(exc),
            )
            state.add_trace_step(trace_step)
            raise

    @abstractmethod
    async def _compute_decision(self, state: CortexState) -> DecisionOutcome:
        """Implement decision evaluation (calls Strands Decider 2B)."""
        ...

    @abstractmethod
    def apply_to_state(self, state: CortexState, outcome: DecisionOutcome) -> None:
        """Update the appropriate field on CortexState based on the decision outcome."""
        ...
