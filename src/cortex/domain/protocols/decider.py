"""Decider Protocol defining the interface for Strands Decider 2B modules.

Adheres to Interface Segregation (ISP) and Dependency Inversion (DIP) Principles.
"""

from typing import Protocol, runtime_checkable
from cortex.domain.models.decision import DecisionOutcome, DecisionType
from cortex.domain.models.state import CortexState


@runtime_checkable
class DecisionModuleProtocol(Protocol):
    """Protocol for all 11 Strands Decider 2B decision modules.

    Each module evaluates the evolving CortexState, produces a DecisionOutcome,
    and contributes to updating the state before passing to the next module.
    """

    @property
    def module_id(self) -> str:
        """Unique identifier for this decision module (e.g. 'decider_01_intent')."""
        ...

    @property
    def name(self) -> str:
        """Human-readable name of the decision module."""
        ...

    @property
    def decision_type(self) -> DecisionType:
        """Category of decision this module handles."""
        ...

    @property
    def order(self) -> int:
        """Sequential execution order (1 through 11)."""
        ...

    async def decide(self, state: CortexState) -> DecisionOutcome:
        """Evaluate current state and return a decision outcome.

        Strict Architecture Invariant:
        Must NOT generate responses. Strictly outputs decisions.
        """
        ...
