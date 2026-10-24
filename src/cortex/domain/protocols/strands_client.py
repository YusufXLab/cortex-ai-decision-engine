"""Protocol for Strands Decider 2B client interactions.

Adheres to Dependency Inversion (DIP) and Interface Segregation (ISP).
"""

from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class StrandsDeciderClientProtocol(Protocol):
    """Protocol for sending decision requests to Strands Decider 2B."""

    async def decide(
        self,
        task: str,
        user_input: str,
        allowed_choices: list[str],
        system_instruction: str,
        parameters: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Invoke Strands Decider 2B model for classification or routing decision.

        Args:
            task: Name of the decision task (e.g. 'intent_classification').
            user_input: The user prompt or contextual state to evaluate.
            allowed_choices: Strict list of permissible choices/categories.
            system_instruction: Decision guideline and reasoning criteria.
            parameters: Optional temperature, top_p, or inference params.

        Returns:
            Dict containing parsed model output (e.g. decision, confidence, reasoning).

        Raises:
            DeciderTimeoutError: If the inference request times out.
            DeciderInferenceError: If the model output cannot be parsed or HTTP fails.
        """
        ...
