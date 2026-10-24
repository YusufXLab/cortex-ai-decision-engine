"""Bedrock Client Protocol defining the interface for generation services.

Adheres to Interface Segregation and Dependency Inversion Principles.
"""

from typing import Protocol, runtime_checkable
from cortex.domain.models.generation import GenerationInput, GenerationOutcome


@runtime_checkable
class BedrockClientProtocol(Protocol):
    """Protocol for Amazon Bedrock generation clients.

    Strict Rule: Only handles text/content generation. Does not make routing decisions.
    """

    async def generate(self, input_data: GenerationInput) -> GenerationOutcome:
        """Call Bedrock foundation model to generate response text."""
        ...

    async def check_health(self) -> bool:
        """Verify connectivity to AWS Bedrock service."""
        ...
