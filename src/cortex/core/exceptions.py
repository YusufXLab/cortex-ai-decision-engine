"""Domain and application exception hierarchy.

Enforces clean separation between decider errors, generation errors, and orchestrator errors.
"""

from typing import Any


class CortexException(Exception):
    """Base exception for all Cortex AI Decision Engine errors."""

    def __init__(
        self,
        message: str,
        code: str = "INTERNAL_ERROR",
        status_code: int = 500,
        details: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.message = message
        self.code = code
        self.status_code = status_code
        self.details = details or {}


class ConfigurationError(CortexException):
    """Raised when environment or runtime configuration is invalid."""

    def __init__(self, message: str, details: dict[str, Any] | None = None) -> None:
        super().__init__(
            message=message,
            code="CONFIG_ERROR",
            status_code=500,
            details=details,
        )


class DeciderException(CortexException):
    """Base exception for Strands Decider 2B decision plane failures."""

    def __init__(
        self,
        message: str,
        code: str = "DECIDER_ERROR",
        status_code: int = 502,
        details: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message=message, code=code, status_code=status_code, details=details)


class DeciderInferenceError(DeciderException):
    """Raised when Strands Decider 2B inference fails to produce a valid decision."""

    def __init__(self, message: str, details: dict[str, Any] | None = None) -> None:
        super().__init__(
            message=message,
            code="DECIDER_INFERENCE_FAILED",
            status_code=502,
            details=details,
        )


class DeciderTimeoutError(DeciderException):
    """Raised when Strands Decider 2B call times out."""

    def __init__(self, message: str = "Decider request timed out", details: dict[str, Any] | None = None) -> None:
        super().__init__(
            message=message,
            code="DECIDER_TIMEOUT",
            status_code=504,
            details=details,
        )


class BedrockException(CortexException):
    """Base exception for Amazon Bedrock generation plane failures."""

    def __init__(
        self,
        message: str,
        code: str = "BEDROCK_ERROR",
        status_code: int = 502,
        details: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message=message, code=code, status_code=status_code, details=details)


class BedrockRateLimitError(BedrockException):
    """Raised when Amazon Bedrock invokes throttling limits."""

    def __init__(self, message: str = "Bedrock rate limit exceeded", details: dict[str, Any] | None = None) -> None:
        super().__init__(
            message=message,
            code="BEDROCK_THROTTLED",
            status_code=429,
            details=details,
        )


class OrchestrationException(CortexException):
    """Base exception for errors in the Decision Orchestrator pipeline."""

    def __init__(
        self,
        message: str,
        code: str = "ORCHESTRATION_ERROR",
        status_code: int = 500,
        details: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message=message, code=code, status_code=status_code, details=details)


class UnauthorizedError(CortexException):
    """Raised when request lacks required authentication credentials."""

    def __init__(self, message: str = "Invalid or missing API Key") -> None:
        super().__init__(
            message=message,
            code="UNAUTHORIZED",
            status_code=401,
        )
