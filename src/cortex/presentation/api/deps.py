"""FastAPI dependency injection providers."""

import logging
from collections.abc import Generator
from fastapi import Depends, Header, HTTPException, status
from cortex.application.orchestrators.decision_orchestrator import DecisionOrchestrator
from cortex.core.config import Settings, get_settings
from cortex.core.logging import get_logger
from cortex.infrastructure.bedrock.client import MockBedrockClient
from cortex.infrastructure.deciders.registry import get_default_decision_modules


def get_app_settings() -> Settings:
    """Dependency providing the cached application settings."""
    return get_settings()


def get_request_logger() -> Generator[logging.Logger, None, None]:
    """Dependency providing a logger bound to the API request context."""
    logger = get_logger("cortex.api")
    yield logger


def get_orchestrator(
    settings: Settings = Depends(get_app_settings),
) -> DecisionOrchestrator:
    """Dependency injecting the configured Decision Orchestrator."""
    modules = get_default_decision_modules(
        default_model_id=settings.bedrock.default_model_id,
    )
    bedrock_client = MockBedrockClient()
    return DecisionOrchestrator(
        modules=modules,
        bedrock_client=bedrock_client,
        settings=settings,
    )


async def verify_api_key(
    x_api_key: str | None = Header(default=None, alias="X-API-Key"),
) -> str | None:
    """Validates the API key if enforced in production."""
    settings = get_settings()
    if settings.app.environment == "production":
        if not x_api_key or x_api_key != settings.app.cortex_api_key:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid or missing X-API-Key header",
            )
    return x_api_key
