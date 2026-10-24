"""Health and readiness check endpoints."""

from fastapi import APIRouter, Depends, status
from cortex.core.config import Settings
from cortex.presentation.api.deps import get_app_settings
from cortex.presentation.schemas.common import APIResponse
from cortex.presentation.schemas.health import ComponentStatus, HealthResponse

router = APIRouter(prefix="/health", tags=["Health & Diagnostics"])


@router.get(
    "",
    response_model=APIResponse[HealthResponse],
    status_code=status.HTTP_200_OK,
    summary="Aggregated System Health",
    description="Returns aggregate health status and subsystem dependencies.",
)
async def get_health(
    settings: Settings = Depends(get_app_settings),
) -> APIResponse[HealthResponse]:
    components = [
        ComponentStatus(
            name="strands_decider_2b",
            status="mocked" if settings.strands.mock_mode else "healthy",
            details=f"Endpoint: {settings.strands.endpoint} (Decision Plane)",
        ),
        ComponentStatus(
            name="amazon_bedrock",
            status="mocked" if settings.bedrock.mock_mode else "healthy",
            details=f"Region: {settings.bedrock.aws_region} (Generation Plane)",
        ),
    ]

    health_payload = HealthResponse(
        status="healthy",
        version=settings.app.app_version,
        environment=settings.app.environment,
        components=components,
    )

    return APIResponse(data=health_payload)


@router.get(
    "/live",
    status_code=status.HTTP_200_OK,
    summary="Liveness Probe",
    description="Kubernetes / Docker liveness probe verifying HTTP server responsiveness.",
)
async def get_liveness() -> dict[str, str]:
    return {"status": "alive"}


@router.get(
    "/ready",
    status_code=status.HTTP_200_OK,
    summary="Readiness Probe",
    description="Kubernetes / Docker readiness probe verifying readiness to accept traffic.",
)
async def get_readiness() -> dict[str, str]:
    return {"status": "ready"}
