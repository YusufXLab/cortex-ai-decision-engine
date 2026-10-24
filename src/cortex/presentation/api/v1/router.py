"""API v1 master router."""

from fastapi import APIRouter
from cortex.presentation.api.v1.endpoints.demo import router as demo_router
from cortex.presentation.api.v1.endpoints.health import router as health_router

api_v1_router = APIRouter()

# Mount endpoints
api_v1_router.include_router(health_router)
api_v1_router.include_router(demo_router)
