"""Integration tests for health, readiness, and liveness endpoints."""

import pytest
from httpx import AsyncClient


@pytest.mark.asyncio
async def test_root_endpoint(async_client: AsyncClient) -> None:
    """Ensure root path responds with app metadata."""
    response = await async_client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert "app" in data
    assert data["status"] == "online"


@pytest.mark.asyncio
async def test_health_endpoint(async_client: AsyncClient) -> None:
    """Ensure aggregate health endpoint returns 200 and component breakdown."""
    response = await async_client.get("/api/v1/health")
    assert response.status_code == 200

    envelope = response.json()
    assert envelope["success"] is True
    assert "data" in envelope

    data = envelope["data"]
    assert data["status"] == "healthy"
    assert len(data["components"]) == 2

    component_names = [c["name"] for c in data["components"]]
    assert "strands_decider_2b" in component_names
    assert "amazon_bedrock" in component_names


@pytest.mark.asyncio
async def test_liveness_endpoint(async_client: AsyncClient) -> None:
    """Ensure liveness probe succeeds."""
    response = await async_client.get("/api/v1/health/live")
    assert response.status_code == 200
    assert response.json() == {"status": "alive"}


@pytest.mark.asyncio
async def test_readiness_endpoint(async_client: AsyncClient) -> None:
    """Ensure readiness probe succeeds."""
    response = await async_client.get("/api/v1/health/ready")
    assert response.status_code == 200
    assert response.json() == {"status": "ready"}
