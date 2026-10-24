"""Pytest configuration and shared fixtures for Cortex AI Decision Engine."""

from collections.abc import AsyncGenerator, Generator
import pytest
from fastapi.testclient import TestClient
from httpx import ASGITransport, AsyncClient
from cortex.core.config import Settings, get_settings
from cortex.main import create_app


@pytest.fixture(scope="session")
def test_settings() -> Settings:
    """Fixture providing testing settings overrides."""
    return Settings(
        app={
            "environment": "test",
            "app_name": "Cortex AI Decision Engine [Test]",
            "debug": True,
        },
        strands={
            "mock_mode": True,
        },
        bedrock={
            "mock_mode": True,
        },
    )


@pytest.fixture
def sync_client(test_settings: Settings) -> Generator[TestClient, None, None]:
    """Synchronous test client for quick validation."""
    app = create_app(settings=test_settings)
    with TestClient(app) as client:
        yield client


@pytest.fixture
async def async_client(test_settings: Settings) -> AsyncGenerator[AsyncClient, None]:
    """Asynchronous HTTPX client for testing async FastAPI endpoints."""
    app = create_app(settings=test_settings)
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://testserver") as client:
        yield client
