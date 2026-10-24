"""Integration test for demo pipeline endpoint returning full CortexState."""

import pytest
from httpx import AsyncClient


@pytest.mark.asyncio
async def test_demo_endpoint_pipeline_execution(async_client: AsyncClient) -> None:
    """Ensure demo endpoint executes the 11 modules and returns complete CortexState."""
    custom_correlation_id = "test-corr-id-12345"
    response = await async_client.post(
        "/api/v1/demo/decide-and-generate",
        json={"query": "Evaluate cloud migration risk", "enforce_guardrail": True},
        headers={"X-Correlation-ID": custom_correlation_id},
    )

    assert response.status_code == 200

    # Verify custom middleware headers
    assert response.headers.get("X-Correlation-ID") == custom_correlation_id
    assert "X-Process-Time-Ms" in response.headers

    # Verify envelope and CortexState data
    payload = response.json()
    assert payload["success"] is True
    data = payload["data"]

    # Verify state fields
    assert data["user_input"] == "Evaluate cloud migration risk"
    assert data["status"] == "completed"
    assert data["intent"] in {"ANALYSIS", "GENERAL_QUERY"}
    assert data["selected_agent"] in {"RiskAnalysisAgent", "general_assistant_agent"}
    assert data["privacy_level"] in {"PUBLIC", "public"}
    assert data["risk_level"] == "low"
    assert data["complexity"] == "moderate"
    assert "claude" in data["selected_model"].lower()
    assert data["cost_strategy"] == "balanced"
    assert data["confidence"] >= 0.75

    # Verify 11 trace steps
    assert len(data["execution_trace"]) == 11
    assert data["generation_output"] is not None
