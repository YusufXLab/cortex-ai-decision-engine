"""Unit tests for the Configuration system."""

from cortex.core.config import Settings, get_settings


def test_settings_defaults() -> None:
    """Ensure default configuration loads cleanly."""
    settings = get_settings()
    assert settings.app.app_name == "Cortex AI Decision Engine"
    assert settings.app.api_v1_prefix == "/api/v1"
    assert settings.strands.model_name == "strands-decider-2b"
    assert settings.bedrock.aws_region == "us-east-1"


def test_settings_environment_override() -> None:
    """Ensure custom settings overrides work as expected."""
    custom_settings = Settings(
        app={"environment": "production", "debug": False},
        strands={"mock_mode": False},
    )
    assert custom_settings.app.environment == "production"
    assert custom_settings.app.debug is False
    assert custom_settings.strands.mock_mode is False
