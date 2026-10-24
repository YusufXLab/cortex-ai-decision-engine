"""Application Configuration using Pydantic Settings v2.

Supports reading from environment variables, .env file, and defaults.
"""

from functools import lru_cache
from typing import Literal
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class AppSettings(BaseSettings):
    """Core application settings."""

    environment: Literal["development", "staging", "production", "test"] = Field(
        default="development",
        description="Current deployment environment.",
    )
    app_name: str = Field(
        default="Cortex AI Decision Engine",
        description="Public display name of the service.",
    )
    app_version: str = Field(
        default="0.1.0",
        description="Application semver version.",
    )
    debug: bool = Field(
        default=False,
        description="Enable debug mode and verbose stack traces.",
    )
    host: str = Field(
        default="0.0.0.0",
        description="Server bind host.",
    )
    port: int = Field(
        default=8000,
        description="Server listen port.",
    )
    api_v1_prefix: str = Field(
        default="/api/v1",
        description="Prefix path for API v1 routes.",
    )
    cortex_api_key: str = Field(
        default="cortex-dev-api-key-change-in-production",
        description="API key for securing administrative / ingestion endpoints.",
    )
    cors_origins: list[str] = Field(
        default=["*"],
        description="Allowed CORS origin origins.",
    )


class LoggingSettings(BaseSettings):
    """Structured logging configuration."""

    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = Field(
        default="INFO",
        description="Log output filtering level.",
    )
    log_json_format: bool = Field(
        default=False,
        description="Emit logs in structured JSON format (recommended for production).",
    )


class StrandsDeciderSettings(BaseSettings):
    """Strands Decider 2B configuration (Decision plane only).

    Strict Rule: Strands Decider ONLY makes intelligent decisions
    (routing, classification, gating, policy evaluation) and NEVER generates final responses.
    """

    endpoint: str = Field(
        default="http://localhost:8080/v1/decide",
        description="Endpoint URL for the Strands Decider 2B model runner.",
    )
    model_name: str = Field(
        default="strands-decider-2b",
        description="Identifier of the Strands Decider 2B model checkpoint.",
    )
    timeout_seconds: float = Field(
        default=5.0,
        description="Inference timeout for decision-making.",
    )
    max_retries: int = Field(
        default=2,
        description="Retry attempts on transient decision errors.",
    )
    mock_mode: bool = Field(
        default=True,
        description="Simulate Strands Decider 2B for local offline development.",
    )


class BedrockSettings(BaseSettings):
    """Amazon Bedrock configuration (Generation plane only).

    Strict Rule: Amazon Bedrock ONLY generates output responses based on
    the decisions provided by the Strands Decider 2B plane.
    """

    aws_region: str = Field(
        default="us-east-1",
        description="AWS region where Bedrock foundation models are hosted.",
    )
    aws_access_key_id: str | None = Field(
        default=None,
        description="Optional explicit AWS Access Key (prefer IAM roles in AWS).",
    )
    aws_secret_access_key: str | None = Field(
        default=None,
        description="Optional explicit AWS Secret Access Key.",
    )
    aws_session_token: str | None = Field(
        default=None,
        description="Optional explicit AWS Session Token.",
    )
    aws_bedrock_assume_role_arn: str | None = Field(
        default=None,
        description="IAM Role ARN to assume when invoking Bedrock.",
    )
    default_model_id: str = Field(
        default="anthropic.claude-3-5-sonnet-20241022-v2:0",
        description="Default Bedrock foundation model ID for generation.",
    )
    timeout_seconds: float = Field(
        default=30.0,
        description="Bedrock generation invocation timeout.",
    )
    max_retries: int = Field(
        default=3,
        description="Maximum retry attempts on rate limits / throttling.",
    )
    mock_mode: bool = Field(
        default=True,
        description="Simulate Bedrock for local offline development.",
    )


class OrchestratorSettings(BaseSettings):
    """Settings for the Decision Orchestrator and Agent Pipeline."""

    decision_confidence_threshold: float = Field(
        default=0.75,
        ge=0.0,
        le=1.0,
        description="Confidence threshold below which fallback decisions trigger.",
    )
    max_parallel_decision_modules: int = Field(
        default=11,
        description="Maximum concurrent Strands Decider 2B modules executed in parallel.",
    )
    benchmarking_enabled: bool = Field(
        default=True,
        description="Track latency breakdown between deciders and generation.",
    )


class Settings(BaseSettings):
    """Root configuration combining all domain settings."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    app: AppSettings = Field(default_factory=AppSettings)
    logging: LoggingSettings = Field(default_factory=LoggingSettings)
    strands: StrandsDeciderSettings = Field(default_factory=StrandsDeciderSettings)
    bedrock: BedrockSettings = Field(default_factory=BedrockSettings)
    orchestrator: OrchestratorSettings = Field(default_factory=OrchestratorSettings)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Singleton getter for application configuration.

    Cached via LRU cache so environment variables are parsed only once.
    """
    return Settings()
