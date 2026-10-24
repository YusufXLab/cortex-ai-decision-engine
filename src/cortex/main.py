"""FastAPI application factory and main server entry point for Cortex AI Decision Engine."""

from contextlib import asynccontextmanager
from typing import AsyncGenerator
from fastapi import FastAPI, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from cortex.core.config import Settings, get_settings
from cortex.core.exceptions import CortexException
from cortex.core.logging import correlation_id_ctx, get_logger, setup_logging
from cortex.presentation.api.v1.router import api_v1_router
from cortex.presentation.middleware.correlation import CorrelationIdMiddleware
from cortex.presentation.middleware.timing import ProcessTimeMiddleware
from cortex.presentation.schemas.common import APIErrorResponse, ErrorDetail

logger = get_logger("cortex.main")


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Lifespan event handler for startup and graceful shutdown."""
    settings = get_settings()

    # 1. Configure logging
    setup_logging(
        log_level=settings.logging.log_level,
        json_format=settings.logging.log_json_format,
    )
    logger.info(
        "Starting %s v%s [Environment: %s]",
        settings.app.app_name,
        settings.app.app_version,
        settings.app.environment,
    )
    logger.info(
        "Strands Decider 2B plane: %s (mock=%s)",
        settings.strands.endpoint,
        settings.strands.mock_mode,
    )
    logger.info(
        "Amazon Bedrock plane: region=%s, default_model=%s (mock=%s)",
        settings.bedrock.aws_region,
        settings.bedrock.default_model_id,
        settings.bedrock.mock_mode,
    )

    yield

    logger.info("Gracefully shutting down %s", settings.app.app_name)


def create_app(settings: Settings | None = None) -> FastAPI:
    """Application factory for Cortex AI Decision Engine."""
    if settings is None:
        settings = get_settings()

    app = FastAPI(
        title=settings.app.app_name,
        version=settings.app.app_version,
        description=(
            "Cortex AI Decision Engine: Orchestrating Strands Decider 2B "
            "for intelligent gating & routing before Amazon Bedrock generation."
        ),
        docs_url="/docs",
        redoc_url="/redoc",
        openapi_url="/openapi.json",
        lifespan=lifespan,
    )

    # --------------------------------------------------------------------------
    # Middleware Stack (Executed in reverse order of addition)
    # --------------------------------------------------------------------------
    # 1. Timing middleware (outermost)
    app.add_middleware(ProcessTimeMiddleware)

    # 2. Correlation ID tracking
    app.add_middleware(CorrelationIdMiddleware)

    # 3. CORS
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.app.cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # --------------------------------------------------------------------------
    # Exception Handlers
    # --------------------------------------------------------------------------
    @app.exception_handler(CortexException)
    async def cortex_exception_handler(request: Request, exc: CortexException) -> JSONResponse:
        logger.error("Domain exception: code=%s message=%s", exc.code, exc.message)
        error_payload = APIErrorResponse(
            error=ErrorDetail(code=exc.code, message=exc.message, details=exc.details),
            correlation_id=correlation_id_ctx.get(),
        )
        return JSONResponse(
            status_code=exc.status_code,
            content=error_payload.model_dump(mode="json"),
        )

    @app.exception_handler(RequestValidationError)
    async def validation_exception_handler(request: Request, exc: RequestValidationError) -> JSONResponse:
        logger.warning("Request validation failed: %s", exc.errors())
        error_payload = APIErrorResponse(
            error=ErrorDetail(
                code="VALIDATION_ERROR",
                message="Request payload validation failed",
                details={"errors": exc.errors()},
            ),
            correlation_id=correlation_id_ctx.get(),
        )
        return JSONResponse(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            content=error_payload.model_dump(mode="json"),
        )

    @app.exception_handler(Exception)
    async def unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
        logger.exception("Unhandled internal exception: %s", str(exc))
        error_payload = APIErrorResponse(
            error=ErrorDetail(
                code="INTERNAL_SERVER_ERROR",
                message="An unexpected server error occurred",
                details={"error_type": type(exc).__name__} if settings.app.debug else {},
            ),
            correlation_id=correlation_id_ctx.get(),
        )
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content=error_payload.model_dump(mode="json"),
        )

    # --------------------------------------------------------------------------
    # Route Registration
    # --------------------------------------------------------------------------
    app.include_router(api_v1_router, prefix=settings.app.api_v1_prefix)

    # Root endpoint for basic health verification
    @app.get("/", tags=["Root"])
    async def root() -> dict[str, str]:
        return {
            "app": settings.app.app_name,
            "version": settings.app.app_version,
            "status": "online",
            "docs": "/docs",
        }

    return app


# Application instance for ASGI servers
app = create_app()

if __name__ == "__main__":
    import uvicorn

    config = get_settings()
    uvicorn.run(
        "cortex.main:app",
        host=config.app.host,
        port=config.app.port,
        reload=config.app.debug,
    )
