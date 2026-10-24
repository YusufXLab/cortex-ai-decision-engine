"""Correlation ID middleware for distributed tracing across logs and requests."""

import uuid
from collections.abc import Callable
from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware
from cortex.core.logging import correlation_id_ctx

CORRELATION_ID_HEADER = "X-Correlation-ID"


class CorrelationIdMiddleware(BaseHTTPMiddleware):
    """Extracts or generates an X-Correlation-ID for each incoming request."""

    async def dispatch(self, request: Request, call_next: Callable[[Request], Response]) -> Response:
        # Check incoming header or generate a new UUID
        correlation_id = request.headers.get(CORRELATION_ID_HEADER) or str(uuid.uuid4())

        # Set contextvar for the current asyncio task
        token = correlation_id_ctx.set(correlation_id)

        try:
            response = await call_next(request)
            response.headers[CORRELATION_ID_HEADER] = correlation_id
            return response
        finally:
            correlation_id_ctx.reset(token)
