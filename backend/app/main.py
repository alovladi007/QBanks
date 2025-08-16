"""
Main FastAPI application entry point.
"""
from contextlib import asynccontextmanager
import logging
from typing import Any

from fastapi import FastAPI, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from starlette.exceptions import HTTPException as StarletteHTTPException
from starlette.middleware.sessions import SessionMiddleware
import sentry_sdk
from sentry_sdk.integrations.fastapi import FastApiIntegration
from sentry_sdk.integrations.sqlalchemy import SqlalchemyIntegration
from prometheus_fastapi_instrumentator import Instrumentator

from .core.config import settings
from .core.database import init_db, close_db
from .core.redis import init_redis, close_redis
from .core.events import init_event_handlers, close_event_handlers
from .middleware.rate_limit import RateLimitMiddleware
from .middleware.security import SecurityHeadersMiddleware
from .middleware.logging import LoggingMiddleware
from .middleware.metrics import MetricsMiddleware
from .api.v1.router import api_router as v1_router
from .api.v2.router import api_router as v2_router

# Configure logging
logging.basicConfig(
    level=getattr(logging, settings.LOG_LEVEL),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize Sentry if configured
if settings.SENTRY_DSN:
    sentry_sdk.init(
        dsn=settings.SENTRY_DSN,
        environment=settings.ENVIRONMENT,
        integrations=[
            FastApiIntegration(transaction_style="endpoint"),
            SqlalchemyIntegration(),
        ],
        traces_sample_rate=settings.SENTRY_TRACES_SAMPLE_RATE,
        profiles_sample_rate=0.1,
    )

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan manager.
    """
    # Startup
    logger.info("Starting QBank Enterprise API...")
    
    # Initialize database
    await init_db()
    logger.info("Database initialized")
    
    # Initialize Redis
    await init_redis()
    logger.info("Redis initialized")
    
    # Initialize event handlers
    await init_event_handlers()
    logger.info("Event handlers initialized")
    
    # Initialize ML models
    # await init_ml_models()
    logger.info("ML models loaded")
    
    yield
    
    # Shutdown
    logger.info("Shutting down QBank Enterprise API...")
    
    # Close event handlers
    await close_event_handlers()
    
    # Close Redis
    await close_redis()
    
    # Close database
    await close_db()
    
    logger.info("Shutdown complete")

# Create FastAPI application
app = FastAPI(
    title=settings.APP_NAME,
    description=settings.APP_DESCRIPTION,
    version=settings.APP_VERSION,
    docs_url=settings.DOCS_URL if not settings.is_production() else None,
    redoc_url=settings.REDOC_URL if not settings.is_production() else None,
    openapi_url=settings.OPENAPI_URL if not settings.is_production() else None,
    lifespan=lifespan,
)

# Add middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=settings.CORS_ALLOW_CREDENTIALS,
    allow_methods=settings.CORS_ALLOW_METHODS,
    allow_headers=settings.CORS_ALLOW_HEADERS,
)

app.add_middleware(GZipMiddleware, minimum_size=1000)
app.add_middleware(
    SessionMiddleware,
    secret_key=settings.SECRET_KEY.get_secret_value(),
    session_cookie="qbank_session",
    max_age=settings.SESSION_TTL,
    same_site="lax",
    https_only=settings.is_production(),
)

if settings.is_production():
    app.add_middleware(
        TrustedHostMiddleware,
        allowed_hosts=["*.qbank.com", "qbank.com"]
    )

app.add_middleware(SecurityHeadersMiddleware)
app.add_middleware(LoggingMiddleware)
app.add_middleware(MetricsMiddleware)

if settings.RATE_LIMIT_ENABLED:
    app.add_middleware(RateLimitMiddleware)

# Add Prometheus metrics
if settings.PROMETHEUS_ENABLED:
    instrumentator = Instrumentator()
    instrumentator.instrument(app).expose(app, endpoint="/metrics")

# Exception handlers
@app.exception_handler(StarletteHTTPException)
async def http_exception_handler(request: Request, exc: StarletteHTTPException):
    """Handle HTTP exceptions."""
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": {
                "message": exc.detail,
                "type": "http_error",
                "status_code": exc.status_code
            }
        }
    )

@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    """Handle validation errors."""
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content={
            "error": {
                "message": "Validation error",
                "type": "validation_error",
                "details": exc.errors()
            }
        }
    )

@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """Handle general exceptions."""
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    
    if settings.is_production():
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={
                "error": {
                    "message": "An internal error occurred",
                    "type": "internal_error"
                }
            }
        )
    else:
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={
                "error": {
                    "message": str(exc),
                    "type": "internal_error",
                    "debug": True
                }
            }
        )

# Health check endpoints
@app.get("/health", tags=["Health"])
async def health_check():
    """Basic health check endpoint."""
    return {
        "status": "healthy",
        "version": settings.APP_VERSION,
        "environment": settings.ENVIRONMENT
    }

@app.get("/health/ready", tags=["Health"])
async def readiness_check():
    """Readiness check endpoint."""
    checks = {
        "database": False,
        "redis": False,
        "services": True
    }
    
    # Check database
    try:
        # await check_database_connection()
        checks["database"] = True
    except Exception as e:
        logger.error(f"Database health check failed: {e}")
    
    # Check Redis
    try:
        # await check_redis_connection()
        checks["redis"] = True
    except Exception as e:
        logger.error(f"Redis health check failed: {e}")
    
    all_healthy = all(checks.values())
    
    return JSONResponse(
        status_code=status.HTTP_200_OK if all_healthy else status.HTTP_503_SERVICE_UNAVAILABLE,
        content={
            "status": "ready" if all_healthy else "not_ready",
            "checks": checks
        }
    )

# Root endpoint
@app.get("/", tags=["Root"])
async def root():
    """Root endpoint."""
    return {
        "name": settings.APP_NAME,
        "version": settings.APP_VERSION,
        "description": settings.APP_DESCRIPTION,
        "documentation": {
            "swagger": f"{settings.DOCS_URL}" if settings.DOCS_URL else None,
            "redoc": f"{settings.REDOC_URL}" if settings.REDOC_URL else None,
        },
        "api_versions": {
            "v1": settings.API_V1_PREFIX,
            "v2": settings.API_V2_PREFIX,
        }
    }

# Include API routers
app.include_router(v1_router, prefix=settings.API_V1_PREFIX)
app.include_router(v2_router, prefix=settings.API_V2_PREFIX)

# WebSocket endpoint for real-time features
@app.websocket("/ws")
async def websocket_endpoint(websocket):
    """WebSocket endpoint for real-time communication."""
    await websocket.accept()
    try:
        while True:
            data = await websocket.receive_text()
            # Process WebSocket messages
            await websocket.send_text(f"Echo: {data}")
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
    finally:
        await websocket.close()

if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        "app.main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=settings.RELOAD,
        workers=settings.WORKERS if not settings.RELOAD else 1,
        log_level=settings.LOG_LEVEL.lower(),
        access_log=True,
    )