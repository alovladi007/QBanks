# =====================================================
# qbank-backend/requirements.txt
# =====================================================
"""
fastapi==0.115.0
uvicorn[standard]==0.30.6
pydantic==2.8.2
pydantic-settings==2.3.4
python-dotenv==1.0.1
redis==5.0.8
hiredis==2.3.2
rq==1.16.2
kafka-python==2.0.2
psycopg2-binary==2.9.9
asyncpg==0.29.0
SQLAlchemy==2.0.32
alembic==1.13.2
PyJWT==2.9.0
passlib[bcrypt]==1.7.4
python-multipart==0.0.9
httpx==0.27.2
orjson==3.10.7
python-jose[cryptography]==3.3.0
email-validator==2.2.0
celery==5.3.4
flower==2.0.1
numpy==1.26.4
scipy==1.14.0
pandas==2.2.2
scikit-learn==1.5.1
statsmodels==0.14.2
pymc==5.16.2
torch==2.4.0
sentence-transformers==3.0.1
elasticsearch[async]==8.14.0
clickhouse-driver==0.2.8
prometheus-client==0.20.0
opentelemetry-api==1.25.0
opentelemetry-sdk==1.25.0
opentelemetry-instrumentation-fastapi==0.46b0
sentry-sdk==2.12.0
structlog==24.4.0
python-json-logger==2.0.7
tenacity==8.5.0
cachetools==5.4.0
pytest==8.3.2
pytest-asyncio==0.23.8
pytest-cov==5.0.0
black==24.8.0
ruff==0.6.2
mypy==1.11.1
pre-commit==3.8.0
"""

# =====================================================
# qbank-backend/app/core/config.py
# =====================================================
import os
from typing import List, Optional
from pydantic_settings import BaseSettings
from pydantic import Field, SecretStr, PostgresDsn, RedisDsn

class Settings(BaseSettings):
    # Application
    APP_NAME: str = "QBank API"
    APP_VERSION: str = "10.0.0"
    DEBUG: bool = Field(default=False)
    ENVIRONMENT: str = Field(default="development")
    API_V1_PREFIX: str = "/v1"
    
    # Security
    SECRET_KEY: SecretStr = Field(default="change-me-in-production")
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 30
    REFRESH_TOKEN_EXPIRE_DAYS: int = 7
    CORS_ORIGINS: List[str] = ["*"]
    
    # Database
    DATABASE_URL: PostgresDsn = Field(
        default="postgresql+asyncpg://qbank:qbank@localhost:5432/qbank"
    )
    DATABASE_POOL_SIZE: int = 20
    DATABASE_MAX_OVERFLOW: int = 40
    DATABASE_POOL_TIMEOUT: int = 30
    
    # Redis
    REDIS_URL: RedisDsn = Field(default="redis://localhost:6379/0")
    REDIS_POOL_SIZE: int = 50
    REDIS_DECODE_RESPONSES: bool = True
    CACHE_TTL: int = 3600  # 1 hour
    
    # Kafka
    KAFKA_BOOTSTRAP_SERVERS: str = "localhost:9092"
    KAFKA_TOPIC_EVENTS: str = "events.qbank"
    KAFKA_TOPIC_ANALYTICS: str = "analytics.qbank"
    KAFKA_CONSUMER_GROUP: str = "qbank-backend"
    
    # Elasticsearch
    ELASTICSEARCH_URL: str = "http://localhost:9200"
    ELASTICSEARCH_INDEX_QUESTIONS: str = "qbank-questions"
    ELASTICSEARCH_INDEX_ANALYTICS: str = "qbank-analytics"
    
    # ClickHouse
    CLICKHOUSE_URL: str = "clickhouse://localhost:9000/qbank"
    
    # Multi-tenancy
    TENANT_ID: str = Field(default="00000000-0000-0000-0000-000000000001")
    ENABLE_MULTI_TENANCY: bool = True
    
    # Queue/Worker
    RQ_QUEUE: str = "calibration"
    RQ_WORKER_CONCURRENCY: int = 4
    CELERY_BROKER_URL: str = "redis://localhost:6379/1"
    CELERY_RESULT_BACKEND: str = "redis://localhost:6379/2"
    
    # Exposure Control
    MAX_DAILY_EXPOSURES: int = 500
    EXPOSURE_CONTROL_ENABLED: bool = True
    DEFAULT_SH_P: float = 1.0
    
    # IRT Settings
    IRT_MODEL: str = "3PL"  # 2PL or 3PL
    IRT_MIN_RESPONSES: int = 200
    ADAPTIVE_ENABLED: bool = True
    
    # ML Models
    EMBEDDINGS_MODEL: str = "sentence-transformers/all-MiniLM-L6-v2"
    DIFFICULTY_PREDICTION_ENABLED: bool = True
    
    # Observability
    ENABLE_METRICS: bool = True
    ENABLE_TRACING: bool = True
    JAEGER_AGENT_HOST: str = "localhost"
    JAEGER_AGENT_PORT: int = 6831
    
    # Rate Limiting
    RATE_LIMIT_ENABLED: bool = True
    RATE_LIMIT_PER_MINUTE: int = 60
    RATE_LIMIT_PER_HOUR: int = 1000
    
    # Feature Flags
    FEATURE_SEMANTIC_SEARCH: bool = True
    FEATURE_RECOMMENDATIONS: bool = True
    FEATURE_BULK_IMPORT: bool = True
    FEATURE_ADVANCED_ANALYTICS: bool = True
    
    class Config:
        env_file = ".env"
        case_sensitive = True

settings = Settings()

# =====================================================
# qbank-backend/app/core/database.py
# =====================================================
from typing import AsyncGenerator
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine, async_sessionmaker
from sqlalchemy.orm import DeclarativeBase
from sqlalchemy.pool import NullPool
from app.core.config import settings
import logging

logger = logging.getLogger(__name__)

# Create async engine
engine = create_async_engine(
    str(settings.DATABASE_URL),
    echo=settings.DEBUG,
    pool_size=settings.DATABASE_POOL_SIZE,
    max_overflow=settings.DATABASE_MAX_OVERFLOW,
    pool_timeout=settings.DATABASE_POOL_TIMEOUT,
    pool_pre_ping=True,
    poolclass=None if settings.ENVIRONMENT != "test" else NullPool,
)

# Create async session factory
AsyncSessionLocal = async_sessionmaker(
    engine,
    class_=AsyncSession,
    expire_on_commit=False,
    autocommit=False,
    autoflush=False,
)

class Base(DeclarativeBase):
    pass

async def get_db() -> AsyncGenerator[AsyncSession, None]:
    """Dependency for getting async database session."""
    async with AsyncSessionLocal() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()

async def init_db():
    """Initialize database, create tables if they don't exist."""
    async with engine.begin() as conn:
        # In production, use Alembic migrations instead
        await conn.run_sync(Base.metadata.create_all)

async def close_db():
    """Close database connections."""
    await engine.dispose()

# =====================================================
# qbank-backend/app/core/cache.py
# =====================================================
import redis.asyncio as redis
from typing import Optional, Any, Union
import json
import pickle
from datetime import datetime, timedelta
from app.core.config import settings
import hashlib
import logging

logger = logging.getLogger(__name__)

class RedisCache:
    def __init__(self):
        self.redis: Optional[redis.Redis] = None
        
    async def connect(self):
        """Initialize Redis connection pool."""
        self.redis = await redis.from_url(
            str(settings.REDIS_URL),
            encoding="utf-8",
            decode_responses=settings.REDIS_DECODE_RESPONSES,
            max_connections=settings.REDIS_POOL_SIZE,
        )
        
    async def disconnect(self):
        """Close Redis connection."""
        if self.redis:
            await self.redis.close()
    
    def _make_key(self, *args) -> str:
        """Generate cache key from arguments."""
        key_parts = [str(arg) for arg in args]
        return ":".join(key_parts)
    
    async def get(self, key: str, default: Any = None) -> Any:
        """Get value from cache."""
        try:
            value = await self.redis.get(key)
            if value is None:
                return default
            # Try to deserialize JSON
            try:
                return json.loads(value)
            except (json.JSONDecodeError, TypeError):
                return value
        except Exception as e:
            logger.error(f"Cache get error: {e}")
            return default
    
    async def set(
        self, 
        key: str, 
        value: Any, 
        expire: Optional[int] = None,
        nx: bool = False,
        xx: bool = False
    ) -> bool:
        """Set value in cache with optional expiration."""
        try:
            # Serialize to JSON if possible
            if isinstance(value, (dict, list)):
                value = json.dumps(value)
            
            return await self.redis.set(
                key, 
                value, 
                ex=expire or settings.CACHE_TTL,
                nx=nx,
                xx=xx
            )
        except Exception as e:
            logger.error(f"Cache set error: {e}")
            return False
    
    async def delete(self, *keys: str) -> int:
        """Delete keys from cache."""
        try:
            return await self.redis.delete(*keys)
        except Exception as e:
            logger.error(f"Cache delete error: {e}")
            return 0
    
    async def exists(self, *keys: str) -> int:
        """Check if keys exist."""
        return await self.redis.exists(*keys)
    
    async def incr(self, key: str, amount: int = 1) -> int:
        """Increment counter."""
        return await self.redis.incr(key, amount)
    
    async def decr(self, key: str, amount: int = 1) -> int:
        """Decrement counter."""
        return await self.redis.decr(key, amount)
    
    async def expire(self, key: str, seconds: int) -> bool:
        """Set expiration on key."""
        return await self.redis.expire(key, seconds)
    
    async def ttl(self, key: str) -> int:
        """Get time to live for key."""
        return await self.redis.ttl(key)
    
    # Exposure control methods
    def exposure_key(self, question_id: int, version: int) -> str:
        """Generate exposure control key."""
        day = datetime.utcnow().strftime("%Y%m%d")
        return f"exp:{day}:{question_id}:{version}"
    
    async def can_serve(self, question_id: int, version: int) -> bool:
        """Check if question can be served based on exposure limits."""
        if not settings.EXPOSURE_CONTROL_ENABLED:
            return True
        
        key = self.exposure_key(question_id, version)
        count = await self.get(key, 0)
        return int(count) < settings.MAX_DAILY_EXPOSURES
    
    async def bump_exposure(self, question_id: int, version: int) -> None:
        """Increment exposure count for question."""
        if not settings.EXPOSURE_CONTROL_ENABLED:
            return
        
        key = self.exposure_key(question_id, version)
        pipe = self.redis.pipeline()
        pipe.incr(key, 1)
        pipe.expire(key, 86400)  # 24 hours
        await pipe.execute()
    
    # Session management
    async def get_session(self, session_id: str) -> Optional[dict]:
        """Get session data."""
        key = f"session:{session_id}"
        return await self.get(key)
    
    async def set_session(
        self, 
        session_id: str, 
        data: dict, 
        expire: int = 7200
    ) -> bool:
        """Set session data with expiration."""
        key = f"session:{session_id}"
        return await self.set(key, data, expire=expire)
    
    # Rate limiting
    async def check_rate_limit(
        self, 
        identifier: str, 
        limit: int, 
        window: int
    ) -> tuple[bool, int]:
        """Check if rate limit is exceeded."""
        key = f"rate_limit:{identifier}"
        
        try:
            pipe = self.redis.pipeline()
            pipe.incr(key)
            pipe.expire(key, window)
            results = await pipe.execute()
            
            current_count = results[0]
            return current_count <= limit, current_count
        except Exception as e:
            logger.error(f"Rate limit check error: {e}")
            return True, 0
    
    # Caching decorator
    def cache_result(
        self, 
        prefix: str, 
        expire: Optional[int] = None,
        key_builder: Optional[callable] = None
    ):
        """Decorator for caching function results."""
        def decorator(func):
            async def wrapper(*args, **kwargs):
                # Build cache key
                if key_builder:
                    cache_key = key_builder(*args, **kwargs)
                else:
                    # Simple key from function name and args
                    key_data = f"{func.__name__}:{args}:{kwargs}"
                    key_hash = hashlib.md5(key_data.encode()).hexdigest()
                    cache_key = f"{prefix}:{key_hash}"
                
                # Try to get from cache
                cached = await self.get(cache_key)
                if cached is not None:
                    return cached
                
                # Execute function
                result = await func(*args, **kwargs)
                
                # Store in cache
                await self.set(cache_key, result, expire=expire)
                
                return result
            return wrapper
        return decorator

# Global cache instance
cache = RedisCache()

# =====================================================
# qbank-backend/app/models/orm.py
# =====================================================
from sqlalchemy import (
    BigInteger, Integer, String, Text, Boolean, Float, 
    ForeignKey, JSON, DateTime, UniqueConstraint, Index,
    CheckConstraint, Enum as SQLEnum, func
)
from sqlalchemy.orm import Mapped, mapped_column, relationship
from sqlalchemy.dialects.postgresql import UUID, JSONB, ARRAY, TSVECTOR
from datetime import datetime
from typing import Optional, List, Dict, Any
import uuid
import enum
from app.core.database import Base

class QuestionState(str, enum.Enum):
    DRAFT = "draft"
    IN_REVIEW = "in_review"
    APPROVED = "approved"
    PUBLISHED = "published"
    ARCHIVED = "archived"

class QuizMode(str, enum.Enum):
    TUTOR = "tutor"
    EXAM = "exam"
    PRACTICE = "practice"
    DIAGNOSTIC = "diagnostic"

class DifficultyLevel(str, enum.Enum):
    VERY_EASY = "very_easy"
    EASY = "easy"
    MEDIUM = "medium"
    HARD = "hard"
    VERY_HARD = "very_hard"

# ========== Content Models ==========

class Topic(Base):
    __tablename__ = "topics"
    __table_args__ = (
        Index("idx_topics_tenant", "tenant_id"),
        Index("idx_topics_parent", "parent_id"),
        Index("idx_topics_blueprint", "blueprint_code"),
    )
    
    id: Mapped[int] = mapped_column(BigInteger, primary_key=True)
    tenant_id: Mapped[UUID] = mapped_column(UUID(as_uuid=True), nullable=False)
    parent_id: Mapped[Optional[int]] = mapped_column(
        BigInteger, ForeignKey("topics.id", ondelete="CASCADE")
    )
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    blueprint_code: Mapped[Optional[str]] = mapped_column(String(50))
    description: Mapped[Optional[str]] = mapped_column(Text)
    weight: Mapped[float] = mapped_column(Float, default=1.0)
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)
    metadata: Mapped[Dict[str, Any]] = mapped_column(JSONB, default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=func.now())
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, default=func.now(), onupdate=func.now()
    )
    
    # Relationships
    children: Mapped[List["Topic"]] = relationship(
        back_populates="parent", cascade="all, delete-orphan"
    )
    parent: Mapped[Optional["Topic"]] = relationship(
        back_populates="children", remote_side=[id]
    )
    questions: Mapped[List["QuestionVersion"]] = relationship(
        back_populates="topic"
    )

class Question(Base):
    __tablename__ = "questions"
    __table_args__ = (
        Index("idx_questions_tenant", "tenant_id"),
        Index("idx_questions_external_ref", "external_ref"),
        Index("idx_questions_created_by", "created_by"),
        Index("idx_questions_created_at", "created_at"),
        UniqueConstraint("tenant_id", "external_ref", name="uq_questions_external_ref"),
    )
    
    id: Mapped[int] = mapped_column(BigInteger, primary_key=True)
    tenant_id: Mapped[UUID] = mapped_column(UUID(as_uuid=True), nullable=False)
    external_ref: Mapped[Optional[str]] = mapped_column(String(100))
    created_by: Mapped[str] = mapped_column(String(255), nullable=False)
    is_deleted: Mapped[bool] = mapped_column(Boolean, default=False)
    metadata: Mapped[Dict[str, Any]] = mapped_column(JSONB, default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=func.now())
    deleted_at: Mapped[Optional[datetime]] = mapped_column(DateTime)
    
    # Relationships
    versions: Mapped[List["QuestionVersion"]] = relationship(
        back_populates="question", cascade="all, delete-orphan"
    )
    publications: Mapped[List["QuestionPublication"]] = relationship(
        back_populates="question", cascade="all, delete-orphan"
    )

class QuestionVersion(Base):
    __tablename__ = "question_versions"
    __table_args__ = (
        Index("idx_qv_question", "question_id"),
        Index("idx_qv_topic", "topic_id"),
        Index("idx_qv_state", "state"),
        Index("idx_qv_version", "version"),
        Index("idx_qv_search", "search_vector", postgresql_using="gin"),
        UniqueConstraint("question_id", "version", name="uq_question_version"),
    )
    
    id: Mapped[int] = mapped_column(BigInteger, primary_key=True)
    question_id: Mapped[int] = mapped_column(
        BigInteger, ForeignKey("questions.id", ondelete="CASCADE"), nullable=False
    )
    version: Mapped[int] = mapped_column(Integer, nullable=False)
    state: Mapped[QuestionState] = mapped_column(
        SQLEnum(QuestionState), nullable=False, default=QuestionState.DRAFT
    )
    stem_md: Mapped[str] = mapped_column(Text, nullable=False)
    lead_in: Mapped[str] = mapped_column(Text, nullable=False)
    rationale_md: Mapped[str] = mapped_column(Text, nullable=False)
    difficulty_label: Mapped[Optional[DifficultyLevel]] = mapped_column(
        SQLEnum(DifficultyLevel)
    )
    bloom_level: Mapped[Optional[int]] = mapped_column(Integer)
    topic_id: Mapped[Optional[int]] = mapped_column(
        BigInteger, ForeignKey("topics.id", ondelete="SET NULL")
    )
    tags: Mapped[List[str]] = mapped_column(ARRAY(String), default=list)
    assets: Mapped[List[Dict]] = mapped_column(JSONB, default=list)
    references: Mapped[List[Dict]] = mapped_column(JSONB, default=list)
    search_vector: Mapped[Optional[str]] = mapped_column(TSVECTOR)
    embedding: Mapped[Optional[List[float]]] = mapped_column(ARRAY(Float))
    metadata: Mapped[Dict[str, Any]] = mapped_column(JSONB, default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=func.now())
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, default=func.now(), onupdate=func.now()
    )
    reviewed_by: Mapped[Optional[str]] = mapped_column(String(255))
    reviewed_at: Mapped[Optional[datetime]] = mapped_column(DateTime)
    
    # Relationships
    question: Mapped["Question"] = relationship(back_populates="versions")
    topic: Mapped[Optional["Topic"]] = relationship(back_populates="questions")
    options: Mapped[List["QuestionOption"]] = relationship(
        back_populates="question_version", cascade="all, delete-orphan"
    )
    calibrations: Mapped[List["ItemCalibration"]] = relationship(
        back_populates="question_version"
    )

class QuestionOption(Base):
    __tablename__ = "question_options"
    __table_args__ = (
        Index("idx_qo_question_version", "question_version_id"),
        UniqueConstraint(
            "question_version_id", "option_label", 
            name="uq_question_option"
        ),
    )
    
    id: Mapped[int] = mapped_column(BigInteger, primary_key=True)
    question_version_id: Mapped[int] = mapped_column(
        BigInteger, ForeignKey("question_versions.id", ondelete="CASCADE"), 
        nullable=False
    )
    option_label: Mapped[str] = mapped_column(String(1), nullable=False)
    option_text_md: Mapped[str] = mapped_column(Text, nullable=False)
    is_correct: Mapped[bool] = mapped_column(Boolean, nullable=False)
    explanation_md: Mapped[Optional[str]] = mapped_column(Text)
    metadata: Mapped[Dict[str, Any]] = mapped_column(JSONB, default=dict)
    
    # Relationships
    question_version: Mapped["QuestionVersion"] = relationship(
        back_populates="options"
    )

class QuestionPublication(Base):
    __tablename__ = "question_publications"
    __table_args__ = (
        Index("idx_qp_question", "question_id"),
        Index("idx_qp_exam_code", "exam_code"),
        Index("idx_qp_tenant", "tenant_id"),
        UniqueConstraint(
            "question_id", "exam_code", 
            name="uq_question_publication"
        ),
    )
    
    id: Mapped[int] = mapped_column(BigInteger, primary_key=True)
    question_id: Mapped[int] = mapped_column(
        BigInteger, ForeignKey("questions.id", ondelete="CASCADE"), 
        nullable=False
    )
    live_version: Mapped[int] = mapped_column(Integer, nullable=False)
    exam_code: Mapped[str] = mapped_column(String(50), nullable=False)
    tenant_id: Mapped[UUID] = mapped_column(UUID(as_uuid=True), nullable=False)
    published_at: Mapped[datetime] = mapped_column(DateTime, default=func.now())
    published_by: Mapped[str] = mapped_column(String(255), nullable=False)
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)
    metadata: Mapped[Dict[str, Any]] = mapped_column(JSONB, default=dict)
    
    # Relationships
    question: Mapped["Question"] = relationship(back_populates="publications")

# ========== Delivery Models ==========

class QuizSession(Base):
    __tablename__ = "quiz_sessions"
    __table_args__ = (
        Index("idx_qs_user", "user_id"),
        Index("idx_qs_tenant", "tenant_id"),
        Index("idx_qs_started", "started_at"),
        Index("idx_qs_mode", "mode"),
        CheckConstraint(
            "expires_at > started_at", 
            name="ck_quiz_session_expires"
        ),
    )
    
    id: Mapped[UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    user_id: Mapped[str] = mapped_column(String(255), nullable=False)
    tenant_id: Mapped[UUID] = mapped_column(UUID(as_uuid=True), nullable=False)
    mode: Mapped[QuizMode] = mapped_column(
        SQLEnum(QuizMode), nullable=False, default=QuizMode.PRACTICE
    )
    adaptive: Mapped[bool] = mapped_column(Boolean, default=False)
    exam_code: Mapped[Optional[str]] = mapped_column(String(50))
    config: Mapped[Dict[str, Any]] = mapped_column(JSONB, default=dict)
    started_at: Mapped[datetime] = mapped_column(DateTime, default=func.now())
    expires_at: Mapped[datetime] = mapped_column(DateTime, nullable=False)
    completed_at: Mapped[Optional[datetime]] = mapped_column(DateTime)
    score: Mapped[Optional[float]] = mapped_column(Float)
    metadata: Mapped[Dict[str, Any]] = mapped_column(JSONB, default=dict)
    
    # Relationships
    items: Mapped[List["QuizItem"]] = relationship(
        back_populates="session", cascade="all, delete-orphan"
    )
    responses: Mapped[List["UserResponse"]] = relationship(
        back_populates="session", cascade="all, delete-orphan"
    )

class QuizItem(Base):
    __tablename__ = "quiz_items"
    __table_args__ = (
        Index("idx_qi_quiz", "quiz_id"),
        Index("idx_qi_question", "question_id"),
        UniqueConstraint("quiz_id", "position", name="uq_quiz_item_position"),
    )
    
    id: Mapped[int] = mapped_column(BigInteger, primary_key=True)
    quiz_id: Mapped[UUID] = mapped_column(
        UUID(as_uuid=True), 
        ForeignKey("quiz_sessions.id", ondelete="CASCADE"), 
        nullable=False
    )
    question_id: Mapped[int] = mapped_column(BigInteger, nullable=False)
    version: Mapped[int] = mapped_column(Integer, nullable=False)
    position: Mapped[int] = mapped_column(Integer, nullable=False)
    served_at: Mapped[datetime] = mapped_column(DateTime, default=func.now())
    metadata: Mapped[Dict[str, Any]] = mapped_column(JSONB, default=dict)
    
    # Relationships
    session: Mapped["QuizSession"] = relationship(back_populates="items")

class UserResponse(Base):
    __tablename__ = "user_responses"
    __table_args__ = (
        Index("idx_ur_quiz", "quiz_id"),
        Index("idx_ur_user", "user_id"),
        Index("idx_ur_question", "question_id", "version"),
        Index("idx_ur_created", "created_at"),
        UniqueConstraint(
            "quiz_id", "question_id", 
            name="uq_user_response"
        ),
    )
    
    id: Mapped[int] = mapped_column(BigInteger, primary_key=True)
    quiz_id: Mapped[UUID] = mapped_column(
        UUID(as_uuid=True), 
        ForeignKey("quiz_sessions.id", ondelete="CASCADE"), 
        nullable=False
    )
    user_id: Mapped[str] = mapped_column(String(255), nullable=False)
    question_id: Mapped[int] = mapped_column(BigInteger, nullable=False)
    version: Mapped[int] = mapped_column(Integer, nullable=False)
    option_label: Mapped[str] = mapped_column(String(1), nullable=False)
    is_correct: Mapped[bool] = mapped_column(Boolean, nullable=False)
    time_taken_ms: Mapped[Optional[int]] = mapped_column(Integer)
    confidence: Mapped[Optional[int]] = mapped_column(Integer)
    metadata: Mapped[Dict[str, Any]] = mapped_column(JSONB, default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=func.now())
    
    # Relationships
    session: Mapped["QuizSession"] = relationship(back_populates="responses")

# ========== Analytics Models ==========

class ItemCalibration(Base):
    __tablename__ = "item_calibration"
    __table_args__ = (
        Index("idx_ic_question", "question_id", "version"),
        Index("idx_ic_model", "model"),
        Index("idx_ic_calibrated", "calibrated_at"),
    )
    
    question_id: Mapped[int] = mapped_column(BigInteger, primary_key=True)
    version: Mapped[int] = mapped_column(Integer, primary_key=True)
    model: Mapped[str] = mapped_column(String(10), primary_key=True)
    a: Mapped[Optional[float]] = mapped_column(Float)  # Discrimination
    b: Mapped[Optional[float]] = mapped_column(Float)  # Difficulty
    c: Mapped[Optional[float]] = mapped_column(Float)  # Guessing
    se_a: Mapped[Optional[float]] = mapped_column(Float)  # Standard errors
    se_b: Mapped[Optional[float]] = mapped_column(Float)
    se_c: Mapped[Optional[float]] = mapped_column(Float)
    n_respondents: Mapped[Optional[int]] = mapped_column(Integer)
    fit_statistics: Mapped[Dict[str, Any]] = mapped_column(JSONB, default=dict)
    calibrated_at: Mapped[datetime] = mapped_column(DateTime, default=func.now())
    
    # Relationships
    question_version: Mapped["QuestionVersion"] = relationship(
        foreign_keys="[ItemCalibration.question_id, ItemCalibration.version]",
        primaryjoin="and_(ItemCalibration.question_id==QuestionVersion.question_id, "
                   "ItemCalibration.version==QuestionVersion.version)",
        viewonly=True,
    )

class ItemExposureControl(Base):
    __tablename__ = "item_exposure_control"
    __table_args__ = (
        Index("idx_iec_question", "question_id", "version"),
        Index("idx_iec_updated", "updated_at"),
    )
    
    question_id: Mapped[int] = mapped_column(BigInteger, primary_key=True)
    version: Mapped[int] = mapped_column(Integer, primary_key=True)
    sh_p: Mapped[float] = mapped_column(Float, default=1.0, nullable=False)
    exposure_count: Mapped[int] = mapped_column(Integer, default=0)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, default=func.now(), onupdate=func.now()
    )

class UserAbility(Base):
    __tablename__ = "user_abilities"
    __table_args__ = (
        Index("idx_ua_user", "user_id"),
        Index("idx_ua_topic", "topic_id"),
        Index("idx_ua_updated", "updated_at"),
        UniqueConstraint("user_id", "topic_id", name="uq_user_ability"),
    )
    
    id: Mapped[int] = mapped_column(BigInteger, primary_key=True)
    user_id: Mapped[str] = mapped_column(String(255), nullable=False)
    topic_id: Mapped[Optional[int]] = mapped_column(BigInteger)  # NULL = global
    theta: Mapped[float] = mapped_column(Float, default=0.0)
    theta_se: Mapped[float] = mapped_column(Float, default=1.0)
    n_responses: Mapped[int] = mapped_column(Integer, default=0)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, default=func.now(), onupdate=func.now()
    )

# ========== Governance Models ==========

class FeatureFlag(Base):
    __tablename__ = "feature_flags"
    __table_args__ = (
        Index("idx_ff_key", "key"),
    )
    
    key: Mapped[str] = mapped_column(String(100), primary_key=True)
    enabled: Mapped[bool] = mapped_column(Boolean, default=True)
    value_json: Mapped[Dict[str, Any]] = mapped_column(JSONB, default=dict)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, default=func.now(), onupdate=func.now()
    )

class CohortAssignment(Base):
    __tablename__ = "cohort_assignments"
    __table_args__ = (
        Index("idx_ca_user", "user_id"),
        Index("idx_ca_cohort", "cohort_key"),
    )
    
    user_id: Mapped[str] = mapped_column(String(255), primary_key=True)
    cohort_key: Mapped[str] = mapped_column(String(100), primary_key=True)
    cohort_value: Mapped[str] = mapped_column(String(255), nullable=False)
    assigned_at: Mapped[datetime] = mapped_column(DateTime, default=func.now())

class CalibrationRun(Base):
    __tablename__ = "calibration_runs"
    __table_args__ = (
        Index("idx_cr_exam", "exam_code"),
        Index("idx_cr_status", "status"),
        Index("idx_cr_created", "created_at"),
    )
    
    id: Mapped[UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    exam_code: Mapped[str] = mapped_column(String(50), nullable=False)
    status: Mapped[str] = mapped_column(String(20), nullable=False)
    params: Mapped[Dict[str, Any]] = mapped_column(JSONB, nullable=False)
    history: Mapped[List[Dict]] = mapped_column(JSONB, default=list)
    result: Mapped[Optional[Dict[str, Any]]] = mapped_column(JSONB)
    error: Mapped[Optional[str]] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=func.now())
    started_at: Mapped[Optional[datetime]] = mapped_column(DateTime)
    finished_at: Mapped[Optional[datetime]] = mapped_column(DateTime)

class AuditLog(Base):
    __tablename__ = "audit_logs"
    __table_args__ = (
        Index("idx_al_user", "user_id"),
        Index("idx_al_entity", "entity_type", "entity_id"),
        Index("idx_al_created", "created_at"),
    )
    
    id: Mapped[int] = mapped_column(BigInteger, primary_key=True)
    user_id: Mapped[str] = mapped_column(String(255), nullable=False)
    action: Mapped[str] = mapped_column(String(50), nullable=False)
    entity_type: Mapped[str] = mapped_column(String(50), nullable=False)
    entity_id: Mapped[str] = mapped_column(String(255), nullable=False)
    changes: Mapped[Dict[str, Any]] = mapped_column(JSONB, default=dict)
    metadata: Mapped[Dict[str, Any]] = mapped_column(JSONB, default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=func.now())
