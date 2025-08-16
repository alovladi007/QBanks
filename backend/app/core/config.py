"""
Application configuration management with environment-based settings.
"""
import os
from typing import List, Optional, Dict, Any
from pydantic_settings import BaseSettings
from pydantic import Field, SecretStr, PostgresDsn, RedisDsn, HttpUrl, validator
from functools import lru_cache

class Settings(BaseSettings):
    """Main application settings."""
    
    # ============= Application Settings =============
    APP_NAME: str = "QBank Enterprise"
    APP_VERSION: str = "2.0.0"
    APP_DESCRIPTION: str = "Enterprise Question Bank Management System"
    DEBUG: bool = Field(default=False)
    ENVIRONMENT: str = Field(default="development")
    API_V1_PREFIX: str = "/api/v1"
    API_V2_PREFIX: str = "/api/v2"
    DOCS_URL: Optional[str] = "/docs"
    REDOC_URL: Optional[str] = "/redoc"
    OPENAPI_URL: Optional[str] = "/openapi.json"
    
    # ============= Server Settings =============
    HOST: str = Field(default="0.0.0.0")
    PORT: int = Field(default=8000)
    WORKERS: int = Field(default=4)
    RELOAD: bool = Field(default=False)
    
    # ============= Security Settings =============
    SECRET_KEY: SecretStr = Field(default="change-me-in-production-use-strong-key")
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 30
    REFRESH_TOKEN_EXPIRE_DAYS: int = 7
    PASSWORD_MIN_LENGTH: int = 8
    PASSWORD_REQUIRE_UPPERCASE: bool = True
    PASSWORD_REQUIRE_NUMBERS: bool = True
    PASSWORD_REQUIRE_SPECIAL: bool = True
    
    # Two-Factor Authentication
    ENABLE_2FA: bool = True
    OTP_ISSUER: str = "QBank Enterprise"
    OTP_VALIDITY_SECONDS: int = 30
    
    # OAuth2 Settings
    OAUTH2_ENABLED: bool = True
    GOOGLE_CLIENT_ID: Optional[str] = None
    GOOGLE_CLIENT_SECRET: Optional[SecretStr] = None
    MICROSOFT_CLIENT_ID: Optional[str] = None
    MICROSOFT_CLIENT_SECRET: Optional[SecretStr] = None
    
    # CORS Settings
    CORS_ORIGINS: List[str] = ["http://localhost:3000", "http://localhost:8000"]
    CORS_ALLOW_CREDENTIALS: bool = True
    CORS_ALLOW_METHODS: List[str] = ["*"]
    CORS_ALLOW_HEADERS: List[str] = ["*"]
    
    # ============= Database Settings =============
    DATABASE_URL: PostgresDsn = Field(
        default="postgresql+asyncpg://qbank:qbank@localhost:5432/qbank"
    )
    DATABASE_POOL_SIZE: int = 20
    DATABASE_MAX_OVERFLOW: int = 40
    DATABASE_POOL_TIMEOUT: int = 30
    DATABASE_ECHO: bool = False
    DATABASE_CONNECT_TIMEOUT: int = 10
    
    # Read Replica Settings
    DATABASE_READ_REPLICA_URL: Optional[PostgresDsn] = None
    ENABLE_READ_REPLICA: bool = False
    
    # ============= Redis Settings =============
    REDIS_URL: RedisDsn = Field(default="redis://localhost:6379/0")
    REDIS_POOL_SIZE: int = 50
    REDIS_DECODE_RESPONSES: bool = True
    REDIS_SOCKET_TIMEOUT: int = 5
    REDIS_CONNECTION_TIMEOUT: int = 5
    CACHE_TTL: int = 3600  # 1 hour
    SESSION_TTL: int = 86400  # 24 hours
    
    # ============= Message Queue Settings =============
    # Kafka
    KAFKA_BOOTSTRAP_SERVERS: str = "localhost:9092"
    KAFKA_TOPIC_EVENTS: str = "qbank.events"
    KAFKA_TOPIC_ANALYTICS: str = "qbank.analytics"
    KAFKA_TOPIC_NOTIFICATIONS: str = "qbank.notifications"
    KAFKA_CONSUMER_GROUP: str = "qbank-backend"
    KAFKA_AUTO_OFFSET_RESET: str = "earliest"
    
    # Celery
    CELERY_BROKER_URL: str = "redis://localhost:6379/1"
    CELERY_RESULT_BACKEND: str = "redis://localhost:6379/2"
    CELERY_TASK_SERIALIZER: str = "json"
    CELERY_RESULT_SERIALIZER: str = "json"
    CELERY_ACCEPT_CONTENT: List[str] = ["json"]
    CELERY_TIMEZONE: str = "UTC"
    CELERY_ENABLE_UTC: bool = True
    
    # ============= Search Settings =============
    ELASTICSEARCH_URL: str = "http://localhost:9200"
    ELASTICSEARCH_INDEX_QUESTIONS: str = "qbank-questions"
    ELASTICSEARCH_INDEX_USERS: str = "qbank-users"
    ELASTICSEARCH_INDEX_ANALYTICS: str = "qbank-analytics"
    ELASTICSEARCH_TIMEOUT: int = 30
    
    # ============= Analytics Settings =============
    CLICKHOUSE_URL: str = "clickhouse://localhost:9000"
    CLICKHOUSE_DATABASE: str = "qbank_analytics"
    CLICKHOUSE_USER: str = "default"
    CLICKHOUSE_PASSWORD: SecretStr = Field(default="")
    
    # ============= AI/ML Settings =============
    OPENAI_API_KEY: Optional[SecretStr] = None
    OPENAI_MODEL: str = "gpt-4-turbo-preview"
    OPENAI_EMBEDDING_MODEL: str = "text-embedding-3-large"
    OPENAI_MAX_TOKENS: int = 4000
    OPENAI_TEMPERATURE: float = 0.7
    
    # Anthropic Settings
    ANTHROPIC_API_KEY: Optional[SecretStr] = None
    ANTHROPIC_MODEL: str = "claude-3-opus-20240229"
    
    # Local ML Settings
    ML_MODEL_PATH: str = "/app/models"
    ENABLE_GPU: bool = False
    CUDA_DEVICE: int = 0
    
    # ============= Storage Settings =============
    # S3/MinIO Settings
    S3_ENDPOINT_URL: Optional[str] = "http://localhost:9000"
    S3_ACCESS_KEY_ID: Optional[str] = None
    S3_SECRET_ACCESS_KEY: Optional[SecretStr] = None
    S3_BUCKET_NAME: str = "qbank-assets"
    S3_REGION: str = "us-east-1"
    S3_USE_SSL: bool = False
    
    # Local Storage
    UPLOAD_DIR: str = "/app/uploads"
    MAX_UPLOAD_SIZE: int = 10 * 1024 * 1024  # 10MB
    ALLOWED_EXTENSIONS: List[str] = [".pdf", ".docx", ".txt", ".md", ".png", ".jpg", ".jpeg"]
    
    # ============= Payment Settings =============
    STRIPE_SECRET_KEY: Optional[SecretStr] = None
    STRIPE_PUBLISHABLE_KEY: Optional[str] = None
    STRIPE_WEBHOOK_SECRET: Optional[SecretStr] = None
    STRIPE_PRICE_ID_BASIC: Optional[str] = None
    STRIPE_PRICE_ID_PREMIUM: Optional[str] = None
    STRIPE_PRICE_ID_ENTERPRISE: Optional[str] = None
    
    PAYPAL_CLIENT_ID: Optional[str] = None
    PAYPAL_CLIENT_SECRET: Optional[SecretStr] = None
    PAYPAL_MODE: str = "sandbox"  # sandbox or live
    
    # ============= Email Settings =============
    SMTP_HOST: str = "localhost"
    SMTP_PORT: int = 587
    SMTP_USER: Optional[str] = None
    SMTP_PASSWORD: Optional[SecretStr] = None
    SMTP_FROM: str = "noreply@qbank.com"
    SMTP_FROM_NAME: str = "QBank Enterprise"
    SMTP_TLS: bool = True
    SMTP_SSL: bool = False
    
    # SendGrid Settings
    SENDGRID_API_KEY: Optional[SecretStr] = None
    SENDGRID_FROM_EMAIL: Optional[str] = None
    
    # ============= Monitoring Settings =============
    SENTRY_DSN: Optional[str] = None
    SENTRY_ENVIRONMENT: Optional[str] = None
    SENTRY_TRACES_SAMPLE_RATE: float = 0.1
    
    PROMETHEUS_ENABLED: bool = True
    PROMETHEUS_PORT: int = 9090
    
    JAEGER_ENABLED: bool = False
    JAEGER_AGENT_HOST: str = "localhost"
    JAEGER_AGENT_PORT: int = 6831
    
    # ============= Rate Limiting =============
    RATE_LIMIT_ENABLED: bool = True
    RATE_LIMIT_PER_MINUTE: int = 60
    RATE_LIMIT_PER_HOUR: int = 1000
    RATE_LIMIT_PER_DAY: int = 10000
    
    # ============= Feature Flags =============
    FEATURE_AI_QUESTION_GENERATION: bool = True
    FEATURE_PLAGIARISM_CHECK: bool = True
    FEATURE_AUTO_GRADING: bool = True
    FEATURE_ADAPTIVE_TESTING: bool = True
    FEATURE_REAL_TIME_ANALYTICS: bool = True
    FEATURE_EXPORT_IMPORT: bool = True
    FEATURE_MULTI_LANGUAGE: bool = True
    FEATURE_WEBHOOKS: bool = True
    
    # ============= Business Settings =============
    # Subscription Tiers
    TIER_FREE_QUESTIONS: int = 100
    TIER_FREE_TESTS: int = 5
    TIER_FREE_STUDENTS: int = 50
    
    TIER_BASIC_QUESTIONS: int = 1000
    TIER_BASIC_TESTS: int = 50
    TIER_BASIC_STUDENTS: int = 500
    
    TIER_PREMIUM_QUESTIONS: int = 10000
    TIER_PREMIUM_TESTS: int = 500
    TIER_PREMIUM_STUDENTS: int = 5000
    
    TIER_ENTERPRISE_QUESTIONS: int = -1  # Unlimited
    TIER_ENTERPRISE_TESTS: int = -1  # Unlimited
    TIER_ENTERPRISE_STUDENTS: int = -1  # Unlimited
    
    # ============= Logging Settings =============
    LOG_LEVEL: str = "INFO"
    LOG_FORMAT: str = "json"  # json or text
    LOG_FILE: Optional[str] = "/app/logs/qbank.log"
    LOG_ROTATION: str = "1 day"
    LOG_RETENTION: str = "30 days"
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = True
        
    @validator("CORS_ORIGINS", pre=True)
    def assemble_cors_origins(cls, v: str | List[str]) -> List[str]:
        if isinstance(v, str):
            return [i.strip() for i in v.split(",")]
        return v
    
    @validator("DATABASE_URL", pre=True)
    def assemble_db_connection(cls, v: str, values: Dict[str, Any]) -> str:
        if isinstance(v, str):
            return v
        return PostgresDsn.build(
            scheme="postgresql+asyncpg",
            user=values.get("POSTGRES_USER"),
            password=values.get("POSTGRES_PASSWORD"),
            host=values.get("POSTGRES_HOST", "localhost"),
            port=values.get("POSTGRES_PORT", "5432"),
            path=f"/{values.get('POSTGRES_DB', 'qbank')}",
        )
    
    def get_redis_url(self) -> str:
        """Get Redis URL as string."""
        return str(self.REDIS_URL)
    
    def is_production(self) -> bool:
        """Check if running in production."""
        return self.ENVIRONMENT.lower() == "production"
    
    def is_development(self) -> bool:
        """Check if running in development."""
        return self.ENVIRONMENT.lower() == "development"
    
    def is_testing(self) -> bool:
        """Check if running in testing."""
        return self.ENVIRONMENT.lower() == "testing"

@lru_cache()
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()

# Export settings instance
settings = get_settings()