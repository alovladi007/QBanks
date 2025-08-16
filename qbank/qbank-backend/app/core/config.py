import os
from dotenv import load_dotenv
load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL", "postgresql+psycopg2://qbank:qbank@localhost:5432/qbank")
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")
RQ_QUEUE = os.getenv("RQ_QUEUE", "calibration")
RQ_WORKER_CONCURRENCY = int(os.getenv("RQ_WORKER_CONCURRENCY", "1"))
KAFKA_BOOTSTRAP = os.getenv("KAFKA_BOOTSTRAP", "localhost:9092")
KAFKA_TOPIC_EVENTS = os.getenv("KAFKA_TOPIC_EVENTS", "events.qbank")
TENANT_ID = os.getenv("APP_TENANT_ID", "00000000-0000-0000-0000-000000000001")
APP_SECRET = os.getenv("APP_SECRET", "dev-secret-change-me")
MAX_DAILY_EXPOSURES = int(os.getenv("MAX_DAILY_EXPOSURES", "500"))