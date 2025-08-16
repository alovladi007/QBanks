import redis
from datetime import datetime
from app.core.config import REDIS_URL, MAX_DAILY_EXPOSURES

redis_client = redis.Redis.from_url(REDIS_URL, decode_responses=True)

def exposure_key(question_id: int, version: int) -> str:
    day = datetime.utcnow().strftime("%Y%m%d")
    return f"exp:{day}:{question_id}:{version}"

def can_serve(question_id: int, version: int) -> bool:
    key = exposure_key(question_id, version)
    count = int(redis_client.get(key) or 0)
    return count < MAX_DAILY_EXPOSURES

def bump_exposure(question_id: int, version: int) -> None:
    key = exposure_key(question_id, version)
    pipe = redis_client.pipeline()
    pipe.incr(key, 1)
    pipe.expire(key, 86400)
    pipe.execute()