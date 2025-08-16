from rq import Queue
from redis import Redis
from app.core.config import REDIS_URL, RQ_QUEUE
redis = Redis.from_url(REDIS_URL)
queue = Queue(RQ_QUEUE, connection=redis)