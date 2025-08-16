from rq import Worker
from app.jobs.queue import queue, redis
from app.core.config import RQ_QUEUE
if __name__ == "__main__":
    w = Worker([RQ_QUEUE], connection=redis)
    w.work(with_scheduler=True)