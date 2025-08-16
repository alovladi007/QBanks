from fastapi import APIRouter, Depends
from pydantic import BaseModel
from rq.job import Job
from app.core.auth import require_roles
from app.jobs.queue import redis

router = APIRouter()

class CalibStatus(BaseModel):
    state: str
    current_iter: int
    total_iters: int
    avg_exp: float | None = None
    max_over: float | None = None
    result: dict | None = None

@router.get("/exposure/calibrate_sh/status", response_model=CalibStatus, dependencies=[Depends(require_roles("admin"))])
def calib_status(job_id: str):
    job = Job.fetch(job_id, connection=redis)
    meta = job.meta or {}
    state = meta.get("state") or job.get_status()
    return CalibStatus(
        state=state,
        current_iter=int(meta.get("current_iter") or 0),
        total_iters=int(meta.get("total_iters") or 0),
        avg_exp=float(meta.get("avg_exp") or 0.0) if meta.get("avg_exp") is not None else None,
        max_over=float(meta.get("max_over") or 0.0) if meta.get("max_over") is not None else None,
        result=job.result if state == "done" else None
    )