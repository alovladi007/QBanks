from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from typing import List, Optional
from sqlalchemy.orm import Session
from sqlalchemy import text
import os, json, uuid, psycopg2
from app.core.database import get_db
from app.core.auth import require_roles
from app.jobs.queue import queue
from app.jobs.calibration_job import calibrate_job
from app.models.orm import ItemExposureControl

router = APIRouter()

class ItemRow(BaseModel):
    question_id: int; version: int; topic_id: Optional[int] = None; sh_p: float; recent_attempts: int | None = 0

@router.get("/exposure/items", response_model=List[ItemRow], dependencies=[Depends(require_roles("admin"))])
def list_items(limit: int = 100, db: Session = Depends(get_db)):
    rows = db.execute(text("""
      SELECT qv.question_id, qv.version, qv.topic_id, COALESCE(iec.sh_p,1.0) sh_p,
        (SELECT count(*) FROM user_responses ur WHERE ur.question_id=qv.question_id AND ur.version=qv.version AND ur.created_at>now()-interval '7 days') recent_attempts
      FROM question_versions qv
      LEFT JOIN item_exposure_control iec ON iec.question_id=qv.question_id AND iec.version=qv.version
      WHERE qv.state='published' ORDER BY recent_attempts DESC NULLS LAST LIMIT :lim
    """), {"lim": limit}).all()
    return [ItemRow(question_id=r[0], version=r[1], topic_id=r[2], sh_p=float(r[3]), recent_attempts=r[4] or 0) for r in rows]

class SetSh(BaseModel): question_id:int; version:int; sh_p:float

@router.post("/exposure/set", dependencies=[Depends(require_roles("admin"))])
def set_sh(payload: SetSh, db: Session = Depends(get_db)):
    if payload.sh_p < 0 or payload.sh_p > 1: raise HTTPException(400, "sh_p must be in [0,1]")
    row = db.get(ItemExposureControl, {"question_id": payload.question_id, "version": payload.version})
    if row: row.sh_p = payload.sh_p
    else: db.add(ItemExposureControl(question_id=payload.question_id, version=payload.version, sh_p=payload.sh_p))
    db.commit(); return {"ok": True}

class StartCalib(BaseModel):
    exam_code: str; tau: float = 0.2; n:int=400; test_len:int=25; iters:int=5; alpha:float=0.6
    theta_dist:str="normal0,1"; floor:float=0.02; ceil:float=1.0
    topic_tau: Optional[dict] = None; topic_weights: Optional[dict] = None; dry_run: bool = False

@router.post("/exposure/calibrate_sh/start", dependencies=[Depends(require_roles("admin"))])
def start_calibration(payload: StartCalib):
    dsn = os.getenv("DATABASE_URL", "postgresql+psycopg2://qbank:qbank@localhost:5432/qbank")
    run_id = str(uuid.uuid4())
    conn = psycopg2.connect(dsn); cur=conn.cursor()
    cur.execute("INSERT INTO calibration_runs(id,exam_code,status,params,created_at) VALUES (%s,%s,%s,%s,now())",
      (run_id, payload.exam_code, "queued", json.dumps(payload.model_dump())) )
    conn.commit(); cur.close(); conn.close()
    job = queue.enqueue(calibrate_job, payload.exam_code, dsn, payload.tau, payload.n, payload.test_len, payload.iters,
                        payload.alpha, payload.theta_dist, payload.floor, payload.ceil,
                        payload.topic_tau, payload.topic_weights, payload.dry_run, run_id, job_timeout=3600)
    return {"job_id": job.get_id(), "run_id": run_id}