from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel
from typing import List, Optional
from sqlalchemy.orm import Session
from sqlalchemy import text
from app.core.database import get_db
from app.core.auth import require_roles

router = APIRouter()

class RunRow(BaseModel):
    id: str; exam_code: str; status: str; created_at: str; started_at: Optional[str]=None; finished_at: Optional[str]=None

@router.get("/exposure/calibrate_sh/runs", response_model=List[RunRow], dependencies=[Depends(require_roles("admin"))])
def list_runs(exam_code: Optional[str] = None, start: Optional[str] = Query(None), end: Optional[str] = Query(None),
              page: int = Query(1, ge=1), page_size: int = Query(25, ge=1, le=200), db: Session = Depends(get_db)):
    where=[]; params={}
    if exam_code: where.append("exam_code = :exam"); params["exam"]=exam_code
    if start: where.append("created_at >= :start"); params["start"]=start
    if end: where.append("created_at < :end"); params["end"]=end
    where_sql=("WHERE "+ " AND ".join(where)) if where else ""
    offset=(page-1)*page_size
    q=f"""
      SELECT id::text, exam_code, status, created_at::text, started_at::text, finished_at::text
      FROM calibration_runs {where_sql}
      ORDER BY created_at DESC LIMIT :lim OFFSET :off
    """
    params.update({"lim":page_size,"off":offset})
    rows=db.execute(text(q), params).all()
    return [RunRow(id=r[0], exam_code=r[1], status=r[2], created_at=r[3], started_at=r[4], finished_at=r[5]) for r in rows]

class RunDetail(BaseModel):
    id: str; exam_code: str; status: str; params: dict; history: list
    result: Optional[dict]=None; error: Optional[str]=None; created_at: str; started_at: Optional[str]=None; finished_at: Optional[str]=None

@router.get("/exposure/calibrate_sh/runs/{run_id}", response_model=RunDetail, dependencies=[Depends(require_roles("admin"))])
def run_detail(run_id: str, db: Session = Depends(get_db)):
    row=db.execute(text("""
      SELECT id::text, exam_code, status, params, history, result, error, created_at::text, started_at::text, finished_at::text
      FROM calibration_runs WHERE id=:id
    """), {"id": run_id}).first()
    if not row: raise HTTPException(404, "Run not found")
    return RunDetail(id=row[0], exam_code=row[1], status=row[2], params=row[3], history=row[4] or [], result=row[5], error=row[6], created_at=row[7], started_at=row[8], finished_at=row[9])