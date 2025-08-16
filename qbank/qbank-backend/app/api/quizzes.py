from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel, Field, constr
from typing import List, Optional, Literal
from uuid import uuid4
from datetime import datetime, timedelta
import json
from sqlalchemy.orm import Session
from sqlalchemy import select
from app.core.cache import redis_client, bump_exposure
from app.core.database import get_db
from app.core.auth import require_roles, TokenData
from app.models.orm import QuestionVersion, QuestionOption, QuestionPublication, ItemCalibration, ItemExposureControl
from app.services.selector_choice import get_selector_for_user
from app.services.adaptive import select_vanilla, select_sympson_hetter

router = APIRouter()

class QuizFilters(BaseModel):
  topics: Optional[List[str]] = None
  difficulty: Optional[List[Literal["easy","medium","hard"]]] = None
  num_questions: int = Field(ge=1, le=120, default=40)
  mode: Literal["tutor","exam"] = "tutor"
  exam_code: Optional[str] = "DEMO-EXAM"

class QuizCreate(BaseModel):
  tenant_id: constr(min_length=8)
  filters: QuizFilters
  adaptive: bool = True

class QuizCreated(BaseModel):
  quiz_id: str
  question_ids: List[int]
  expires_at: datetime
  mode: Literal["tutor","exam"]

class NextQuestion(BaseModel):
  question_id: int
  version: int
  payload: dict

class AnswerSubmit(BaseModel):
  question_id: int
  selected: constr(min_length=1, max_length=1)
  time_taken_ms: Optional[int] = 0
  client_latency_ms: Optional[int] = 0

class AnswerResult(BaseModel):
  correct: bool
  correct_option: constr(min_length=1, max_length=1)
  explanation: dict
  difficulty: float

def _rk(qid: str, suf: str) -> str: return f"quiz:{qid}:{suf}"

@router.post("", response_model=QuizCreated, status_code=201, dependencies=[Depends(require_roles("student","admin"))])
def create_quiz(payload: QuizCreate, user: TokenData = Depends(require_roles("student","admin")), db: Session = Depends(get_db)):
  quiz_id = str(uuid4()); mode = payload.filters.mode
  expires_at = datetime.utcnow() + timedelta(hours=2)
  stmt = select(QuestionPublication, QuestionVersion).join(
    QuestionVersion,
    (QuestionVersion.question_id == QuestionPublication.question_id) & (QuestionVersion.version == QuestionPublication.live_version)
  ).where(QuestionPublication.exam_code == (payload.filters.exam_code or "DEMO-EXAM"), QuestionVersion.state == "published")
  rows = db.execute(stmt).all()
  versions = [r[1] for r in rows]
  vcache = [{"q": v.question_id, "v": v.version, "t": v.topic_id, "d": v.difficulty_label} for v in versions]
  redis_client.set(_rk(quiz_id, "versions"), json.dumps(vcache), ex=7200)
  redis_client.set(_rk(quiz_id, "cursor"), 0, ex=7200)
  redis_client.set(_rk(quiz_id, "mode"), mode, ex=7200)
  redis_client.set(_rk(quiz_id, "user"), user.sub, ex=7200)
  selector = get_selector_for_user(db, user.sub)
  redis_client.set(_rk(quiz_id, "selector"), selector, ex=7200)
  qids = list({v["q"] for v in vcache})[:payload.filters.num_questions]
  return QuizCreated(quiz_id=quiz_id, question_ids=qids, expires_at=expires_at, mode=mode)

@router.get("/{quiz_id}/next", response_model=NextQuestion, dependencies=[Depends(require_roles("student","admin"))])
def next_question(quiz_id: str, db: Session = Depends(get_db)):
  import json
  raw = redis_client.get(_rk(quiz_id,"versions"))
  if not raw: raise HTTPException(404, "Quiz not found or expired")
  versions = json.loads(raw)
  selector = redis_client.get(_rk(quiz_id,"selector")) or "vanilla"
  curk = _rk(quiz_id, "cursor"); cur = int(redis_client.get(curk) or 0)
  if cur >= len(versions): raise HTTPException(404, "No more questions")
  window = versions[cur : min(cur+20, len(versions))]
  candidates = []
  for w in window:
    ic = db.scalar(select(ItemCalibration).where(ItemCalibration.question_id==w["q"], ItemCalibration.version==w["v"]).limit(1))
    exp = db.scalar(select(ItemExposureControl).where(ItemExposureControl.question_id==w["q"], ItemExposureControl.version==w["v"]).limit(1))
    a = (ic.a if ic and ic.a is not None else 1.0) if ic else 1.0
    b = (ic.b if ic and ic.b is not None else 0.0) if ic else 0.0
    c = (ic.c if ic and ic.c is not None else 0.2) if ic else 0.2
    sh_p = exp.sh_p if exp else 1.0
    candidates.append({"question_id": w["q"], "version": w["v"], "topic_id": w["t"], "a": a, "b": b, "c": c, "sh_p": sh_p})
  best = (select_sympson_hetter if selector=="sympson_hetter" else select_vanilla)(candidates, theta=0.0) or candidates[0]
  redis_client.set(curk, cur+1)
  qv = db.scalar(select(QuestionVersion).where(QuestionVersion.question_id==best["question_id"], QuestionVersion.version==best["version"]))
  if not qv: raise HTTPException(500, "Item not found")
  opts = db.execute(select(QuestionOption).where(QuestionOption.question_version_id==qv.id)).scalars().all()
  bump_exposure(best["question_id"], best["version"])
  payload = {"stem_md": qv.stem_md, "lead_in": qv.lead_in, "options": [{"label": o.option_label, "text": o.option_text_md} for o in opts]}
  return NextQuestion(question_id=best["question_id"], version=best["version"], payload=payload)

@router.post("/{quiz_id}/answers", response_model=AnswerResult, dependencies=[Depends(require_roles("student","admin"))])
def submit_answer(quiz_id: str, payload: AnswerSubmit, db: Session = Depends(get_db)):
  qv = db.scalar(select(QuestionVersion).where(QuestionVersion.question_id==payload.question_id).order_by(QuestionVersion.version.desc()))
  if not qv: raise HTTPException(404, "Question not found")
  opts = db.execute(select(QuestionOption).where(QuestionOption.question_version_id==qv.id)).scalars().all()
  correct = next((o.option_label for o in opts if o.is_correct), None)
  if not correct: raise HTTPException(500, "No correct option set")
  ok = (payload.selected.upper() == correct)
  return AnswerResult(correct=ok, correct_option=correct, explanation={"rationale_md": qv.rationale_md}, difficulty=0.5)