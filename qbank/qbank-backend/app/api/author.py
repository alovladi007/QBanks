from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, constr
from typing import List, Optional
from sqlalchemy.orm import Session
from sqlalchemy import select, func
from app.core.database import get_db
from app.core.config import TENANT_ID
from app.core.auth import require_roles, TokenData
from app.models.orm import Topic, Question, QuestionVersion, QuestionOption, QuestionPublication

router = APIRouter()

class OptionIn(BaseModel):
    label: constr(min_length=1, max_length=1)
    text_md: str
    is_correct: bool

class QuestionCreate(BaseModel):
    external_ref: Optional[str] = None
    topic_name: str
    exam_code: str = "DEMO-EXAM"
    stem_md: str
    lead_in: str
    rationale_md: str
    difficulty_label: Optional[str] = "medium"
    options: List[OptionIn]

@router.post("/questions", dependencies=[Depends(require_roles("author","admin"))])
def create_question(payload: QuestionCreate, user: TokenData = Depends(require_roles("author","admin")), db: Session = Depends(get_db)):
    t = db.scalar(select(Topic).where(Topic.name == payload.topic_name))
    if not t:
        t = Topic(tenant_id=TENANT_ID, parent_id=None, name=payload.topic_name, blueprint_code=None)
        db.add(t); db.flush()
    q = Question(tenant_id=TENANT_ID, external_ref=payload.external_ref, created_by=user.sub, is_deleted=False)
    db.add(q); db.flush()
    next_v = (db.scalar(select(func.coalesce(func.max(QuestionVersion.version), 0)).where(QuestionVersion.question_id == q.id)) or 0) + 1
    qv = QuestionVersion(
        question_id=q.id, version=next_v, state="published",
        stem_md=payload.stem_md, lead_in=payload.lead_in, rationale_md=payload.rationale_md,
        difficulty_label=payload.difficulty_label, topic_id=t.id, tags={}, assets=[], references=[]
    )
    db.add(qv); db.flush()
    for o in payload.options:
        db.add(QuestionOption(question_version_id=qv.id, option_label=o.label.upper(), option_text_md=o.text_md, is_correct=o.is_correct))
    db.add(QuestionPublication(question_id=q.id, live_version=next_v, exam_code=payload.exam_code, tenant_id=TENANT_ID))
    db.commit()
    return {"question_id": q.id, "version": next_v, "topic_id": t.id}