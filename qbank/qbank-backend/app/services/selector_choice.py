from sqlalchemy.orm import Session
from sqlalchemy import select
from app.models.orm import FeatureFlag, CohortAssignment

def get_selector_for_user(db: Session, user_id: str) -> str:
    ff = db.scalar(select(FeatureFlag).where(FeatureFlag.key == "selector_strategy"))
    default = "sympson_hetter" if (ff and ff.enabled and (ff.value_json or {}).get("value") == "sympson_hetter") else "vanilla"
    cohort = db.scalar(select(CohortAssignment).where(CohortAssignment.user_id==user_id, CohortAssignment.cohort_key=="selector_strategy"))
    return cohort.cohort_value if cohort else default