from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column
from sqlalchemy import BigInteger, Integer, String, Text, Boolean, ForeignKey, JSON, Float

class Base(DeclarativeBase): pass

class Topic(Base):
    __tablename__ = "topics"
    id: Mapped[int] = mapped_column(BigInteger, primary_key=True)
    tenant_id: Mapped[str] = mapped_column(String)
    parent_id: Mapped[int | None] = mapped_column(BigInteger, ForeignKey("topics.id"), nullable=True)
    name: Mapped[str] = mapped_column(String)
    blueprint_code: Mapped[str | None] = mapped_column(String, nullable=True)

class Question(Base):
    __tablename__ = "questions"
    id: Mapped[int] = mapped_column(BigInteger, primary_key=True)
    tenant_id: Mapped[str] = mapped_column(String)
    external_ref: Mapped[str | None] = mapped_column(String, nullable=True)
    created_by: Mapped[str] = mapped_column(String)
    is_deleted: Mapped[bool] = mapped_column(Boolean, default=False)

class QuestionVersion(Base):
    __tablename__ = "question_versions"
    id: Mapped[int] = mapped_column(BigInteger, primary_key=True)
    question_id: Mapped[int] = mapped_column(BigInteger, ForeignKey("questions.id"))
    version: Mapped[int] = mapped_column(Integer)
    state: Mapped[str] = mapped_column(String)
    stem_md: Mapped[str] = mapped_column(Text)
    lead_in: Mapped[str] = mapped_column(Text)
    rationale_md: Mapped[str] = mapped_column(Text)
    difficulty_label: Mapped[str | None] = mapped_column(String, nullable=True)
    bloom_level: Mapped[int | None] = mapped_column(Integer, nullable=True)
    topic_id: Mapped[int | None] = mapped_column(BigInteger, ForeignKey("topics.id"), nullable=True)
    tags: Mapped[dict] = mapped_column(JSON)
    assets: Mapped[list] = mapped_column(JSON)
    references: Mapped[list] = mapped_column(JSON)

class QuestionOption(Base):
    __tablename__ = "question_options"
    id: Mapped[int] = mapped_column(BigInteger, primary_key=True)
    question_version_id: Mapped[int] = mapped_column(BigInteger, ForeignKey("question_versions.id"))
    option_label: Mapped[str] = mapped_column(String(1))
    option_text_md: Mapped[str] = mapped_column(Text)
    is_correct: Mapped[bool] = mapped_column(Boolean)

class QuestionPublication(Base):
    __tablename__ = "question_publications"
    id: Mapped[int] = mapped_column(BigInteger, primary_key=True)
    question_id: Mapped[int] = mapped_column(BigInteger, ForeignKey("questions.id"))
    live_version: Mapped[int] = mapped_column(Integer)
    exam_code: Mapped[str] = mapped_column(String)
    tenant_id: Mapped[str] = mapped_column(String)

class QuizSession(Base):
    __tablename__ = "quiz_sessions"
    id: Mapped[str] = mapped_column(String, primary_key=True)
    user_id: Mapped[str] = mapped_column(String)
    tenant_id: Mapped[str] = mapped_column(String)
    mode: Mapped[str] = mapped_column(String)
    adaptive: Mapped[bool] = mapped_column(Boolean, default=True)
    exam_code: Mapped[str | None] = mapped_column(String, nullable=True)

class QuizItem(Base):
    __tablename__ = "quiz_items"
    id: Mapped[int] = mapped_column(BigInteger, primary_key=True)
    quiz_id: Mapped[str] = mapped_column(String)
    question_id: Mapped[int] = mapped_column(BigInteger)
    version: Mapped[int] = mapped_column(Integer)
    position: Mapped[int] = mapped_column(Integer)

class UserResponse(Base):
    __tablename__ = "user_responses"
    id: Mapped[int] = mapped_column(BigInteger, primary_key=True)
    quiz_id: Mapped[str] = mapped_column(String)
    user_id: Mapped[str] = mapped_column(String)
    question_id: Mapped[int] = mapped_column(BigInteger)
    version: Mapped[int] = mapped_column(Integer)
    option_label: Mapped[str] = mapped_column(String(1))
    is_correct: Mapped[bool] = mapped_column(Boolean)
    time_taken_ms: Mapped[int | None] = mapped_column(Integer, nullable=True)

class ItemCalibration(Base):
    __tablename__ = "item_calibration"
    question_id: Mapped[int] = mapped_column(BigInteger, primary_key=True)
    version: Mapped[int] = mapped_column(Integer, primary_key=True)
    model: Mapped[str] = mapped_column(String, primary_key=True)
    a: Mapped[float | None] = mapped_column(Float)
    b: Mapped[float | None] = mapped_column(Float)
    c: Mapped[float | None] = mapped_column(Float)
    n_respondents: Mapped[int | None] = mapped_column(Integer)

class ItemExposureControl(Base):
    __tablename__ = "item_exposure_control"
    question_id: Mapped[int] = mapped_column(BigInteger, primary_key=True)
    version: Mapped[int] = mapped_column(Integer, primary_key=True)
    sh_p: Mapped[float] = mapped_column(Float)

class FeatureFlag(Base):
    __tablename__ = "feature_flags"
    key: Mapped[str] = mapped_column(String, primary_key=True)
    enabled: Mapped[bool] = mapped_column(Boolean, default=True)
    value_json: Mapped[dict] = mapped_column(JSON)

class CohortAssignment(Base):
    __tablename__ = "cohort_assignments"
    user_id: Mapped[str] = mapped_column(String, primary_key=True)
    cohort_key: Mapped[str] = mapped_column(String, primary_key=True)
    cohort_value: Mapped[str] = mapped_column(String)