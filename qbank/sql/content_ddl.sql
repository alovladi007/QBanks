CREATE TABLE IF NOT EXISTS topics(
  id BIGINT PRIMARY KEY GENERATED ALWAYS AS IDENTITY,
  tenant_id TEXT NOT NULL,
  parent_id BIGINT NULL,
  name TEXT NOT NULL,
  blueprint_code TEXT NULL
);
CREATE TABLE IF NOT EXISTS questions(
  id BIGINT PRIMARY KEY GENERATED ALWAYS AS IDENTITY,
  tenant_id TEXT NOT NULL,
  external_ref TEXT,
  created_by TEXT NOT NULL,
  is_deleted BOOLEAN NOT NULL DEFAULT FALSE
);
CREATE TABLE IF NOT EXISTS question_versions(
  id BIGINT PRIMARY KEY GENERATED ALWAYS AS IDENTITY,
  question_id BIGINT NOT NULL REFERENCES questions(id),
  version INT NOT NULL,
  state TEXT NOT NULL,
  stem_md TEXT NOT NULL,
  lead_in TEXT NOT NULL,
  rationale_md TEXT NOT NULL,
  difficulty_label TEXT,
  bloom_level INT,
  topic_id BIGINT REFERENCES topics(id),
  tags JSONB NOT NULL DEFAULT '{}'::jsonb,
  assets JSONB NOT NULL DEFAULT '[]'::jsonb,
  references JSONB NOT NULL DEFAULT '[]'::jsonb
);
CREATE TABLE IF NOT EXISTS question_options(
  id BIGINT PRIMARY KEY GENERATED ALWAYS AS IDENTITY,
  question_version_id BIGINT NOT NULL REFERENCES question_versions(id),
  option_label TEXT NOT NULL,
  option_text_md TEXT NOT NULL,
  is_correct BOOLEAN NOT NULL
);
CREATE TABLE IF NOT EXISTS question_publications(
  id BIGINT PRIMARY KEY GENERATED ALWAYS AS IDENTITY,
  question_id BIGINT NOT NULL REFERENCES questions(id),
  live_version INT NOT NULL,
  exam_code TEXT NOT NULL,
  tenant_id TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS quiz_sessions(
  id TEXT PRIMARY KEY,
  user_id TEXT NOT NULL,
  tenant_id TEXT NOT NULL,
  mode TEXT NOT NULL,
  adaptive BOOLEAN NOT NULL DEFAULT TRUE,
  exam_code TEXT
);
CREATE TABLE IF NOT EXISTS quiz_items(
  id BIGINT PRIMARY KEY GENERATED ALWAYS AS IDENTITY,
  quiz_id TEXT NOT NULL,
  question_id BIGINT NOT NULL,
  version INT NOT NULL,
  position INT NOT NULL
);
CREATE TABLE IF NOT EXISTS user_responses(
  id BIGINT PRIMARY KEY GENERATED ALWAYS AS IDENTITY,
  quiz_id TEXT NOT NULL,
  user_id TEXT NOT NULL,
  question_id BIGINT NOT NULL,
  version INT NOT NULL,
  option_label TEXT NOT NULL,
  is_correct BOOLEAN NOT NULL,
  time_taken_ms INT
);
CREATE TABLE IF NOT EXISTS item_calibration(
  question_id BIGINT NOT NULL,
  version INT NOT NULL,
  model TEXT NOT NULL,
  a DOUBLE PRECISION, b DOUBLE PRECISION, c DOUBLE PRECISION,
  n_respondents INT,
  PRIMARY KEY(question_id, version, model)
);