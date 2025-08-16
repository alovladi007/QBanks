CREATE TABLE IF NOT EXISTS item_exposure_control (
  question_id BIGINT NOT NULL,
  version INT NOT NULL,
  sh_p DOUBLE PRECISION NOT NULL DEFAULT 1.0,
  updated_at TIMESTAMPTZ NOT NULL DEFAULT now(),
  PRIMARY KEY (question_id, version)
);