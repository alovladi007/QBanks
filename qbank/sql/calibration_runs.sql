CREATE TABLE IF NOT EXISTS calibration_runs (
  id UUID PRIMARY KEY,
  exam_code TEXT NOT NULL,
  status TEXT NOT NULL,
  params JSONB NOT NULL,
  history JSONB NOT NULL DEFAULT '[]'::jsonb,
  result JSONB,
  error TEXT,
  created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
  started_at TIMESTAMPTZ,
  finished_at TIMESTAMPTZ
);
CREATE INDEX IF NOT EXISTS idx_cal_runs_exam ON calibration_runs(exam_code);
CREATE INDEX IF NOT EXISTS idx_cal_runs_created ON calibration_runs(created_at DESC);