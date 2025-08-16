# QBank (v9)
- FastAPI backend + Redis/RQ worker
- Full Sympson–Hetter (S–H) iterative calibration
- Admin UI (Next.js) with progress & run history
- Postgres-backed calibration runs + CSV diff
- Filters & pagination on run list

Quick start:
- `psql $DATABASE_URL -f sql/content_ddl.sql`
- `psql $DATABASE_URL -f sql/item_exposure_control.sql`
- `psql $DATABASE_URL -f sql/feature_flags.sql`
- `psql $DATABASE_URL -f sql/calibration_runs.sql`
- `pip install -r qbank-backend/requirements.txt`
- API: `uvicorn app.main:app --reload` (in `qbank-backend/`)
- Worker: `python -m app.jobs.worker`
- Admin UI: `cd admin-ui && npm install && npm run dev`