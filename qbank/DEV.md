## Dev setup
1) Postgres + Redis running; export `DATABASE_URL` and `REDIS_URL` if non-default.
2) Apply schema:

```bash
psql "$DATABASE_URL" -f sql/content_ddl.sql
psql "$DATABASE_URL" -f sql/item_exposure_control.sql
psql "$DATABASE_URL" -f sql/feature_flags.sql
psql "$DATABASE_URL" -f sql/calibration_runs.sql
```

3) Backend:

```bash
cd qbank-backend
pip install -r requirements.txt
uvicorn app.main:app --reload
```

4) Worker:

```bash
python -m app.jobs.worker
```

5) Admin UI:

```bash
cd admin-ui
npm install
npm run dev # http://localhost:4000
```

6) Get admin JWT:

```bash
POST /v1/auth/mock-login {"user_id":"admin","roles":["admin","author","publisher","student"]}
```

7) Start calibration in `/calibration`; review history & export CSV in `/calibration-history`.