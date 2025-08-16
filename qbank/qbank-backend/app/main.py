from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.api.quizzes import router as quizzes_router
from app.api.author import router as author_router
from app.api.auth import router as auth_router
from app.api.admin import router as admin_router
from app.api.admin_runs import router as runs_router
from app.api.admin_status import router as status_router

app = FastAPI(title="QBank API v9", version="9.0.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])
app.include_router(auth_router, prefix="/v1/auth", tags=["auth"])
app.include_router(quizzes_router, prefix="/v1/quizzes", tags=["quizzes"])
app.include_router(author_router, prefix="/v1/author", tags=["authoring"])
app.include_router(admin_router, prefix="/v1/admin", tags=["admin"])
app.include_router(runs_router, prefix="/v1/admin", tags=["calibration-runs"])
app.include_router(status_router, prefix="/v1/admin", tags=["calibration-status"])

@app.get("/health")
def health(): return {"status": "ok"}