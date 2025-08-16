from fastapi import APIRouter
from pydantic import BaseModel
from typing import List
from app.core.auth import create_token

router = APIRouter()

class MockLogin(BaseModel):
    user_id: str
    roles: List[str]

@router.post("/mock-login")
def mock_login(payload: MockLogin):
    token = create_token(payload.user_id, payload.roles)
    return {"access_token": token, "token_type": "bearer", "roles": payload.roles}