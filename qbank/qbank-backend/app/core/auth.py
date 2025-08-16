from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel
from typing import List
import jwt
from datetime import datetime, timedelta, timezone
from app.core.config import APP_SECRET

class TokenData(BaseModel):
    sub: str
    roles: List[str]

bearer = HTTPBearer()

def create_token(user_id: str, roles: List[str], ttl_minutes: int = 120) -> str:
    now = datetime.now(timezone.utc)
    payload = {"sub": user_id, "roles": roles, "iat": int(now.timestamp()), "exp": int((now + timedelta(minutes=ttl_minutes)).timestamp())}
    return jwt.encode(payload, APP_SECRET, algorithm="HS256")

def get_current_user(creds: HTTPAuthorizationCredentials = Depends(bearer)) -> TokenData:
    try:
        payload = jwt.decode(creds.credentials, APP_SECRET, algorithms=["HS256"])
        return TokenData(sub=payload["sub"], roles=payload.get("roles", []))
    except Exception:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid or expired token")

def require_roles(*required: str):
    def checker(user: TokenData = Depends(get_current_user)):
        roles = set(user.roles)
        if not roles.intersection(set(required)):
            raise HTTPException(status_code=403, detail="Insufficient role")
        return user
    return checker