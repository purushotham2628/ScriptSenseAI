from fastapi import APIRouter

from backend.core.security import create_access_token, hash_password

router = APIRouter()


@router.post("/token")
async def token(username: str, password: str):
    # Template auth endpoint. In production, verify against PostgreSQL users table.
    return {"access_token": create_access_token(username), "token_type": "bearer"}


@router.post("/register")
async def register(email: str, password: str):
    return {"email": email, "hashed_password_preview": hash_password(password)[:16], "status": "template_created"}
