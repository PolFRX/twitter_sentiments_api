from fastapi import HTTPException, status

from config import API_KEY


def api_key_auth(api_key: str):
    if api_key != API_KEY:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Forbidden"
        )
