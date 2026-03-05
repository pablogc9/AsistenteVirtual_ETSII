from __future__ import annotations

import os
from datetime import datetime, timedelta, timezone
from typing import Optional

from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from jose import JWTError, jwt

# --------------------------
# Configuración
# --------------------------

SECRET_KEY                  = os.getenv("JWT_SECRET_KEY", "dev_secret_inseguro")
ALGORITHM                   = "HS256"
ACCESS_TOKEN_EXPIRE_HOURS   = 8

# Le dice a FastAPI dónde est´´a el endpoint de login para obtener el token
# Esto habilita el botón "Authorize" en /docs automaticamente
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/login")

# --------------------------
# Creación de tokens
# --------------------------

def create_access_token(
    data: dict,
    expires_delta: Optional[timedelta] = None,
) -> str:
    """
    Genera un JWT firmado con los datos proporcionados y una fecha de expiración
    """
    to_encode = data.copy()
    expire = datetime.now(timezone.utc) + (
        expires_delta or timedelta(hours=ACCESS_TOKEN_EXPIRE_HOURS)
    )
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)


# --------------------------
# Dependencia de FastAPI
# --------------------------

def get_current_admin(token: str = Depends(oauth2_scheme)) -> str:
    """
    Dependencia que protege los endpoints administrativos.

    Extrae y valida el JWT de la cabecera Authorization: Bearer <token>.
    Lanza HTTP 401 si el token es inválido, ha expierado o no contiene 'sub'.
    """
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Token inválido o expirado. Inicia sesión de nuevo.",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username: str | None = payload.get("sub")
        if username is None:
            raise credentials_exception
    except JWTError:
        raise credentials_exception

    return username


__all__ = ["create_access_token", "get_current_admin"]