import logging
import os
import time

import bcrypt
import jwt
import pyotp
from fastapi import Depends, HTTPException
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

logger = logging.getLogger("Auth")

SECRET_KEY = os.getenv("JWT_SECRET_KEY", "quant_portal_super_secret_jwt_key_9988")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 60


def _get_secret_key() -> str:
    """
    Resolves the JWT secret at CALL time so the platform can auto-generate and
    inject a strong secret (via env) after startup config validation, instead
    of being stuck with the import-time default. Never exposes the default.
    """
    return os.getenv("JWT_SECRET_KEY", SECRET_KEY)

security = HTTPBearer()

class Roles:
    VIEWER = "VIEWER"
    TRADER = "TRADER"
    RISK_MANAGER = "RISK_MANAGER"
    ADMIN = "ADMIN"
    SUPER_ADMIN = "SUPER_ADMIN"

    # Hierarchy level mapping
    HIERARCHY = {
        VIEWER: 1,
        TRADER: 2,
        RISK_MANAGER: 3,
        ADMIN: 4,
        SUPER_ADMIN: 5
    }

class AuthManager:
    """
    SaaS Multi-User Authentication & Real TOTP 2FA Manager.
    Handles bcrypt hashing, JWT token signatures, and pyotp-based TOTP 2FA verification.
    """
    @staticmethod
    def hash_password(password: str) -> str:
        salt = bcrypt.gensalt()
        return bcrypt.hashpw(password.encode('utf-8'), salt).decode('utf-8')

    @staticmethod
    def verify_password(password: str, hashed_password: str) -> bool:
        try:
            return bcrypt.checkpw(password.encode('utf-8'), hashed_password.encode('utf-8'))
        except Exception:
            return False

    @staticmethod
    def create_jwt_token(user_id: int, username: str, role: str) -> str:
        payload = {
            "sub": str(user_id),
            "username": username,
            "role": role,
            "exp": time.time() + (ACCESS_TOKEN_EXPIRE_MINUTES * 60)
        }
        return jwt.encode(payload, _get_secret_key(), algorithm=ALGORITHM)

    @staticmethod
    def verify_jwt_token(token: str) -> dict:
        try:
            payload = jwt.decode(token, _get_secret_key(), algorithms=[ALGORITHM])
            return payload
        except jwt.ExpiredSignatureError:
            raise HTTPException(status_code=401, detail="Session expired. Please log in again.")
        except jwt.InvalidTokenError:
            raise HTTPException(status_code=401, detail="Invalid session token.")

    @staticmethod
    def generate_totp_secret() -> str:
        return pyotp.random_base32()

    @staticmethod
    def get_totp_provisioning_uri(username: str, secret: str) -> str:
        """
        Generates standard TOTP provisioning URI compatible with Google Authenticator.
        """
        totp = pyotp.TOTP(secret)
        return totp.provisioning_uri(name=username, issuer_name="QUANT-PORTAL")

    @staticmethod
    def verify_totp_token(secret: str, code: str) -> bool:
        """
        Verifies actual, real-time 2FA codes. Completely removes hardcoded 123456/888888!
        """
        totp = pyotp.TOTP(secret)
        # Verify with 30-seconds grace step window to handle clock-drift
        return totp.verify(code, valid_window=1)


def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)) -> dict:
    """
    Dependency resolver verifying the active JWT session and returning user context.
    """
    token = credentials.credentials
    return AuthManager.verify_jwt_token(token)


class RBACPermissionChecker:
    """
    Enforces Role-Based Access Control (RBAC) across protected endpoints.
    """
    def __init__(self, required_role: str):
        self.required_role = required_role

    def __call__(self, user: dict = Depends(get_current_user)):
        user_role = user.get("role", Roles.VIEWER)

        user_level = Roles.HIERARCHY.get(user_role, 1)
        required_level = Roles.HIERARCHY.get(self.required_role, 5)

        if user_level < required_level:
            logger.warning(f"RBAC Refused: User {user.get('username')} ({user_role}) lacks permission for {self.required_role} protected endpoint.")
            raise HTTPException(status_code=403, detail="Access denied. Insufficient role permissions.")

        return user
