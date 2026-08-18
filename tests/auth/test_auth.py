import pytest
import time
from database.auth import AuthManager, Roles

def test_password_hashing():
    pwd = "MySecretPassword123"
    hashed = AuthManager.hash_password(pwd)
    
    assert hashed != pwd
    assert AuthManager.verify_password(pwd, hashed) is True
    assert AuthManager.verify_password("wrong_password", hashed) is False

def test_jwt_token_flow():
    user_id = 99
    username = "quant_trader"
    role = Roles.TRADER
    
    token = AuthManager.create_jwt_token(user_id, username, role)
    assert isinstance(token, str)
    
    payload = AuthManager.verify_jwt_token(token)
    assert payload["sub"] == str(user_id)
    assert payload["username"] == username
    assert payload["role"] == role

def test_totp_token_flow():
    secret = AuthManager.generate_totp_secret()
    assert len(secret) == 32
    
    uri = AuthManager.get_totp_provisioning_uri("test_user", secret)
    assert "otpauth://totp/" in uri
    assert "secret=" in uri
