"""Password hashing and short-lived admin JWTs using explicit algorithms."""

from __future__ import annotations

import base64
from datetime import datetime, timezone
import hmac
import re
import uuid

from argon2 import Parameters, PasswordHasher, extract_parameters
from argon2.exceptions import HashingError, InvalidHashError, VerificationError
from argon2.low_level import Type
import jwt

PASSWORD_MEMORY_KIB = 65_536
PASSWORD_TIME_COST = 3
PASSWORD_PARALLELISM = 1
_PASSWORD_HASHER = PasswordHasher(time_cost=PASSWORD_TIME_COST, memory_cost=PASSWORD_MEMORY_KIB,
                                  parallelism=PASSWORD_PARALLELISM, salt_len=16, hash_len=32, type=Type.ID)
JWT_ISSUER = "banglapsg-api"
JWT_AUDIENCE = "banglapsg-admin"
JWT_CITIZEN_AUDIENCE = "banglapsg-citizen"


def validate_password_hash(encoded: str) -> Parameters:
    """Accept bounded, canonical Argon2id PHC hashes; never downgrade silently.

    Old PBKDF2 deployments must regenerate the administrator hash with the
    provisioning helper. A trusted configuration string still receives cost
    bounds so a typo cannot request unbounded allocation before login.
    """
    message = ("BACKEND_ADMIN_PASSWORD_HASH must be a generated Argon2id hash; "
               "regenerate legacy hashes with python scripts/create_admin.py")
    if not isinstance(encoded, str) or len(encoded) > 512:
        raise ValueError(message)
    try:
        match = re.fullmatch(r"\$argon2id\$v=19\$m=([0-9]+),t=([0-9]+),p=([0-9]+)\$([A-Za-z0-9+/]+)\$([A-Za-z0-9+/]+)", encoded)
        if match is None:
            raise ValueError("Unsupported password-hash encoding")
        parameters = extract_parameters(encoded)
        if (parameters.type != Type.ID or parameters.version != 19
                or not 19_456 <= parameters.memory_cost <= 262_144
                or not 2 <= parameters.time_cost <= 6 or not 1 <= parameters.parallelism <= 4
                or not 16 <= parameters.salt_len <= 64 or not 32 <= parameters.hash_len <= 64):
            raise ValueError("Unsupported Argon2id parameters")
        for value in match.group(4, 5):
            decoded = base64.b64decode(value + "=" * (-len(value) % 4), validate=True)
            if base64.b64encode(decoded).decode("ascii").rstrip("=") != value:
                raise ValueError("Noncanonical password-hash encoding")
    except (ValueError, TypeError, InvalidHashError) as exc:
        raise ValueError(message) from exc
    return parameters


def validate_jwt_secret(secret: str) -> None:
    """Reject missing/example signing keys; random provisioning remains required.

    This is not an entropy proof. The helper creates a random URL-safe secret;
    these checks also catch common copied placeholders and repeated strings.
    Length is measured in characters, not merely multibyte UTF-8 bytes.
    """
    message = "BACKEND_JWT_SECRET must be a random, non-placeholder secret of at least 32 characters"
    if (not isinstance(secret, str) or not 32 <= len(secret) <= 4096 or secret != secret.strip()
            or any(ord(char) < 32 or ord(char) == 127 for char in secret)):
        raise ValueError(message)
    try:
        secret.encode("utf-8")
    except UnicodeError as exc:
        raise ValueError(message) from exc
    lowered = re.sub(r"[^a-z0-9]", "", secret.casefold())
    placeholders = ("changeme", "changethis", "replaceme", "replacewith", "placeholder", "yoursecret",
                    "yourjwtsecret", "yourrandomsecret", "defaultsecret", "supersecret", "insecuresecret",
                    "thisisasecret", "examplesecret", "secretkeyhere", "jwtsecretgoeshere")
    if any(value in lowered for value in placeholders) or len(set(secret)) < 8:
        raise ValueError(message)
    if lowered in {"abcdefghijklmnopqrstuvwxyz0123456789", "0123456789abcdefghijklmnopqrstuvwxyz",
                   "0123456789abcdef0123456789abcdef"}:
        raise ValueError(message)
    for period in range(1, min(len(secret) // 2, 64) + 1):
        if secret == (secret[:period] * ((len(secret) + period - 1) // period))[:len(secret)]:
            raise ValueError(message)


def hash_password(password: str) -> str:
    if not isinstance(password, str) or not 8 <= len(password) <= 256:
        raise ValueError("Admin password must contain 8–256 characters")
    try:
        password.encode("utf-8")
        return _PASSWORD_HASHER.hash(password)
    except (UnicodeError, HashingError) as exc:
        raise ValueError("The administrator password could not be hashed safely") from exc


def verify_password(password: str, encoded: str) -> bool:
    if not isinstance(password, str) or not 1 <= len(password) <= 256:
        return False
    try:
        validate_password_hash(encoded)
        return _PASSWORD_HASHER.verify(encoded, password)
    except (ValueError, VerificationError, InvalidHashError, UnicodeError):
        return False


def authenticate_admin(username: str, password: str, settings) -> bool:
    """Always verify the password, including for a wrong username."""
    try:
        name_matches = isinstance(username, str) and hmac.compare_digest(username.encode("utf-8"), settings.admin_username.encode("utf-8"))
    except UnicodeError:
        name_matches = False
    password_matches = verify_password(password, settings.admin_password_hash)
    return bool(name_matches and password_matches)


def create_access_token(username: str, settings) -> tuple[str, int]:
    settings.validate()
    validate_jwt_secret(settings.jwt_secret)
    if not isinstance(username, str) or not hmac.compare_digest(username.encode("utf-8"), settings.admin_username.encode("utf-8")):
        raise ValueError("Only the configured administrator can receive a token")
    now = int(datetime.now(timezone.utc).timestamp())
    lifetime = settings.jwt_expire_minutes * 60
    token = jwt.encode({"sub": username, "role": "admin", "iat": now, "nbf": now, "exp": now + lifetime,
                        "iss": JWT_ISSUER, "aud": JWT_AUDIENCE, "jti": str(uuid.uuid4())},
                       settings.jwt_secret, algorithm="HS256")
    return token, lifetime


def verify_access_token(token: str, settings) -> str:
    if not isinstance(token, str) or not token or len(token) > 8192:
        raise ValueError("Invalid administrator token")
    try:
        validate_jwt_secret(settings.jwt_secret)
        claims = jwt.decode(token, settings.jwt_secret, algorithms=["HS256"], issuer=JWT_ISSUER,
                            audience=JWT_AUDIENCE, options={"require": ["sub", "role", "exp", "iat", "nbf", "iss", "aud", "jti"]})
        if (any(type(claims.get(name)) is not int for name in ("iat", "nbf", "exp"))
                or not 0 < claims["exp"] - claims["iat"] <= settings.jwt_expire_minutes * 60):
            raise ValueError("Invalid token lifetime")
        if claims.get("role") != "admin" or not isinstance(claims.get("sub"), str) or not isinstance(claims.get("jti"), str) or not claims["jti"]:
            raise ValueError("Invalid administrator claims")
        if not hmac.compare_digest(claims["sub"].encode("utf-8"), settings.admin_username.encode("utf-8")):
            raise ValueError("Unknown administrator")
    except (jwt.PyJWTError, ValueError, TypeError) as exc:
        raise ValueError("Invalid or expired administrator token") from exc
    return claims["sub"]


def hash_citizen_password(password: str) -> str:
    """Hash a citizen password with the same bounded Argon2id settings."""
    if not isinstance(password, str) or not 10 <= len(password) <= 256:
        raise ValueError("Password must contain 10–256 characters")
    try:
        password.encode("utf-8")
        return _PASSWORD_HASHER.hash(password)
    except (UnicodeError, HashingError) as exc:
        raise ValueError("The password could not be hashed safely") from exc


def create_citizen_access_token(account_id: str, settings) -> tuple[str, int]:
    """Create a citizen-only token with an audience separate from admin JWTs."""
    settings.validate()
    validate_jwt_secret(settings.jwt_secret)
    try:
        canonical_id = str(uuid.UUID(account_id))
    except (ValueError, TypeError, AttributeError) as exc:
        raise ValueError("Invalid citizen account ID") from exc
    now = int(datetime.now(timezone.utc).timestamp())
    lifetime = settings.jwt_expire_minutes * 60
    token = jwt.encode(
        {"sub": canonical_id, "role": "citizen", "iat": now, "nbf": now, "exp": now + lifetime,
         "iss": JWT_ISSUER, "aud": JWT_CITIZEN_AUDIENCE, "jti": str(uuid.uuid4())},
        settings.jwt_secret,
        algorithm="HS256",
    )
    return token, lifetime


def verify_citizen_access_token(token: str, settings) -> str:
    """Validate citizen tokens; admin tokens fail on the separate audience/role."""
    if not isinstance(token, str) or not token or len(token) > 8192:
        raise ValueError("Invalid citizen token")
    try:
        validate_jwt_secret(settings.jwt_secret)
        claims = jwt.decode(
            token,
            settings.jwt_secret,
            algorithms=["HS256"],
            issuer=JWT_ISSUER,
            audience=JWT_CITIZEN_AUDIENCE,
            options={"require": ["sub", "role", "exp", "iat", "nbf", "iss", "aud", "jti"]},
        )
        if (any(type(claims.get(name)) is not int for name in ("iat", "nbf", "exp"))
                or not 0 < claims["exp"] - claims["iat"] <= settings.jwt_expire_minutes * 60
                or claims.get("role") != "citizen"
                or not isinstance(claims.get("jti"), str)
                or not claims["jti"]):
            raise ValueError("Invalid citizen claims")
        return str(uuid.UUID(claims["sub"]))
    except (jwt.PyJWTError, ValueError, TypeError, AttributeError) as exc:
        raise ValueError("Invalid or expired citizen token") from exc
