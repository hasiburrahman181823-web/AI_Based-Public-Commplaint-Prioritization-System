"""Explicit backend configuration; no usable default admin credentials."""

from __future__ import annotations

from dataclasses import dataclass, field
import os
from pathlib import Path
from urllib.parse import urlsplit

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]


def _boolean(value: str, name: str) -> bool:
    if value.strip().lower() in ("1", "true", "yes", "on"):
        return True
    if value.strip().lower() in ("0", "false", "no", "off"):
        return False
    raise ValueError(f"{name} must be true or false")


@dataclass(frozen=True)
class Settings:
    database_url: str = "sqlite:///./backend/data/backend.sqlite3"
    jwt_secret: str = ""
    admin_username: str = "admin"
    admin_password_hash: str = ""
    jwt_expire_minutes: int = 30
    environment: str = "development"
    allowed_origins: tuple[str, ...] = field(default_factory=tuple)
    trust_proxy_headers: bool = False
    rate_limit_storage_uri: str = "memory://"
    submission_rate_limit: str = "5/minute"
    analysis_rate_limit: str = "10/minute"
    login_rate_limit: str = "3/minute"
    max_text_length: int = 1000
    abuse_max_urls: int = 2
    max_body_bytes: int = 32768
    category_model: str | None = "artifacts/category/char_svm.joblib"
    severity_model: str | None = "artifacts/severity/char_svm.joblib"
    review_policy: str | None = "artifacts/review_policy.json"
    priority_ranker: str | None = "artifacts/priority_ranker.joblib"
    duplicate_model: str | None = "artifacts/duplicates/P2.joblib"
    rules_config: str = "config/priority_rules.yaml"
    embedding_cache_dir: str = ".cache/backend"
    duplicate_top_k: int = 100
    require_ml_ready: bool = False
    allow_public_reads: bool = False

    def validate(self) -> "Settings":
        from backend.security import validate_jwt_secret, validate_password_hash

        if self.environment not in ("development", "production", "test"):
            raise ValueError("BACKEND_ENVIRONMENT must be development, test or production")
        validate_jwt_secret(self.jwt_secret)
        if not isinstance(self.admin_username, str) or not self.admin_username.strip() or self.admin_username != self.admin_username.strip() or len(self.admin_username) > 128:
            raise ValueError("BACKEND_ADMIN_USERNAME must be a nonempty, trimmed name of at most 128 characters")
        validate_password_hash(self.admin_password_hash)
        for name, value, low, high in (
            ("jwt_expire_minutes", self.jwt_expire_minutes, 1, 60),
            ("max_text_length", self.max_text_length, 1, 1000),
            ("abuse_max_urls", self.abuse_max_urls, 0, 20),
            ("max_body_bytes", self.max_body_bytes, 1024, 1024 * 1024),
            ("duplicate_top_k", self.duplicate_top_k, 1, 1000),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or not low <= value <= high:
                raise ValueError(f"{name} must be an integer in [{low}, {high}]")
        if not isinstance(self.database_url, str) or not self.database_url:
            raise ValueError("BACKEND_DATABASE_URL is required")
        if not self.database_url.startswith(("sqlite://", "sqlite+pysqlite://", "postgresql://", "postgresql+psycopg://")):
            raise ValueError("Use a SQLite development URL or a PostgreSQL production URL")
        if not isinstance(self.rate_limit_storage_uri, str) or not self.rate_limit_storage_uri.startswith(("memory://", "redis://", "rediss://")):
            raise ValueError("Rate-limit storage must be memory:// or a Redis URL")
        for name in ("submission_rate_limit", "analysis_rate_limit", "login_rate_limit"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{name} must be a nonempty slowapi rate specification")
            # Parse at startup where the optional backend dependency is present.
            try:
                from limits import parse
                parsed = parse(value)
                if parsed.amount < 1:
                    raise ValueError("positive rate required")
            except ImportError:
                import re
                if not re.fullmatch(r"[1-9]\d*/(?:second|minute|hour|day)", value):
                    raise ValueError(f"Invalid {name}; install backend requirements for full rate syntax")
            except Exception as exc:
                raise ValueError(f"Invalid {name}") from exc
        for origin in self.allowed_origins:
            parsed = urlsplit(origin)
            if parsed.scheme not in ("http", "https") or not parsed.hostname or parsed.username or parsed.password or parsed.path not in ("", "/") or parsed.query or parsed.fragment:
                raise ValueError("Allowed CORS origins must be explicit http(s) origins without paths or credentials")
        for name in ("trust_proxy_headers", "require_ml_ready", "allow_public_reads"):
            if not isinstance(getattr(self, name), bool):
                raise ValueError(f"{name} must be boolean")
        if self.environment == "production":
            if not self.database_url.startswith(("postgresql://", "postgresql+psycopg://")):
                raise ValueError("Production requires PostgreSQL; SQLite is for local development/tests")
            if not self.rate_limit_storage_uri.startswith(("redis://", "rediss://")):
                raise ValueError("Production requires shared Redis rate-limit storage")
            if not self.require_ml_ready:
                raise ValueError("Production requires ready ML artifacts (BACKEND_REQUIRE_ML_READY=true)")
        return self

    @classmethod
    def from_env(cls) -> "Settings":
        load_dotenv(ROOT / ".env", override=False)
        text_fields = {
            "database_url": "DATABASE_URL", "jwt_secret": "JWT_SECRET", "admin_username": "ADMIN_USERNAME",
            "admin_password_hash": "ADMIN_PASSWORD_HASH", "environment": "ENVIRONMENT",
            "rate_limit_storage_uri": "RATE_LIMIT_STORAGE_URI", "submission_rate_limit": "SUBMISSION_RATE_LIMIT",
            "analysis_rate_limit": "ANALYSIS_RATE_LIMIT", "login_rate_limit": "LOGIN_RATE_LIMIT",
            "category_model": "CATEGORY_MODEL", "severity_model": "SEVERITY_MODEL", "review_policy": "REVIEW_POLICY",
            "priority_ranker": "PRIORITY_RANKER", "duplicate_model": "DUPLICATE_MODEL", "rules_config": "RULES_CONFIG",
            "embedding_cache_dir": "EMBEDDING_CACHE_DIR",
        }
        defaults = cls()
        values = {field: os.getenv("BACKEND_" + env_name, getattr(defaults, field)) for field, env_name in text_fields.items()}
        for name in ("category_model", "severity_model", "review_policy", "priority_ranker", "duplicate_model"):
            if values[name] == "":
                values[name] = None
        for name in ("jwt_expire_minutes", "max_text_length", "abuse_max_urls", "max_body_bytes", "duplicate_top_k"):
            try:
                values[name] = int(os.getenv("BACKEND_" + name.upper(), str(getattr(defaults, name))))
            except ValueError as exc:
                raise ValueError(f"BACKEND_{name.upper()} must be an integer") from exc
        for name in ("trust_proxy_headers", "allow_public_reads"):
            values[name] = _boolean(os.getenv("BACKEND_" + name.upper(), str(getattr(defaults, name))), name)
        values["require_ml_ready"] = _boolean(os.getenv("BACKEND_REQUIRE_ML_READY", "true" if values["environment"] == "production" else "false"), "require_ml_ready")
        values["allowed_origins"] = tuple(value.strip() for value in os.getenv("BACKEND_ALLOWED_ORIGINS", "").split(",") if value.strip())
        return cls(**values).validate()
