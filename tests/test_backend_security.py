"""Admin-token, configuration and intake validation without public ML claims."""

from dataclasses import replace
from datetime import datetime, timezone
from functools import lru_cache
import os
import secrets
from uuid import uuid4

import jwt
import pytest
from pydantic import ValidationError

from backend import config
from backend.config import Settings
from backend.schemas import AdminLogin, AnalyzeRequest, ComplaintCreate, ComplaintPatch, GroupConfirm, GroupSplit, validate_submission
from argon2.low_level import Type

from backend.security import (JWT_AUDIENCE, JWT_ISSUER, PASSWORD_MEMORY_KIB, PASSWORD_TIME_COST,
                              PASSWORD_PARALLELISM, authenticate_admin, create_access_token, hash_password,
                              validate_jwt_secret, validate_password_hash, verify_access_token, verify_password)
from backend.scripts import create_admin


@lru_cache(maxsize=1)
def encoded():
    return hash_password("test administrator password")


def settings(**updates):
    return replace(Settings(jwt_secret=secrets.token_urlsafe(48), admin_password_hash=encoded()), **updates)


def test_hashes_are_salted_verify_exact_password_and_hide_plaintext():
    first, second = encoded(), hash_password("test administrator password")
    assert first != second
    assert "test administrator password" not in first
    assert verify_password("test administrator password", first)
    assert not verify_password("wrong password", first)
    parameters = validate_password_hash(first)
    assert first.startswith("$argon2id$v=19$")
    assert parameters.type == Type.ID and parameters.version == 19
    assert parameters.memory_cost == PASSWORD_MEMORY_KIB and parameters.time_cost == PASSWORD_TIME_COST
    assert parameters.parallelism == PASSWORD_PARALLELISM and parameters.salt_len >= 16 and parameters.hash_len == 32
    assert not verify_password(None, first)


@pytest.mark.parametrize("value", ["", "short", "x" * 257, None])
def test_password_creation_rejects_invalid_values(value):
    with pytest.raises(ValueError):
        hash_password(value)


@pytest.mark.parametrize("value", ["", "change-me", "pbkdf2_sha256$1$AAAA$BBBB", "pbkdf2_sha256$999999999999$AAAA$BBBB", "pbkdf2_sha256$600000$bad!$bad!", "bcrypt$600000$AAAA$BBBB", "$argon2id$v=19$m=65536,t=3,p=1$bad!$bad!"])
def test_malformed_password_hashes_do_not_authorize(value):
    with pytest.raises(ValueError):
        validate_password_hash(value)
    assert not verify_password("valid candidate password", value)


@pytest.mark.parametrize("replacement", ["m=8,t=3,p=1", "m=999999999,t=3,p=1", "m=65536,t=1,p=1",
                                          "m=65536,t=999,p=1", "m=65536,t=3,p=0", "m=65536,t=3,p=999"])
def test_hash_cost_bounds_are_checked_before_password_verification(replacement, monkeypatch):
    from backend import security
    from types import SimpleNamespace
    unsafe = encoded().replace("m=65536,t=3,p=1", replacement)
    def must_not_verify(*args):
        raise AssertionError("Unsafe hash reached the Argon2 verifier")
    monkeypatch.setattr(security, "_PASSWORD_HASHER", SimpleNamespace(verify=must_not_verify))
    with pytest.raises(ValueError, match="Argon2id"):
        validate_password_hash(unsafe)
    assert not verify_password("test administrator password", unsafe)


def test_legacy_pbkdf2_hash_requires_explicit_reprovisioning():
    import base64
    import hashlib
    salt = b"synthetic-salt-16"
    digest = hashlib.pbkdf2_hmac("sha256", b"old password", salt, 600000)
    def part(value):
        return base64.urlsafe_b64encode(value).decode("ascii").rstrip("=")
    legacy = f"pbkdf2_sha256$600000${part(salt)}${part(digest)}"
    with pytest.raises(ValueError, match="regenerate legacy hashes"):
        validate_password_hash(legacy)
    assert not verify_password("old password", legacy)


@pytest.mark.parametrize("value", ["", "short", "আ" * 16, "s" * 48, "0123456789" * 5,
                                   "change-me-to-a-long-random-secret-now", "replace-with-32-character-secret-value",
                                   "your-jwt-secret-at-least-32-characters", "abcdefghijklmnopqrstuvwxyz0123456789"])
def test_weak_or_example_jwt_secrets_are_rejected(value):
    with pytest.raises(ValueError, match="BACKEND_JWT_SECRET"):
        validate_jwt_secret(value)
    with pytest.raises(ValueError, match="BACKEND_JWT_SECRET"):
        settings(jwt_secret=value).validate()


def test_generated_jwt_secret_meets_character_length_and_example_checks():
    validate_jwt_secret(secrets.token_urlsafe(24))
    validate_jwt_secret(secrets.token_urlsafe(48))


def test_wrong_username_still_verifies_the_password(monkeypatch):
    from backend import security
    calls = []
    monkeypatch.setattr(security, "verify_password", lambda password, value: calls.append((password, value)) or True)
    assert not authenticate_admin("wrong-user", "secret", settings())
    assert authenticate_admin("admin", "secret", settings())
    assert len(calls) == 2


def test_invalid_unicode_username_does_not_skip_password_verification(monkeypatch):
    from backend import security
    calls = []
    monkeypatch.setattr(security, "verify_password", lambda password, value: calls.append(password) or True)
    assert not authenticate_admin("\ud800", "candidate", settings())
    assert calls == ["candidate"]


def test_admin_token_has_required_role_scope_and_short_expiration():
    configuration = settings()
    token, duration = create_access_token("admin", configuration)
    assert verify_access_token(token, configuration) == "admin"
    claims = jwt.decode(token, configuration.jwt_secret, algorithms=["HS256"], issuer=JWT_ISSUER, audience=JWT_AUDIENCE)
    assert claims["role"] == "admin" and claims["sub"] == "admin"
    assert claims["exp"] - claims["iat"] == duration == configuration.jwt_expire_minutes * 60
    assert claims["jti"]
    with pytest.raises(ValueError):
        create_access_token("citizen", configuration)


def claims():
    now = int(datetime.now(timezone.utc).timestamp())
    return {"sub": "admin", "role": "admin", "iat": now, "nbf": now, "exp": now + 1800,
            "iss": JWT_ISSUER, "aud": JWT_AUDIENCE, "jti": str(uuid4())}


@pytest.mark.parametrize("change", [{"role": "citizen"}, {"sub": "other-admin"}, {"iss": "other-api"}, {"aud": "public"}, {"exp": 1}, {"jti": ""}, {"iat": "not-a-time"}])
def test_modified_claims_are_rejected_even_with_valid_signature(change):
    configuration = settings()
    payload = {**claims(), **change}
    token = jwt.encode(payload, configuration.jwt_secret, algorithm="HS256")
    with pytest.raises(ValueError):
        verify_access_token(token, configuration)


def test_missing_claim_wrong_secret_and_unsigned_tokens_are_rejected():
    configuration = settings()
    payload = claims()
    payload.pop("role")
    bad_tokens = [jwt.encode(payload, configuration.jwt_secret, algorithm="HS256"),
                  jwt.encode(claims(), "another-synthetic-secret-at-least-32-bytes", algorithm="HS256"),
                  jwt.encode(claims(), "", algorithm="none"), "not.a.jwt", "x" * 9000]
    for token in bad_tokens:
        with pytest.raises(ValueError):
            verify_access_token(token, configuration)


def test_configuration_has_no_usable_default_credentials():
    with pytest.raises(ValueError, match="JWT_SECRET"):
        Settings().validate()
    with pytest.raises(ValueError, match="PASSWORD_HASH"):
        Settings(jwt_secret=secrets.token_urlsafe(48)).validate()
    assert settings().validate().trust_proxy_headers is False


@pytest.mark.parametrize("updates", [{"jwt_expire_minutes": 0}, {"jwt_expire_minutes": True}, {"max_text_length": 4001}, {"max_body_bytes": 12}, {"duplicate_top_k": 0}, {"rate_limit_storage_uri": "http://not-redis"}, {"submission_rate_limit": "unlimited"}, {"allowed_origins": ("*",)}, {"allowed_origins": ("https://admin.example/path",)}, {"environment": "unknown"}])
def test_invalid_configuration_fails_before_serving(updates):
    with pytest.raises(ValueError):
        settings(**updates).validate()


def test_production_requires_database_shared_limiter_and_ready_models():
    with pytest.raises(ValueError, match="PostgreSQL"):
        settings(environment="production").validate()
    with pytest.raises(ValueError, match="Redis"):
        settings(environment="production", database_url="postgresql+psycopg://db/complaints").validate()
    with pytest.raises(ValueError, match="ready ML"):
        settings(environment="production", database_url="postgresql+psycopg://db/complaints", rate_limit_storage_uri="redis://redis/0").validate()
    assert settings(environment="production", database_url="postgresql+psycopg://db/complaints", rate_limit_storage_uri="rediss://redis/0", require_ml_ready=True).validate()


def test_env_loading_is_explicit_and_parses_optional_model_paths(monkeypatch):
    monkeypatch.setattr(config, "load_dotenv", lambda *args, **kwargs: None)
    for name in list(os.environ):
        if name.startswith("BACKEND_"):
            monkeypatch.delenv(name)
    monkeypatch.setenv("BACKEND_JWT_SECRET", secrets.token_urlsafe(48))
    monkeypatch.setenv("BACKEND_ADMIN_PASSWORD_HASH", encoded())
    monkeypatch.setenv("BACKEND_CATEGORY_MODEL", "")
    monkeypatch.setenv("BACKEND_ALLOWED_ORIGINS", "https://admin.example,http://localhost:5173")
    monkeypatch.setenv("BACKEND_DUPLICATE_TOP_K", "25")
    configuration = Settings.from_env()
    assert configuration.category_model is None and configuration.duplicate_top_k == 25
    assert configuration.allowed_origins == ("https://admin.example", "http://localhost:5173")
    monkeypatch.setenv("BACKEND_ALLOW_PUBLIC_READS", "maybe")
    with pytest.raises(ValueError):
        Settings.from_env()


@pytest.mark.parametrize("text", ["মিরপুর ১০ এ ৩ দিন ধরে পানি নেই", "mirpur-10 e 3 din dhore pani nai", "No water for three days in Mirpur 10.", "The hospital has a dangerous exposed wire; the office is ignoring this complaint."])
def test_normal_multilingual_grievances_and_critical_language_are_allowed(text):
    assert ComplaintCreate(text=text).text == text
    assert AnalyzeRequest(text=text).text == text


@pytest.mark.parametrize("text", ["", "   ", "https://example.com", "... !!!", "a" * 40, "pani nai " * 30, "\x00\x01\x02", "x" * 1001, "ﬃ" * 1400])
def test_basic_spam_or_invalid_payloads_are_rejected(text):
    with pytest.raises((ValueError, ValidationError)):
        ComplaintCreate(text=text)


def test_control_characters_are_removed_without_erasing_surrounding_complaint_words():
    assert ComplaintCreate(text="No\x00 water\tnear\nschool\r").text == "No  water near school"


def test_intake_accepts_1000_characters_and_rejects_1001_after_normalization():
    text = " ".join(f"area{index}" for index in range(180))[:999] + "."
    assert len(text) == 1000 and ComplaintCreate(text=text).text == text
    with pytest.raises(ValidationError):
        ComplaintCreate(text=text + "!")


def test_urls_and_configured_length_have_bounded_limits():
    assert validate_submission("No water. https://example.com")
    with pytest.raises(ValueError, match="URLs"):
        validate_submission("No water " + " ".join("https://example.com/" + str(index) for index in range(4)))
    with pytest.raises(ValueError):
        validate_submission("an otherwise valid grievance", max_length=10)


def test_citizens_cannot_inject_prediction_or_admin_fields():
    with pytest.raises(ValidationError):
        ComplaintCreate(text="No water", priority_score=100, category="Healthcare", status="confirmed")
    with pytest.raises(ValidationError):
        AdminLogin(username="admin", password="password", role="admin")


def test_timestamps_require_timezone_and_are_normalized_to_utc():
    complaint = ComplaintCreate(text="No water", timestamp="2026-10-30T10:00:00+06:00")
    assert complaint.timestamp.isoformat() == "2026-10-30T04:00:00+00:00"
    for value in ("2026-10-30", "2026-10-30T10:00:00", 12345, True):
        with pytest.raises(ValidationError):
            ComplaintCreate(text="No water", timestamp=value)


def test_patch_requires_reason_version_and_an_actual_nonnull_change():
    valid = ComplaintPatch(expected_version=1, reason="Reviewed evidence", priority_score=8, priority_label="high")
    assert valid.priority_score == 8
    assert valid.model_fields_set == {"expected_version", "reason", "priority_score", "priority_label"}
    for body in ({"reason": "Reviewed", "status": "resolved"}, {"expected_version": True, "reason": "Reviewed", "status": "resolved"},
                 {"expected_version": 1, "reason": "   ", "status": "resolved"},
                 {"expected_version": 1, "reason": "Reviewed"},
                 {"expected_version": 1, "reason": "Reviewed", "priority_score": None},
                 {"expected_version": 1, "reason": "Reviewed", "priority_score": float("nan")},
                 {"expected_version": 1, "reason": "Reviewed", "priority_score": True}):
        with pytest.raises(ValidationError):
            ComplaintPatch(**body)


def test_group_payloads_have_unique_canonical_uuid_members():
    first, second = str(uuid4()), str(uuid4())
    assert GroupConfirm(expected_version=1, reason="Reviewed").complaint_ids is None
    assert GroupConfirm(expected_version=1, reason="Reviewed", complaint_ids=[first.upper(), second]).complaint_ids == [first, second]
    assert GroupSplit(expected_version=1, reason="Different incidents", complaint_ids=[first]).complaint_ids == [first]
    for identifiers in ([], [first, first.upper()], ["not-a-uuid"]):
        with pytest.raises(ValidationError):
            GroupSplit(expected_version=1, reason="Reviewed", complaint_ids=identifiers)


def test_admin_generation_uses_password_env_without_printing_credentials(monkeypatch, capsys, tmp_path):
    from dotenv import dotenv_values

    monkeypatch.setenv("SYNTHETIC_ADMIN_PASSWORD", "synthetic long admin pass")
    destination = tmp_path / ".env.admin"
    assert create_admin.main(["--username", "reviewer", "--password-env", "SYNTHETIC_ADMIN_PASSWORD", "--output", str(destination)]) == 0
    output = capsys.readouterr().out
    assert "synthetic long admin pass" not in output
    values = dotenv_values(destination)
    assert values["BACKEND_ADMIN_USERNAME"] == "reviewer"
    assert values["BACKEND_ADMIN_PASSWORD_HASH"].startswith("$argon2id$")
    assert verify_password("synthetic long admin pass", values["BACKEND_ADMIN_PASSWORD_HASH"])
    secret = values["BACKEND_JWT_SECRET"]
    assert len(secret) >= 32
    validate_jwt_secret(secret)
    assert secret not in output and values["BACKEND_ADMIN_PASSWORD_HASH"] not in output
    original = destination.read_bytes()
    with pytest.raises(SystemExit):
        create_admin.main(["--password-env", "SYNTHETIC_ADMIN_PASSWORD", "--output", str(destination)])
    assert destination.read_bytes() == original


def test_admin_helper_prompts_for_password_and_confirmation_without_a_default(monkeypatch, capsys, tmp_path):
    prompts = []
    monkeypatch.setattr(create_admin.getpass, "getpass", lambda prompt: prompts.append(prompt) or "interactive synthetic password")
    destination = tmp_path / ".env.admin"
    assert create_admin.main(["--username", "reviewer", "--output", str(destination)]) == 0
    output = capsys.readouterr().out
    assert len(prompts) == 2 and "interactive synthetic password" not in output
    assert "$argon2id$" not in output
    assert "BACKEND_ADMIN_PASSWORD_HASH='$argon2id$" in destination.read_text(encoding="utf-8")


@pytest.mark.parametrize("filename", [".env.example", "credentials.txt", ".env"])
def test_admin_helper_refuses_public_or_unignored_output(monkeypatch, tmp_path, filename):
    monkeypatch.setenv("SYNTHETIC_ADMIN_PASSWORD", "synthetic long admin pass")
    destination = tmp_path / filename
    with pytest.raises(SystemExit):
        create_admin.main(["--password-env", "SYNTHETIC_ADMIN_PASSWORD", "--output", str(destination)])
    assert not destination.exists()


def test_admin_helper_rejects_a_mismatched_confirmation(monkeypatch, capsys):
    passwords = iter(["first synthetic password", "different synthetic password"])
    monkeypatch.setattr(create_admin.getpass, "getpass", lambda prompt: next(passwords))
    with pytest.raises(SystemExit) as stopped:
        create_admin.main([])
    assert stopped.value.code == 2
    assert "BACKEND_ADMIN_PASSWORD_HASH=" not in capsys.readouterr().out


@pytest.mark.parametrize("change", [{"exp": 4102444800}, {"iat": True}, {"nbf": True}])
def test_signed_tokens_cannot_override_time_claim_types_or_configured_ttl(change):
    configuration = settings()
    token = jwt.encode({**claims(), **change}, configuration.jwt_secret, algorithm="HS256")
    with pytest.raises(ValueError):
        verify_access_token(token, configuration)
