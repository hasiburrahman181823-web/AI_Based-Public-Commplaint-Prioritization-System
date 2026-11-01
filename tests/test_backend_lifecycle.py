"""Real rule-only startup and failure contracts, without pretrained downloads."""

from dataclasses import replace
import secrets

from fastapi.testclient import TestClient
import pytest
from sqlalchemy import func, select

from backend.app import create_app
from backend.config import Settings
from backend.db import make_engine, make_session_factory
from backend.ml_service import MLService
from backend.models import Base, Complaint
from backend.security import hash_password


PASSWORD_HASH = hash_password("synthetic-lifecycle-password")


def settings_for(tmp_path, **updates):
    return replace(Settings(database_url=f"sqlite:///{(tmp_path / 'lifecycle.sqlite3').as_posix()}",
                            jwt_secret=secrets.token_hex(32), admin_password_hash=PASSWORD_HASH,
                            environment="test", category_model=None, severity_model=None,
                            review_policy=None, priority_ranker=None, duplicate_model=None,
                            analysis_rate_limit="1000/minute", submission_rate_limit="1000/minute"), **updates)


def test_actual_rule_only_service_starts_once_and_marks_missing_predictions_for_review(tmp_path, monkeypatch):
    settings = settings_for(tmp_path)
    engine = make_engine(settings.database_url)
    Base.metadata.create_all(engine)
    service = MLService(settings)
    original, loads = service._make_analyzer, []

    def counted(records):
        loads.append(1)
        return original(records)

    monkeypatch.setattr(service, "_make_analyzer", counted)
    app = create_app(settings, service, engine)
    try:
        with TestClient(app) as client:
            assert client.get("/health").json()["status"] == "degraded"
            payload = {"text": "No water for three days", "lang": "en", "location": "Mirpur 10"}
            for _ in range(2):
                response = client.post("/analyze", json=payload)
                assert response.status_code == 200, response.text
                body = response.json()
                assert body["category"] is body["severity"] is body["confidence"] is None
                assert body["review_required"] and body["review_status"] == "needs_human_review"
            created = client.post("/complaints", json=payload)
            assert created.status_code == 201, created.text
            assert created.json()["review_required"] is True
            assert len(loads) == 1
            with make_session_factory(engine)() as session:
                assert session.scalar(select(func.count()).select_from(Complaint)) == 1
    finally:
        engine.dispose()


def test_startup_requires_explicit_migration(tmp_path):
    settings = settings_for(tmp_path)
    engine = make_engine(settings.database_url)
    try:
        with pytest.raises(RuntimeError, match="run python -m alembic upgrade head"):
            with TestClient(create_app(settings, engine=engine)):
                pass
    finally:
        engine.dispose()


def test_strict_ml_readiness_rejects_missing_artifacts(tmp_path):
    settings = settings_for(tmp_path, require_ml_ready=True)
    engine = make_engine(settings.database_url)
    Base.metadata.create_all(engine)
    try:
        with pytest.raises(RuntimeError, match="not ready"):
            with TestClient(create_app(settings, engine=engine)):
                pass
    finally:
        engine.dispose()


def test_unavailable_limiter_storage_prevents_startup(tmp_path, monkeypatch):
    settings = settings_for(tmp_path)
    engine = make_engine(settings.database_url)
    Base.metadata.create_all(engine)
    app = create_app(settings, engine=engine)
    monkeypatch.setattr(app.state.limiter._storage, "check", lambda: False)
    try:
        with pytest.raises(RuntimeError, match="Rate-limit storage is unavailable"):
            with TestClient(app):
                pass
    finally:
        engine.dispose()


def test_rate_store_failure_fails_closed_without_saving_or_disclosing_details(tmp_path, monkeypatch):
    settings = settings_for(tmp_path)
    engine = make_engine(settings.database_url)
    Base.metadata.create_all(engine)
    app = create_app(settings, engine=engine)

    def unavailable(*args, **kwargs):
        raise ConnectionError("synthetic-sensitive-store-credential")

    try:
        with TestClient(app, raise_server_exceptions=False) as client:
            monkeypatch.setattr(app.state.limiter._storage, "incr", unavailable)
            response = client.post("/complaints", json={"text": "Water unavailable", "lang": "en"})
            assert response.status_code == 503
            assert "synthetic-sensitive-store-credential" not in response.text
            assert response.json()["request_id"]
            with make_session_factory(engine)() as session:
                assert session.scalar(select(func.count()).select_from(Complaint)) == 0
    finally:
        engine.dispose()


def test_database_failure_returns_sanitized_unavailable_health(tmp_path):
    settings = settings_for(tmp_path)
    engine = make_engine(settings.database_url)
    Base.metadata.create_all(engine)
    try:
        with TestClient(create_app(settings, engine=engine)) as client:
            Complaint.__table__.drop(engine)
            response = client.get("/health")
            assert response.status_code == 503
            assert response.json()["status"] == "unavailable"
            assert "sqlite" not in response.text and "SELECT" not in response.text
    finally:
        engine.dispose()
