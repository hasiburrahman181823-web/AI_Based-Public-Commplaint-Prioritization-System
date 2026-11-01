"""Security behavior through real TestClient requests and disposable SQLite."""
from dataclasses import replace
from uuid import uuid4

from fastapi.testclient import TestClient
import pytest
from sqlalchemy import func, select

from backend.app import create_app
from backend.models import AuditLog, Complaint
from test_backend_api import auth, build_client, counts, submit


@pytest.fixture
def hardened(tmp_path):
    context = build_client(tmp_path, allowed_origins=("https://portal.example",))
    with TestClient(context.app) as client:
        context.client = client
        yield context
    context.engine.dispose()


def test_origin_allowlist_blocks_unknown_preflight_and_write_before_persistence(hardened):
    client = hardened.client
    for method in ("OPTIONS", "POST"):
        response = client.request(method, "/complaints", json={"text": "Water unavailable"}, headers={
            "Origin": "https://untrusted.example", "Access-Control-Request-Method": "POST"})
        assert response.status_code == 403
        assert "access-control-allow-origin" not in response.headers
    allowed = client.options("/complaints", headers={"Origin": "https://portal.example", "Access-Control-Request-Method": "POST"})
    assert allowed.status_code == 200
    assert allowed.headers["access-control-allow-origin"] == "https://portal.example"
    assert counts(hardened)["complaints"] == 0


def test_complaint_limit_and_control_stripping(hardened):
    text = ("Water unavailable near the school. " * 30)[:1000]
    assert len(text) == 1000
    good = hardened.client.post("/complaints", json={"text": text})
    assert good.status_code == 201, good.text
    before = counts(hardened)
    for endpoint in ("/complaints", "/analyze"):
        response = hardened.client.post(endpoint, json={"text": text + "x"})
        assert response.status_code == 422
    assert counts(hardened) == before
    cleaned = hardened.client.post("/complaints", json={"text": " No\x00water\x1favailable\n"})
    assert cleaned.status_code == 201
    assert cleaned.json()["text"] == "No water available"
    assert hardened.client.post("/complaints", json={"text": "\x00\x1f \t"}).status_code == 422


@pytest.mark.parametrize("path,body", [
    ("/analyze", {"text": "No water"}),
    ("/complaints", {"text": "No water"}),
    ("/auth/token", {"username": "unknown", "password": "invalid-test-only"}),
])
def test_limits_trigger_for_each_sensitive_public_post(tmp_path, path, body):
    context = build_client(tmp_path, analysis_rate_limit="2/minute", submission_rate_limit="2/minute", login_rate_limit="2/minute")
    with TestClient(context.app) as client:
        assert client.post(path, json=body).status_code != 429
        assert client.post(path, json=body).status_code != 429
        assert client.post(path, json=body).status_code == 429
    context.engine.dispose()


@pytest.mark.parametrize("method,path,body", [
    ("GET", "/complaints", None), ("GET", "/analytics", None), ("GET", "/duplicate-groups", None),
    ("PATCH", "/complaints/" + str(uuid4()), {"expected_version": 1, "reason": "Synthetic review", "status": "confirmed"}),
    ("POST", "/duplicate-groups/" + str(uuid4()) + "/confirm", {"expected_version": 1, "reason": "Synthetic review"}),
    ("POST", "/duplicate-groups/" + str(uuid4()) + "/split", {"expected_version": 1, "reason": "Synthetic review", "complaint_ids": [str(uuid4())]}),
    ("POST", "/admin/data-deletion", {"complaint_ids": [str(uuid4())], "reason": "Synthetic request"}),
])
def test_admin_operations_require_auth_and_reject_invalid_jwt(hardened, method, path, body):
    assert hardened.client.request(method, path, json=body).status_code == 401
    invalid = hardened.client.request(method, path, json=body, headers={"Authorization": "Bearer invalid.jwt.signature"})
    assert invalid.status_code == 401


def test_all_security_headers_cover_json_validation_and_origin_errors(hardened):
    responses = [hardened.client.get("/health"), hardened.client.post("/complaints", json={"text": ""}),
                 hardened.client.get("/health", headers={"Origin": "https://untrusted.example"})]
    for response in responses:
        assert response.headers["x-content-type-options"] == "nosniff"
        assert response.headers["x-frame-options"] == "DENY"
        assert response.headers["referrer-policy"] == "no-referrer"
        assert "default-src 'none'" in response.headers["content-security-policy"]
        assert "frame-ancestors 'none'" in response.headers["content-security-policy"]


def test_production_docs_routes_are_disabled_without_contacting_live_services(tmp_path):
    context = build_client(tmp_path)
    settings = replace(context.settings, environment="production", database_url="postgresql://unused.invalid/test",
                       rate_limit_storage_uri="redis://127.0.0.1:6379", require_ml_ready=True)
    app = create_app(settings=settings, engine=context.engine, ml_service=context.service)
    # These route checks deliberately do not enter lifespan or claim a real
    # PostgreSQL/Redis production startup has been tested.
    client = TestClient(app)
    try:
        for path in ("/docs", "/redoc", "/openapi.json"):
            assert client.get(path).status_code == 404
    finally:
        client.close()
        context.engine.dispose()


def test_deletion_is_preview_first_then_audited_and_tracking_becomes_missing(hardened):
    created = submit(hardened)
    headers = auth(hardened)
    payload = {"complaint_ids": [created["id"]], "reason": "Verified synthetic privacy request"}
    preview = hardened.client.post("/admin/data-deletion", json=payload, headers=headers)
    assert preview.status_code == 200 and preview.json()["dry_run"] is True
    assert hardened.client.get("/tracking/" + created["id"]).status_code == 200
    before = counts(hardened)
    assert hardened.client.post("/admin/data-deletion", json={**payload, "apply": "false"}, headers=headers).status_code == 422
    assert counts(hardened) == before
    applied = hardened.client.post("/admin/data-deletion", json={**payload, "apply": True}, headers=headers)
    assert applied.status_code == 200, applied.text
    assert applied.json()["deleted_count"] == 1 and applied.json()["dry_run"] is False
    assert hardened.client.get("/tracking/" + created["id"]).status_code == 404
    with hardened.session_factory() as session:
        assert session.scalar(select(func.count()).select_from(Complaint)) == 0
        audit = session.get(AuditLog, applied.json()["audit_event_id"])
        assert audit.actor == hardened.settings.admin_username and audit.created_at
        assert payload["reason"] not in str(audit.changes)
