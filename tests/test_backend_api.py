"""API integration with SQLite and synthetic semantic inference.

The deterministic analyzer exercises transport/persistence/grouping contracts;
it is not a test or performance claim about a real pretrained checkpoint.
"""

from copy import deepcopy
from datetime import datetime, timezone
import json
import secrets
from types import SimpleNamespace
from uuid import uuid4

import jwt
import pytest
from fastapi.testclient import TestClient
from sqlalchemy import func, select

from backend.config import Settings
from backend.db import make_engine, make_session_factory
from backend.ml_service import MLService
from backend.models import AdminReview, AuditLog, Base, Complaint
from backend.security import hash_password
from ml.analysis import infer_language
from ml.location import LocationGazetteer


PASSWORD = "synthetic-tests-only-password"
PASSWORD_HASH = hash_password(PASSWORD)


class SemanticTestAnalyzer:
    """Water meaning is a fixture, independent of area and script.

    It intentionally proposes the same high score for different areas so the
    real backend guard, rather than a cooperative test scorer, must veto them.
    """

    ready = True

    def __init__(self):
        self.existing = {}
        self.finder = None
        self.gazetteer = LocationGazetteer()
        self.calls = []

    @staticmethod
    def water(text):
        lowered = text.casefold()
        return any(word in lowered for word in ("পানি", "water", "pani", "pany"))

    def analyze_record(self, record):
        self.calls.append(deepcopy(record))
        candidates = []
        for identifier, previous in self.existing.items():
            if identifier != record.get("id") and self.water(record["text"]) and self.water(previous["text"]):
                candidates.append({"id": identifier, "score": 0.99, "threshold": 0.8, "is_duplicate": True})
        return {
            "id": record.get("id", "preview"), "lang": record.get("lang") or infer_language(record["text"])["lang"],
            "category": "Water" if self.water(record["text"]) else "Road & Transport",
            "severity": "High", "priority_score": 9.0, "priority_label": "high", "confidence": 0.95,
            "review_required": False, "review_status": "scored", "review_reasons": [],
            "explanation": {"rule_breakdown": [], "notes": ["Synthetic inference fixture"]},
            "duplicate_candidates": candidates,
        }


def build_settings(tmp_path, **updates):
    values = dict(database_url=f"sqlite:///{(tmp_path / 'api.sqlite3').as_posix()}",
                  environment="test", jwt_secret=secrets.token_hex(32), admin_username="test-admin",
                  admin_password_hash=PASSWORD_HASH, submission_rate_limit="1000/minute",
                  analysis_rate_limit="1000/minute", login_rate_limit="1000/minute",
                  category_model=None, severity_model=None, review_policy=None,
                  priority_ranker=None, duplicate_model=None, require_ml_ready=False)
    values.update(updates)
    return Settings(**values)


def build_client(tmp_path, **updates):
    # Import after root's app implementation is present; no module-level ASGI
    # app or production credentials are needed for test collection.
    from backend.app import create_app

    settings = build_settings(tmp_path, **updates)
    engine = make_engine(settings.database_url)
    Base.metadata.create_all(engine)
    analyzer = SemanticTestAnalyzer()
    service = MLService(settings, analyzer)
    app = create_app(settings=settings, ml_service=service, engine=engine)
    return SimpleNamespace(settings=settings, engine=engine, session_factory=make_session_factory(engine),
                           analyzer=analyzer, service=service, app=app)


@pytest.fixture
def api(tmp_path):
    context = build_client(tmp_path)
    with TestClient(context.app) as client:
        context.client = client
        yield context
    context.engine.dispose()


def auth(api):
    response = api.client.post("/auth/token", json={"username": api.settings.admin_username, "password": PASSWORD})
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["token_type"] == "bearer" and body["expires_in"] > 0
    return {"Authorization": "Bearer " + body["access_token"]}


def submit(api, text="No water for three days", lang="en", location="Mirpur 10", **updates):
    body = {"text": text, "lang": lang, "location": location}
    body.update(updates)
    response = api.client.post("/complaints", json=body)
    assert response.status_code == 201, response.text
    return response.json()


def groups(api, headers):
    response = api.client.get("/duplicate-groups", headers=headers)
    assert response.status_code == 200, response.text
    return response.json()["items"]


def counts(api):
    with api.session_factory() as session:
        return {model.__tablename__: session.scalar(select(func.count()).select_from(model))
                for model in (Complaint, AdminReview, AuditLog)}


def test_health_stats_and_openapi_are_available_without_citizen_auth(api):
    health = api.client.get("/health")
    assert health.status_code == 200
    assert health.json()["status"] in ("ready", "degraded")
    assert "database" in health.json() and "ml" in health.json()
    stats = api.client.get("/stats")
    assert stats.status_code == 200
    assert "text" not in stats.text and "password" not in stats.text and "jwt_secret" not in stats.text
    specification = api.client.get("/openapi.json")
    assert specification.status_code == 200
    paths = specification.json()["paths"]
    assert {"/complaints", "/complaints/{complaint_id}", "/duplicate-groups", "/stats", "/health", "/analyze"} <= set(paths)
    assert api.client.get("/docs").status_code == 200


def test_anonymous_submission_persists_analysis_with_uuid_and_singleton_group(api):
    created = submit(api)
    assert created["id"] and created["group_id"] and created["version"] == 1
    assert created["category"] == "Water" and created["priority_label"] == "high"
    assert created["area_id"] == "dhaka_mirpur_10"
    assert created["lang"] == "en"
    assert created["status"] in ("submitted", "in_review")
    with api.session_factory() as session:
        stored = session.get(Complaint, created["id"])
        assert stored is not None and stored.analysis["category"] == "Water"
        assert session.scalar(select(func.count()).select_from(AuditLog)) >= 1


def test_cross_language_water_reports_group_and_different_area_never_joins(api):
    one = submit(api, "মিরপুর ১০ এ তিন দিন ধরে পানি নেই", "bn", "মিরপুর ১০")
    two = submit(api, "mirpur-10 e 3 din dhore pani nai", "banglish", "Mirpur-10")
    three = submit(api, "No water in Mirpur 10 for three days", "en", "Mirpur 10")
    different = submit(api, "No water in Mirpur 10 for three days", "en", "Uttara sector 4")
    assert one["group_id"] == two["group_id"] == three["group_id"]
    assert different["group_id"] != one["group_id"]
    listed = groups(api, auth(api))
    matched = next(group for group in listed if group["id"] == one["group_id"])
    assert set(matched["complaint_ids"]) == {one["id"], two["id"], three["id"]}
    assert matched["size"] == 3
    with api.session_factory() as session:
        stored = session.get(Complaint, different["id"])
        assert all(not item["auto_group_eligible"] for item in stored.analysis["duplicate_candidates"])


def test_public_submission_and_preview_do_not_disclose_existing_candidate_ids(api):
    existing = submit(api)
    created = submit(api, "pani nai 3 din dhore", "banglish", "Mirpur 10")
    candidate_items = created.get("analysis", {}).get("duplicate_candidates", [])
    assert all("id" not in item for item in candidate_items)
    preview = api.client.post("/analyze", json={"text": "No water", "lang": "en", "location": "Mirpur 10"})
    assert preview.status_code == 200
    assert all("id" not in item for item in preview.json().get("duplicate_candidates", []))
    assert existing["id"] not in preview.text


def test_preview_analysis_does_not_write_complaints_feedback_or_audit(api):
    submit(api)
    before = counts(api)
    response = api.client.post("/analyze", json={"text": "পানি নেই", "lang": "bn", "location": "Mirpur 10"})
    assert response.status_code == 200
    assert response.json()["category"] == "Water"
    assert counts(api) == before


def test_admin_reads_filter_paginate_and_get_details(api):
    first = submit(api)
    submit(api, "Road has a pothole", "en", "Mohammadpur")
    headers = auth(api)
    response = api.client.get("/complaints", params={"category": "Water", "priority": "high", "area": "Mirpur 10", "status": first["status"], "page_size": 1}, headers=headers)
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["total"] == 1 and body["page"] == 1 and body["page_size"] == 1
    assert body["items"][0]["id"] == first["id"]
    detail = api.client.get(f"/complaints/{first['id']}", headers=headers)
    assert detail.status_code == 200 and detail.json()["text"] == first["text"]
    assert api.client.get("/complaints", params={"page_size": 101}, headers=headers).status_code == 422
    assert api.client.get("/complaints/not-a-uuid", headers=headers).status_code == 422
    assert api.client.get(f"/complaints/{uuid4()}", headers=headers).status_code == 404


@pytest.mark.parametrize("path", ["/complaints", "/duplicate-groups"])
def test_private_reads_require_admin_token_by_default(api, path):
    assert api.client.get(path).status_code == 401


def test_priority_status_override_is_feedback_and_preserves_original_model_analysis(api):
    created = submit(api)
    headers = auth(api)
    response = api.client.patch(f"/complaints/{created['id']}", headers=headers,
        json={"expected_version": created["version"], "reason": "Expert confirmed urgent essential-service disruption",
              "priority_score": 15, "priority_label": "urgent", "status": "in_progress"})
    assert response.status_code == 200, response.text
    updated = response.json()
    assert updated["priority_score"] == 15 and updated["priority_label"] == "urgent"
    assert updated["status"] == "in_progress" and updated["version"] == created["version"] + 1
    with api.session_factory() as session:
        review = session.scalar(select(AdminReview).where(AdminReview.complaint_id == created["id"]))
        stored = session.get(Complaint, created["id"])
        assert review.actor == api.settings.admin_username and review.action == "complaint_update"
        assert review.previous_values["priority_score"] == created["priority_score"]
        assert review.new_values["priority_score"] == 15 and review.feedback_exportable
        assert stored.analysis["priority_score"] == created["priority_score"]


def test_stale_override_is_a_conflict_and_does_not_create_feedback_or_audit(api):
    created = submit(api)
    headers = auth(api)
    endpoint = f"/complaints/{created['id']}"
    body = {"expected_version": created["version"], "reason": "Confirmed source complaint status", "status": "confirmed"}
    assert api.client.patch(endpoint, headers=headers, json=body).status_code == 200
    before = counts(api)
    assert api.client.patch(endpoint, headers=headers, json=body).status_code == 409
    assert counts(api) == before


def test_group_confirm_and_proper_subset_split_record_admin_decisions(api):
    one, two, three = submit(api), submit(api, "pani nai", "banglish"), submit(api, "পানি নেই", "bn")
    headers = auth(api)
    group = next(item for item in groups(api, headers) if item["id"] == one["group_id"])
    confirmed = api.client.post(f"/duplicate-groups/{group['id']}/confirm", headers=headers,
                               json={"expected_version": group["version"], "reason": "Same source service incident reviewed across languages"})
    assert confirmed.status_code == 200, confirmed.text
    current = confirmed.json()
    assert current["status"] == "confirmed" and current["version"] > group["version"]
    split = api.client.post(f"/duplicate-groups/{group['id']}/split", headers=headers,
                           json={"expected_version": current["version"], "reason": "Manual evidence shows one distinct incident",
                                 "complaint_ids": [three["id"]]})
    assert split.status_code == 200, split.text
    body = split.json()
    assert set(body["original_group"]["complaint_ids"]) == {one["id"], two["id"]}
    assert body["new_group"]["complaint_ids"] == [three["id"]]
    assert body["original_group"]["status"] == body["new_group"]["status"] == "reviewed_split"
    with api.session_factory() as session:
        actions = set(session.scalars(select(AdminReview.action)))
        assert {"group_confirm", "group_split"} <= actions


def test_invalid_or_stale_group_mutations_are_rejected(api):
    one, two = submit(api), submit(api, "pani nai", "banglish")
    headers = auth(api)
    group = next(item for item in groups(api, headers) if item["id"] == one["group_id"])
    path = f"/duplicate-groups/{group['id']}/split"
    reason = "Explicit reviewed split request"
    assert api.client.post(path, headers=headers, json={"expected_version": group["version"], "reason": reason, "complaint_ids": []}).status_code == 422
    all_members = api.client.post(path, headers=headers, json={"expected_version": group["version"], "reason": reason, "complaint_ids": [one["id"], two["id"]]})
    assert all_members.status_code in (409, 422)
    assert api.client.post(f"/duplicate-groups/{group['id']}/confirm", headers=headers,
                          json={"expected_version": 0, "reason": reason}).status_code == 422
    confirmed = api.client.post(f"/duplicate-groups/{group['id']}/confirm", headers=headers,
                                json={"expected_version": group["version"], "reason": reason})
    assert confirmed.status_code == 200
    assert api.client.post(f"/duplicate-groups/{group['id']}/confirm", headers=headers,
                          json={"expected_version": group["version"], "reason": reason}).status_code == 409


def test_reviewed_split_cannot_be_undone_by_next_automatic_similarity_match(api):
    one, two = submit(api), submit(api, "pani nai", "banglish")
    headers = auth(api)
    group = next(item for item in groups(api, headers) if item["id"] == one["group_id"])
    split = api.client.post(f"/duplicate-groups/{group['id']}/split", headers=headers,
                           json={"expected_version": group["version"], "reason": "Separate incidents despite similar wording", "complaint_ids": [two["id"]]})
    assert split.status_code == 200
    next_report = submit(api, "পানি নেই", "bn")
    assert next_report["group_id"] not in {split.json()["original_group"]["id"], split.json()["new_group"]["id"]}
    assert next_report["review_required"] is True


@pytest.mark.parametrize("body", [
    {"text": ""}, {"text": "x" * 4001}, {"text": "https://example.com"},
    {"text": "repeat " * 30}, {"text": "aaa" * 30},
    {"text": "water unavailable", "priority_score": 100},
    {"text": "water unavailable", "timestamp": "2026-01-01T12:00:00"},
])
def test_invalid_or_spam_submissions_are_rejected_without_persistence(api, body):
    before = counts(api)
    response = api.client.post("/complaints", json=body)
    assert response.status_code in (413, 422), response.text
    assert counts(api) == before


def test_payload_size_limit_and_invalid_priority_override_are_enforced(api):
    response = api.client.post("/complaints", content=json.dumps({"text": "a" * 50000}), headers={"Content-Type": "application/json"})
    assert response.status_code == 413
    created = submit(api)
    headers = auth(api)
    endpoint = f"/complaints/{created['id']}"
    for change in ({"priority_score": -1}, {"priority_score": 1001}, {"status": "unknown"}, {"priority_label": None}):
        assert api.client.patch(endpoint, headers=headers, json={"expected_version": created["version"], "reason": "Invalid override validation test", **change}).status_code == 422
    assert api.client.patch(endpoint, json={"expected_version": 1, "reason": "No credentials", "status": "confirmed"}).status_code == 401


def test_wrong_expired_and_wrong_role_tokens_cannot_authorize_admin_requests(api):
    assert api.client.post("/auth/token", json={"username": api.settings.admin_username, "password": "wrong-password"}).status_code == 401
    assert api.client.post("/auth/token", json={"username": "not-admin", "password": PASSWORD}).status_code == 401
    valid = auth(api)["Authorization"].split(" ", 1)[1]
    claims = jwt.decode(valid, options={"verify_signature": False})
    now = int(datetime.now(timezone.utc).timestamp())
    for edits in ({"role": "citizen"}, {"exp": now - 10}, {"sub": "other-admin"}):
        token = jwt.encode({**claims, **edits}, api.settings.jwt_secret, algorithm="HS256")
        assert api.client.get("/complaints", headers={"Authorization": "Bearer " + token}).status_code == 401
    assert api.client.get("/complaints", headers={"Authorization": "Bearer invalid"}).status_code == 401


def test_rate_limit_is_effective_and_untrusted_x_forwarded_for_does_not_bypass(tmp_path):
    context = build_client(tmp_path, submission_rate_limit="2/minute")
    with TestClient(context.app) as client:
        context.client = client
        for address in ("198.51.100.1", "198.51.100.2"):
            response = client.post("/complaints", json={"text": "No water", "lang": "en", "location": "Mirpur 10"}, headers={"X-Forwarded-For": address})
            assert response.status_code == 201, response.text
        blocked = client.post("/complaints", json={"text": "No water", "lang": "en", "location": "Mirpur 10"}, headers={"X-Forwarded-For": "198.51.100.3"})
        assert blocked.status_code == 429, blocked.text
    context.engine.dispose()


def test_public_read_opt_in_uses_redacted_views(tmp_path):
    context = build_client(tmp_path, allow_public_reads=True)
    with TestClient(context.app) as client:
        context.client = client
        created = submit(context, "No water, private detailed wording", "en", "Mirpur 10")
        listing = client.get("/complaints")
        assert listing.status_code == 200
        row = listing.json()["items"][0]
        assert "text" not in row and "analysis" not in row and "location" not in row
        detail = client.get(f"/complaints/{created['id']}")
        assert detail.status_code == 200
        assert "text" not in detail.json() and "analysis" not in detail.json()
    context.engine.dispose()


def test_unknown_area_requires_review_instead_of_automatic_grouping(api):
    first = submit(api, location="An area outside the gazetteer")
    second = submit(api, "pani nai", "banglish", "An area outside the gazetteer")
    assert first["group_id"] != second["group_id"]
    assert second["review_required"] is True
    with api.session_factory() as session:
        stored = session.get(Complaint, second["id"])
        assert stored.analysis["duplicate_candidates"][0]["area_guard_status"] == "unknown"


def test_multiple_existing_groups_are_not_silently_merged(api):
    first = submit(api)
    second = submit(api, "pani nai", "banglish", "Uttara sector 4")
    # Simulate later location correction/import into two pre-existing groups.
    # The next submission must not merge those groups on a transitive guess.
    with api.session_factory.begin() as session:
        corrected = session.get(Complaint, second["id"])
        corrected.location, corrected.area_id = "Mirpur 10", "dhaka_mirpur_10"
    third = submit(api, "পানি নেই", "bn")
    assert third["group_id"] not in {first["group_id"], second["group_id"]}
    assert third["review_required"] is True


def test_admin_can_explicitly_merge_memberships_and_export_actual_decision(api):
    first = submit(api)
    other = submit(api, "pani nai", "banglish", "Uttara sector 4")
    headers = auth(api)
    target = next(item for item in groups(api, headers) if item["id"] == first["group_id"])
    response = api.client.post(f"/duplicate-groups/{target['id']}/confirm", headers=headers,
                              json={"expected_version": target["version"],
                                    "reason": "Expert checked reporting-area mismatch against source evidence",
                                    "complaint_ids": [other["id"]]})
    assert response.status_code == 200, response.text
    assert set(response.json()["complaint_ids"]) == {first["id"], other["id"]}
    with api.session_factory() as session:
        moved = session.get(Complaint, other["id"])
        review = session.scalar(select(AdminReview).where(AdminReview.group_id == target["id"]))
        assert moved.group_id == target["id"] and moved.version > other["version"]
        assert review.action == "group_confirm" and review.actor == api.settings.admin_username
        assert review.previous_values != review.new_values


def test_rejected_reports_are_excluded_from_future_duplicate_index(api):
    first = submit(api)
    response = api.client.patch(f"/complaints/{first['id']}", headers=auth(api),
                               json={"expected_version": first["version"], "reason": "Spam report confirmed by administrator",
                                     "status": "rejected"})
    assert response.status_code == 200, response.text
    second = submit(api, "pani nai", "banglish")
    assert second["group_id"] != first["group_id"]
    with api.session_factory() as session:
        stored = session.get(Complaint, second["id"])
        assert all(item["id"] != first["id"] for item in stored.analysis["duplicate_candidates"])


def test_two_workers_serialize_concurrent_group_assignment(tmp_path):
    from concurrent.futures import ThreadPoolExecutor

    first, second = build_client(tmp_path), build_client(tmp_path)
    examples = [("মিরপুর ১০ এ পানি নেই", "bn"), ("mirpur-10 e pani nai", "banglish"),
                ("No water in Mirpur 10", "en")] * 2
    with TestClient(first.app) as first_client, TestClient(second.app) as second_client:
        def post(index):
            text, lang = examples[index]
            client = (first_client, second_client)[index % 2]
            return client.post("/complaints", json={"text": text, "lang": lang, "location": "Mirpur 10"})

        with ThreadPoolExecutor(max_workers=6) as pool:
            responses = list(pool.map(post, range(len(examples))))
        assert all(response.status_code == 201 for response in responses), [response.text for response in responses]
        assert len({response.json()["group_id"] for response in responses}) == 1
        with first.session_factory() as session:
            assert session.scalar(select(func.count()).select_from(Complaint)) == len(examples)
    first.engine.dispose()
    second.engine.dispose()


def test_explicit_member_move_preserves_other_group_split_protection(api):
    first, second, third = submit(api), submit(api, "pani nai", "banglish"), submit(api, "পানি নেই", "bn")
    headers = auth(api)
    group = next(item for item in groups(api, headers) if item["id"] == first["group_id"])
    split = api.client.post(f"/duplicate-groups/{group['id']}/split", headers=headers,
                           json={"expected_version": group["version"], "reason": "Evidence supports two distinct incidents",
                                 "complaint_ids": [third["id"]]})
    assert split.status_code == 200, split.text
    target = split.json()["new_group"]
    merged = api.client.post(f"/duplicate-groups/{target['id']}/confirm", headers=headers,
                            json={"expected_version": target["version"],
                                  "reason": "One source report belongs to the second reviewed incident",
                                  "complaint_ids": [second["id"]]})
    assert merged.status_code == 200, merged.text
    original = next(item for item in groups(api, headers) if item["id"] == group["id"])
    assert original["complaint_ids"] == [first["id"]]
    assert original["status"] == "reviewed_split"


def test_contact_is_admin_only_and_never_reaches_ml_or_feedback(api, tmp_path):
    from backend.scripts.export_feedback import export_feedback

    contact = "synthetic-private-contact@example.invalid"
    created = submit(api, contact="  " + contact + "  ")
    assert "contact" not in created and contact not in json.dumps(created)
    assert all("contact" not in call for call in api.analyzer.calls)
    headers = auth(api)
    detail = api.client.get(f"/complaints/{created['id']}", headers=headers)
    assert detail.status_code == 200 and detail.json()["contact"] == contact
    with api.session_factory() as session:
        complaint = session.get(Complaint, created["id"])
        assert complaint.contact == contact and contact not in json.dumps(complaint.analysis)
    changed = api.client.patch(f"/complaints/{created['id']}", headers=headers,
                              json={"expected_version": created["version"], "status": "confirmed", "reason": "Reviewed current service status"})
    assert changed.status_code == 200
    path = tmp_path / "private-feedback.jsonl"
    export_feedback(api.settings.database_url, path)
    assert contact not in path.read_text(encoding="utf-8")
    assert '"contact"' not in path.read_text(encoding="utf-8")


@pytest.mark.parametrize("contact", ["a" * 257, "admin@example.invalid\nextra", "\u0001"])
def test_contact_validation_rejects_excess_length_and_controls(api, contact):
    before = counts(api)
    response = api.client.post("/complaints", json={"text": "Water unavailable", "contact": contact})
    assert response.status_code == 422
    assert counts(api) == before


def test_public_tracking_returns_status_summary_without_private_fields(api):
    created = submit(api, contact="private-contact@example.invalid")
    tracked = api.client.get(f"/tracking/{created['id']}")
    assert tracked.status_code == 200
    assert set(tracked.json()) == {"id", "status", "category", "severity", "priority_score", "priority_label",
                                   "review_required", "area_id", "lang", "created_at", "updated_at"}
    assert tracked.json()["id"] == created["id"] and tracked.json()["status"] == created["status"]
    assert created["text"] not in tracked.text and "private-contact" not in tracked.text
    assert api.client.get(f"/tracking/{uuid4()}").status_code == 404
    assert api.client.get("/tracking/not-a-uuid").status_code == 422


def test_tracking_rate_limit_is_enforced(tmp_path):
    context = build_client(tmp_path, analysis_rate_limit="2/minute")
    try:
        with TestClient(context.app) as client:
            context.client = client
            created = submit(context)
            endpoint = f"/tracking/{created['id']}"
            assert client.get(endpoint).status_code == 200
            assert client.get(endpoint).status_code == 200
            assert client.get(endpoint).status_code == 429
    finally:
        context.engine.dispose()


def test_public_read_opt_in_still_omits_contact(tmp_path):
    context = build_client(tmp_path, allow_public_reads=True)
    try:
        with TestClient(context.app) as client:
            context.client = client
            created = submit(context, contact="private-contact@example.invalid")
            detail = client.get(f"/complaints/{created['id']}")
            assert detail.status_code == 200 and "contact" not in detail.json()
            listing = client.get("/complaints")
            assert "private-contact" not in listing.text and "contact" not in listing.json()["items"][0]
    finally:
        context.engine.dispose()


def test_review_filter_and_direct_admin_group_lookup(api):
    first = submit(api)
    second = submit(api, "Road blocked", "en", "Unknown area")
    headers = auth(api)
    marked = api.client.patch(f"/complaints/{first['id']}", headers=headers,
                             json={"expected_version": first["version"], "status": "in_review", "reason": "Needs field team verification"})
    assert marked.status_code == 200
    response = api.client.get("/complaints", params={"review_required": True}, headers=headers)
    assert response.status_code == 200 and response.json()["total"] == 1
    assert response.json()["items"][0]["id"] == first["id"]
    group = api.client.get(f"/duplicate-groups/{first['group_id']}", headers=headers)
    assert group.status_code == 200 and first["id"] in group.json()["complaint_ids"]
    assert api.client.get(f"/duplicate-groups/{first['group_id']}").status_code == 401
    assert api.client.get(f"/duplicate-groups/{uuid4()}", headers=headers).status_code == 404
    filtered = api.client.get("/duplicate-groups", params={"group_id": first["group_id"]}, headers=headers)
    assert filtered.status_code == 200 and filtered.json()["total"] == 1
    assert filtered.json()["items"][0]["id"] != second["group_id"]


def test_admin_analytics_uses_full_corpus_utc_dates_and_current_review_state(api):
    first = submit(api)
    second = submit(api, "Road has a pothole", "en", "Mohammadpur")
    with api.session_factory.begin() as session:
        one, two = session.get(Complaint, first["id"]), session.get(Complaint, second["id"])
        one.created_at = datetime(2026, 10, 30, 0, 30, tzinfo=timezone.utc)
        two.created_at = datetime(2026, 10, 31, 12, 0, tzinfo=timezone.utc)
        one.priority_label = "urgent"
        two.status, two.review_required = "in_review", True
    assert api.client.get("/analytics").status_code == 401
    response = api.client.get("/analytics", headers=auth(api))
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["total_complaints"] == 2 and body["by_category"] == {"Water": 1, "Road & Transport": 1}
    assert body["by_priority"] == {"high": 1, "urgent": 1}
    assert body["needs_human_review"] == 1
    assert body["priority_over_time"] == [{"date": "2026-10-30", "priority": "urgent", "count": 1},
                                           {"date": "2026-10-31", "priority": "high", "count": 1}]
    assert sum(item["count"] for item in body["review_workload"]) == 2
    assert {"status": "in_review", "review_required": True, "count": 1} in body["review_workload"]
    assert first["text"] not in response.text and "contact" not in body
