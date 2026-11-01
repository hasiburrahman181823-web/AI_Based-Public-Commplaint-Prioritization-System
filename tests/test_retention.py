"""Synthetic erasure fixtures; no real complaint or contact data is deleted."""

from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
import sqlite3

import pytest
from sqlalchemy import func, select

from backend.db import make_engine, make_session_factory
from backend.models import AdminReview, AuditLog, Base, Complaint, DuplicateGroup, new_id
from backend.retention import coordinated_session, delete_complaints, purge_embedding_cache, select_retention_candidates
from backend.scripts.delete_data import main, run_deletion


@pytest.fixture
def storage(tmp_path):
    url = "sqlite:///" + (tmp_path / "records.sqlite3").as_posix()
    engine = make_engine(url)
    Base.metadata.create_all(engine)
    factory = make_session_factory(engine)
    yield url, engine, factory
    engine.dispose()


def complaint(identifier=None, **updates):
    return Complaint(id=identifier or new_id(), text="Synthetic service complaint", lang="en", **updates)


def cache_at(directory: Path):
    directory.mkdir()
    path = directory / "embeddings.sqlite3"
    with sqlite3.connect(path) as connection:
        connection.executescript("CREATE TABLE namespaces(namespace TEXT PRIMARY KEY,dimension INTEGER);"
                                 "CREATE TABLE embeddings(namespace TEXT,text_key TEXT,dimension INTEGER,vector BLOB);")
        connection.execute("INSERT INTO namespaces VALUES ('test-model',2)")
        connection.execute("INSERT INTO embeddings VALUES ('test-model','private-source-hash',2,?)", (b"\x00" * 8,))
    return path


def test_erasure_removes_source_private_fields_and_known_linked_copies(storage):
    _, _, factory = storage
    group_id, other_group = new_id(), new_id()
    target_id, survivor_id, unrelated_id = new_id(), new_id(), new_id()
    with factory() as session, session.begin():
        session.add_all([DuplicateGroup(id=group_id, status="confirmed", confirmed_by="reviewer"), DuplicateGroup(id=other_group)])
        session.flush()
        target = complaint(target_id, group_id=group_id, contact="private-person@example.invalid", location="Private-original-location",
                           analysis={"text_copy": "private-original-content"})
        target.text = "private-original-content"
        session.add_all([target, complaint(survivor_id, group_id=group_id, source_report_id=survivor_id,
                                           analysis={"duplicate_candidates": [{"id": target_id, "text": "private-original-content"}]}),
                         complaint(unrelated_id, group_id=other_group, analysis={"own_evidence": "unrelated-kept"})])
        session.flush()
        session.add_all([
            AdminReview(complaint_id=target_id, actor="reviewer", action="complaint_update", reason="private-original-content"),
            AdminReview(group_id=group_id, actor="reviewer", action="group_confirm", reason="private-person@example.invalid",
                        previous_values={"complaint_ids": [target_id, survivor_id]}),
            AdminReview(group_id=other_group, actor="reviewer", action="group_confirm", reason="historical move",
                        new_values={"moved_from": {"id": target_id, "text": "private-original-content"}}),
            AdminReview(complaint_id=unrelated_id, actor="reviewer", action="complaint_update", reason="unrelated-kept"),
            AuditLog(actor="anonymous", action="complaint_submit", entity_type="complaint", entity_id=target_id,
                     request_id="old-request", changes={"text": "private-original-content"}),
            AuditLog(actor="reviewer", action="group_confirm", entity_type="duplicate_group", entity_id=group_id,
                     changes={"reason": "private-person@example.invalid"}),
            AuditLog(actor="reviewer", action="group_member_move", entity_type="duplicate_group", entity_id=other_group,
                     changes={"id": target_id}),
            AuditLog(actor="reviewer", action="complaint_update", entity_type="complaint", entity_id=unrelated_id,
                     changes={"reason": "unrelated-kept"}),
        ])
    with coordinated_session(factory) as session:
        result = delete_complaints(session, [target_id], "privacy-admin", "Do not copy private-person@example.invalid into audit", "new-request")
    assert result["deleted_count"] == 1 and result["deleted_ids"] == [target_id]
    assert result["deleted_reviews"] == 1 and result["redacted_reviews"] == 2
    assert result["redacted_audit_logs"] == 3 and result["redacted_survivor_analyses"] == 1
    with factory() as session:
        assert session.get(Complaint, target_id) is None
        survivor = session.get(Complaint, survivor_id)
        assert survivor.review_required and survivor.analysis["privacy_redacted"] and survivor.version == 2
        assert survivor.text == "Synthetic service complaint"
        assert session.get(Complaint, unrelated_id).analysis == {"own_evidence": "unrelated-kept"}
        assert session.get(DuplicateGroup, group_id).status == "proposed"
        review = session.scalar(select(AdminReview).where(AdminReview.group_id == group_id))
        assert review.feedback_exportable is False and review.previous_values == {"privacy_redacted": True}
        audit = session.get(AuditLog, result["audit_event_id"])
        assert audit.actor == "privacy-admin" and audit.action == "complaints_delete" and audit.entity_id is None
        assert audit.changes["deleted_count"] == 1 and audit.created_at.tzinfo is not None
        serialized = json.dumps({"complaints": list(session.execute(select(Complaint.text, Complaint.contact, Complaint.analysis))),
                                 "reviews": list(session.execute(select(AdminReview.reason, AdminReview.previous_values, AdminReview.new_values))),
                                 "audits": list(session.execute(select(AuditLog.entity_id, AuditLog.changes)))}, default=str)
        for removed in (target_id, "private-original-content", "private-person@example.invalid", "Private-original-location"):
            assert removed not in serialized
        assert "unrelated-kept" in serialized


def test_missing_ids_are_idempotent_and_do_not_create_audit(storage):
    _, _, factory = storage
    identifier = new_id()
    with coordinated_session(factory) as session:
        result = delete_complaints(session, [identifier], "admin", "verified request")
    assert result["deleted_count"] == 0 and result["missing_ids"] == [identifier] and result["audit_event_id"] is None
    with factory() as session:
        assert session.scalar(select(func.count()).select_from(AuditLog)) == 0


def test_failure_rolls_back_source_review_and_audit_changes(storage):
    _, _, factory = storage
    identifier = new_id()
    with factory() as session, session.begin():
        session.add(complaint(identifier))
        session.flush()
        session.add(AdminReview(complaint_id=identifier, actor="reviewer", action="complaint_update", reason="keep original review"))
    with pytest.raises(RuntimeError):
        with coordinated_session(factory) as session:
            delete_complaints(session, [identifier], "admin", "verified deletion")
            raise RuntimeError("force rollback")
    with factory() as session:
        assert session.get(Complaint, identifier) is not None
        assert session.scalar(select(AdminReview.reason)) == "keep original review"
        assert session.scalar(select(func.count()).select_from(AuditLog)) == 0


def test_reviewed_split_protection_and_unrelated_reports_survive(storage):
    _, _, factory = storage
    group_id, target, other = new_id(), new_id(), new_id()
    with factory() as session, session.begin():
        session.add(DuplicateGroup(id=group_id, status="reviewed_split"))
        session.flush()
        session.add_all([complaint(target, group_id=group_id), complaint(other, group_id=group_id)])
    with coordinated_session(factory) as session:
        delete_complaints(session, [target], "admin", "verified deletion")
    with factory() as session:
        assert session.get(DuplicateGroup, group_id).status == "reviewed_split"
        assert session.get(Complaint, other) is not None


def test_surviving_lineage_is_not_silently_cascade_deleted(storage):
    _, _, factory = storage
    parent, derived = new_id(), new_id()
    with factory() as session, session.begin():
        session.add_all([complaint(parent), complaint(derived, source_report_id=parent.upper())])
    with coordinated_session(factory) as session:
        result = delete_complaints(session, [parent], "admin", "verified deletion")
    assert result["lineage_links_removed"] == 1 and result["warnings"]
    with factory() as session:
        retained = session.get(Complaint, derived)
        assert retained is not None and retained.source_report_id is None and retained.review_required


def test_default_retention_selects_only_old_closed_last_updates(storage):
    _, _, factory = storage
    now = datetime(2026, 10, 31, tzinfo=timezone.utc)
    old, recent, boundary = now - timedelta(days=91), now - timedelta(days=5), now - timedelta(days=90)
    identifiers = [new_id() for _ in range(5)]
    with factory() as session, session.begin():
        session.add_all([complaint(identifiers[0], status="resolved", updated_at=old),
                         complaint(identifiers[1], status="rejected", updated_at=old),
                         complaint(identifiers[2], status="submitted", updated_at=old),
                         complaint(identifiers[3], status="resolved", updated_at=recent),
                         complaint(identifiers[4], status="resolved", updated_at=boundary)])
    with factory() as session:
        assert set(select_retention_candidates(session, now=now)) == set(identifiers[:2])
        assert len(select_retention_candidates(session, now=now, limit=1)) == 1


@pytest.mark.parametrize("options", [
    {"older_than": datetime(2099, 1, 1, tzinfo=timezone.utc)},
    {"older_than": datetime(2026, 1, 1)}, {"days": 0}, {"days": True}, {"limit": 1001},
])
def test_invalid_retention_cutoffs_or_limits_are_rejected(storage, options):
    _, _, factory = storage
    with factory() as session, pytest.raises(ValueError):
        select_retention_candidates(session, **options)


@pytest.mark.parametrize("ids", [[], ["not-a-uuid"], "not-a-list"])
def test_invalid_delete_requests_are_rejected_before_writes(storage, ids):
    _, _, factory = storage
    with coordinated_session(factory) as session, pytest.raises(ValueError):
        delete_complaints(session, ids, "admin", "verified request")


def test_deletion_requires_an_explicit_transaction(storage):
    _, _, factory = storage
    with factory() as session, pytest.raises(ValueError, match="coordinated transaction"):
        delete_complaints(session, [new_id()], "admin", "verified request")


def test_cli_default_dry_run_keeps_rows_feedback_audit_and_cache(storage, tmp_path):
    url, _, factory = storage
    identifier = new_id()
    with factory() as session, session.begin():
        session.add(complaint(identifier, status="resolved", updated_at=datetime(2020, 1, 1, tzinfo=timezone.utc)))
    cache = cache_at(tmp_path / "cache")
    report = run_deletion(url, cache_dir=cache.parent)
    assert report["dry_run"] and report["candidate_ids"] == [identifier] and not report["writes_applied"]
    with factory() as session:
        assert session.get(Complaint, identifier) is not None
        assert session.scalar(select(func.count()).select_from(AuditLog)) == 0
    with sqlite3.connect(cache) as connection:
        assert connection.execute("SELECT count(*) FROM embeddings").fetchone()[0] == 1


def test_cli_apply_purges_recognized_cache_and_physically_erases_rows(storage, tmp_path):
    url, _, factory = storage
    identifier = new_id()
    with factory() as session, session.begin():
        session.add(complaint(identifier, contact="synthetic-private@example.invalid"))
    cache = cache_at(tmp_path / "cache")
    report = run_deletion(url, complaint_ids=[identifier], apply=True, actor="privacy-admin", reason="verified privacy request", cache_dir=cache.parent)
    assert report["deleted_ids"] == [identifier] and report["cache_purged"] and report["writes_applied"]
    assert report["cache"]["deleted_embeddings"] == 1 and report["remaining_operator_actions"]
    with factory() as session:
        assert session.get(Complaint, identifier) is None
    with sqlite3.connect(cache) as connection:
        assert connection.execute("SELECT count(*) FROM embeddings").fetchone()[0] == 0
        assert connection.execute("SELECT count(*) FROM namespaces").fetchone()[0] == 0


def test_cache_failure_rolls_back_domain_deletion_and_does_not_clear_unrelated_database(storage, tmp_path):
    url, _, factory = storage
    identifier = new_id()
    with factory() as session, session.begin():
        session.add(complaint(identifier))
    cache_dir = tmp_path / "not-cache"
    cache_dir.mkdir()
    path = cache_dir / "embeddings.sqlite3"
    with sqlite3.connect(path) as connection:
        connection.executescript("CREATE TABLE unrelated(value TEXT); INSERT INTO unrelated VALUES ('keep me');")
    with pytest.raises(ValueError, match="recognized embedding-cache"):
        run_deletion(url, complaint_ids=[identifier], apply=True, actor="admin", reason="verified deletion", cache_dir=cache_dir)
    with factory() as session:
        assert session.get(Complaint, identifier) is not None
        assert session.scalar(select(func.count()).select_from(AuditLog)) == 0
    with sqlite3.connect(path) as connection:
        assert connection.execute("SELECT value FROM unrelated").fetchone()[0] == "keep me"


def test_purge_does_not_create_files_or_recursively_remove_other_cache_material(tmp_path):
    root = tmp_path / "cache"
    assert purge_embedding_cache(root) == {"status": "absent", "deleted_embeddings": 0}
    assert not root.exists()
    cache = cache_at(root)
    other = root / "keep-model.json"
    other.write_text("keep", encoding="utf-8")
    purge_embedding_cache(root)
    assert cache.exists() and other.read_text(encoding="utf-8") == "keep"


def test_cli_rejects_apply_without_actor_reason_and_never_prints_db_credentials(storage, monkeypatch, capsys):
    url, _, _ = storage
    monkeypatch.setenv("BACKEND_DATABASE_URL", url)
    assert main(["--apply"]) == 2
    output = capsys.readouterr()
    assert "--actor and --reason" in output.err and url not in output.err


def test_cli_clear_failure_does_not_create_a_missing_database(tmp_path, monkeypatch, capsys):
    path = tmp_path / "missing.sqlite3"
    monkeypatch.setenv("BACKEND_DATABASE_URL", "sqlite:///" + path.as_posix())
    assert main([]) == 2
    assert "does not exist" in capsys.readouterr().err and not path.exists()
