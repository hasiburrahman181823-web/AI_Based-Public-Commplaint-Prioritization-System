"""Streaming, non-mutating human-feedback event exports."""

import csv
from datetime import datetime, timezone
import hashlib
import json

import pytest
from sqlalchemy import func, select

from backend.db import make_engine, make_session_factory, session_scope
from backend.models import AdminReview, AuditLog, Base, Complaint, DuplicateGroup
from backend.scripts.export_feedback import export_feedback, main


@pytest.fixture
def feedback(tmp_path):
    database = tmp_path / "backend.sqlite3"
    url = "sqlite:///" + database.as_posix()
    engine = make_engine(url)
    Base.metadata.create_all(engine)
    factory = make_session_factory(engine)
    with session_scope(factory) as session:
        group = DuplicateGroup(status="confirmed", confirmed_by="reviewer")
        session.add(group)
        session.flush()
        complaint = Complaint(text="মিরপুর ১০ এ পানি নেই", lang="bn", area_id="dhaka_mirpur_10", group_id=group.id,
                              source_report_id="source-001", priority_score=8,
                              analysis={"priority_score": 3, "priority_label": "low", "category": "Water", "severity": "Low"})
        session.add(complaint)
        session.flush()
        records = [
            AdminReview(complaint_id=complaint.id, actor="admin-one", action="complaint_update", reason='Actual impact reviewed,\n"more urgent"',
                        previous_values={"priority_score": 3, "status": "submitted"}, new_values={"priority_score": 8, "status": "in_review"},
                        created_at=datetime(2026, 10, 30, 6, tzinfo=timezone.utc)),
            AdminReview(group_id=group.id, actor="admin-two", action="group_confirm", reason="Reviewed same-incident evidence",
                        previous_values={"member_ids": [complaint.id], "status": "proposed"},
                        new_values={"member_ids": [complaint.id], "status": "confirmed"},
                        created_at=datetime(2026, 10, 30, 7, tzinfo=timezone.utc)),
            AdminReview(group_id=group.id, actor="admin-two", action="group_split", reason="Different location evidence",
                        previous_values={"member_ids": [complaint.id]}, new_values={"detached_ids": [complaint.id]},
                        created_at=datetime(2026, 10, 30, 8, tzinfo=timezone.utc)),
            AdminReview(complaint_id=complaint.id, actor="admin-one", action="complaint_update", reason="Nonexportable review",
                        feedback_exportable=False, created_at=datetime(2026, 10, 30, 9, tzinfo=timezone.utc)),
        ]
        session.add_all(records)
        session.add(AuditLog(actor="admin-one", action="complaint_update", entity_type="complaint", entity_id=complaint.id,
                             changes={"event": "audit stays unchanged"}))
    yield {"url": url, "database": database, "factory": factory, "engine": engine}
    engine.dispose()


def counts(factory):
    with factory() as session:
        return {entity.__tablename__: session.scalar(select(func.count()).select_from(entity)) for entity in (Complaint, AdminReview, DuplicateGroup, AuditLog)}


def test_jsonl_export_keeps_exact_events_and_automatic_snapshot_without_mutation(feedback, tmp_path):
    before = counts(feedback["factory"])
    destination = tmp_path / "feedback.jsonl"
    manifest = export_feedback(feedback["url"], destination)
    rows = [json.loads(line) for line in destination.read_text(encoding="utf-8").splitlines()]
    assert manifest["rows"] == len(rows) == 3
    assert counts(feedback["factory"]) == before
    assert rows[0]["reason"] == 'Actual impact reviewed,\n"more urgent"'
    assert rows[0]["previous_values"]["priority_score"] == 3 and rows[0]["new_values"]["priority_score"] == 8
    assert rows[0]["original_analysis"]["priority_score"] == 3
    assert rows[0]["original_analysis"]["severity"] == "Low"
    assert rows[0]["source_report_id"] == "source-001"
    assert all("text" not in row for row in rows)
    assert rows[1]["action"] == "group_confirm" and rows[1]["complaint_id"] is None
    assert rows[1]["original_analysis"] is None and "member_ids" in rows[1]["new_values"]
    assert rows[2]["action"] == "group_split" and "detached_ids" in rows[2]["new_values"]
    assert all(row["created_at"].endswith("Z") for row in rows)
    assert manifest["sha256"] == hashlib.sha256(destination.read_bytes()).hexdigest()
    assert "database_url" not in manifest
    assert json.loads(destination.with_name(destination.name + ".manifest.json").read_text(encoding="utf-8")) == manifest


def test_csv_preserves_json_and_quotes_and_text_requires_explicit_flag(feedback, tmp_path):
    destination = tmp_path / "feedback.csv"
    export_feedback(feedback["url"], destination, "csv", include_text=True)
    with destination.open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert rows[0]["text"] == "মিরপুর ১০ এ পানি নেই"
    assert json.loads(rows[0]["original_analysis"])["priority_score"] == 3
    assert json.loads(rows[0]["new_values"])["priority_score"] == 8
    assert rows[0]["reason"] == 'Actual impact reviewed,\n"more urgent"'
    assert rows[1]["text"] == "" and json.loads(rows[1]["original_analysis"]) is None


def test_since_is_inclusive_utc_with_offset_and_excludes_older_events(feedback, tmp_path):
    destination = tmp_path / "since.jsonl"
    result = export_feedback(feedback["url"], destination, since="2026-10-30T13:00:00+06:00")
    rows = [json.loads(line) for line in destination.read_text(encoding="utf-8").splitlines()]
    assert result["rows"] == 2
    assert rows[0]["created_at"] == "2026-10-30T07:00:00Z"
    assert result["since_utc_inclusive"] == "2026-10-30T07:00:00Z"


def test_empty_exports_have_valid_jsonl_and_csv_header(feedback, tmp_path):
    jsonl, csv_path = tmp_path / "empty.jsonl", tmp_path / "empty.csv"
    assert export_feedback(feedback["url"], jsonl, since="2027-01-01T00:00:00Z")["rows"] == 0
    assert jsonl.read_bytes() == b""
    assert export_feedback(feedback["url"], csv_path, "csv", since="2027-01-01T00:00:00Z")["rows"] == 0
    with csv_path.open(encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        assert "feedback_id" in reader.fieldnames and "text" not in reader.fieldnames
        assert list(reader) == []


@pytest.mark.parametrize("value", ["2026-10-30", "2026-10-30T10:00:00", "nonsense", 4])
def test_since_requires_explicit_valid_timezone(feedback, tmp_path, value):
    with pytest.raises(ValueError, match="since"):
        export_feedback(feedback["url"], tmp_path / "bad.jsonl", since=value)
    assert not (tmp_path / "bad.jsonl").exists()


def test_existing_outputs_and_database_journals_are_protected(feedback, tmp_path):
    destination = tmp_path / "feedback.jsonl"
    export_feedback(feedback["url"], destination)
    with pytest.raises(ValueError, match="exists"):
        export_feedback(feedback["url"], destination)
    export_feedback(feedback["url"], destination, overwrite=True)
    for suffix in ("", "-wal", "-shm", "-journal"):
        with pytest.raises(ValueError, match="database"):
            export_feedback(feedback["url"], str(feedback["database"]) + suffix, overwrite=True)


def test_missing_or_memory_sqlite_does_not_create_a_database(tmp_path):
    missing = tmp_path / "absent.sqlite3"
    with pytest.raises(ValueError, match="does not exist"):
        export_feedback("sqlite:///" + missing.as_posix(), tmp_path / "out.jsonl")
    assert not missing.exists()
    with pytest.raises(ValueError, match="persistent"):
        export_feedback("sqlite:///:memory:", tmp_path / "out.jsonl")


def test_cli_exports_with_environment_url_and_suppresses_credentials(feedback, tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("BACKEND_DATABASE_URL", feedback["url"])
    destination = tmp_path / "cli.jsonl"
    assert main(["--output", str(destination)]) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["rows"] == 3 and "database_url" not in result
    assert main(["--output", str(destination)]) == 2
    assert "ERROR:" in capsys.readouterr().err


def test_bad_format_and_flags_rejected_before_output(feedback, tmp_path):
    with pytest.raises(ValueError, match="format"):
        export_feedback(feedback["url"], tmp_path / "out", "xml")
    with pytest.raises(ValueError, match="boolean"):
        export_feedback(feedback["url"], tmp_path / "out", include_text="yes")
