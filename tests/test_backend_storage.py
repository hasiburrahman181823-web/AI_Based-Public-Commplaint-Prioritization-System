"""Database constraints, transactions and migration consistency."""

from datetime import datetime, timedelta, timezone
from pathlib import Path
import uuid

import pytest
from sqlalchemy import inspect, select, text, update
from sqlalchemy.exc import IntegrityError

from backend.db import make_engine, make_session_factory, session_scope
from backend.models import AdminReview, AuditLog, Base, Complaint, DuplicateGroup


@pytest.fixture
def storage():
    engine = make_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    factory = make_session_factory(engine)
    yield engine, factory
    engine.dispose()


def complaint(**changes):
    return Complaint(text="মিরপুর ১০ এ পানি নেই", lang="bn", **changes)


def test_exact_domain_tables_indexes_and_connection_constraints(storage):
    engine, _ = storage
    database = inspect(engine)
    assert set(database.get_table_names()) == {
        "complaints", "duplicate_groups", "admin_reviews", "audit_log",
        "citizen_accounts", "citizen_complaint_links",
    }
    assert {row["name"] for row in database.get_indexes("complaints")} >= {
        "ix_complaints_filters", "ix_complaints_group_id", "ix_complaints_status", "ix_complaints_area_id",
    }
    assert "ix_citizen_accounts_email" in {row["name"] for row in database.get_indexes("citizen_accounts")}
    with engine.connect() as connection:
        assert connection.execute(text("PRAGMA foreign_keys")).scalar_one() == 1
        assert connection.execute(text("PRAGMA busy_timeout")).scalar_one() == 5000


def test_defaults_uuid_and_independent_json_snapshots_persist(storage):
    _, factory = storage
    with session_scope(factory) as session:
        first, second = complaint(), complaint()
        session.add_all([first, second])
        session.flush()
        uuid.UUID(first.id)
        assert first.id != second.id
        assert first.status == "submitted" and first.review_required
        assert first.priority_score == 0 and first.priority_label == "low" and first.version == 1
        assert first.analysis == second.analysis == {} and first.analysis is not second.analysis
        first.analysis = {"review_required": True, "category": None}
        identifier = first.id
    with factory() as session:
        stored = session.get(Complaint, identifier)
        assert stored.analysis == {"review_required": True, "category": None}
        assert stored.created_at.utcoffset() == timedelta(0)
        assert stored.updated_at.utcoffset() == timedelta(0)


@pytest.mark.parametrize("values", [
    {"priority_score": -1}, {"confidence": -0.1}, {"confidence": 1.1},
    {"version": 0}, {"lang": "unknown"}, {"status": "deleted"},
    {"text": ""}, {"text": "a" * 4001},
])
def test_complaint_invalid_direct_writes_are_rejected(storage, values):
    _, factory = storage
    with pytest.raises(IntegrityError):
        with session_scope(factory) as session:
            session.add(Complaint(text=values.get("text", "water problem"),
                                  lang=values.get("lang", "en"),
                                  **{key: value for key, value in values.items() if key not in ("text", "lang")}))
    with factory() as session:
        assert session.scalars(select(Complaint)).all() == []


def test_missing_group_foreign_key_is_rejected(storage):
    _, factory = storage
    with pytest.raises(IntegrityError):
        with session_scope(factory) as session:
            session.add(complaint(group_id=str(uuid.uuid4())))


def test_group_delete_detaches_complaints_without_deleting_them(storage):
    _, factory = storage
    with session_scope(factory) as session:
        group = DuplicateGroup()
        session.add(group)
        session.flush()
        row = complaint(group_id=group.id)
        session.add(row)
        session.flush()
        group_id, identifier = group.id, row.id
    with session_scope(factory) as session:
        session.delete(session.get(DuplicateGroup, group_id))
    with factory() as session:
        assert session.get(Complaint, identifier).group_id is None
        assert session.get(DuplicateGroup, group_id) is None


def test_review_target_constraint_and_feedback_defaults(storage):
    _, factory = storage
    with pytest.raises(IntegrityError):
        with session_scope(factory) as session:
            session.add(AdminReview(actor="reviewer", action="complaint_update", reason="No target"))
    with session_scope(factory) as session:
        row = complaint()
        session.add(row)
        session.flush()
        review = AdminReview(complaint_id=row.id, actor="reviewer", action="complaint_update",
                             reason="Confirmed reported impact", previous_values={"priority_score": 0},
                             new_values={"priority_score": 5})
        session.add(review)
        session.flush()
        assert review.feedback_exportable
        identifier = review.id
    with factory() as session:
        stored = session.get(AdminReview, identifier)
        assert stored.actor == "reviewer" and stored.previous_values["priority_score"] == 0
        assert stored.new_values["priority_score"] == 5
        assert stored.created_at.utcoffset() == timedelta(0)


@pytest.mark.parametrize("values", [{"status": "invalid"}, {"version": 0}])
def test_group_status_version_constraints(storage, values):
    _, factory = storage
    with pytest.raises(IntegrityError):
        with session_scope(factory) as session:
            session.add(DuplicateGroup(**values))


def test_audit_and_review_transaction_rollback_is_atomic(storage):
    _, factory = storage
    with pytest.raises(RuntimeError, match="forced failure"):
        with session_scope(factory) as session:
            row = complaint()
            session.add(row)
            session.flush()
            session.add_all([
                AdminReview(complaint_id=row.id, actor="reviewer", action="complaint_update", reason="Review"),
                AuditLog(actor="reviewer", action="complaint_update", entity_type="complaint", entity_id=row.id,
                         changes={"before": 0, "after": 1}),
            ])
            session.flush()
            raise RuntimeError("forced failure")
    with factory() as session:
        for entity in (Complaint, AdminReview, AuditLog):
            assert session.scalars(select(entity)).all() == []


def test_explicit_compare_and_swap_can_reject_stale_versions(storage):
    _, factory = storage
    with session_scope(factory) as session:
        row = complaint()
        session.add(row)
        session.flush()
        identifier = row.id
    with session_scope(factory) as session:
        result = session.execute(update(Complaint).where(Complaint.id == identifier, Complaint.version == 1)
                                 .values(status="in_review", version=2))
        assert result.rowcount == 1
    with session_scope(factory) as session:
        stale = session.execute(update(Complaint).where(Complaint.id == identifier, Complaint.version == 1)
                                .values(status="rejected", version=2))
        assert stale.rowcount == 0
    with factory() as session:
        assert session.get(Complaint, identifier).status == "in_review"


def test_file_database_parent_wal_and_timestamp_offset_roundtrip(tmp_path):
    path = tmp_path / "nested" / "backend.sqlite3"
    engine = make_engine("sqlite:///" + path.as_posix())
    Base.metadata.create_all(engine)
    factory = make_session_factory(engine)
    supplied = datetime(2026, 10, 30, 12, 0, tzinfo=timezone(timedelta(hours=6)))
    with session_scope(factory) as session:
        row = complaint(timestamp=supplied)
        session.add(row)
        session.flush()
        identifier = row.id
    with engine.connect() as connection:
        assert connection.execute(text("PRAGMA journal_mode")).scalar_one().lower() == "wal"
    with factory() as session:
        assert session.get(Complaint, identifier).timestamp == datetime(2026, 10, 30, 6, 0, tzinfo=timezone.utc)
    engine.dispose()
    assert path.is_file()


def test_engine_factory_does_not_create_tables_implicitly():
    engine = make_engine("sqlite:///:memory:")
    assert inspect(engine).get_table_names() == []
    engine.dispose()


def test_alembic_upgrade_matches_orm_and_can_downgrade(tmp_path, monkeypatch):
    from alembic import command
    from alembic.autogenerate import compare_metadata
    from alembic.config import Config
    from alembic.migration import MigrationContext

    repo = Path(__file__).resolve().parents[1]
    url = "sqlite:///" + (tmp_path / "migration" / "backend.sqlite3").as_posix()
    monkeypatch.setenv("BACKEND_DATABASE_URL", url)
    config = Config(str(repo / "alembic.ini"))
    command.upgrade(config, "head")
    engine = make_engine(url)
    assert set(inspect(engine).get_table_names()) == {*Base.metadata.tables, "alembic_version"}
    with engine.connect() as connection:
        assert connection.execute(text("SELECT version_num FROM alembic_version")).scalar_one() == "0003_citizen_accounts"
        differences = compare_metadata(MigrationContext.configure(connection, opts={"compare_type": True}), Base.metadata)
        assert differences == []
    engine.dispose()
    command.downgrade(config, "base")
    engine = make_engine(url)
    assert inspect(engine).get_table_names() == ["alembic_version"]
    engine.dispose()


def test_contact_migration_preserves_phase6_rows_and_can_return_to_phase6(tmp_path, monkeypatch):
    from alembic import command
    from alembic.config import Config

    repo = Path(__file__).resolve().parents[1]
    url = "sqlite:///" + (tmp_path / "upgrade.sqlite3").as_posix()
    monkeypatch.setenv("BACKEND_DATABASE_URL", url)
    config = Config(str(repo / "alembic.ini"))
    command.upgrade(config, "0001_initial")
    engine = make_engine(url)
    identifier = str(uuid.uuid4())
    with engine.begin() as connection:
        connection.execute(text("INSERT INTO complaints (id,text,lang,created_at,updated_at) VALUES (:id,:text,'en',:created,:updated)"),
                           {"id": identifier, "text": "Existing Phase 6 report", "created": "2026-10-30 12:00:00", "updated": "2026-10-30 12:00:00"})
    command.upgrade(config, "head")
    with make_session_factory(engine)() as session:
        record = session.get(Complaint, identifier)
        assert record.text == "Existing Phase 6 report" and record.contact is None
    with make_session_factory(engine).begin() as session:
        session.add(AdminReview(complaint_id=identifier, actor="migration-test", action="complaint_update",
                                reason="Existing linked feedback must survive downgrade", previous_values={}, new_values={}))
    command.downgrade(config, "0001_initial")
    with engine.connect() as connection:
        assert "contact" not in {column["name"] for column in inspect(connection).get_columns("complaints")}
        assert connection.execute(text("SELECT text FROM complaints WHERE id=:id"), {"id": identifier}).scalar_one() == "Existing Phase 6 report"
        assert connection.execute(text("SELECT count(*) FROM admin_reviews WHERE complaint_id=:id"), {"id": identifier}).scalar_one() == 1
    engine.dispose()
