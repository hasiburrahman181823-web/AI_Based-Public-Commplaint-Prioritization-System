"""Reviewed, transactional complaint erasure and closed-case retention selection.

The caller owns authentication, the coordinated write transaction, the ML lock
and post-commit index refresh. This module never commits the domain transaction.
Offline exports, provider logs/backups, external copies and trained models need
the operator's separate deletion ledger/runbook; they are not silently erased.
"""

from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
import sqlite3
from uuid import UUID

from sqlalchemy import delete, select, text

from backend.models import AdminReview, AuditLog, Complaint, DuplicateGroup, new_id, utcnow

ROOT = Path(__file__).resolve().parents[1]
WRITE_LOCK_ID = 0x42414E474C415053
CLOSED_STATUSES = ("resolved", "rejected")
REDACTED_REASON = "Narrative removed during authorized complaint-data deletion."


def _aware_utc(value: datetime | None, name: str) -> datetime:
    if value is None:
        return utcnow()
    if not isinstance(value, datetime) or value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(f"{name} requires an explicit timezone")
    return value.astimezone(timezone.utc)


def _ids(values) -> list[str]:
    if isinstance(values, (str, bytes)):
        raise ValueError("Complaint IDs must be a list of unique UUIDs")
    try:
        identifiers = [str(UUID(value)) for value in values]
    except (TypeError, ValueError, AttributeError) as exc:
        raise ValueError("Complaint IDs must be a list of unique UUIDs") from exc
    if not 1 <= len(identifiers) <= 1000 or len(set(identifiers)) != len(identifiers):
        raise ValueError("Supply 1–1000 unique complaint UUIDs")
    return identifiers


def _references(value, identifiers: set[str]) -> bool:
    if value is None:
        return False
    rendered = (value if isinstance(value, str) else json.dumps(value, ensure_ascii=False, default=str)).casefold()
    return any(identifier in rendered for identifier in identifiers)


@contextmanager
def coordinated_session(factory):
    """Use the same cross-worker write lock as the complaint API coordinator."""
    with factory() as session, session.begin():
        dialect = session.get_bind().dialect.name
        if dialect == "sqlite":
            session.execute(text("BEGIN IMMEDIATE"))
        elif dialect == "postgresql":
            session.execute(text("SELECT pg_advisory_xact_lock(:key)"), {"key": WRITE_LOCK_ID})
        else:
            raise ValueError("Retention supports SQLite development or PostgreSQL")
        yield session


def select_retention_candidates(session, *, days: int = 90, older_than: datetime | None = None,
                                now: datetime | None = None, limit: int = 1000) -> list[str]:
    """Select closed cases by last update, never open cases or future cutoffs.

    The schema has no historical closed_at column. updated_at is a conservative
    last-change proxy: any subsequent review postpones automatic eligibility.
    Explicit, verified privacy requests may use delete_complaints for open cases.
    """
    if isinstance(days, bool) or not isinstance(days, int) or not 1 <= days <= 36500:
        raise ValueError("Retention days must be an integer in [1, 36500]")
    if isinstance(limit, bool) or not isinstance(limit, int) or not 1 <= limit <= 1000:
        raise ValueError("Retention limit must be an integer in [1, 1000]")
    current = _aware_utc(now, "Retention clock")
    cutoff = _aware_utc(older_than, "Retention cutoff") if older_than is not None else current - timedelta(days=days)
    if cutoff > current:
        raise ValueError("Retention cutoff must not be in the future")
    return list(session.scalars(select(Complaint.id)
                               .where(Complaint.status.in_(CLOSED_STATUSES), Complaint.updated_at < cutoff)
                               .order_by(Complaint.updated_at, Complaint.id).limit(limit)))


def delete_complaints(session, complaint_ids, actor: str, reason: str, request_id: str | None = None) -> dict:
    """Erase selected rows and known linked content inside an existing transaction.

    Free-text request reasons are deliberately not copied into the deletion
    audit, where they could reintroduce the information being erased. The
    operator retains a minimal, separately protected request rationale/ledger.
    Missing IDs are reported idempotently; no content is returned.
    """
    identifiers = _ids(complaint_ids)
    if not isinstance(actor, str) or not actor.strip() or len(actor) > 128:
        raise ValueError("Deletion requires an authorized actor of at most 128 characters")
    if not isinstance(reason, str) or not 3 <= len(reason.strip()) <= 2000:
        raise ValueError("Deletion requires a reviewed reason of 3–2000 characters")
    if request_id is not None and (not isinstance(request_id, str) or len(request_id) > 64):
        raise ValueError("Deletion request ID must contain at most 64 characters")
    if not session.in_transaction():
        raise ValueError("Deletion must run inside a coordinated transaction")
    rows = session.scalars(select(Complaint).where(Complaint.id.in_(identifiers))).all()
    deleted_ids = {row.id for row in rows}
    groups = {row.group_id for row in rows if row.group_id}
    result = {"deleted_ids": sorted(deleted_ids), "missing_ids": sorted(set(identifiers) - deleted_ids),
              "deleted_count": len(deleted_ids), "deleted_reviews": 0, "redacted_reviews": 0,
              "redacted_audit_logs": 0, "redacted_survivor_analyses": 0,
              "affected_groups": len(groups), "lineage_links_removed": 0,
              "warnings": [], "audit_event_id": None}
    if not rows:
        return result

    # Direct review rows have restrictive complaint FKs and may contain copied
    # narratives. Delete those rows before deleting their source complaints.
    direct_reviews = session.scalars(select(AdminReview).where(AdminReview.complaint_id.in_(deleted_ids))).all()
    for review in direct_reviews:
        session.delete(review)
    result["deleted_reviews"] = len(direct_reviews)
    session.flush()

    # Historical group moves can mention the target in another group's JSON.
    # Scan bounded ORM batches rather than relying on dialect-specific JSON SQL.
    reviews = session.scalars(select(AdminReview).execution_options(yield_per=500))
    for review in reviews:
        if review.group_id in groups or _references([review.reason, review.previous_values, review.new_values], deleted_ids):
            review.reason = REDACTED_REASON
            review.previous_values = {"privacy_redacted": True}
            review.new_values = {"privacy_redacted": True}
            review.feedback_exportable = False
            result["redacted_reviews"] += 1

    audits = session.scalars(select(AuditLog).execution_options(yield_per=500))
    for audit in audits:
        direct = isinstance(audit.entity_id, str) and audit.entity_id.casefold() in deleted_ids
        linked_group = audit.entity_type == "duplicate_group" and audit.entity_id in groups
        if direct or linked_group or _references(audit.changes, deleted_ids):
            audit.changes = {"privacy_redacted": True}
            audit.request_id = None
            if direct:
                audit.entity_id = None
            result["redacted_audit_logs"] += 1

    # Retained reports must not expose cached candidate IDs or stale group-size
    # evidence through their stored automatic analysis. Keep their own source
    # complaint/contact, but remove affected snapshots and require review.
    survivors = session.scalars(select(Complaint).where(Complaint.id.not_in(deleted_ids)).execution_options(yield_per=500))
    for survivor in survivors:
        lineage_reference = isinstance(survivor.source_report_id, str) and survivor.source_report_id.casefold() in deleted_ids
        if survivor.group_id in groups or lineage_reference or _references(survivor.analysis, deleted_ids):
            survivor.analysis = {"privacy_redacted": True, "review_required": True,
                                 "review_status": "needs_human_review",
                                 "review_reasons": ["related_complaint_data_deleted"], "duplicate_candidates": []}
            survivor.review_required = True
            survivor.version += 1
            survivor.updated_at = utcnow()
            if lineage_reference:
                survivor.source_report_id = None
                result["lineage_links_removed"] += 1
            result["redacted_survivor_analyses"] += 1

    for group_id in groups:
        group = session.get(DuplicateGroup, group_id)
        if group is not None:
            group.version += 1
            group.updated_at, group.confirmed_by = utcnow(), None
            if group.status != "reviewed_split":
                group.status = "proposed"

    session.flush()
    if result["lineage_links_removed"]:
        result["warnings"].append("Other records had source-lineage references to deleted IDs. References/analyses were cleared; separately verify derived content and authorized deletion scope.")
    session.execute(delete(Complaint).where(Complaint.id.in_(deleted_ids)).execution_options(synchronize_session="fetch"))
    event_id = new_id()
    session.add(AuditLog(id=event_id, actor=actor.strip(), action="complaints_delete", entity_type="data_deletion",
                         entity_id=None, request_id=request_id,
                         changes={"reason_code": "authorized_privacy_or_retention_deletion",
                                  **{key: value for key, value in result.items() if key.endswith("count") or key in (
                                      "deleted_reviews", "redacted_reviews", "redacted_audit_logs", "redacted_survivor_analyses", "affected_groups")}}))
    session.flush()
    result["audit_event_id"] = event_id
    return result


def purge_embedding_cache(cache_dir: str | Path) -> dict:
    """Clear only the configured fixed-name cache, never recursively delete paths.

    A recognized cache schema is mandatory. WAL checkpoint/VACUUM require a
    maintenance window without other cache writers. This is logical cache
    erasure and compaction, not a guarantee about SSD remnants or backups.
    """
    configured = Path(cache_dir).expanduser()
    configured = configured if configured.is_absolute() else ROOT / configured
    path_chain = [configured.absolute(), *configured.absolute().parents]
    if any(path.is_symlink() or bool(getattr(path, "is_junction", lambda: False)()) for path in path_chain):
        raise ValueError("Embedding cache root must not be a link or junction")
    root = configured.resolve()
    path = root / "embeddings.sqlite3"
    if path.is_symlink() or path.resolve().parent != root:
        raise ValueError("Embedding cache target must remain inside its configured root")
    if any(Path(str(path) + suffix).is_symlink() for suffix in ("-wal", "-shm", "-journal")):
        raise ValueError("Embedding cache journal files must not be links")
    if not path.exists():
        return {"status": "absent", "deleted_embeddings": 0}
    if not path.is_file():
        raise ValueError("Embedding cache target must be a regular database file")
    try:
        connection = sqlite3.connect(path.as_uri() + "?mode=rw", uri=True, timeout=5)
        try:
            tables = {row[0] for row in connection.execute("SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'")}
            executable = connection.execute("SELECT count(*) FROM sqlite_master WHERE type IN ('trigger','view')").fetchone()[0]
            embedding_columns = {row[1] for row in connection.execute("PRAGMA table_info(embeddings)")}
            namespace_columns = {row[1] for row in connection.execute("PRAGMA table_info(namespaces)")}
            if (tables != {"embeddings", "namespaces"} or executable or
                    embedding_columns != {"namespace", "text_key", "dimension", "vector"} or
                    namespace_columns != {"namespace", "dimension"}):
                raise ValueError("Configured file is not the recognized embedding-cache schema; it was not cleared")
            connection.execute("PRAGMA secure_delete=ON")
            count = connection.execute("SELECT count(*) FROM embeddings").fetchone()[0]
            with connection:
                connection.execute("DELETE FROM embeddings")
                connection.execute("DELETE FROM namespaces")
            checkpoint = connection.execute("PRAGMA wal_checkpoint(TRUNCATE)").fetchone()
            if checkpoint and checkpoint[0]:
                raise ValueError("Embedding cache is busy; stop other workers and retry deletion")
            connection.execute("VACUUM")
            connection.execute("PRAGMA wal_checkpoint(TRUNCATE)")
            return {"status": "purged", "deleted_embeddings": count}
        finally:
            connection.close()
    except sqlite3.Error as exc:
        raise ValueError("Embedding cache could not be cleared; stop other workers and retry") from exc
