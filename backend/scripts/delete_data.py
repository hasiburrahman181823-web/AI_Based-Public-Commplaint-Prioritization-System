"""Dry-run or apply reviewed complaint erasure during an operator maintenance window."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv
from sqlalchemy import select, text
from sqlalchemy.engine import make_url
from sqlalchemy.exc import SQLAlchemyError

from backend.db import make_engine, make_session_factory
from backend.models import Complaint
from backend.retention import _ids, coordinated_session, delete_complaints, purge_embedding_cache, select_retention_candidates


def _cutoff(value: str) -> datetime:
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise argparse.ArgumentTypeError("Use an ISO-8601 cutoff with an explicit timezone") from exc
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise argparse.ArgumentTypeError("Use an ISO-8601 cutoff with an explicit timezone")
    return parsed.astimezone(timezone.utc)


def run_deletion(database_url: str, *, complaint_ids=None, retention_days: int = 90,
                 older_than: datetime | None = None, apply: bool = False,
                 actor: str | None = None, reason: str | None = None,
                 cache_dir: str | Path = ".cache/backend", limit: int = 1000) -> dict:
    """Operator-only path; apply requires all API workers stopped or coordinated.

    Credentials are supplied by the environment, never emitted in the receipt.
    Database writes and cache erasure precede the domain commit. A cache failure
    rolls back domain erasure; a successfully emptied cache can be rebuilt.
    """
    if not isinstance(apply, bool):
        raise ValueError("Apply must be boolean")
    if apply and (not actor or not reason):
        raise ValueError("Applying deletion requires --actor and --reason")
    requested = _ids(complaint_ids) if complaint_ids else None
    if requested and older_than is not None:
        raise ValueError("Choose explicit complaint IDs or a retention cutoff, not both")
    url = make_url(database_url)
    if url.get_backend_name() == "sqlite":
        if url.database in (None, "", ":memory:") or str(url.database).startswith("file:"):
            raise ValueError("Deletion requires an existing persistent SQLite database")
        if not Path(url.database).expanduser().resolve().is_file():
            raise ValueError("SQLite source database does not exist; apply migrations first")
    elif url.get_backend_name() != "postgresql":
        raise ValueError("Deletion supports SQLite development or PostgreSQL")
    engine = make_engine(database_url)
    factory = make_session_factory(engine)
    try:
        if not apply:
            with factory() as session:
                session.execute(text("SET TRANSACTION READ ONLY") if engine.dialect.name == "postgresql" else text("PRAGMA query_only=ON"))
                selected = requested or select_retention_candidates(session, days=retention_days, older_than=older_than, limit=limit)
                existing = list(session.scalars(select(Complaint.id).where(Complaint.id.in_(selected)).order_by(Complaint.id))) if selected else []
            return {"dry_run": True, "candidate_ids": existing, "candidate_count": len(existing),
                    "missing_ids": sorted(set(requested or []) - set(existing)),
                    "selection": "explicit_verified_request" if requested else "closed_cases_by_last_update",
                    "retention_days": retention_days if requested is None and older_than is None else None,
                    "cache_purged": False, "writes_applied": False}
        with coordinated_session(factory) as session:
            selected = requested or select_retention_candidates(session, days=retention_days, older_than=older_than, limit=limit)
            if not selected:
                return {"dry_run": False, "deleted_count": 0, "deleted_ids": [], "missing_ids": [],
                        "cache_purged": False, "writes_applied": False}
            result = delete_complaints(session, selected, actor=actor, reason=reason)
            cache = purge_embedding_cache(cache_dir) if result["deleted_count"] else {"status": "not_needed", "deleted_embeddings": 0}
        return {"dry_run": False, **result, "cache": cache, "cache_purged": cache["status"] == "purged",
                "writes_applied": result["deleted_count"] > 0,
                "remaining_operator_actions": ["Restart all API workers to discard old in-memory indexes.",
                                               "Review and remove authorized private exports/derived copies; expire backups and replay the deletion ledger after restore.",
                                               "Assess any governed research datasets or model versions separately; database erasure is not machine unlearning."]}
    finally:
        engine.dispose()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument("--complaint-id", action="append", help="Repeat for verified complaint UUIDs")
    selection.add_argument("--older-than", type=_cutoff, help="Closed-case last-update cutoff, ISO-8601 with timezone")
    parser.add_argument("--retention-days", type=int, default=90)
    parser.add_argument("--limit", type=int, default=1000, help="Maximum closed cases per retention batch, 1–1000")
    parser.add_argument("--apply", action="store_true", help="Perform irreversible erasure; default is dry-run")
    parser.add_argument("--actor", help="Authorized operator identifier; required with --apply")
    parser.add_argument("--reason", help="Reviewed rationale; required with --apply, not copied into DB audit")
    args = parser.parse_args(argv)
    load_dotenv(ROOT / ".env", override=False)
    database_url = os.getenv("BACKEND_DATABASE_URL", "sqlite:///./backend/data/backend.sqlite3")
    if database_url.startswith(("sqlite:///./", "sqlite+pysqlite:///./")):
        prefix = "sqlite+pysqlite:///" if database_url.startswith("sqlite+pysqlite") else "sqlite:///"
        database_url = prefix + (ROOT / database_url.split("///./", 1)[1]).as_posix()
    try:
        result = run_deletion(database_url, complaint_ids=args.complaint_id, retention_days=args.retention_days,
                              older_than=args.older_than, apply=args.apply, actor=args.actor, reason=args.reason,
                              cache_dir=os.getenv("BACKEND_EMBEDDING_CACHE_DIR", ".cache/backend"), limit=args.limit)
    except ValueError as exc:
        print("Deletion not completed: " + str(exc), file=sys.stderr)
        return 2
    except (OSError, SQLAlchemyError):
        print("Deletion not completed: database/cache unavailable or migration missing; no credentials are shown.", file=sys.stderr)
        return 2
    print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
