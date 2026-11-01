"""Stream human admin-review events for a separately reviewed retraining dataset.

This operator-only database export does not mutate reviews, audit logs, source
severity, or automatic-analysis snapshots. Group reviews remain group events;
they are never expanded into invented pair labels or expert priority grades.
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv
from sqlalchemy import select, text
from sqlalchemy.engine import make_url
from sqlalchemy.exc import SQLAlchemyError

from backend.db import make_engine, make_session_factory
from backend.models import AdminReview, Complaint

FIELDS = (
    "feedback_id", "action", "complaint_id", "group_id", "actor", "reason", "created_at",
    "previous_values", "new_values", "original_analysis", "lang", "area_id", "source_report_id",
)
JSON_FIELDS = {"previous_values", "new_values", "original_analysis"}


def _utc(value: datetime) -> str:
    if value.tzinfo is None:
        value = value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def _since(value: str | datetime | None) -> datetime | None:
    if value is None:
        return None
    if isinstance(value, str):
        try:
            value = datetime.fromisoformat(value.strip().replace("Z", "+00:00"))
        except ValueError as exc:
            raise ValueError("--since must be an ISO-8601 timestamp with an explicit timezone") from exc
    if not isinstance(value, datetime) or value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("--since must be an ISO-8601 timestamp with an explicit timezone")
    return value.astimezone(timezone.utc)


def export_feedback(
    database_url: str, output: str | Path,
    output_format: str = "jsonl", since: str | datetime | None = None,
    overwrite: bool = False, include_text: bool = False,
) -> dict:
    """Export feedback in bounded batches using a read-only transaction.

    Load only through an operator account authorized to read review data. No
    database URL/credential is returned or written to the manifest. Complaint
    body text is omitted unless ``include_text`` is explicitly enabled.
    """
    if output_format not in ("jsonl", "csv"):
        raise ValueError("Feedback format must be jsonl or csv")
    if not isinstance(include_text, bool) or not isinstance(overwrite, bool):
        raise ValueError("include_text and overwrite must be boolean")
    since_value = _since(since)
    destination = Path(output).expanduser().resolve()
    manifest_path = destination.with_name(destination.name + ".manifest.json")
    url = make_url(database_url)
    if url.get_backend_name() == "sqlite":
        if url.database in (None, "", ":memory:"):
            raise ValueError("Feedback export needs an existing persistent SQLite database")
        if not url.database.startswith("file:"):
            source_path = Path(url.database).expanduser().resolve()
            if not source_path.is_file():
                raise ValueError("SQLite source database does not exist; run the API migrations first")
            protected = {Path(str(source_path) + suffix) for suffix in ("", "-wal", "-shm", "-journal")}
            if destination in protected or manifest_path in protected:
                raise ValueError("Feedback output must not overwrite the database or its journal files")
    if not overwrite and (destination.exists() or manifest_path.exists()):
        raise ValueError("Feedback output exists; use a new output or --overwrite")
    if destination.exists() and not destination.is_file():
        raise ValueError("Feedback output must be a file")
    destination.parent.mkdir(parents=True, exist_ok=True)
    engine = make_engine(database_url)
    factory = make_session_factory(engine)
    count, digest = 0, hashlib.sha256()
    columns = [*FIELDS, *(["text"] if include_text else [])]
    try:
        with tempfile.TemporaryDirectory(prefix=".feedback-export-", dir=destination.parent) as temporary:
            staged = Path(temporary) / "feedback"
            with staged.open("w", encoding="utf-8", newline="") as stream:
                class Writer:
                    def write(self, content):
                        digest.update(content.encode("utf-8"))
                        return stream.write(content)
                hashed = Writer()
                csv_writer = csv.DictWriter(hashed, fieldnames=columns, lineterminator="\n") if output_format == "csv" else None
                if csv_writer:
                    csv_writer.writeheader()
                with factory() as session:
                    if engine.dialect.name == "postgresql":
                        session.execute(text("SET TRANSACTION READ ONLY"))
                    elif engine.dialect.name == "sqlite":
                        session.execute(text("PRAGMA query_only = ON"))
                    statement = (
                        select(AdminReview, Complaint)
                        .outerjoin(Complaint, AdminReview.complaint_id == Complaint.id)
                        .where(AdminReview.feedback_exportable.is_(True))
                        .order_by(AdminReview.created_at, AdminReview.id)
                        .execution_options(yield_per=1000)
                    )
                    if since_value is not None:
                        statement = statement.where(AdminReview.created_at >= since_value)
                    for review, complaint in session.execute(statement):
                        row = {
                            "feedback_id": review.id, "action": review.action,
                            "complaint_id": review.complaint_id, "group_id": review.group_id,
                            "actor": review.actor, "reason": review.reason, "created_at": _utc(review.created_at),
                            "previous_values": review.previous_values, "new_values": review.new_values,
                            "original_analysis": complaint.analysis if complaint is not None else None,
                            "lang": complaint.lang if complaint is not None else None,
                            "area_id": complaint.area_id if complaint is not None else None,
                            "source_report_id": complaint.source_report_id if complaint is not None else None,
                        }
                        if include_text:
                            row["text"] = complaint.text if complaint is not None else None
                        if csv_writer:
                            csv_writer.writerow({key: json.dumps(value, ensure_ascii=False, allow_nan=False) if key in JSON_FIELDS else value
                                                 for key, value in row.items()})
                        else:
                            hashed.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
                        count += 1
            manifest = {
                "kind": "admin_feedback_event_export", "schema_version": 1,
                "database_backend": engine.dialect.name, "format": output_format,
                "rows": count, "sha256": digest.hexdigest(), "output": str(destination),
                "since_utc_inclusive": _utc(since_value) if since_value is not None else None,
                "include_text": include_text, "created_at_utc": _utc(datetime.now(timezone.utc)),
                "interpretation": "One immutable review event per row. Human overrides and group decisions require adjudication before retraining; no priority grades or duplicate pair labels are inferred.",
            }
            staged_manifest = Path(temporary) / "manifest.json"
            staged_manifest.write_text(json.dumps(manifest, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
            os.replace(staged, destination)
            os.replace(staged_manifest, manifest_path)
        return manifest
    finally:
        engine.dispose()


def main(argv: list[str] | None = None) -> int:
    load_dotenv(ROOT / ".env", override=False)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--database-url", default=os.getenv("BACKEND_DATABASE_URL", "sqlite:///./backend/data/backend.sqlite3"))
    parser.add_argument("--output", default="backend/exports/admin_feedback.jsonl")
    parser.add_argument("--format", choices=("jsonl", "csv"), default="jsonl")
    parser.add_argument("--since", help="Inclusive ISO-8601 review timestamp with timezone")
    parser.add_argument("--include-text", action="store_true", help="Explicitly include submitted complaint bodies")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)
    output = Path(args.output).expanduser()
    if not output.is_absolute():
        output = ROOT / output
    try:
        manifest = export_feedback(args.database_url, output, args.format, args.since, args.overwrite, args.include_text)
        print(json.dumps(manifest, ensure_ascii=True, allow_nan=False))
        return 0
    except SQLAlchemyError as exc:
        print(f"ERROR: Database feedback export failed ({type(exc).__name__}); check the connection and migrations.", file=sys.stderr)
        return 2
    except (ValueError, OSError, RuntimeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
