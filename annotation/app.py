"""Run a local annotation workspace: python -m annotation.app --help.

Annotator names are self-asserted local pseudonyms, not authenticated accounts.
The CLI binds to 127.0.0.1. Use one server for all browser tabs sharing a database.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import csv
from datetime import datetime, timezone
import hashlib
import io
import json
from pathlib import Path
import re
import sqlite3
from typing import Literal

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse, Response
from pydantic import BaseModel, Field, StrictInt

from annotation.agreement import cohen_kappa

ROOT = Path(__file__).resolve().parents[1]
NAME_RE = re.compile(r"[A-Za-z0-9_.-]{1,64}\Z")
PAIR_COLUMNS = (
    "text_a", "text_b", "loc_a", "loc_b", "time_a", "time_b", "label", "pair_type",
    "pair_id", "split", "id_a", "id_b", "original_id_a", "original_id_b",
)


def resolve_path(path: str | Path) -> Path:
    path = Path(path).expanduser()
    return path if path.is_absolute() else ROOT / path


def read_csv(path: Path, required: tuple[str, ...]) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle, strict=True)
        header = reader.fieldnames or []
        if len(set(header)) != len(header):
            raise ValueError(f"Duplicate headers in {path}")
        missing = set(required) - set(header)
        if missing:
            raise ValueError(f"{path}: missing columns {sorted(missing)}")
        rows = []
        for index, row in enumerate(reader, 2):
            if None in row or any(value is None for value in row.values()):
                raise ValueError(f"{path}: wrong number of fields at CSV record {index}")
            rows.append({key: value.strip() for key, value in row.items()})
    return rows


def build_items(pairs_path: Path | None, complaints_paths: list[Path]) -> dict[str, dict]:
    items: dict[str, dict] = {}
    group_splits: dict[str, str] = {}

    def check_split(original_id: str, split: str) -> None:
        if original_id:
            previous = group_splits.setdefault(original_id, split)
            if previous != split:
                raise ValueError(f"original_id {original_id!r} crosses splits {previous!r} and {split!r}")

    def insert(item: dict) -> None:
        key = item["item_id"]
        if key in items and items[key] != item:
            raise ValueError(f"Conflicting records share item ID {key!r}")
        items[key] = item

    if pairs_path:
        for row in read_csv(pairs_path, PAIR_COLUMNS):
            for key in ("pair_id", "text_a", "text_b", "id_a", "id_b", "original_id_a", "original_id_b", "split"):
                if not row[key]:
                    raise ValueError(f"{pairs_path}: pair field {key!r} must be nonempty")
            check_split(row["original_id_a"], row["split"])
            check_split(row["original_id_b"], row["split"])
            # Keep only source content; heuristic labels/types are deliberately
            # absent from annotation items so they cannot anchor human judgments.
            payload = {key: row[key] for key in PAIR_COLUMNS if key not in ("label", "pair_type", "pair_id")}
            insert({"item_id": "pair:" + row["pair_id"], "task_type": "duplicate", "payload": payload})
    for path in complaints_paths:
        for row in read_csv(path, ("text", "lang", "category", "location", "timestamp")):
            if not row["text"] or not row["category"] or row["lang"] not in ("bn", "en", "banglish"):
                raise ValueError(f"{path}: complaints need nonempty text/category and lang bn/en/banglish")
            split = row.get("split") or (path.stem if path.stem in ("train", "val", "test") else "natural_test")
            original_id = row.get("original_id", "")
            check_split(original_id, split)
            identifier = row.get("id") or "natural_" + hashlib.sha256(
                json.dumps([row[key] for key in ("text", "lang", "category", "location", "timestamp")], ensure_ascii=False).encode("utf-8")
            ).hexdigest()
            payload = {key: row[key] for key in ("text", "lang", "category", "location", "timestamp")}
            payload.update(id=identifier, original_id=original_id, split=split)
            insert({"item_id": "complaint:" + identifier, "task_type": "priority", "payload": payload})
    if not items:
        raise ValueError("No annotation items found. Supply a nonempty --pairs or --complaints CSV.")
    return items


class AnnotationInput(BaseModel):
    item_id: str = Field(min_length=1, max_length=512)
    annotator: str = Field(min_length=1, max_length=64)
    label: StrictInt | None = None
    status: Literal["labeled", "skipped"] = "labeled"
    reason: str = Field(min_length=1, max_length=4000)
    expected_revision: int = Field(default=0, ge=0)


def create_app(
    pairs_path: str | Path | None = None,
    complaints_paths: list[str | Path] | None = None,
    db_path: str | Path = "annotation/annotations.sqlite3",
) -> FastAPI:
    """Create an app with a validated, immutable snapshot of loaded items."""
    items = build_items(
        resolve_path(pairs_path) if pairs_path else None,
        [resolve_path(path) for path in (complaints_paths or [])],
    )
    database = resolve_path(db_path)
    database.parent.mkdir(parents=True, exist_ok=True)

    @contextmanager
    def connect():
        connection = sqlite3.connect(database, timeout=10)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        try:
            with connection:
                yield connection
        finally:
            connection.close()

    with connect() as connection:
        connection.execute("PRAGMA journal_mode = WAL")
        connection.executescript("""
            CREATE TABLE IF NOT EXISTS items (
                item_id TEXT PRIMARY KEY,
                task_type TEXT NOT NULL,
                payload TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS annotations (
                item_id TEXT NOT NULL REFERENCES items(item_id),
                annotator TEXT NOT NULL,
                label INTEGER,
                status TEXT NOT NULL,
                reason TEXT NOT NULL,
                revision INTEGER NOT NULL,
                updated_at TEXT NOT NULL,
                PRIMARY KEY (item_id, annotator)
            );
        """)
        for item_id, item in items.items():
            payload = json.dumps(item["payload"], ensure_ascii=False, sort_keys=True)
            existing = connection.execute("SELECT * FROM items WHERE item_id = ?", (item_id,)).fetchone()
            if existing and (existing["payload"] != payload or existing["task_type"] != item["task_type"]):
                raise ValueError(
                    f"Item {item_id!r} changed since this database was created. "
                    "Use a new database for a changed dataset to preserve annotation provenance."
                )
            connection.execute("INSERT OR IGNORE INTO items VALUES (?, ?, ?)", (item_id, item["task_type"], payload))
    app = FastAPI(title="Incident Review", version="0.2.0")
    app.state.items = items
    app.state.database = database

    def validate_name(name: str) -> None:
        if not NAME_RE.fullmatch(name):
            raise HTTPException(422, "Annotator ID must use 1–64 letters, digits, '.', '_' or '-'.")

    def active_annotations(annotator: str | None = None) -> list[dict]:
        with connect() as connection:
            if annotator is None:
                rows = connection.execute("SELECT * FROM annotations ORDER BY item_id, annotator").fetchall()
            else:
                rows = connection.execute("SELECT * FROM annotations WHERE annotator=? ORDER BY item_id", (annotator,)).fetchall()
        return [dict(row) for row in rows if row["item_id"] in items]

    @app.get("/")
    def index():
        return FileResponse(Path(__file__).with_name("index.html"))

    @app.get("/api/health")
    def health():
        return {"status": "ok", "items": len(items)}

    @app.get("/api/items")
    def queue(
        annotator: str,
        task: Literal["duplicate", "priority"] = "duplicate",
        include_labeled: bool = False,
        offset: int = Query(default=0, ge=0),
        limit: int = Query(default=50, ge=1, le=200),
    ):
        validate_name(annotator)
        own = {row["item_id"]: row for row in active_annotations(annotator)}
        task_items = [item for _, item in sorted(items.items()) if item["task_type"] == task]
        available = [item for item in task_items if include_labeled or item["item_id"] not in own]
        return {
            "total": len(task_items),
            "labeled": sum(own.get(item["item_id"], {}).get("status") == "labeled" for item in task_items),
            "skipped": sum(own.get(item["item_id"], {}).get("status") == "skipped" for item in task_items),
            "available": len(available),
            "items": [dict(item, annotation=own.get(item["item_id"])) for item in available[offset:offset + limit]],
        }

    @app.post("/api/annotations")
    def save_annotation(body: AnnotationInput):
        validate_name(body.annotator)
        if body.item_id not in items:
            raise HTTPException(404, "Unknown item ID")
        reason = body.reason.strip()
        if not reason:
            raise HTTPException(422, "Add a reason for your judgment or skip.")
        task = items[body.item_id]["task_type"]
        valid = (0, 1) if task == "duplicate" else (1, 2, 3, 4, 5)
        if body.status == "labeled" and body.label not in valid:
            raise HTTPException(422, f"Label must be one of {valid} for {task}.")
        if body.status == "skipped" and body.label is not None:
            raise HTTPException(422, "A skipped judgment must not contain a label.")
        with connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            existing = connection.execute(
                "SELECT revision FROM annotations WHERE item_id=? AND annotator=?",
                (body.item_id, body.annotator),
            ).fetchone()
            revision = int(existing["revision"]) if existing else 0
            if revision != body.expected_revision:
                raise HTTPException(409, "This judgment was edited in another tab. Reload before saving.")
            updated = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
            connection.execute("""
                INSERT INTO annotations VALUES (?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(item_id, annotator) DO UPDATE SET
                    label=excluded.label, status=excluded.status, reason=excluded.reason,
                    revision=excluded.revision, updated_at=excluded.updated_at
            """, (body.item_id, body.annotator, body.label, body.status, reason, revision + 1, updated))
        return {"saved": True, "revision": revision + 1, "updated_at": updated}

    @app.get("/api/agreement")
    def agreement(annotator_a: str, annotator_b: str):
        validate_name(annotator_a)
        validate_name(annotator_b)
        if annotator_a == annotator_b:
            raise HTTPException(422, "Choose two different annotators.")
        left = {row["item_id"]: row for row in active_annotations(annotator_a) if row["status"] == "labeled"}
        right = {row["item_id"]: row for row in active_annotations(annotator_b) if row["status"] == "labeled"}
        reports = {}
        for task, labels in (("duplicate", (0, 1)), ("priority", (1, 2, 3, 4, 5))):
            overlap = sorted(key for key in left.keys() & right.keys() if items[key]["task_type"] == task)
            a = [left[key]["label"] for key in overlap]
            b = [right[key]["label"] for key in overlap]
            reports[task] = cohen_kappa(a, b, labels, "unweighted" if task == "duplicate" else "quadratic")
            if task == "priority":
                reports["priority_unweighted"] = cohen_kappa(a, b, labels)
        return {"annotator_a": annotator_a, "annotator_b": annotator_b, **reports}

    @app.get("/api/export/annotations.csv")
    def export_annotations():
        fields = (
            "item_id", "task_type", "annotator", "label", "status", "reason", "revision", "updated_at",
            "split", "id_a", "id_b", "original_id_a", "original_id_b", "id", "original_id",
            "text_a", "text_b", "loc_a", "loc_b", "time_a", "time_b", "text", "lang", "category", "location", "timestamp",
        )
        stream = io.StringIO(newline="")
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in active_annotations():
            item = items[row["item_id"]]
            writer.writerow({**item["payload"], **row, "task_type": item["task_type"]})
        return Response(
            stream.getvalue(), media_type="text/csv; charset=utf-8",
            headers={"Content-Disposition": 'attachment; filename="annotations.csv"'},
        )

    return app


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pairs", type=Path, help="Generated pairs.csv; omit for priority-only annotation")
    parser.add_argument("--complaints", action="append", type=Path, default=[], help="Complaint CSV; repeat for multiple splits or natural collection")
    parser.add_argument("--db", type=Path, default=Path("annotation/annotations.sqlite3"))
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args(argv)
    if not args.pairs and not args.complaints:
        parser.error("provide --pairs and/or --complaints")
    if not 1 <= args.port <= 65535:
        parser.error("--port must be between 1 and 65535")
    try:
        app = create_app(args.pairs, args.complaints, args.db)
    except (ValueError, OSError, csv.Error, sqlite3.Error) as exc:
        parser.exit(2, f"ERROR: {exc}\n")
    import uvicorn
    print(f"Local annotation workspace: http://127.0.0.1:{args.port}", flush=True)
    uvicorn.run(app, host="127.0.0.1", port=args.port)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
