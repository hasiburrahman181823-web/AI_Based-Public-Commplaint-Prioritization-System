"""Cluster saved duplicate scores without joining different dataset splits.

The threshold must be the saved validation-selected value or an explicit value
chosen independently of test labels. Connected components create candidate
groups, including possible transitive chain errors; no cluster gold is inferred.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ml.duplicate_index import connected_components


def _path(value: str | Path) -> Path:
    value = Path(value).expanduser()
    return value.resolve() if value.is_absolute() else (ROOT / value).resolve()


def _rows(path: Path, required: set[str]) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream, strict=True)
        header = reader.fieldnames or []
        if len(set(header)) != len(header) or required - set(header):
            raise ValueError(f"{path}: required distinct columns {sorted(required)}")
        rows = []
        for number, row in enumerate(reader, 2):
            if None in row or any(value is None for value in row.values()):
                raise ValueError(f"{path}: wrong field count in CSV record {number}")
            rows.append({key: value.strip() for key, value in row.items()})
    return rows


def cluster_predictions(predictions: str | Path, threshold: float | None = None,
                        all_complaints: list[str | Path] | None = None,
                        model: str | None = None) -> tuple[list[dict], dict]:
    rows = _rows(Path(predictions), {"id_a", "id_b", "score"})
    if model:
        if rows and "model" not in rows[0]:
            raise ValueError("--model requires a model column in predictions")
        rows = [row for row in rows if row.get("model") == model]
    model_names = {row.get("model", "") for row in rows} - {""}
    if len(model_names) > 1:
        raise ValueError("Predictions contain multiple models; select one with --model")
    if threshold is None:
        if not rows or any(not row.get("threshold", "") for row in rows):
            raise ValueError("Supply --threshold or predictions with one saved threshold value")
        choices = {float(row["threshold"]) for row in rows}
        if len(choices) != 1:
            raise ValueError("Supply --threshold or predictions with one saved threshold value")
        threshold = next(iter(choices))
    if not math.isfinite(threshold):
        raise ValueError("Clustering threshold must be finite")
    partitions: dict[str, list[tuple[str, str, float]]] = {}
    vertices: dict[str, set[str]] = {}
    id_split: dict[str, str] = {}
    original_split: dict[str, str] = {}

    def assign(identifier: str, split: str) -> None:
        if not identifier:
            raise ValueError("Complaint IDs must be nonempty")
        previous = id_split.setdefault(identifier, split)
        if previous != split:
            raise ValueError(f"Complaint id {identifier!r} crosses {previous} and {split}; clustering will not cross splits")
        vertices.setdefault(split, set()).add(identifier)

    for row in rows:
        split = row.get("split") or "unspecified"
        assign(row["id_a"], split)
        assign(row["id_b"], split)
        for field in ("original_id_a", "original_id_b"):
            if row.get(field):
                previous = original_split.setdefault(row[field], split)
                if previous != split:
                    raise ValueError(f"original_id {row[field]!r} crosses dataset splits")
        score = float(row["score"])
        if not math.isfinite(score):
            raise ValueError("Prediction scores must be finite")
        partitions.setdefault(split, []).append((row["id_a"], row["id_b"], score))
    for path in all_complaints or []:
        path = Path(path)
        for row in _rows(path, {"id"}):
            split = row.get("split") or (path.stem if path.stem in ("train", "val", "test") else None)
            if not split:
                existing_splits = set(vertices) | set(partitions)
                if len(existing_splits) > 1:
                    raise ValueError("Singleton complaint files need a split column when predictions contain multiple splits")
                split = next(iter(existing_splits), "unspecified")
            assign(row["id"], split)
            if row.get("original_id"):
                previous = original_split.setdefault(row["original_id"], split)
                if previous != split:
                    raise ValueError(f"original_id {row['original_id']!r} crosses dataset splits")
    output, summary = [], {"threshold": threshold, "method": "connected_components", "splits": {}}
    for split in sorted(vertices):
        groups = connected_components(partitions.get(split, []), threshold, sorted(vertices[split]))
        for group in groups:
            identity = hashlib.sha256(json.dumps([split, group], ensure_ascii=False, separators=(",", ":")).encode("utf-8")).hexdigest()[:20]
            output.extend({"id": identifier, "duplicate_group_id": "dup_" + identity, "split": split,
                           "group_size": len(group)} for identifier in group)
        summary["splits"][split] = {"complaints": len(vertices[split]), "groups": len(groups),
                                    "scored_edges": len(partitions.get(split, [])),
                                    "threshold_edges": sum(score >= threshold for _, _, score in partitions.get(split, []))}
    summary["warning"] = "Connected components can merge chains without direct pair evidence; review clusters and evaluate cluster validity separately."
    return output, summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--threshold", type=float, help="Otherwise use the single saved threshold column")
    parser.add_argument("--all-complaints", action="append", default=[], help="Repeat CSVs to retain unpaired singleton complaints")
    parser.add_argument("--model", help="Select one model if the prediction table contains several")
    parser.add_argument("--output", default="results/duplicates/clusters.csv")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)
    try:
        destination = _path(args.output)
        source = _path(args.predictions)
        complaint_paths = [_path(path) for path in args.all_complaints]
        if destination in [source, *complaint_paths]:
            raise ValueError("Cluster output must not overwrite an input CSV")
        if destination.exists() and not args.overwrite:
            raise ValueError(f"Output already exists: {destination}; use --overwrite or a new path")
        output, summary = cluster_predictions(source, args.threshold, complaint_paths, args.model)
        destination.parent.mkdir(parents=True, exist_ok=True)
        with destination.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=("id", "duplicate_group_id", "split", "group_size"))
            writer.writeheader()
            writer.writerows(output)
        print(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False))
        print(f"Saved candidate duplicate groups: {destination}")
        return 0
    except (ValueError, OSError, csv.Error) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
