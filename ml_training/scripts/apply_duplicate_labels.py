"""Attach one explicitly adjudicated gold label to every candidate pair.

No voting, label inference or automatic conflict resolution is performed. The
labels CSV must contain pair_id,gold_label; all source pairs must be covered.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd

from ml_training.scripts.train_classifiers import _publish, resolve_path


def _read(path: Path, required: set[str]) -> pd.DataFrame:
    with path.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.reader(stream, strict=True)
        header = next(reader, [])
        if len(set(header)) != len(header) or not required <= set(header):
            raise ValueError(f"{path}: need unique CSV headers including {sorted(required)}")
        records = []
        for number, fields in enumerate(reader, 2):
            if len(fields) != len(header):
                raise ValueError(f"{path}: malformed CSV record {number}")
            records.append(fields)
    return pd.DataFrame(records, columns=header, dtype=str)


def apply_labels(pairs_dir: Path, labels_path: Path, output_dir: Path, overwrite: bool = False) -> dict:
    pairs_dir, labels_path, output_dir = map(lambda path: Path(path).resolve(), (pairs_dir, labels_path, output_dir))
    if output_dir == pairs_dir or output_dir in pairs_dir.parents or pairs_dir in output_dir.parents:
        raise ValueError("Use separate non-nested source and gold-pair directories")
    if output_dir == labels_path or output_dir in labels_path.parents:
        raise ValueError("Gold output must not contain the labels input")
    if output_dir.exists() and (not output_dir.is_dir() or any(output_dir.iterdir())) and not overwrite:
        raise ValueError("Gold output exists; choose a new directory or use --overwrite")
    labels = _read(labels_path, {"pair_id", "gold_label"})
    if labels["pair_id"].str.strip().eq("").any() or labels["pair_id"].ne(labels["pair_id"].str.strip()).any() or labels["pair_id"].duplicated().any():
        raise ValueError("Adjudicated pair IDs must be nonempty and unique; resolve conflicts manually")
    if not labels["gold_label"].isin(("0", "1")).all():
        raise ValueError("Every gold_label must be explicitly 0 or 1; resolve missing/skip decisions first")
    lookup = dict(zip(labels["pair_id"], labels["gold_label"]))
    frames = []
    seen = set()
    for split in ("train", "val", "test"):
        frame = _read(pairs_dir / split / "pairs.csv", {"pair_id", "id_a", "id_b"})
        if frame["pair_id"].str.strip().eq("").any() or frame["pair_id"].ne(frame["pair_id"].str.strip()).any() or frame["pair_id"].duplicated().any() or seen & set(frame["pair_id"]):
            raise ValueError("Source pair IDs must be globally unique and nonempty")
        seen.update(frame["pair_id"])
        if "split" in frame and not frame["split"].eq(split).all():
            raise ValueError("Source split column disagrees with directory")
        frame["split"] = split
        missing = set(frame["pair_id"]) - set(lookup)
        if missing:
            raise ValueError(f"{split}: {len(missing)} pairs lack adjudicated labels; no rows are silently dropped")
        if "gold_label" in frame:
            supplied = frame["pair_id"].map(lookup)
            if not frame["gold_label"].eq(supplied).all():
                raise ValueError("Source already contains differing gold labels; reconcile the adjudication round")
        frame["gold_label"] = frame["pair_id"].map(lookup)
        frames.append(frame)
    if set(lookup) - seen:
        raise ValueError("Adjudication file includes unknown pair IDs; use the matching source candidate set")
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    summary = {"label_status": "explicit_adjudicated_input", "pairs": len(seen),
               "splits": {split: len(frame) for split, frame in zip(("train", "val", "test"), frames)}}
    with tempfile.TemporaryDirectory(prefix=".gold-pairs-", dir=output_dir.parent) as temporary:
        stage = Path(temporary)
        for split, frame in zip(("train", "val", "test"), frames):
            (stage / split).mkdir()
            frame.to_csv(stage / split / "pairs.csv", index=False, encoding="utf-8")
        pd.concat(frames, ignore_index=True).to_csv(stage / "pairs.csv", index=False, encoding="utf-8")
        (stage / "label_manifest.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
        _publish(stage, output_dir)
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pairs-dir", default="ml_training/data/pairs")
    parser.add_argument("--labels", required=True, help="One row per pair: pair_id,gold_label")
    parser.add_argument("--output-dir", default="ml_training/data/pairs_gold")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)
    try:
        result = apply_labels(resolve_path(args.pairs_dir), resolve_path(args.labels), resolve_path(args.output_dir), args.overwrite)
        print(json.dumps(result, ensure_ascii=False))
        return 0
    except (ValueError, OSError, csv.Error) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
