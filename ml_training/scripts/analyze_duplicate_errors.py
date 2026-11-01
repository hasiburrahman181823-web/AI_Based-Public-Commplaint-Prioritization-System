"""Export the highest-scoring false positives and lowest-scoring false negatives."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd


def resolve_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else ROOT / path


def export_errors(predictions: str | Path, output_dir: str | Path, top_k: int = 50, overwrite: bool = False) -> dict:
    """Keep provenance and rank errors within this fixed model/score scale.

    Scores from different models are not pooled: a cosine and a classifier
    probability do not share a confidence scale. Sorting by signed threshold
    margin is equivalent to sorting raw scores within a fixed model threshold.
    """
    if isinstance(top_k, bool) or not isinstance(top_k, int) or top_k < 1:
        raise ValueError("top_k must be a positive integer")
    source, output = Path(predictions), Path(output_dir)
    with source.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.reader(handle, strict=True)
        header = next(reader, [])
        required = {"pair_id", "label", "score", "threshold", "prediction"}
        if len(header) != len(set(header)) or not required <= set(header):
            raise ValueError(f"Predictions CSV must have unique headers including {sorted(required)}")
        rows = []
        for number, row in enumerate(reader, 2):
            if len(row) != len(header):
                raise ValueError(f"Wrong CSV field count at prediction record {number}")
            rows.append(row)
    frame = pd.DataFrame(rows, columns=header, dtype=str)
    if frame["pair_id"].eq("").any() or frame["pair_id"].duplicated().any():
        raise ValueError("Prediction pair IDs must be nonempty and unique")
    for column in ("label", "prediction"):
        if not frame[column].isin(("0", "1")).all():
            raise ValueError(f"{column} must contain 0/1 labels")
        frame[column] = frame[column].astype(int)
    for column in ("score", "threshold"):
        frame[column] = pd.to_numeric(frame[column], errors="raise")
        if not np.isfinite(frame[column]).all():
            raise ValueError(f"{column} must be finite")
    if not frame.empty and frame["threshold"].nunique() != 1:
        raise ValueError("Analyze one model/threshold at a time; mixed thresholds cannot be ranked together")
    if "model" in frame and frame["model"].nunique() > 1:
        raise ValueError("Analyze one model/threshold at a time; mixed model score scales cannot be ranked together")
    if not (frame["prediction"].to_numpy() == (frame["score"].to_numpy() >= frame["threshold"].to_numpy()).astype(int)).all():
        raise ValueError("Predictions disagree with score >= threshold")
    frame["score_margin"] = frame["score"] - frame["threshold"]
    destinations = [output / "false_positives.csv", output / "false_negatives.csv", output / "error_summary.json"]
    if source.resolve() in {path.resolve() for path in destinations}:
        raise ValueError("Error outputs must not overwrite the source predictions CSV")
    if not overwrite and any(path.exists() for path in destinations):
        raise ValueError("Error analysis outputs already exist; use another output directory or --overwrite")
    output.mkdir(parents=True, exist_ok=True)
    positive_errors = frame.loc[frame["label"].eq(0) & frame["prediction"].eq(1)].sort_values(["score_margin", "pair_id"], ascending=[False, True], kind="stable")
    negative_errors = frame.loc[frame["label"].eq(1) & frame["prediction"].eq(0)].sort_values(["score_margin", "pair_id"], ascending=[True, True], kind="stable")
    positive_errors.head(top_k).to_csv(destinations[0], index=False, encoding="utf-8")
    negative_errors.head(top_k).to_csv(destinations[1], index=False, encoding="utf-8")
    summary = {"input": str(source.resolve()), "top_k_per_error_type": top_k,
               "total_pairs": len(frame), "false_positives": len(positive_errors), "false_negatives": len(negative_errors),
               "exported_false_positives": min(top_k, len(positive_errors)), "exported_false_negatives": min(top_k, len(negative_errors)),
               "ranking": "FP highest score-threshold first; FN lowest score-threshold first; pair_id breaks ties",
               "files": [str(path) for path in destinations]}
    destinations[2].write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--output-dir", default="results/duplicate_errors")
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)
    try:
        result = export_errors(resolve_path(args.predictions), resolve_path(args.output_dir), args.top_k, args.overwrite)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0
    except (ValueError, OSError, csv.Error) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
