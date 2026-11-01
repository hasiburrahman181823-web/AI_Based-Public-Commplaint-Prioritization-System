"""Analyze one complaint or export a batch for explanation and fairness review."""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd

from ml.analysis import PriorityAnalyzer
from ml.classifier_runtime import file_sha256
from ml_training.scripts.train_classifiers import resolve_path


def read_records(path: str | Path) -> list[dict]:
    with Path(path).open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.reader(stream, strict=True)
        header = next(reader, [])
        if len(header) != len(set(header)) or "text" not in header:
            raise ValueError("Input needs unique CSV headers including text")
        records = []
        for number, row in enumerate(reader, 2):
            if len(row) != len(header):
                raise ValueError(f"Malformed complaint CSV record {number}")
            records.append(dict(zip(header, row)))
    return records


def analyze_batch(analyzer: PriorityAnalyzer, input_path: str | Path, output_path: str | Path,
                  csv_output: str | Path | None = None, overwrite: bool = False) -> dict:
    source, destination = Path(input_path).resolve(), Path(output_path).resolve()
    tabular = Path(csv_output).resolve() if csv_output else None
    outputs = [destination] + ([tabular] if tabular else [])
    if source in outputs or len(set(outputs)) != len(outputs):
        raise ValueError("Analysis outputs must be distinct from the input and each other")
    if not overwrite and any(path.exists() for path in outputs):
        raise ValueError("Analysis output exists; use a new output or --overwrite")
    records = read_records(source)
    if not records:
        raise ValueError("Complaint input is empty")
    source_hash = file_sha256(source)
    seen, results, rows = set(), [], []
    for index, record in enumerate(records):
        record = dict(record)
        record["id"] = record.get("id") or f"batch_{source_hash[:16]}_{index}"
        if record["id"].strip() in seen:
            raise ValueError("Complaint input IDs must be unique")
        seen.add(record["id"].strip())
        result = analyzer.analyze_record(record)
        results.append(result)
        rows.append({"id": result["id"], "original_id": record.get("original_id", ""),
                     "text": record["text"], "lang": result["lang"], "area_id": result["area_id"],
                     "location": record.get("location", ""), "timestamp": record.get("timestamp", ""),
                     "category_pred": result["category"], "severity_pred": result["severity"],
                     "category_true": record.get("category_true", record.get("category", "")),
                     "severity_true": record.get("severity_true", record.get("severity", "")),
                     "priority_score": result["priority_score"], "priority_label": result["priority_label"],
                     "ranking_score": result["ranking_score"], "ranking_method": result["ranking_method"],
                     "confidence": result["confidence"], "review_required": result["review_required"],
                     "review_status": result["review_status"], "review_reasons": json.dumps(result["review_reasons"]),
                     "language_inference_method": result["language_inference"]["method"],
                     "duplicate_group_size": result["duplicate_group_size"]})
    for path in outputs:
        path.parent.mkdir(parents=True, exist_ok=True)
    # Compute the whole batch before publishing any rows; no partially analyzed
    # file is presented as a completed evaluation.
    with tempfile.TemporaryDirectory(prefix=".analysis-batch-", dir=destination.parent) as temporary:
        staged = Path(temporary) / "predictions.jsonl"
        with staged.open("w", encoding="utf-8", newline="") as stream:
            for result in results:
                stream.write(json.dumps(result, ensure_ascii=False, allow_nan=False) + "\n")
        if tabular:
            with tempfile.TemporaryDirectory(prefix=".analysis-csv-", dir=tabular.parent) as csv_tmp:
                csv_stage = Path(csv_tmp) / "predictions.csv"
                pd.DataFrame(rows).to_csv(csv_stage, index=False, encoding="utf-8")
                os.replace(csv_stage, tabular)
        os.replace(staged, destination)
    return {"n": len(results), "n_needs_human_review": sum(result["review_required"] for result in results),
            "input_sha256": source_hash, "jsonl": str(destination), "csv": str(tabular) if tabular else None}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    inputs = parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument("--text")
    inputs.add_argument("--input")
    parser.add_argument("--location")
    parser.add_argument("--timestamp")
    parser.add_argument("--category-model", default="artifacts/category/char_svm.joblib")
    parser.add_argument("--severity-model", default="artifacts/severity/char_svm.joblib")
    parser.add_argument("--review-policy", default="artifacts/review_policy.json")
    parser.add_argument("--rules-config", default="config/priority_rules.yaml")
    parser.add_argument("--ranker", default="artifacts/priority_ranker.joblib")
    parser.add_argument("--duplicate-model", default="artifacts/duplicates/P2.joblib")
    parser.add_argument("--existing", help="Complaint CSV to index once before analysis")
    parser.add_argument("--cache-dir", default=".cache/analysis_duplicates")
    parser.add_argument("--top-k", type=int, default=100)
    parser.add_argument("--output", default="results/analysis/predictions.jsonl")
    parser.add_argument("--csv-output")
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.top_k < 1:
            raise ValueError("--top-k must be positive")
        analyzer = PriorityAnalyzer(category_model=args.category_model, severity_model=args.severity_model,
                                    review_policy=args.review_policy, rules_config=args.rules_config, ranker=args.ranker,
                                    duplicate_model=args.duplicate_model, cache_dir=args.cache_dir, top_k=args.top_k,
                                    existing_complaints=read_records(resolve_path(args.existing)) if args.existing else None)
        if args.text is not None:
            print(json.dumps(analyzer.analyze_complaint(args.text, args.location, args.timestamp), ensure_ascii=True, allow_nan=False))
        else:
            print(json.dumps(analyze_batch(analyzer, resolve_path(args.input), resolve_path(args.output),
                                          resolve_path(args.csv_output) if args.csv_output else None, args.overwrite)))
        return 0
    except (ValueError, OSError, RuntimeError, ImportError, csv.Error) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
