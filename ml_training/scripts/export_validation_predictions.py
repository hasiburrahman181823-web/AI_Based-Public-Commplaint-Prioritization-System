"""Export joint validation predictions bound to saved calibrated classifiers."""

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

from ml.classifier_runtime import ClassifierRuntime, file_sha256
from ml_training.scripts.train_classifiers import resolve_path


def export_validation_predictions(
    input_path: str | Path, category_model: str | Path, severity_model: str | Path,
    output_path: str | Path, allow_unverified_validation: bool = False,
    overwrite: bool = False,
) -> dict:
    source, category, severity, destination = [Path(value).resolve() for value in (input_path, category_model, severity_model, output_path)]
    if destination in (source, category, severity):
        raise ValueError("Prediction output must not overwrite input data or classifier files")
    if destination.exists() and not overwrite:
        raise ValueError("Validation prediction output exists; use a new path or --overwrite")
    with source.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.reader(stream, strict=True)
        header = next(reader, [])
        required = {"id", "original_id", "text", "lang", "category", "severity"}
        if len(header) != len(set(header)) or not required <= set(header):
            raise ValueError(f"Validation CSV needs unique headers including {sorted(required)}")
        records = []
        for number, row in enumerate(reader, 2):
            if len(row) != len(header):
                raise ValueError(f"Malformed validation CSV record {number}")
            records.append([value.strip() for value in row])
    frame = pd.DataFrame(records, columns=header, dtype=str)
    if frame.empty:
        raise ValueError("Validation data is empty")
    for name in required:
        if frame[name].eq("").any():
            raise ValueError(f"Validation {name} contains blank values")
    if frame["id"].duplicated().any():
        raise ValueError("Validation IDs must be unique")
    if not frame["lang"].isin(("bn", "en", "banglish")).all():
        raise ValueError("Validation lang must be bn, en or banglish")
    if "split" in frame:
        if not frame["split"].eq("val").all():
            raise ValueError("Threshold selection accepts only split=val, never train or test")
    elif source.stem != "val":
        raise ValueError("A validation file must be named val.csv or contain explicit split=val rows")
    source_hash = file_sha256(source)
    runtimes = {"category": ClassifierRuntime(category, "category"), "severity": ClassifierRuntime(severity, "severity")}
    warnings, proof = [], {}
    for target, runtime in runtimes.items():
        unknown = set(frame[target]) - set(runtime.labels)
        if unknown:
            raise ValueError(f"Validation {target} labels absent from the saved classifier: {sorted(unknown)}")
        manifest_path = runtime.root / "experiment.json"
        verified = False
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8-sig"))
            if isinstance(manifest, dict) and manifest.get("experiment_kind") == "calibrated_classifiers":
                verified = any(isinstance(entry, dict) and Path(entry.get("path", "")).stem == "val"
                               and entry.get("sha256") == source_hash for entry in manifest.get("inputs", []))
        except (OSError, ValueError, TypeError):
            pass
        if not verified:
            message = f"{target}: validation input fingerprint is not verified against the saved Phase 3 experiment manifest"
            if not allow_unverified_validation:
                raise ValueError(message + "; use --allow-unverified-validation only for a deliberate, audited alternative validation cohort")
            warnings.append(message)
        proof[target] = "saved_validation_input_hash_match" if verified else "caller_declared_validation_only"
    predictions = {target: runtime.predict_many(frame["text"].tolist(), frame["lang"].tolist()) for target, runtime in runtimes.items()}
    output = frame.loc[:, ["id", "original_id", "text", "lang"]].copy()
    output["split"] = "val"
    output["source_validation_sha256"] = source_hash
    for target, runtime in runtimes.items():
        output[target + "_true"] = frame[target]
        output[target + "_pred"] = [row["prediction"] for row in predictions[target]]
        output[target + "_confidence"] = [row["confidence"] for row in predictions[target]]
        output[target + "_probabilities"] = [json.dumps(row["probabilities"], ensure_ascii=False) for row in predictions[target]]
        output[target + "_model_sha256"] = runtime.fingerprint
        output[target + "_resources_sha256"] = runtime.resource_fingerprint
        output[target + "_validation_provenance"] = proof[target]
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".validation-export-", dir=destination.parent) as temporary:
        staged = Path(temporary) / "predictions.csv"
        output.to_csv(staged, index=False, encoding="utf-8")
        os.replace(staged, destination)
    return {"rows": len(output), "original_groups": int(frame["original_id"].nunique()), "output": str(destination),
            "source_validation_sha256": source_hash, "validation_provenance": proof, "warnings": warnings}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--category-model", required=True)
    parser.add_argument("--severity-model", required=True)
    parser.add_argument("--output", default="results/uncertainty/validation_joint.csv")
    parser.add_argument("--allow-unverified-validation", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)
    try:
        result = export_validation_predictions(resolve_path(args.input), resolve_path(args.category_model), resolve_path(args.severity_model),
                                               resolve_path(args.output), args.allow_unverified_validation, args.overwrite)
        print(json.dumps(result, ensure_ascii=False))
        return 0
    except (ValueError, OSError, RuntimeError, ImportError, csv.Error) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
