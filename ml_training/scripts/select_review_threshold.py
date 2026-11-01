"""Select an empirical human-review policy from validation predictions only."""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd

from ml.uncertainty import coverage_error_curve, select_review_threshold
from ml_training.scripts.train_classifiers import _publish, resolve_path

REQUIRED = {"id", "original_id", "split", "category_true", "category_pred", "severity_true",
            "severity_pred", "category_confidence", "severity_confidence"}


def fingerprint(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_validation_predictions(path: Path) -> pd.DataFrame:
    with path.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.reader(stream, strict=True)
        header = next(reader, [])
        if len(set(header)) != len(header) or not REQUIRED <= set(header):
            raise ValueError(f"Validation CSV needs unique headers including {sorted(REQUIRED)}")
        records = []
        for number, row in enumerate(reader, 2):
            if len(row) != len(header):
                raise ValueError(f"Invalid CSV field count at validation record {number}")
            records.append(row)
    frame = pd.DataFrame(records, columns=header, dtype=str)
    if frame.empty:
        raise ValueError("Validation prediction file is empty")
    for field in REQUIRED:
        if frame[field].str.strip().eq("").any():
            raise ValueError(f"Validation field {field!r} contains blank values")
    if not frame["split"].eq("val").all():
        raise ValueError("Review threshold selection accepts split=val only; never select on train/test/natural labels")
    if frame["id"].duplicated().any():
        raise ValueError("Validation prediction IDs must be unique")
    for column in ("category_confidence", "severity_confidence"):
        try:
            frame[column] = pd.to_numeric(frame[column], errors="raise")
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{column} must contain finite numeric confidence values between 0 and 1") from exc
    for target in ("category", "severity"):
        column = target + "_probabilities"
        if column not in frame:
            continue
        for index, raw in frame[column].items():
            try:
                values = json.loads(raw)
                if not isinstance(values, dict) or not values or any(
                    not isinstance(key, str) or not key or isinstance(value, bool)
                    or not isinstance(value, (int, float)) or not math.isfinite(value) or not 0 <= value <= 1
                    for key, value in values.items()
                ) or not math.isclose(sum(values.values()), 1, abs_tol=1e-6):
                    raise ValueError("invalid probability distribution")
                predicted = frame.at[index, target + "_pred"]
                if predicted not in values or frame.at[index, target + "_true"] not in values:
                    raise ValueError("label absent from probability classes")
                confidence = float(frame.at[index, target + "_confidence"])
                if not math.isclose(values[predicted], max(values.values()), abs_tol=1e-9) or not math.isclose(confidence, max(values.values()), abs_tol=1e-9):
                    raise ValueError("prediction/confidence does not match probability argmax")
            except (ValueError, TypeError) as exc:
                raise ValueError(f"Invalid {target} probabilities at validation CSV record {index + 2}: {exc}") from exc
    # The utility validates finiteness, bounds and aligned errors as well.
    return frame


def _plot_curve(curve: list[dict], selection: dict, path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    observed = [row for row in curve if row["n_accepted"]]
    figure, axis = plt.subplots(figsize=(7, 5))
    axis.plot([row["coverage"] for row in observed], [row["error_rate"] for row in observed],
              marker=".", color="#244e65", label="Observed validation operating points")
    axis.axhline(selection["max_error_rate"], linestyle="--", color="gray", label="Requested empirical error ceiling")
    if selection["n_accepted"]:
        axis.scatter([selection["coverage"]], [selection["error_rate"]], s=65, color="#b04b31", zorder=4, label="Selected policy")
    axis.set(xlim=(0, 1.02), ylim=(0, 1.02), xlabel="Accepted coverage", ylabel="Error among accepted predictions",
             title="Validation coverage versus classification error")
    axis.grid(alpha=0.2)
    axis.legend(loc="upper center", bbox_to_anchor=(0.5, -0.18), fontsize=8)
    figure.tight_layout()
    figure.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(figure)


def build_review_policy(
    predictions: str | Path, category_model: str | Path, severity_model: str | Path,
    output_dir: str | Path, policy_output: str | Path,
    max_error_rate: float = 0.1, min_accepted: int = 10, overwrite: bool = False,
) -> dict:
    source, category, severity, output, destination = map(lambda value: Path(value).resolve(),
                                                        (predictions, category_model, severity_model, output_dir, policy_output))
    inputs = {source, category, severity}
    files = [output / name for name in ("coverage_error_curve.csv", "coverage_error_curve.json", "coverage_error_curve.png", "selection.json")]
    if any(path in inputs for path in [*files, destination]):
        raise ValueError("Review outputs must not overwrite predictions or model inputs")
    if destination in files:
        raise ValueError("Policy artifact must not collide with a curve/report filename")
    if not overwrite and any(path.exists() for path in [*files, destination]):
        raise ValueError("Review-policy output exists; use a fresh output directory or --overwrite")
    frame = read_validation_predictions(source)
    fingerprints = {"category": fingerprint(category), "severity": fingerprint(severity)}
    resource_fingerprints = {}
    warnings = []
    for target in ("category", "severity"):
        column = target + "_model_sha256"
        if column in frame:
            if not frame[column].eq(fingerprints[target]).all():
                raise ValueError(f"Validation predictions' {column} does not match the supplied classifier")
        else:
            warnings.append(f"{column} is absent: input split/probability/model provenance is asserted by the caller, not independently verified.")
        resource_column = target + "_resources_sha256"
        if resource_column in frame:
            from ml.classifier_runtime import ClassifierRuntime
            runtime = ClassifierRuntime(category if target == "category" else severity, target)
            if not frame[resource_column].eq(runtime.resource_fingerprint).all():
                raise ValueError(f"Validation predictions' {resource_column} does not match the classifier resources")
            resource_fingerprints[target] = runtime.resource_fingerprint
        else:
            warnings.append(f"{resource_column} absent: this policy is not resource-bound and cannot authorize automatic scoring in the Phase 5 analyzer.")
    confidence = frame[["category_confidence", "severity_confidence"]].min(axis=1).to_numpy()
    # Validate BOTH confidence columns, not only their minimum (an invalid 1.2
    # must not be hidden by a smaller valid score in the other target).
    errors = ((frame["category_true"] != frame["category_pred"]) | (frame["severity_true"] != frame["severity_pred"])).astype(int).to_numpy()
    for target in ("category", "severity"):
        coverage_error_curve(frame[target + "_confidence"], errors)
    curve = coverage_error_curve(confidence, errors)
    selection = select_review_threshold(confidence, errors, max_error_rate, min_accepted)
    policy = {
        "kind": "priority_review_policy", "schema_version": 1, **selection,
        "classifier_fingerprints": fingerprints,
        "classifier_resource_fingerprints": resource_fingerprints,
        "source_validation_file": str(source), "source_validation_sha256": fingerprint(source),
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "error_definition": "category incorrect OR severity incorrect on the same validation row",
        "n_validation_original_groups": int(frame["original_id"].nunique()),
        "warnings": [*warnings, "Selecting on validation data already used for probability calibration can be optimistic. Preserve a separate policy-validation cohort for stronger claims.",
                     "Empirical accepted-error constraints do not establish future or group-specific risk guarantees."],
    }
    output.mkdir(parents=True, exist_ok=True)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".review-policy-", dir=output) as temporary:
        stage = Path(temporary)
        pd.DataFrame(curve).to_csv(stage / "coverage_error_curve.csv", index=False, encoding="utf-8")
        (stage / "coverage_error_curve.json").write_text(json.dumps(curve, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        (stage / "selection.json").write_text(json.dumps(policy, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        _plot_curve(curve, selection, stage / "coverage_error_curve.png")
        _publish(stage, output)
    with tempfile.TemporaryDirectory(prefix=".policy-artifact-", dir=destination.parent) as temporary:
        staged_policy = Path(temporary) / "policy.json"
        staged_policy.write_text(json.dumps(policy, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        os.replace(staged_policy, destination)
    return policy


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", required=True, help="Joint validation CSV exported from the saved calibrated classifiers")
    parser.add_argument("--category-model", required=True)
    parser.add_argument("--severity-model", required=True)
    parser.add_argument("--output-dir", default="results/uncertainty")
    parser.add_argument("--policy-output", default="artifacts/review_policy.json")
    parser.add_argument("--max-error-rate", type=float, default=0.1)
    parser.add_argument("--min-accepted", type=int, default=10)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)
    try:
        policy = build_review_policy(resolve_path(args.predictions), resolve_path(args.category_model), resolve_path(args.severity_model),
                                     resolve_path(args.output_dir), resolve_path(args.policy_output), args.max_error_rate, args.min_accepted, args.overwrite)
        print(json.dumps(policy, ensure_ascii=False, indent=2, allow_nan=False))
        return 0
    except (ValueError, OSError, RuntimeError, ImportError, csv.Error) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
