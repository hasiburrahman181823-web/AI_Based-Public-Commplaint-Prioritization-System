"""Incident-isolated, validation-thresholded duplicate-detection ablations.

Every selected model sees the same test pairs. B1/B2 use cosine scores; C0/P1/P2
learn a symmetric pair scorer on training pairs only. Thresholds maximize
validation F1, with higher thresholds breaking ties. No test-based selection.
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_sample_weight

from ml.duplicate_features import FEATURE_SETS, feature_matrix, preprocess_complaint
from ml.duplicate_index import EmbeddingCache
from ml.location import DEFAULT_GAZETTEER_PATH, LocationGazetteer
from ml.normalize import DEFAULT_VARIANTS_PATH, load_variants
from ml_training.duplicate_data import load_duplicate_data
from ml_training.duplicate_metrics import evaluate_duplicate_scores, tune_threshold
from ml_training.modeling import DEFAULT_SENTENCE_MODEL, SentenceEncoder
from ml_training.scripts.train_classifiers import _publish, _versions, resolve_path

MODEL_NAMES = ("B1", "B2", "C0", "P1", "P2")


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def pair_fingerprint(frame: pd.DataFrame) -> str:
    """Fingerprint ordered observations, labels and metadata used for scoring."""
    columns = [name for name in frame.columns if name not in ("candidate_label",)]
    payload = json.dumps(frame.loc[:, columns].to_dict("records"), ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def pair_records(frame: pd.DataFrame, variants_path: Path) -> tuple[list[dict], list[dict]]:
    result = {"a": [], "b": []}
    for row in frame.to_dict("records"):
        for side in ("a", "b"):
            result[side].append(preprocess_complaint({
                "id": row[f"id_{side}"], "original_id": row[f"original_id_{side}"],
                "text": row[f"text_{side}"], "lang": row[f"lang_{side}"],
                "category": row[f"category_{side}"], "location": row[f"loc_{side}"],
                "timestamp": row[f"time_{side}"],
            }, variants_path))
    return result["a"], result["b"]


def fit_pair_estimator(features, truth, kind: str, seed: int):
    if kind == "lr":
        model = Pipeline([("scale", StandardScaler()), ("classifier", LogisticRegression(
            class_weight="balanced", max_iter=3000, random_state=seed,
        ))])
        return model.fit(features, truth)
    if kind == "gradient_boosting":
        model = GradientBoostingClassifier(n_estimators=200, learning_rate=0.05,
                                           max_depth=3, random_state=seed)
        return model.fit(features, truth, sample_weight=compute_sample_weight("balanced", truth))
    raise ValueError("classifier must be lr or gradient_boosting")


def positive_scores(estimator, features) -> np.ndarray:
    classes = list(estimator.classes_)
    if classes.count(1) != 1:
        raise ValueError("Learned duplicate scorer needs positive class 1")
    scores = np.asarray(estimator.predict_proba(features), dtype=float)[:, classes.index(1)]
    if not np.isfinite(scores).all():
        raise ValueError("Duplicate scorer produced non-finite scores")
    return scores


def _related(left: Path, right: Path) -> bool:
    return left == right or left in right.parents or right in left.parents


def validate_paths(args, results: Path, artifacts: Path, cache: Path) -> None:
    outputs = (results, artifacts, cache)
    if any(_related(left, right) for i, left in enumerate(outputs) for right in outputs[i + 1:]):
        raise ValueError("Results, artifacts and cache directories must be separate, non-nested paths")
    pair_path = resolve_path(args.pairs)
    inputs = [resolve_path(path) for path in args.complaints] + [resolve_path(args.gazetteer)]
    if pair_path.is_dir():
        if any(_related(pair_path, output) for output in outputs):
            raise ValueError("Output/cache directories must be outside the pair input tree")
    else:
        inputs.append(pair_path)
    for source in inputs:
        if any(output == source or output in source.parents for output in outputs):
            raise ValueError("An output/cache directory contains an input file")
    for output in outputs:
        if output.exists() and not output.is_dir():
            raise ValueError(f"Output/cache path is a file: {output}")
    for output in (results, artifacts):
        if output.is_dir() and any(path.name != ".gitkeep" for path in output.iterdir()) and not args.overwrite:
            raise ValueError(f"Output already exists: {output}. Use a new run directory or --overwrite.")


def _relative_report(metrics: dict, stage: Path) -> None:
    metrics["files"] = {name: Path(path).relative_to(stage).as_posix() for name, path in metrics["files"].items()}


def _table_markdown(rows: list[dict]) -> str:
    columns = ["model", "classifier", "n_test", "threshold", "precision", "recall", "f1", "pr_auc", "false_positive_rate"]
    lines = ["# Duplicate-detection ablation results", "", "Measured on the identical saved test-pair set; thresholds were selected on validation only.", "",
             "| " + " | ".join(columns) + " |", "| " + " | ".join(["---"] * len(columns)) + " |"]
    for row in rows:
        def render(value):
            return "undefined" if value is None else f"{value:.6f}" if isinstance(value, float) else str(value)
        lines.append("| " + " | ".join(render(row.get(name)) for name in columns) + " |")
    lines.extend(["", "PR-AUC denotes average precision; a separate trapezoidal PR area is recorded in JSON/CSV.",
                  "C0 controls for learning and category/language covariates. P1 − C0 adds location; P2 − P1 adds time.", ""])
    return "\n".join(lines)


def run_experiment(args) -> dict:
    if len(set(args.models)) != len(args.models) or not args.models:
        raise ValueError("Select nonempty, distinct model names")
    if not set(args.models) <= set(MODEL_NAMES):
        raise ValueError(f"Supported models: {MODEL_NAMES}")
    if args.embedding_batch_size < 1:
        raise ValueError("--embedding-batch-size must be positive")
    results, artifacts, cache_dir = map(resolve_path, (args.results_dir, args.artifacts_dir, args.cache_dir))
    validate_paths(args, results, artifacts, cache_dir)
    gazetteer_path = resolve_path(args.gazetteer)
    gazetteer = LocationGazetteer(gazetteer_path, fuzzy_threshold=args.fuzzy_threshold)
    load_variants(DEFAULT_VARIANTS_PATH)
    data = load_duplicate_data(resolve_path(args.pairs), [resolve_path(path) for path in args.complaints], args.allow_weak_labels)
    for split in ("train", "val"):
        if data[split].empty or set(data[split]["label"]) != {0, 1}:
            raise ValueError(f"{split} pairs must contain both explicit 0 and 1 labels; no pair-level re-splitting is performed")
    if data["test"].empty:
        raise ValueError("Test pair set is empty; supply a fixed independent pair test set")
    sides = {split: pair_records(data[split], DEFAULT_VARIANTS_PATH) for split in ("train", "val", "test")}
    test_id = pair_fingerprint(data["test"])
    warnings = list(data["warnings"])
    if args.category_source == "provided":
        warnings.append("Category features use provided complaint labels. Gold category metadata is an oracle relative to noisy predicted categories; run --category-source none as a sensitivity analysis.")
    if set(data["test"]["label"]) != {0, 1}:
        warnings.append("The test set has a single true class; ranking PR-AUC will be undefined.")
    input_files = [resolve_path(path) for path in args.complaints]
    pair_path = resolve_path(args.pairs)
    input_files += [pair_path] if pair_path.is_file() else [pair_path / split / "pairs.csv" for split in ("train", "val", "test")]
    input_files.append(gazetteer_path)
    manifest = {
        "schema_version": 1, "experiment_kind": "cross_language_duplicate_detection",
        "created_at_utc": datetime.now(timezone.utc).isoformat(), "seed": args.seed,
        "models_requested": args.models, "classifier": args.classifier,
        "test_pair_fingerprint": test_id, "versions": _versions(),
        "inputs": [{"path": str(path), "sha256": file_hash(path)} for path in input_files],
        "code_sha256": {path.relative_to(ROOT).as_posix(): file_hash(path) for path in (
            Path(__file__), ROOT / "ml" / "duplicate_features.py", ROOT / "ml" / "duplicate_index.py",
            ROOT / "ml" / "location.py", ROOT / "ml" / "time_features.py", ROOT / "ml" / "normalize.py",
            ROOT / "ml_training" / "duplicate_data.py", ROOT / "ml_training" / "duplicate_metrics.py",
        )},
        "design": {
            "training": "train pairs only; B1 vocabulary uses unique complaints appearing in train pairs",
            "validation": "threshold maximizing pairwise F1; higher threshold breaks ties; no test tuning",
            "test": "identical ordered pair rows, labels and metadata for all selected ablations",
            "category_source": args.category_source, "pr_auc": "average precision, distinct from trapezoidal PR area",
            "score_semantics": "cosine or learned ranking score; no probability calibration claim",
            "ablations": "C0=learned cosine/category/language control; P1=C0+location; P2=P1+time",
            "location_distance_flag": "known distinct, nonancestor canonical areas; not geographic distance",
        },
        "counts": {split: {"pairs": len(data[split]), "positives": int(data[split]["label"].sum()),
                           "incident_groups": len(set(data[split]["original_id_a"]) | set(data[split]["original_id_b"]))}
                   for split in ("train", "val", "test")},
        "feature_config": {"category_source": args.category_source, "fuzzy_threshold": args.fuzzy_threshold,
                           "time_gap_unit": "hours", "duration_gap_unit": "hours"},
        "warnings": warnings, "models": [],
    }
    results.parent.mkdir(parents=True, exist_ok=True)
    artifacts.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".duplicate-results-", dir=results.parent) as rtmp, tempfile.TemporaryDirectory(prefix=".duplicate-artifacts-", dir=artifacts.parent) as atmp:
        rstage, astage = Path(rtmp), Path(atmp)
        (astage / "resources").mkdir()
        shutil.copyfile(DEFAULT_VARIANTS_PATH, astage / "resources" / "banglish_variants.json")
        shutil.copyfile(gazetteer_path, astage / "resources" / "location_gazetteer.json")
        resource_hashes = {path.relative_to(astage).as_posix(): file_hash(path) for path in (astage / "resources").iterdir()}
        cosine_scores, all_features = {}, {}
        vectorizer = None
        if "B1" in args.models:
            train_records = {row["id"]: row for side in sides["train"] for row in side}
            vectorizer = TfidfVectorizer(analyzer="char_wb", ngram_range=(1, 5), lowercase=False)
            vectorizer.fit([train_records[key]["text"] for key in sorted(train_records)])
            cosine_scores["B1"] = {}
            for split, (left, right) in sides.items():
                a = vectorizer.transform([row["text"] for row in left])
                b = vectorizer.transform([row["text"] for row in right])
                cosine_scores["B1"][split] = np.asarray(a.multiply(b).sum(axis=1)).reshape(-1).clip(-1, 1)
        encoder_id = None
        if any(name != "B1" for name in args.models):
            encoder = SentenceEncoder(args.embedding_model, args.embedding_device, args.allow_download,
                                      args.embedding_batch_size, args.embedding_revision)
            encoder.save(astage / "minilm_encoder")
            encoder_files = {path.relative_to(astage).as_posix(): file_hash(path) for path in sorted((astage / "minilm_encoder").rglob("*")) if path.is_file()}
            if not encoder_files:
                raise ValueError("Encoder export produced no saved resource files")
            resource_hashes.update(encoder_files)
            encoder_id = hashlib.sha256(json.dumps({"files": encoder_files, "preprocessing": manifest["code_sha256"]["ml/normalize.py"]}, sort_keys=True).encode("utf-8")).hexdigest()
            namespace = hashlib.sha256(json.dumps({"encoder": encoder_id, "resources": resource_hashes}, sort_keys=True).encode("utf-8")).hexdigest()
            cache = EmbeddingCache(cache_dir / "embeddings.sqlite3", namespace)
            unique = {row["text"] for pair_sides in sides.values() for side in pair_sides for row in side}
            texts = sorted(unique)
            encoded = cache.get_or_encode(texts, encoder, batch_size=args.embedding_batch_size)
            lookup = {text: i for i, text in enumerate(texts)}
            cosine_scores["minilm"] = {}
            for split, (left, right) in sides.items():
                # Aligned row dot products, never an all-complaint similarity matrix.
                a = encoded[[lookup[row["text"]] for row in left]]
                b = encoded[[lookup[row["text"]] for row in right]]
                scores = np.einsum("ij,ij->i", a, b).clip(-1, 1)
                cosine_scores["minilm"][split] = scores
                if any(name in ("C0", "P1", "P2") for name in args.models):
                    all_features[split] = feature_matrix(left, right, scores, FEATURE_SETS["P2"], gazetteer, args.category_source)
                    audit = data[split].loc[:, ["pair_id", "label"]].reset_index(drop=True).copy()
                    for index, feature in enumerate(FEATURE_SETS["P2"]):
                        audit[feature] = all_features[split][:, index]
                    audit_dir = rstage / "feature_audits"
                    audit_dir.mkdir(exist_ok=True)
                    audit.to_csv(audit_dir / f"{split}.csv", index=False, encoding="utf-8")
            manifest["embedding"] = {"model": args.embedding_model, "requested_revision": args.embedding_revision,
                                     "fingerprint": encoder_id, "saved_files_sha256": encoder_files,
                                     "allow_download": args.allow_download}
        table = []
        for name in args.models:
            estimator = None
            features = list(FEATURE_SETS[name])
            if name in ("B1", "B2"):
                model_scores = cosine_scores["B1" if name == "B1" else "minilm"]
            else:
                columns = [FEATURE_SETS["P2"].index(feature) for feature in features]
                selected = {split: matrix[:, columns] for split, matrix in all_features.items()}
                estimator = fit_pair_estimator(selected["train"], data["train"]["label"].to_numpy(dtype=int), args.classifier, args.seed)
                model_scores = {split: positive_scores(estimator, values) for split, values in selected.items()}
            selection = tune_threshold(data["val"]["label"], model_scores["val"])
            threshold = selection["threshold"]
            model_record = {"name": name, "feature_names": features, "threshold_selection": selection,
                            "classifier": args.classifier if estimator is not None else "cosine_threshold",
                            "estimator_description": repr(estimator) if estimator is not None else None,
                            "test_pair_fingerprint": test_id, "artifact": f"{name}.joblib"}
            for split, output_name in (("val", "validation"), ("test", "test")):
                report = evaluate_duplicate_scores(data[split], model_scores[split], threshold, rstage / name / output_name, model_name=name)
                _relative_report(report, rstage)
                # The component writes its own JSON; replace staged paths with
                # portable relative references in that file as well.
                (rstage / name / output_name / "metrics.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
                model_record[output_name] = report
            bundle = {
                "schema_version": 1, "kind": "banglapsg_duplicate_model", "model_name": name,
                "estimator": estimator, "positive_class": 1,
                "vectorizer": vectorizer if name == "B1" else None,
                "threshold": threshold, "feature_names": features,
                "artifact_root_relative": ".", "encoder_dir": "minilm_encoder" if name != "B1" else None,
                "encoder_fingerprint": encoder_id if name != "B1" else None,
                "gazetteer_file": "resources/location_gazetteer.json", "variants_file": "resources/banglish_variants.json",
                "resource_sha256": {key: value for key, value in resource_hashes.items() if name != "B1" or not key.startswith("minilm_encoder/")},
                "feature_config": {"category_source": args.category_source, "fuzzy_threshold": args.fuzzy_threshold, "time_gap_unit": "hours"},
                "versions": manifest["versions"], "test_pair_fingerprint": test_id,
            }
            joblib.dump(bundle, astage / f"{name}.joblib", compress=3)
            report = model_record["test"]
            table_row = {"model": name, "classifier": model_record["classifier"], "features": ",".join(features),
                         "n_test": report["n"], "threshold": threshold, "validation_f1": selection["f1"],
                         "test_pair_fingerprint": test_id,
                         **{key: report[key] for key in ("precision", "recall", "f1", "pr_auc", "trapezoidal_pr_auc", "false_positive_rate", "false_negative_rate")}}
            table.append(table_row)
            manifest["models"].append(model_record)
            print(json.dumps(table_row, ensure_ascii=False, allow_nan=False), flush=True)
        pd.DataFrame(table).to_csv(rstage / "ablation_table.csv", index=False, encoding="utf-8")
        (rstage / "ablation_table.md").write_text(_table_markdown(table), encoding="utf-8")
        manifest["resources_sha256"] = resource_hashes
        manifest["result_files"] = sorted(path.relative_to(rstage).as_posix() for path in rstage.rglob("*") if path.is_file()) + ["metrics.json"]
        manifest["artifact_files"] = sorted(path.relative_to(astage).as_posix() for path in astage.rglob("*") if path.is_file()) + ["experiment.json"]
        payload = json.dumps(manifest, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
        (rstage / "metrics.json").write_text(payload, encoding="utf-8")
        (astage / "experiment.json").write_text(payload, encoding="utf-8")
        _publish(astage, artifacts)
        _publish(rstage, results)
    for warning in warnings:
        print(f"WARNING: {warning}", file=sys.stderr)
    print(f"Saved measured duplicate metrics: {results / 'metrics.json'}")
    print(f"Saved ablation tables: {results / 'ablation_table.csv'} and {results / 'ablation_table.md'}")
    print(f"Saved duplicate model bundles: {artifacts}")
    return manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pairs", default="ml_training/data/pairs", help="Gold pair split directory or combined CSV with split column")
    parser.add_argument("--complaints", action="append", required=True, help="Repeat train/val/test metadata CSVs and any manual paraphrase CSV")
    parser.add_argument("--models", nargs="+", choices=MODEL_NAMES, default=list(MODEL_NAMES))
    parser.add_argument("--classifier", choices=("lr", "gradient_boosting"), default="lr")
    parser.add_argument("--results-dir", default="results/duplicates")
    parser.add_argument("--artifacts-dir", default="artifacts/duplicates")
    parser.add_argument("--cache-dir", default=".cache/duplicate_training")
    parser.add_argument("--category-source", choices=("provided", "none"), default="provided")
    parser.add_argument("--gazetteer", default=str(DEFAULT_GAZETTEER_PATH))
    parser.add_argument("--fuzzy-threshold", type=float, default=90)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--embedding-model", default=DEFAULT_SENTENCE_MODEL)
    parser.add_argument("--embedding-revision")
    parser.add_argument("--embedding-device", default="cpu")
    parser.add_argument("--embedding-batch-size", type=int, default=64)
    parser.add_argument("--allow-download", action="store_true")
    parser.add_argument("--allow-weak-labels", action="store_true", help="Use generated suggestions for a clearly marked pipeline experiment, not independent research truth")
    parser.add_argument("--overwrite", action="store_true", help="Replace files generated by this run; prefer fresh directories to avoid stale unselected models")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        run_experiment(args)
        return 0
    except (ValueError, OSError, RuntimeError, ImportError, csv.Error) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
