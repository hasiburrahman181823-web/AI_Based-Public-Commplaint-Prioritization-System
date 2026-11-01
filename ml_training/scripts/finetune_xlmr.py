"""Optional local/GPU XLM-R training with validation-only temperature scaling.

Install requirements-xlmr.txt before running. No model is downloaded unless
--allow-download is supplied. Fixed epochs are used: validation data is reserved
for temperature calibration and is not used to select a checkpoint.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import random
import sys

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def resolve_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else REPO_ROOT / path


def fingerprint_file(path: Path) -> dict:
    """Fingerprint input files in bounded memory for model/data provenance."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return {"path": str(path.resolve()), "sha256": digest.hexdigest(), "bytes": path.stat().st_size}


def validate_output_paths(
    input_dir: Path,
    artifacts: Path,
    results: Path,
    natural_test: Path | None = None,
    overwrite: bool = False,
) -> None:
    """Reject shared/nested data and output trees before importing neural libraries."""
    input_dir, artifacts, results = (Path(path).resolve() for path in (input_dir, artifacts, results))

    def related(left: Path, right: Path) -> bool:
        return left == right or left in right.parents or right in left.parents

    if related(artifacts, results) or related(input_dir, artifacts) or related(input_dir, results):
        raise ValueError("Input, artifact and result directories must be separate, non-nested directories")
    if natural_test is not None:
        natural = Path(natural_test).resolve()
        if any(output == natural or output in natural.parents for output in (artifacts, results)):
            raise ValueError("Natural-test input must be outside artifact and result directories")
    for directory in (artifacts, results):
        if directory.exists() and not directory.is_dir():
            raise ValueError(f"Output path is a file: {directory}")
        if directory.is_dir() and any(child.name != ".gitkeep" for child in directory.iterdir()) and not overwrite:
            raise ValueError(f"Output directory is not empty: {directory}. Use a new path or --overwrite.")


def validate_class_coverage(train_labels, validation_labels, configured_labels) -> None:
    """Use the same train/validation class-support contract as classical models."""
    training, validation, configured = set(train_labels), set(validation_labels), set(configured_labels)
    if training != configured:
        raise ValueError("Every configured target label must have training examples; correct the group split or label order upstream.")
    if validation != training:
        raise ValueError(
            "Validation must contain exactly the training classes for temperature calibration. "
            f"Missing={sorted(training - validation)}; unseen={sorted(validation - training)}. "
            "Correct the upstream group split or collect more validation incidents."
        )


def validate_logits(logits, labels=None) -> tuple[np.ndarray, np.ndarray | None]:
    """Validate numerical model output without importing Torch or Transformers."""
    values = np.asarray(logits, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] < 2 or not np.isfinite(values).all():
        raise ValueError("logits must be a finite two-dimensional array with at least two classes")
    if labels is None:
        return values, None
    targets = np.asarray(labels)
    if targets.ndim != 1 or len(targets) != len(values) or targets.dtype.kind not in "iu":
        raise ValueError("labels must be a one-dimensional integer array aligned with logits")
    if not len(targets) or ((targets < 0) | (targets >= values.shape[1])).any():
        raise ValueError("calibration needs nonempty labels in the logits class range")
    return values, targets.astype(np.int64, copy=False)


def softmax_with_temperature(logits, temperature: float = 1.0) -> np.ndarray:
    """Stable softmax used for both clean and noisy evaluation probabilities."""
    if not math.isfinite(temperature) or temperature <= 0:
        raise ValueError("temperature must be finite and greater than zero")
    values, _ = validate_logits(logits)
    if not len(values):
        return np.empty_like(values)
    scaled = values / temperature
    scaled -= scaled.max(axis=1, keepdims=True)
    exponentials = np.exp(scaled)
    return exponentials / exponentials.sum(axis=1, keepdims=True)


def temperature_nll(logits, labels, temperature: float = 1.0) -> float:
    """Mean validation negative log likelihood, computed without softmax underflow."""
    if not math.isfinite(temperature) or temperature <= 0:
        raise ValueError("temperature must be finite and greater than zero")
    values, targets = validate_logits(logits, labels)
    scaled = values / temperature
    maxima = scaled.max(axis=1)
    log_normalizer = maxima + np.log(np.exp(scaled - maxima[:, None]).sum(axis=1))
    return float(np.mean(log_normalizer - scaled[np.arange(len(targets)), targets]))


def fit_temperature(logits, labels, max_iter: int = 100) -> dict:
    """Fit one positive scalar on validation logits; keep T=1 if fitting worsens NLL.

    The scalar is bounded to exp(-3)..exp(3). Class predictions are unchanged by
    scalar temperature; confidence and ECE may change. No test labels are used.
    """
    import torch

    values, targets = validate_logits(logits, labels)
    x = torch.as_tensor(values, dtype=torch.float64)
    y = torch.as_tensor(targets, dtype=torch.long)
    log_temperature = torch.zeros((), dtype=torch.float64, requires_grad=True)
    optimizer = torch.optim.LBFGS([log_temperature], lr=0.1, max_iter=max_iter, line_search_fn="strong_wolfe")

    def closure():
        optimizer.zero_grad()
        temperature = log_temperature.clamp(-3.0, 3.0).exp()
        loss = torch.nn.functional.cross_entropy(x / temperature, y)
        loss.backward()
        return loss

    before = temperature_nll(values, targets)
    optimizer.step(closure)
    fitted = float(log_temperature.detach().clamp(-3.0, 3.0).exp().item())
    after = temperature_nll(values, targets, fitted)
    accepted = math.isfinite(after) and after <= before + 1e-12
    return {
        "temperature": fitted if accepted else 1.0,
        "validation_nll_before": before,
        "validation_nll_after": after if accepted else before,
        "method": "validation_only_scalar_temperature_lbfgs",
        "accepted_fitted_temperature": accepted,
        "n_calibration": len(targets),
        "temperature_bounds": [math.exp(-3), math.exp(3)],
    }


def collect_logits(model, tokenizer, texts, device, batch_size: int, max_length: int) -> np.ndarray:
    import torch

    model.eval()
    batches = []
    with torch.no_grad():
        for start in range(0, len(texts), batch_size):
            encoded = tokenizer(texts[start:start + batch_size], padding=True, truncation=True,
                                max_length=max_length, return_tensors="pt")
            encoded = {key: value.to(device) for key, value in encoded.items()}
            batches.append(model(**encoded).logits.detach().cpu().numpy())
    return np.concatenate(batches) if batches else np.empty((0, model.config.num_labels), dtype=np.float32)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", choices=("category", "severity"), required=True)
    parser.add_argument("--input-dir", default="ml_training/data/multilingual")
    parser.add_argument("--natural-test", help="Independent CSV; severity evaluation needs a severity column")
    parser.add_argument("--severity-order", nargs="+", help="Explicit low-to-high severity labels, e.g. low medium high")
    parser.add_argument("--model", default="FacebookAI/xlm-roberta-base", help="Cached model name or local model directory")
    parser.add_argument("--revision", help="Pin a model revision/commit when using a Hugging Face model name")
    parser.add_argument("--allow-download", action="store_true", help="Explicitly permit pretrained model/tokenizer download")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--noise-rate", type=float, default=0.15)
    parser.add_argument("--n-bins", type=int, default=10)
    parser.add_argument("--artifacts-dir", help="Default: artifacts/xlmr_<target>")
    parser.add_argument("--results-dir", help="Default: results/xlmr_<target>; separate from classical experiments")
    parser.add_argument("--overwrite", action="store_true")
    return parser


def run(args: argparse.Namespace) -> dict:
    # Keep CUDA workspace configuration ahead of CUDA initialization.
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    if args.epochs < 1 or args.batch_size < 1 or args.max_length < 8:
        raise ValueError("epochs and batch-size must be positive; max-length must be at least 8")
    if not math.isfinite(args.learning_rate) or args.learning_rate <= 0:
        raise ValueError("learning-rate must be finite and positive")
    if not 0 <= args.noise_rate <= 1 or args.n_bins < 2:
        raise ValueError("noise-rate must be in [0,1] and n-bins must be at least 2")
    if args.target == "severity" and not args.severity_order:
        raise ValueError("Severity training requires --severity-order in low-to-high order; do not infer ordinal order alphabetically.")
    input_dir = resolve_path(args.input_dir)
    natural_path = resolve_path(args.natural_test) if args.natural_test else None
    artifacts = resolve_path(args.artifacts_dir or f"artifacts/xlmr_{args.target}")
    results = resolve_path(args.results_dir or f"results/xlmr_{args.target}")
    validate_output_paths(input_dir, artifacts, results, natural_path, args.overwrite)
    try:
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
    except ImportError as exc:
        raise ValueError("Optional XLM-R dependencies are missing. Install requirements-xlmr.txt in a GPU/Colab or local environment.") from exc
    import joblib
    from ml.noise import add_banglish_noise
    from ml.normalize import normalize_banglish
    from ml_training.evaluation import evaluate_predictions
    from ml_training.experiment_data import load_experiment_data

    data = load_experiment_data(input_dir,
                                natural_test=natural_path,
                                severity_order=args.severity_order)
    inputs = {name: fingerprint_file(input_dir / f"{name}.csv")
              for name in ("train", "val", "test")}
    if args.natural_test:
        inputs["natural"] = fingerprint_file(natural_path)
    variants_path = REPO_ROOT / "ml" / "banglish_variants.json"
    variants_bytes = variants_path.read_bytes()
    labels = data["category_labels" if args.target == "category" else "severity_labels"]
    if len(labels) < 2:
        raise ValueError(f"{args.target} training requires at least two classes")
    label_to_id = {str(label): index for index, label in enumerate(labels)}
    labels = [str(label) for label in labels]
    train, validation = data["train"], data["val"]
    if train.empty or validation.empty:
        raise ValueError("XLM-R needs nonempty training and validation splits; validation is reserved for temperature calibration.")
    validate_class_coverage(train[args.target].astype(str), validation[args.target].astype(str), labels)
    for directory in (artifacts, results):
        directory.mkdir(parents=True, exist_ok=True)
    (artifacts / "banglish_variants.json").write_bytes(variants_bytes)
    if args.device == "cuda" and not torch.cuda.is_available():
        raise ValueError("--device cuda requested but CUDA is unavailable; use --device cpu or a GPU runtime.")
    device = "cuda" if args.device == "auto" and torch.cuda.is_available() else "cpu" if args.device == "auto" else args.device
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    torch.use_deterministic_algorithms(True, warn_only=True)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    loading = {"local_files_only": not args.allow_download, "trust_remote_code": False}
    if args.revision:
        loading["revision"] = args.revision
    try:
        tokenizer = AutoTokenizer.from_pretrained(args.model, **loading)
        model = AutoModelForSequenceClassification.from_pretrained(
            args.model, num_labels=len(labels), id2label=dict(enumerate(labels)), label2id=label_to_id, **loading
        ).to(device)
    except (OSError, ValueError) as exc:
        raise ValueError(
            "Cannot load XLM-R weights/tokenizer. Supply a complete local model directory, cache it first, "
            "or explicitly use --allow-download. " + str(exc)
        ) from exc

    training_examples = [{"text": row["text"], "label": label_to_id[str(row[args.target])]}
                         for row in train.to_dict("records")]

    def collate(examples):
        encoded = tokenizer([row["text"] for row in examples], padding=True, truncation=True,
                            max_length=args.max_length, return_tensors="pt")
        encoded["labels"] = torch.tensor([row["label"] for row in examples], dtype=torch.long)
        return encoded

    generator = torch.Generator().manual_seed(args.seed)
    loader = torch.utils.data.DataLoader(training_examples, batch_size=args.batch_size,
                                         shuffle=True, generator=generator, collate_fn=collate, num_workers=0)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=0.01)
    history = []
    for epoch in range(args.epochs):
        model.train()
        total_loss, seen = 0.0, 0
        for batch in loader:
            batch = {key: value.to(device) for key, value in batch.items()}
            optimizer.zero_grad(set_to_none=True)
            loss = model(**batch).loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            batch_count = len(batch["labels"])
            total_loss += float(loss.detach().cpu()) * batch_count
            seen += batch_count
        epoch_loss = total_loss / seen
        history.append({"epoch": epoch + 1, "train_loss": epoch_loss})
        print(f"epoch {epoch + 1}/{args.epochs}: train_loss={epoch_loss:.6f}", flush=True)

    validation_logits = collect_logits(model, tokenizer, validation["text"].tolist(), device, args.batch_size, args.max_length)
    calibration = fit_temperature(validation_logits, [label_to_id[str(value)] for value in validation[args.target]])
    temperature = calibration["temperature"]
    print("Validation calibration: " + json.dumps(calibration, sort_keys=True), flush=True)
    # Persist expensive trained weights before rendering diagnostic figures.
    model_dir = artifacts / "model"
    model.save_pretrained(model_dir)
    tokenizer.save_pretrained(model_dir)
    descriptor = {
        "kind": "huggingface_sequence_classifier", "target": args.target,
        "model_subdir": "model", "labels": labels, "temperature": temperature,
        "max_length": args.max_length,
        "preprocessing": "clean_text; normalize_banglish only when lang=banglish",
        "variants_subpath": "banglish_variants.json",
        "variants_sha256": hashlib.sha256(variants_bytes).hexdigest(),
        "weights_format": "Hugging Face save_pretrained (load relative to this joblib descriptor)",
    }
    joblib.dump(descriptor, artifacts / "model.joblib")
    test, natural = data["test"], data["natural"]
    cohorts = {
        "test_all": test,
        "bangla_test": test.loc[test["lang"].eq("bn")],
        "english_test": test.loc[test["lang"].eq("en")],
        "banglish_test": test.loc[test["lang"].eq("banglish")],
        "natural_test": natural,
    }
    evaluations, table = {}, []

    def evaluate(name, frame, texts=None):
        eligible = frame.loc[frame[args.target].fillna("").astype(str).str.strip().ne("")].copy() if args.target in frame else frame.iloc[0:0].copy()
        if len(eligible) != len(frame) and texts is not None:
            raise ValueError("Noise evaluation must align with fully labeled target rows")
        truth = eligible[args.target].astype(str).tolist() if args.target in eligible else []
        encoded = collect_logits(model, tokenizer, texts if texts is not None else eligible["text"].tolist(),
                                 device, args.batch_size, args.max_length)
        probabilities = softmax_with_temperature(encoded, temperature)
        prediction = [labels[index] for index in probabilities.argmax(axis=1)] if len(probabilities) else []
        metrics = evaluate_predictions(truth, prediction, probabilities, labels, results / name,
                                       ordinal=args.target == "severity", n_bins=args.n_bins)
        metrics["unlabeled_rows_excluded"] = len(frame) - len(eligible)
        evaluations[name] = metrics
        row = {"model": "xlmr", "target": args.target, "cohort": name, **metrics}
        row = {key: value for key, value in row.items() if value is None or isinstance(value, (str, int, float, bool))}
        table.append(row)
        print(f"{args.target}/xlmr/{name}: " + json.dumps(metrics, ensure_ascii=False, sort_keys=True), flush=True)
        return metrics

    for name, frame in cohorts.items():
        evaluate(name, frame)
    banglish = cohorts["banglish_test"]
    rng = random.Random(args.seed)
    noise_source = banglish["raw_text"] if "raw_text" in banglish else banglish["text"]
    noisy_texts = [normalize_banglish(add_banglish_noise(text, rng=rng, rate=args.noise_rate)) for text in noise_source]
    noisy = evaluate("banglish_noisy", banglish, noisy_texts)
    clean_f1 = evaluations["banglish_test"].get("macro_f1")
    noisy_f1 = noisy.get("macro_f1")
    drop = clean_f1 - noisy_f1 if clean_f1 is not None and noisy_f1 is not None else None
    changed = sum(a != b for a, b in zip(banglish["text"], noisy_texts))
    noisy.update(macro_f1_drop=drop, noise_rate=args.noise_rate, changed_rows_after_preprocessing=changed)
    table[-1].update(macro_f1_drop=drop, noise_rate=args.noise_rate, changed_rows_after_preprocessing=changed)
    print("Banglish robustness: " + json.dumps({"macro_f1_drop": drop, "noise_rate": args.noise_rate,
                                                "changed_rows_after_preprocessing": changed}, sort_keys=True), flush=True)

    metadata = {
        "schema_version": 1, "model": "xlmr", "target": args.target,
        "pretrained_model": args.model, "requested_revision": args.revision,
        "resolved_revision": getattr(model.config, "_commit_hash", None),
        "download_permitted": args.allow_download, "seed": args.seed, "device": device,
        "input_files": inputs,
        "banglish_variants_sha256": hashlib.sha256(variants_bytes).hexdigest(),
        "labels": labels, "epochs": args.epochs, "batch_size": args.batch_size,
        "max_length": args.max_length, "learning_rate": args.learning_rate,
        "training_rows": len(train), "training_original_groups": train["original_id"].nunique(),
        "validation_usage": "temperature calibration only; fixed epoch count; no checkpoint selection",
        "calibration": calibration, "training_history": history,
        "warnings": data.get("warnings", []),
        "versions": {"torch": torch.__version__, "transformers": __import__("transformers").__version__},
        "evaluations": evaluations,
    }
    # Convert NumPy scalars through JSON's default hook while rejecting NaN.
    content = json.dumps(metadata, ensure_ascii=False, indent=2, allow_nan=False,
                         default=lambda value: value.item() if isinstance(value, np.generic) else str(value)) + "\n"
    (results / "metrics.json").write_text(content, encoding="utf-8")
    fields = list(dict.fromkeys(key for row in table for key in row))
    with (results / "results_table.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(table)
    metadata["metrics_sha256"] = hashlib.sha256(content.encode("utf-8")).hexdigest()
    (artifacts / "metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2,
                                                       default=lambda value: value.item() if isinstance(value, np.generic) else str(value)) + "\n", encoding="utf-8")
    print(f"Saved weights/tokenizer: {model_dir}")
    print(f"Saved joblib descriptor: {artifacts / 'model.joblib'}")
    print(f"Saved metrics/table: {results / 'metrics.json'} ; {results / 'results_table.csv'}")
    return metadata


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        run(args)
    except (ValueError, OSError, ImportError, RuntimeError, KeyError, csv.Error) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
