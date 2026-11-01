"""Predict with a trusted Phase 3 joblib bundle and its saved local resources."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import joblib
import numpy as np

from ml_training.experiment_data import preprocess_text
from ml_training.modeling import SentenceEncoder
from ml_training.scripts.train_classifiers import aligned_probabilities, resolve_path


def predict_bundle(model_path: str | Path, texts: list[str], langs: list[str]) -> list[dict]:
    """Load only your own trusted joblib; pickle-based files can execute code."""
    if not texts or len(texts) != len(langs):
        raise ValueError("Supply matching, nonempty text and language lists")
    path = Path(model_path).resolve()
    bundle = joblib.load(path)
    if not isinstance(bundle, dict) or bundle.get("kind") != "banglapsg_classifier" or bundle.get("schema_version") != 1:
        raise ValueError("Not a supported classical classifier bundle; XLM-R uses its separate descriptor and model weights")
    artifact_root = (path.parent / bundle["artifact_root_relative"]).resolve()
    variants = artifact_root / bundle["variants_file"]
    if hashlib.sha256(variants.read_bytes()).hexdigest() != bundle["normalization_dictionary_sha256"]:
        raise ValueError("Saved normalization dictionary fingerprint changed; restore the original inference resources")
    cleaned = [preprocess_text(text, lang, variants) for text, lang in zip(texts, langs)]
    if any(not text for text in cleaned):
        raise ValueError("Prediction text is empty after cleaning")
    features = cleaned
    if bundle["feature_type"] == "embedding":
        encoder = SentenceEncoder(str(artifact_root / bundle["encoder_dir"]), allow_download=False)
        features = encoder.encode(cleaned)
    elif bundle["feature_type"] != "tfidf":
        raise ValueError("Unknown feature_type in model bundle")
    labels = bundle["labels"]
    probabilities = aligned_probabilities(bundle["estimator"], features, labels)
    return [
        {"target": bundle["target"], "model": bundle["model_name"], "text_normalized": text,
         "prediction": labels[int(np.argmax(row))], "probabilities": dict(zip(labels, map(float, row)))}
        for text, row in zip(cleaned, probabilities)
    ]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="Trusted model path, e.g. artifacts/category/char_svm.joblib")
    parser.add_argument("--text", required=True)
    parser.add_argument("--lang", choices=("bn", "en", "banglish"), required=True)
    args = parser.parse_args(argv)
    try:
        for result in predict_bundle(resolve_path(args.model), [args.text], [args.lang]):
            print(json.dumps(result, ensure_ascii=False, allow_nan=False))
        return 0
    except (ValueError, OSError, RuntimeError, ImportError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
