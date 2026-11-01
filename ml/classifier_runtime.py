"""Cached Phase 3 inference and honest explanations of its linear base model."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re

import joblib
import numpy as np
from sklearn.pipeline import Pipeline

from ml_training.experiment_data import preprocess_text
from ml_training.modeling import SentenceEncoder
from ml_training.scripts.train_classifiers import aligned_probabilities

_ENCODERS: dict[str, SentenceEncoder] = {}


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _linear_base(calibrated):
    model = calibrated
    seen = set()
    while not isinstance(model, Pipeline) and not hasattr(model, "coef_"):
        if id(model) in seen:
            raise ValueError("Cyclic classifier wrapper")
        seen.add(id(model))
        if getattr(model, "calibrated_classifiers_", None):
            if len(model.calibrated_classifiers_) != 1:
                raise ValueError("Exact base-margin explanations require one frozen calibrated estimator")
            model = model.calibrated_classifiers_[0].estimator
        elif hasattr(model, "estimator"):
            model = model.estimator
        else:
            raise ValueError("Saved classifier has no supported linear base estimator")
    return model


class ClassifierRuntime:
    """Load a trusted classifier once; keep pretrained encoders shared locally.

    TF-IDF contributions explain the *uncalibrated base margin*, not additive
    changes to calibrated probabilities. Embedding dimensions have no word
    semantics; optional token deletion supplies explicitly approximate term
    sensitivity alongside exact linear dimension contributions.
    """

    def __init__(self, model_path: str | Path, expected_target: str | None = None, encoder=None):
        self.path = Path(model_path).resolve()
        self.fingerprint = file_sha256(self.path)
        self.bundle = joblib.load(self.path)  # Trusted self-created joblib only.
        if not isinstance(self.bundle, dict) or self.bundle.get("kind") != "banglapsg_classifier" or self.bundle.get("schema_version") != 1:
            raise ValueError("Expected a Phase 3 classical classifier joblib bundle")
        self.target = self.bundle.get("target")
        if self.target not in ("category", "severity") or expected_target and self.target != expected_target:
            raise ValueError(f"Unexpected classifier target: {self.target!r}")
        if not self.bundle.get("calibration") or not hasattr(self.bundle.get("estimator"), "predict_proba"):
            raise ValueError("A calibrated Phase 3 probability classifier is required")
        self.root = (self.path.parent / self.bundle["artifact_root_relative"]).resolve()
        self.variants_path = (self.root / self.bundle["variants_file"]).resolve()
        if self.root not in self.variants_path.parents:
            raise ValueError("Normalization resource escapes the artifact root")
        if file_sha256(self.variants_path) != self.bundle["normalization_dictionary_sha256"]:
            raise ValueError("Classifier normalization dictionary fingerprint changed")
        resources = {"normalization_dictionary": file_sha256(self.variants_path),
                     "normalizer_code": file_sha256(Path(__file__).with_name("normalize.py")),
                     "preprocessing_code": file_sha256(Path(__file__).resolve().parents[1] / "ml_training" / "experiment_data.py"),
                     "runtime_code": file_sha256(__file__)}
        self.estimator = self.bundle["estimator"]
        self.labels = list(self.bundle["labels"])
        self.base = _linear_base(self.estimator)
        self.encoder = encoder
        if self.bundle["feature_type"] == "embedding":
            path = (self.root / self.bundle["encoder_dir"]).resolve()
            if self.root not in path.parents:
                raise ValueError("Encoder resource escapes the artifact root")
            encoder_files = {file.relative_to(path).as_posix(): file_sha256(file) for file in sorted(path.rglob("*")) if file.is_file()}
            if not encoder_files:
                raise ValueError("Saved sentence encoder resources are missing")
            resources["encoder_files"] = encoder_files
            if self.encoder is None:
                key = str(path) + hashlib.sha256(json.dumps(resources, sort_keys=True).encode("utf-8")).hexdigest()
                self.encoder = _ENCODERS.setdefault(key, SentenceEncoder(str(path), allow_download=False))
        elif self.bundle["feature_type"] not in ("tfidf", "embedding"):
            raise ValueError("Unsupported classifier feature type")
        self.resource_fingerprint = hashlib.sha256(json.dumps(resources, sort_keys=True).encode("utf-8")).hexdigest()

    def predict_many(self, texts: list[str], langs: list[str]) -> list[dict]:
        if len(texts) != len(langs):
            raise ValueError("Texts and languages must align")
        if not texts:
            return []
        cleaned = [preprocess_text(text, lang, self.variants_path) for text, lang in zip(texts, langs)]
        if any(not text for text in cleaned):
            raise ValueError("Text became empty after cleaning")
        features = self.encoder.encode(cleaned) if self.bundle["feature_type"] == "embedding" else cleaned
        probabilities = aligned_probabilities(self.estimator, features, self.labels)
        if not np.isfinite(probabilities).all() or ((probabilities < 0) | (probabilities > 1)).any() or not np.allclose(probabilities.sum(axis=1), 1):
            raise ValueError("Calibrated classifier returned invalid probabilities")
        return [{"prediction": self.labels[int(row.argmax())], "confidence": float(row.max()),
                 "probabilities": dict(zip(self.labels, map(float, row))), "text_normalized": text}
                for text, row in zip(cleaned, probabilities)]

    def predict(self, text: str, lang: str, top_k: int = 8, occlusion_limit: int = 20) -> dict:
        if isinstance(top_k, bool) or not isinstance(top_k, int) or top_k < 1:
            raise ValueError("top_k must be positive")
        result = self.predict_many([text], [lang])[0]
        cleaned = result["text_normalized"]
        if isinstance(self.base, Pipeline):
            vectorizer = self.base.named_steps.get("tfidf")
            classifier = self.base.named_steps.get("classifier")
            if vectorizer is None or classifier is None:
                raise ValueError("Unsupported text-pipeline explanation layout")
            vector = vectorizer.transform([cleaned])
            positions, values = vector.indices, vector.data
            names = vectorizer.get_feature_names_out()[positions]
            feature_kind = "character_ngram" if vectorizer.analyzer in ("char", "char_wb") else "word_token"
        else:
            classifier = self.base
            values = np.asarray(self.encoder.encode([cleaned]), dtype=float)[0]
            positions = np.arange(len(values))
            names = [f"embedding_dimension_{index}" for index in positions]
            feature_kind = "embedding_dimension"
        classes = list(classifier.classes_)
        class_index = classes.index(result["prediction"])
        coefficients = np.asarray(classifier.coef_, dtype=float)
        intercepts = np.asarray(classifier.intercept_, dtype=float)
        if coefficients.shape[0] == 1 and len(classes) == 2:
            sign = 1 if class_index == 1 else -1
            weights, intercept = coefficients[0] * sign, float(intercepts[0] * sign)
        else:
            weights, intercept = coefficients[class_index], float(intercepts[class_index])
        contributions = [{"term": str(name), "feature_kind": feature_kind, "feature_value": float(value),
                          "coefficient": float(weights[index]), "contribution": float(value * weights[index])}
                         for name, index, value in zip(names, positions, values)]
        contributions.sort(key=lambda row: (-abs(row["contribution"]), row["term"]))
        explanation = {"method": "linear_base_margin", "target_class": result["prediction"],
                       "feature_kind": feature_kind, "top_terms": contributions[:top_k] if feature_kind != "embedding_dimension" else [],
                       "top_features": contributions[:top_k], "intercept": intercept,
                       "base_margin": float(intercept + sum(row["contribution"] for row in contributions)),
                       "all_feature_contributions_sum": float(sum(row["contribution"] for row in contributions)),
                       "semantics": "Exact contributions to the selected class's linear base margin; not calibrated probability contributions or causal explanations."}
        if feature_kind == "embedding_dimension" and occlusion_limit > 0:
            matches = list(re.finditer(r"\S+", cleaned))[:occlusion_limit]
            variants, tokens = [], []
            for match in matches:
                reduced = (cleaned[:match.start()] + " " + cleaned[match.end():]).strip()
                if reduced:
                    variants.append(reduced)
                    tokens.append(match.group())
            effects = []
            for token, prediction in zip(tokens, self.predict_many(variants, [lang] * len(variants))):
                effect = result["probabilities"][result["prediction"]] - prediction["probabilities"][result["prediction"]]
                effects.append({"term": token, "probability_drop_on_deletion": float(effect)})
            explanation["top_terms"] = sorted(effects, key=lambda row: (-abs(row["probability_drop_on_deletion"]), row["term"]))[:top_k]
            explanation["term_method"] = "bounded token-deletion sensitivity; approximate, contextual, not SHAP or an additive probability decomposition"
        result["explanation"] = explanation
        return result
