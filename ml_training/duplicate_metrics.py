"""Validation-threshold tuning and auditable duplicate-detection measurements."""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import auc, average_precision_score, precision_recall_curve

from ml_training.duplicate_data import language_pair

REQUIRED_LANGUAGE_PAIRS = ("bn-en", "bn-banglish", "en-banglish")
REQUIRED_PAIR_TYPES = ("positive_cross_language", "positive_paraphrase", "hard_negative", "easy_negative")


def _arrays(y_true, scores) -> tuple[np.ndarray, np.ndarray]:
    raw = np.asarray(list(y_true))
    if raw.ndim != 1 or raw.dtype.kind == "b":
        raise ValueError("Binary truth must be a one-dimensional sequence of 0/1 labels")
    try:
        truth = raw.astype(float)
        values = np.asarray(scores, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError("Labels and scores must be numeric") from exc
    if not np.isfinite(truth).all() or not np.isin(truth, (0, 1)).all():
        raise ValueError("Binary truth must contain only 0 and 1")
    if values.shape != (len(truth),) or not np.isfinite(values).all():
        raise ValueError("Scores must be a finite one-dimensional array aligned with labels")
    return truth.astype(np.int8), values


def tune_threshold(y_true, scores) -> dict:
    """Maximize validation F1; ties prefer the higher, more conservative threshold.

    Predictions use score >= threshold. All observed scores plus the threshold
    immediately above the maximum are candidates. Sorting/cumulative counts
    make this O(n log n), without a quadratic scan over validation rows.
    Call only on the fixed validation split; test labels never enter this step.
    """
    truth, values = _arrays(y_true, scores)
    if not len(truth):
        raise ValueError("Threshold tuning requires nonempty validation pairs")
    order = np.argsort(-values, kind="stable")
    ordered_scores, ordered_truth = values[order], truth[order]
    boundaries = np.r_[np.flatnonzero(ordered_scores[:-1] != ordered_scores[1:]), len(truth) - 1]
    true_positive = np.cumsum(ordered_truth, dtype=np.int64)[boundaries]
    predicted_positive = boundaries + 1
    false_positive = predicted_positive - true_positive
    false_negative = int(truth.sum()) - true_positive
    thresholds = ordered_scores[boundaries]
    above = np.nextafter(float(ordered_scores[0]), math.inf)
    if not np.isfinite(above):
        raise ValueError("Scores are too large for a finite all-negative threshold; rescale them before tuning")
    thresholds = np.r_[above, thresholds]
    tp, fp, fn = np.r_[0, true_positive], np.r_[0, false_positive], np.r_[truth.sum(), false_negative]
    denominator = 2 * tp + fp + fn
    f1 = np.divide(2 * tp, denominator, out=np.zeros(len(tp), dtype=float), where=denominator != 0)
    selected = int(np.argmax(f1))  # descending thresholds make the first tie highest
    precision = tp[selected] / (tp[selected] + fp[selected]) if tp[selected] + fp[selected] else 0.0
    recall = tp[selected] / (tp[selected] + fn[selected]) if tp[selected] + fn[selected] else 0.0
    return {
        "threshold": float(thresholds[selected]), "precision": float(precision),
        "recall": float(recall), "f1": float(f1[selected]), "n": len(truth),
        "n_positives": int(truth.sum()), "candidates_examined": len(thresholds),
        "objective": "validation_pairwise_f1", "tie_break": "higher_threshold",
        "decision_rule": "score >= threshold",
    }


def _statistics(truth: np.ndarray, values: np.ndarray, threshold: float) -> dict:
    if not len(truth):
        return {
            "status": "skipped_empty", "n": 0, "n_positives": 0, "n_negatives": 0,
            "positive_prevalence": None, "precision": None, "recall": None, "f1": None,
            "accuracy": None, "false_positive_rate": None, "false_negative_rate": None,
            "pr_auc": None, "average_precision": None, "trapezoidal_pr_auc": None,
            "pr_auc_undefined_reason": "No evaluation pairs", "confusion_matrix": None,
        }
    prediction = values >= threshold
    positive = truth == 1
    tp, fp = int((prediction & positive).sum()), int((prediction & ~positive).sum())
    fn, tn = int((~prediction & positive).sum()), int((~prediction & ~positive).sum())
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0
    average_precision, trapezoidal, undefined = None, None, None
    if len(np.unique(truth)) < 2:
        undefined = "Only one true label occurs in this slice; ranking discrimination is not estimable."
    else:
        average_precision = float(average_precision_score(truth, values))
        curve_precision, curve_recall, _ = precision_recall_curve(truth, values)
        trapezoidal = float(auc(curve_recall, curve_precision))
    return {
        "status": "ok", "n": len(truth), "n_positives": int(positive.sum()),
        "n_negatives": int((~positive).sum()), "positive_prevalence": float(positive.mean()),
        "precision": float(precision), "recall": float(recall), "f1": float(f1),
        "accuracy": (tp + tn) / len(truth),
        "false_positive_rate": fp / (fp + tn) if fp + tn else None,
        "false_negative_rate": fn / (fn + tp) if fn + tp else None,
        "pr_auc": average_precision, "average_precision": average_precision,
        "trapezoidal_pr_auc": trapezoidal, "pr_auc_undefined_reason": undefined,
        "confusion_matrix": {"tn": tn, "fp": fp, "fn": fn, "tp": tp},
    }


def _plot_pr(truth, scores, output: Path, model_name: str) -> None:
    import matplotlib
    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    precision, recall, _ = precision_recall_curve(truth, scores)
    figure, axis = plt.subplots(figsize=(6.5, 5.5))
    axis.step(recall, precision, where="post", color="#244e65", label="Measured PR curve")
    axis.axhline(float(truth.mean()), color="gray", linestyle="--", label="Positive prevalence")
    axis.set(xlim=(0, 1), ylim=(0, 1.02), xlabel="Recall", ylabel="Precision",
             title=f"Duplicate detection PR curve{': ' + model_name if model_name else ''}")
    axis.legend(loc="best")
    axis.grid(alpha=0.2)
    figure.tight_layout()
    figure.savefig(output, dpi=160, bbox_inches="tight")
    plt.close(figure)


def evaluate_duplicate_scores(
    frame: pd.DataFrame,
    scores,
    threshold: float,
    output_dir: str | Path,
    model_name: str = "",
) -> dict:
    """Evaluate exactly the supplied pair rows and export auditable predictions.

    ``pr_auc`` is explicitly average precision (step-integral), not trapezoidal
    PR area; the latter is provided under its own distinct name. Cosines and
    classifier scores are rankings, not assumed calibrated probabilities.
    Empty strata have null metrics, while single-label strata retain binary
    precision/recall/F1 and state why ranking AUC is undefined.
    """
    if "label" not in frame or "pair_type" not in frame:
        raise ValueError("Pair frame requires label and pair_type columns")
    if isinstance(threshold, bool) or not isinstance(threshold, (int, float, np.floating, np.integer)) or not np.isfinite(threshold):
        raise ValueError("Threshold must be a finite numeric value")
    truth, values = _arrays(frame["label"], scores)
    predictions = frame.copy().reset_index(drop=True)
    if "language_pair" not in predictions:
        if "lang_a" not in predictions or "lang_b" not in predictions:
            raise ValueError("Supply language_pair or independent lang_a/lang_b metadata")
        predictions["language_pair"] = [language_pair(a, b) for a, b in zip(predictions["lang_a"], predictions["lang_b"])]
    valid_pairs = {language_pair(left, right) for left in ("bn", "en", "banglish") for right in ("bn", "en", "banglish")}
    if not predictions["language_pair"].isin(valid_pairs).all():
        raise ValueError("language_pair must use canonical bn-en, bn-banglish, en-banglish or same-language values")
    result = _statistics(truth, values, float(threshold))
    result.update(model=model_name, threshold=float(threshold), decision_rule="score >= threshold",
                  pr_auc_definition="average precision (step-integral of precision over recall)",
                  zero_division_policy="precision/recall/F1 use zero when a denominator is zero; inspect support and false-positive rate for negative-only pair types",
                  score_semantics="ranking score; not necessarily a calibrated probability")
    result["by_pair_type"] = {
        kind: _statistics(truth[mask], values[mask], float(threshold))
        for kind in sorted(set(REQUIRED_PAIR_TYPES) | set(predictions["pair_type"]))
        for mask in [predictions["pair_type"].eq(kind).to_numpy()]
    }
    result["by_language_pair"] = {
        kind: _statistics(truth[mask], values[mask], float(threshold))
        for kind in sorted(set(REQUIRED_LANGUAGE_PAIRS) | set(predictions["language_pair"]))
        for mask in [predictions["language_pair"].eq(kind).to_numpy()]
    }
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    predictions["label"] = truth
    predictions["model"] = model_name
    predictions["score"] = values
    predictions["threshold"] = float(threshold)
    predictions["prediction"] = (values >= threshold).astype(int)
    predictions["score_margin"] = values - float(threshold)
    predictions["error_type"] = np.where((truth == 0) & (values >= threshold), "false_positive",
                                        np.where((truth == 1) & (values < threshold), "false_negative", "correct"))
    predictions.to_csv(output / "predictions.csv", index=False, encoding="utf-8")
    result["files"] = {"predictions.csv": str(output / "predictions.csv"), "metrics.json": str(output / "metrics.json")}
    if len(truth) and len(np.unique(truth)) == 2:
        curve_precision, curve_recall, curve_thresholds = precision_recall_curve(truth, values)
        pd.DataFrame({"precision": curve_precision, "recall": curve_recall,
                      "threshold": [*map(float, curve_thresholds), None]}).to_csv(output / "precision_recall_curve.csv", index=False)
        _plot_pr(truth, values, output / "precision_recall_curve.png", model_name)
        result["files"].update({name: str(output / name) for name in ("precision_recall_curve.csv", "precision_recall_curve.png")})
    (output / "metrics.json").write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return result
