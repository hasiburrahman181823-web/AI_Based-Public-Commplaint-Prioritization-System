"""Headless classification, calibration and ordinal evaluation.

Probabilities must already be calibrated, with columns in the supplied ``labels``
order. F1 uses that fixed label universe even when a language slice lacks a class.
ECE is top-label ECE with equal-width confidence bins, left-closed/right-open
except for the final bin, which includes confidence 1.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Sequence

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
)


def _validate_labels(labels: Sequence[str]) -> list[str]:
    result = list(labels)
    if not result or any(not isinstance(label, str) or not label for label in result):
        raise ValueError("labels must be a nonempty sequence of nonempty strings in probability-column order")
    if len(set(result)) != len(result):
        raise ValueError("labels must be unique")
    return result


def _validated_arrays(y_true, probabilities, labels):
    labels = _validate_labels(labels)
    truth = np.asarray(list(y_true), dtype=object)
    if truth.ndim != 1:
        raise ValueError("y_true must be one-dimensional")
    unknown = set(truth) - set(labels)
    if unknown:
        raise ValueError(f"Evaluation contains labels outside the supplied label universe: {sorted(unknown)}")
    try:
        probabilities = np.asarray(probabilities, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError("probabilities must be a numeric matrix") from exc
    if probabilities.shape != (len(truth), len(labels)):
        raise ValueError(
            f"probabilities must have shape ({len(truth)}, {len(labels)}) in supplied labels order; "
            f"got {probabilities.shape}"
        )
    if not np.isfinite(probabilities).all():
        raise ValueError("probabilities must be finite")
    if ((probabilities < 0) | (probabilities > 1)).any():
        raise ValueError("probabilities must be between 0 and 1")
    if len(truth) and not np.allclose(probabilities.sum(axis=1), 1, rtol=1e-6, atol=1e-6):
        raise ValueError("Each probability row must sum to 1")
    return truth, probabilities, labels


def expected_calibration_error(
    y_true, probabilities, labels: Sequence[str], n_bins: int = 10,
) -> dict:
    """Return top-label ECE and auditable bin counts/accuracy/confidence.

    ECE = sum(count_bin / n * abs(accuracy_bin - mean_confidence_bin)). Empty
    bins have ``None`` accuracy/confidence rather than fabricated observations.
    An empty evaluation set returns ``ece=None``.
    """
    if isinstance(n_bins, bool) or not isinstance(n_bins, int) or n_bins < 1:
        raise ValueError("n_bins must be a positive integer")
    truth, probabilities, labels = _validated_arrays(y_true, probabilities, labels)
    bins = [
        {"bin": index, "lower": index / n_bins, "upper": (index + 1) / n_bins,
         "count": 0, "accuracy": None, "confidence": None}
        for index in range(n_bins)
    ]
    if not len(truth):
        return {"ece": None, "n": 0, "n_bins": n_bins, "bins": bins}
    winning = probabilities.argmax(axis=1)
    predicted = np.asarray(labels, dtype=object)[winning]
    confidence = probabilities[np.arange(len(truth)), winning]
    correct = predicted == truth
    assignments = np.minimum((confidence * n_bins).astype(int), n_bins - 1)
    ece = 0.0
    for index, data in enumerate(bins):
        mask = assignments == index
        count = int(mask.sum())
        data["count"] = count
        if count:
            data["accuracy"] = float(correct[mask].mean())
            data["confidence"] = float(confidence[mask].mean())
            ece += count / len(truth) * abs(data["accuracy"] - data["confidence"])
    return {"ece": float(ece), "n": len(truth), "n_bins": n_bins, "bins": bins}


def quadratic_weighted_kappa(y_true, y_pred, labels: Sequence[str]) -> tuple[float | None, str | None]:
    """Return severity QWK and an explanation when it is undefined.

    The supplied label order defines the ordinal scale; labels are never sorted
    here. Both absent classes and the complete configured severity scale matter.
    """
    labels = _validate_labels(labels)
    truth, predicted = list(y_true), list(y_pred)
    if len(truth) != len(predicted):
        raise ValueError("y_true and y_pred must have the same length")
    unknown = (set(truth) | set(predicted)) - set(labels)
    if unknown:
        raise ValueError(f"Ordinal evaluation contains unknown labels: {sorted(unknown)}")
    if not truth:
        return None, "No evaluation rows; quadratic weighted kappa is undefined."
    if len(labels) < 2:
        return None, "The ordinal scale has fewer than two labels."
    index = {label: i for i, label in enumerate(labels)}
    left = np.bincount([index[label] for label in truth], minlength=len(labels)) / len(truth)
    right = np.bincount([index[label] for label in predicted], minlength=len(labels)) / len(truth)
    distance = (np.arange(len(labels))[:, None] - np.arange(len(labels))[None, :]) ** 2
    if float((distance * left[:, None] * right[None, :]).sum()) == 0:
        return None, "Expected ordinal disagreement is zero; quadratic weighted kappa is undefined."
    value = float(cohen_kappa_score(truth, predicted, labels=labels, weights="quadratic"))
    if not np.isfinite(value):
        return None, "Quadratic weighted kappa returned a non-finite value for this label distribution."
    return value, None


def _plot_font_families() -> list[str]:
    """Prefer an installed Bangla-capable font, without fetching font files.

    Nirmala UI is normally present on Windows; Noto family names cover common
    Linux/Colab installations. DejaVu remains the final Latin fallback. Font
    selection is scoped to each figure call instead of changing process-wide
    matplotlib defaults.
    """
    from matplotlib import font_manager

    installed = {font.name.casefold(): font.name for font in font_manager.fontManager.ttflist}
    for preferred in ("Nirmala UI", "Noto Sans Bengali", "Noto Sans Bangla", "Nirmala Text", "Vrinda"):
        if preferred.casefold() in installed:
            return [installed[preferred.casefold()], "DejaVu Sans"]
    return ["DejaVu Sans"]


def _plots(matrix, labels, calibration, output_dir):
    # Lazy import permits metric-only use without loading a plotting backend.
    import matplotlib
    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    with matplotlib.rc_context({"font.family": _plot_font_families()}):
        _render_plots(matrix, labels, calibration, output_dir, plt)


def _render_plots(matrix, labels, calibration, output_dir, plt):
    size = max(5, min(16, len(labels) * 0.7 + 2))
    figure, axis = plt.subplots(figsize=(size, size))
    image = axis.imshow(matrix, cmap="Blues", interpolation="nearest")
    figure.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
    axis.set(xticks=np.arange(len(labels)), yticks=np.arange(len(labels)),
             xticklabels=labels, yticklabels=labels, xlabel="Predicted label", ylabel="True label",
             title="Confusion matrix (counts)")
    plt.setp(axis.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    if len(labels) <= 30:
        threshold = float(matrix.max()) / 2
        for row in range(len(labels)):
            for column in range(len(labels)):
                axis.text(column, row, str(int(matrix[row, column])), ha="center", va="center",
                          color="white" if matrix[row, column] > threshold else "black", fontsize=9)
    figure.tight_layout()
    figure.savefig(output_dir / "confusion_matrix.png", dpi=160, bbox_inches="tight")
    plt.close(figure)

    figure, (axis, histogram) = plt.subplots(2, 1, figsize=(6.5, 7.5), sharex=True,
                                           gridspec_kw={"height_ratios": [3, 1]})
    axis.plot([0, 1], [0, 1], "--", color="gray", label="Perfect calibration")
    occupied = [item for item in calibration["bins"] if item["count"]]
    axis.plot([item["confidence"] for item in occupied], [item["accuracy"] for item in occupied],
              marker="o", linewidth=1.8, color="#244e65", label="Observed bins")
    axis.set(xlim=(0, 1), ylim=(0, 1), ylabel="Fraction correct",
             title=f"Top-label reliability diagram · ECE={calibration['ece']:.4f}")
    axis.grid(alpha=0.2)
    axis.legend(loc="best")
    histogram.bar([(item["lower"] + item["upper"]) / 2 for item in calibration["bins"]],
                  [item["count"] for item in calibration["bins"]],
                  width=0.9 / calibration["n_bins"], color="#244e65")
    histogram.set(xlabel="Predicted confidence", ylabel="Samples", xlim=(0, 1))
    figure.tight_layout()
    figure.savefig(output_dir / "reliability_diagram.png", dpi=160, bbox_inches="tight")
    plt.close(figure)


def evaluate_predictions(
    y_true, y_pred, probabilities, labels: Sequence[str], output_dir: str | Path,
    ordinal: bool = False, n_bins: int = 10,
) -> dict:
    """Compute and persist reports for one fixed model/target/evaluation slice.

    ``y_pred`` must equal ``labels[argmax(probabilities)]`` so accuracy/F1 and
    calibration assess the same deployed decision rule. Empty slices return
    explicit null metrics and do not create plots or reports.
    """
    truth, probabilities, labels = _validated_arrays(y_true, probabilities, labels)
    predicted = np.asarray(list(y_pred), dtype=object)
    if predicted.ndim != 1 or len(predicted) != len(truth):
        raise ValueError("y_pred must be one-dimensional with the same sample count as y_true")
    unknown = set(predicted) - set(labels)
    if unknown:
        raise ValueError(f"Predictions contain unknown labels: {sorted(unknown)}")
    calibration = expected_calibration_error(truth, probabilities, labels, n_bins=n_bins)
    result = {
        "status": "ok" if len(truth) else "skipped_empty", "n": len(truth),
        "labels": labels, "accuracy": None, "macro_f1": None, "weighted_f1": None,
        "ece": calibration["ece"], "ece_bins": n_bins,
        "quadratic_weighted_kappa": None, "qwk_undefined_reason": None,
        "files": {},
    }
    if not len(truth):
        if ordinal:
            result["qwk_undefined_reason"] = "No evaluation rows; quadratic weighted kappa is undefined."
        return result
    probability_predictions = np.asarray(labels, dtype=object)[probabilities.argmax(axis=1)]
    if not np.array_equal(predicted, probability_predictions):
        raise ValueError("y_pred must match probability argmax in supplied labels order")
    result.update(
        accuracy=float(accuracy_score(truth, predicted)),
        macro_f1=float(f1_score(truth, predicted, labels=labels, average="macro", zero_division=0)),
        weighted_f1=float(f1_score(truth, predicted, labels=labels, average="weighted", zero_division=0)),
    )
    if ordinal:
        result["quadratic_weighted_kappa"], result["qwk_undefined_reason"] = quadratic_weighted_kappa(truth, predicted, labels)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    report = classification_report(truth, predicted, labels=labels, zero_division=0, output_dict=True)
    (output_dir / "classification_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    (output_dir / "classification_report.txt").write_text(
        classification_report(truth, predicted, labels=labels, zero_division=0, digits=4), encoding="utf-8"
    )
    matrix = confusion_matrix(truth, predicted, labels=labels)
    with (output_dir / "confusion_matrix.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["true_label\\predicted_label", *labels])
        writer.writerows([label, *map(int, row)] for label, row in zip(labels, matrix))
    with (output_dir / "reliability_bins.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=("bin", "lower", "upper", "count", "accuracy", "confidence"))
        writer.writeheader()
        writer.writerows(calibration["bins"])
    _plots(matrix, labels, calibration, output_dir)
    result["files"] = {name: str(output_dir / name) for name in (
        "classification_report.json", "classification_report.txt", "confusion_matrix.csv",
        "confusion_matrix.png", "reliability_diagram.png", "reliability_bins.csv",
    )}
    return result
