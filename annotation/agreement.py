"""Cohen's kappa without an external statistics service or sklearn dependency."""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence


def cohen_kappa(
    a: Sequence[int],
    b: Sequence[int],
    labels: Sequence[int] = (0, 1),
    weighting: str = "unweighted",
) -> dict:
    """Compute kappa on aligned, independently annotated items.

    Quadratic weighting uses squared ordinal label distance. No overlap and
    zero expected disagreement are undefined, represented by JSON-safe None.
    ``exact_agreement`` always refers to identical labels, even when weighted.
    """
    if len(a) != len(b):
        raise ValueError("Annotator label sequences must have the same length")
    if weighting not in ("unweighted", "quadratic"):
        raise ValueError("weighting must be unweighted or quadratic")
    if len(labels) < 2 or len(set(labels)) != len(labels):
        raise ValueError("labels must contain at least two distinct values in ordinal order")
    if any(value not in labels for value in (*a, *b)):
        raise ValueError("Annotation values must belong to labels")
    count = len(a)
    matrix = [[sum(x == left and y == right for x, y in zip(a, b)) for right in labels] for left in labels]
    result = {
        "n_overlap": count,
        "labels": list(labels),
        "weighting": weighting,
        "confusion_matrix": matrix,
        "exact_agreement": None,
        "kappa": None,
        "undefined_reason": None,
    }
    if count == 0:
        result["undefined_reason"] = "No items labeled by both annotators (skipped items excluded)."
        return result
    result["exact_agreement"] = sum(left == right for left, right in zip(a, b)) / count
    counts_a, counts_b = Counter(a), Counter(b)
    indices = {label: i for i, label in enumerate(labels)}

    def disagreement(left: int, right: int) -> float:
        if weighting == "unweighted":
            return float(left != right)
        return ((indices[left] - indices[right]) / (len(labels) - 1)) ** 2

    observed = sum(disagreement(left, right) for left, right in zip(a, b)) / count
    expected = sum(
        disagreement(left, right) * counts_a[left] * counts_b[right] / count**2
        for left in labels for right in labels
    )
    if expected <= 1e-15:
        result["undefined_reason"] = "Expected disagreement is zero; kappa is undefined for these constant labels."
    else:
        result["kappa"] = 1 - observed / expected
    return result
