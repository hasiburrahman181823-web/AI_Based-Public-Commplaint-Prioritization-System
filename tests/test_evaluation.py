"""Behavioral tests for calibrated evaluation and paired Banglish noise."""

import csv
import json
import random
import warnings

import numpy as np
import pytest

from ml.noise import add_banglish_noise, perturb_texts
from ml_training.evaluation import (
    evaluate_predictions,
    expected_calibration_error,
    quadratic_weighted_kappa,
)


def test_ece_known_value_and_confidence_one_boundary():
    report = expected_calibration_error(["a", "b", "b", "b"], [[1, 0], [0.1, 0.9], [0.8, 0.2], [0.4, 0.6]], ["a", "b"], n_bins=2)
    assert report["ece"] == pytest.approx(abs(0.75 - 0.825))
    assert sum(item["count"] for item in report["bins"]) == 4
    assert report["bins"][-1]["count"] == 4


def test_ece_perfect_and_empty_bins():
    report = expected_calibration_error(["a", "b"], [[1, 0], [0, 1]], ["a", "b"])
    assert report["ece"] == 0
    assert report["bins"][0]["count"] == 0
    assert report["bins"][0]["accuracy"] is None


@pytest.mark.parametrize("probabilities,fragment", [
    ([[0.4, 0.4]], "sum to 1"),
    ([[float("nan"), 0.5]], "finite"),
    ([[-0.1, 1.1]], "between 0 and 1"),
    ([[1]], "shape"),
])
def test_invalid_probability_values_are_rejected(probabilities, fragment):
    with pytest.raises(ValueError, match=fragment):
        expected_calibration_error(["a"], probabilities, ["a", "b"])


def test_unknown_labels_and_invalid_bins_are_rejected():
    with pytest.raises(ValueError, match="outside"):
        expected_calibration_error(["new"], [[1, 0]], ["a", "b"])
    for invalid in (0, -1, 2.5, True):
        with pytest.raises(ValueError, match="n_bins"):
            expected_calibration_error(["a"], [[1, 0]], ["a", "b"], invalid)


def test_ordinal_kappa_uses_configured_order():
    ordered = ["routine", "moderate", "urgent"]
    truth = ["routine", "moderate", "urgent"]
    predicted = ["routine", "urgent", "moderate"]
    correct_order, _ = quadratic_weighted_kappa(truth, predicted, ordered)
    changed_order, _ = quadratic_weighted_kappa(truth, predicted, ["moderate", "routine", "urgent"])
    assert correct_order == pytest.approx(0.5)
    assert correct_order != changed_order


def test_undefined_kappa_is_none_not_nan():
    result, reason = quadratic_weighted_kappa(["low", "low"], ["low", "low"], ["low", "high"])
    assert result is None and "zero" in reason
    assert quadratic_weighted_kappa([], [], ["low", "high"])[0] is None


def test_ordinal_kappa_perfect_and_opposite():
    labels = ["low", "high"]
    assert quadratic_weighted_kappa(["low", "high"], ["low", "high"], labels)[0] == 1
    assert quadratic_weighted_kappa(["low", "high"], ["high", "low"], labels)[0] == -1


def test_empty_slice_is_explicit_and_does_not_create_fake_plots(tmp_path):
    destination = tmp_path / "missing_language"
    result = evaluate_predictions([], [], np.empty((0, 2)), ["a", "b"], destination, ordinal=True)
    assert result["status"] == "skipped_empty"
    assert result["n"] == 0
    assert result["macro_f1"] is None and result["ece"] is None
    assert result["quadratic_weighted_kappa"] is None
    assert result["files"] == {}
    assert not destination.exists()


def test_reports_and_real_png_files_with_fixed_label_universe(tmp_path):
    result = evaluate_predictions(["a", "a"], ["a", "b"], [[0.9, 0.1], [0.4, 0.6]], ["a", "b"], tmp_path, ordinal=True)
    assert result["macro_f1"] == pytest.approx(1 / 3)
    assert result["n"] == 2
    report = json.loads((tmp_path / "classification_report.json").read_text(encoding="utf-8"))
    assert report["b"]["support"] == 0
    for name in ("confusion_matrix.png", "reliability_diagram.png"):
        assert (tmp_path / name).read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
    with (tmp_path / "reliability_bins.csv").open(encoding="utf-8", newline="") as handle:
        assert sum(int(row["count"]) for row in csv.DictReader(handle)) == 2


def test_prediction_probability_mismatch_is_rejected(tmp_path):
    with pytest.raises(ValueError, match="argmax"):
        evaluate_predictions(["a"], ["b"], [[1, 0]], ["a", "b"], tmp_path)


def test_bangla_plot_labels_use_installed_font_without_missing_glyphs(tmp_path):
    from ml_training.evaluation import _plot_font_families

    if _plot_font_families() == ["DejaVu Sans"]:
        pytest.skip("No Bangla-capable plotting font is installed on this machine")
    import matplotlib
    previous_family = list(matplotlib.rcParams["font.family"])
    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter("always")
        evaluate_predictions(["নিম্ন", "উচ্চ"], ["নিম্ন", "উচ্চ"], [[1, 0], [0, 1]], ["নিম্ন", "উচ্চ"], tmp_path)
    assert not any("Glyph" in str(item.message) and "missing" in str(item.message) for item in captured)
    assert list(matplotlib.rcParams["font.family"]) == previous_family


def test_noise_is_seeded_and_preserves_non_ascii_tokens_numbers_urls_and_spacing():
    source = "pani nei  ১২ 123 abc123 পানি https://example.com/pani?x=123 www.pani.com!"
    left = add_banglish_noise(source, random.Random(23), rate=1)
    assert left == add_banglish_noise(source, random.Random(23), rate=1)
    assert left != source
    for token in ("১২", "123", "abc123", "পানি", "https://example.com/pani?x=123", "www.pani.com!"):
        assert token in left
    assert "  " in left


def test_noise_zero_rate_and_sequence_repeatability():
    assert add_banglish_noise("PANY NEI", random.Random(3), rate=0) == "PANY NEI"
    data = ["pani nei", "bidyut nai"]
    assert perturb_texts(data, seed=7, rate=1) == perturb_texts(data, seed=7, rate=1)


@pytest.mark.parametrize("rate", [-0.01, 1.1, float("nan"), True, "0.5"])
def test_invalid_noise_rate_is_rejected(rate):
    with pytest.raises(ValueError, match="rate"):
        add_banglish_noise("pani", random.Random(0), rate)
