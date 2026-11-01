"""Numerical and CLI checks that do not load or download neural model weights."""

import math

import numpy as np
import pytest

from ml_training.scripts.finetune_xlmr import (
    build_parser,
    fingerprint_file,
    main,
    softmax_with_temperature,
    temperature_nll,
    validate_class_coverage,
    validate_logits,
    validate_output_paths,
)


def test_softmax_is_stable_normalized_and_temperature_changes_confidence():
    logits = np.array([[1000, 1001, 999], [-1000, -1001, -999]], dtype=float)
    cold = softmax_with_temperature(logits)
    warm = softmax_with_temperature(logits, 3)
    assert np.isfinite(cold).all()
    assert np.allclose(cold.sum(axis=1), 1)
    assert np.allclose(warm.sum(axis=1), 1)
    assert np.array_equal(cold.argmax(axis=1), warm.argmax(axis=1))
    assert (warm.max(axis=1) < cold.max(axis=1)).all()


def test_softmax_uniform_logits_and_empty_cohort():
    assert np.allclose(softmax_with_temperature([[5, 5]]), [[0.5, 0.5]])
    assert softmax_with_temperature(np.empty((0, 3))).shape == (0, 3)


def test_temperature_nll_matches_known_cross_entropy_and_handles_extreme_logits():
    assert temperature_nll([[0, 0], [0, 0]], [0, 1]) == pytest.approx(math.log(2))
    assert temperature_nll([[1000, -1000]], [1]) == pytest.approx(2000)
    assert temperature_nll([[8, 0]], [1], temperature=2) < temperature_nll([[8, 0]], [1])


@pytest.mark.parametrize("temperature", [0, -1, float("nan"), float("inf")])
def test_invalid_temperature_rejected(temperature):
    with pytest.raises(ValueError, match="temperature"):
        softmax_with_temperature([[1, 2]], temperature)
    with pytest.raises(ValueError, match="temperature"):
        temperature_nll([[1, 2]], [0], temperature)


@pytest.mark.parametrize("logits", [[1, 2], [[1]], [[float("nan"), 0]], [[0, float("inf")]]])
def test_invalid_logits_rejected(logits):
    with pytest.raises(ValueError, match="logits"):
        validate_logits(logits)


@pytest.mark.parametrize("labels", [[], [2], [-1], [0.5], [[0]], [0, 1]])
def test_invalid_calibration_labels_rejected(labels):
    with pytest.raises(ValueError, match="labels"):
        validate_logits([[1, 2]], labels)


def test_defaults_are_offline_and_output_directories_remain_separate():
    args = build_parser().parse_args(["--target", "category"])
    assert args.allow_download is False
    assert args.model == "FacebookAI/xlm-roberta-base"
    assert args.results_dir is None and args.artifacts_dir is None
    assert args.epochs > 0


def test_help_does_not_require_torch_or_transformers(capsys):
    with pytest.raises(SystemExit) as exit_info:
        main(["--help"])
    assert exit_info.value.code == 0
    assert "--allow-download" in capsys.readouterr().out


def test_fingerprint_streams_exact_file_content(tmp_path):
    import hashlib

    path = tmp_path / "input.csv"
    content = "পানি নেই\n".encode("utf-8")
    path.write_bytes(content)
    result = fingerprint_file(path)
    assert result["sha256"] == hashlib.sha256(content).hexdigest()
    assert result["bytes"] == len(content)
    assert result["path"] == str(path.resolve())


@pytest.mark.parametrize("kind", ["same_outputs", "results_in_artifacts", "artifacts_in_results", "artifacts_in_input", "results_ancestor_input"])
def test_output_tree_isolation_rejects_nested_or_shared_paths(tmp_path, kind):
    source, artifacts, results = tmp_path / "input", tmp_path / "artifacts", tmp_path / "results"
    if kind == "same_outputs":
        results = artifacts
    elif kind == "results_in_artifacts":
        results = artifacts / "reports"
    elif kind == "artifacts_in_results":
        artifacts = results / "models"
    elif kind == "artifacts_in_input":
        artifacts = source / "models"
    else:
        source = results / "source"
    with pytest.raises(ValueError, match="non-nested"):
        validate_output_paths(source, artifacts, results, overwrite=True)


@pytest.mark.parametrize("parent", ["artifacts", "results"])
def test_natural_input_cannot_be_within_outputs(tmp_path, parent):
    with pytest.raises(ValueError, match="Natural-test input"):
        validate_output_paths(tmp_path / "source", tmp_path / "artifacts", tmp_path / "results",
                              tmp_path / parent / "natural.csv", overwrite=True)


def test_output_file_rejected_and_existing_outputs_require_overwrite(tmp_path):
    artifacts, results = tmp_path / "artifacts", tmp_path / "results"
    artifacts.write_text("not a directory")
    with pytest.raises(ValueError, match="is a file"):
        validate_output_paths(tmp_path / "source", artifacts, results)
    artifacts.unlink()
    artifacts.mkdir()
    (artifacts / ".gitkeep").touch()
    validate_output_paths(tmp_path / "source", artifacts, results)
    (artifacts / "model.joblib").touch()
    with pytest.raises(ValueError, match="not empty"):
        validate_output_paths(tmp_path / "source", artifacts, results)
    validate_output_paths(tmp_path / "source", artifacts, results, overwrite=True)


@pytest.mark.parametrize("training,validation,configured,fragment", [
    (["low"], ["low"], ["low", "high"], "configured"),
    (["low", "high"], ["low"], ["low", "high"], "Missing"),
    (["low", "high"], ["low", "other"], ["low", "high"], "unseen"),
])
def test_target_class_coverage_is_explicit(training, validation, configured, fragment):
    with pytest.raises(ValueError, match=fragment):
        validate_class_coverage(training, validation, configured)


def test_matching_class_coverage_and_fresh_isolated_paths_are_accepted(tmp_path):
    validate_class_coverage(["high", "low", "high"], ["low", "high"], ["low", "high"])
    validate_output_paths(tmp_path / "source", tmp_path / "artifacts", tmp_path / "results",
                          tmp_path / "natural.csv")


def test_invalid_output_paths_fail_before_neural_dependencies_are_imported(tmp_path, capsys):
    source = tmp_path / "source"
    assert main(["--target", "category", "--input-dir", str(source),
                 "--artifacts-dir", str(source / "models"),
                 "--results-dir", str(tmp_path / "results")]) == 2
    assert "non-nested" in capsys.readouterr().err
