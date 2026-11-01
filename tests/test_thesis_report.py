"""Evidence collection tests; numeric fixtures below are not research results."""

import hashlib
import json
from pathlib import Path

import pytest

import make_thesis_report as collector
from make_thesis_report import main, make_report


@pytest.fixture
def tree(tmp_path):
    results = tmp_path / "results"
    results.mkdir()
    return results, tmp_path / "docs" / "thesis_results.md"


def write(path: Path, value: str):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(value, encoding="utf-8")
    return path


def test_empty_results_never_become_a_metric_or_support_count(tree):
    source, output = tree
    write(source / ".gitkeep", "")
    result = make_report(source, output)
    report = output.read_text(encoding="utf-8")
    assert "No evaluated research results are saved" in report
    assert "expert-label counts" in report and "unknown counts stay unknown" in report
    assert "Awaiting experiments" in report
    assert result["saved_artifacts"] == 0 and result["research_measurements_computed"] is False
    assert "0.000" not in report and "accuracy: 0" not in report


def test_actual_manifest_shapes_are_preserved_without_aggregation(tree):
    source, output = tree
    payload = {"experiment_kind": "calibrated_classifiers", "runs": [
        {"target": "severity", "model": "char_svm", "cohorts": {
            "bangla_test": {"n": 23, "macro_f1": 0.71, "quadratic_weighted_kappa": None, "ece": 0.09},
            "natural_test": {"n": 0, "macro_f1": None, "status": "skipped"}}}]}
    path = write(source / "classifiers" / "metrics.json", json.dumps(payload))
    make_report(source, output)
    report = output.read_text(encoding="utf-8")
    assert "0.71" in report and "0.09" in report and "23" in report
    assert "undefined (null)" in report and "natural\\_test" in report
    assert hashlib.sha256(path.read_bytes()).hexdigest() in report
    assert "[Source file](../results/classifiers/metrics.json)" in report
    assert "No languages are pooled" in report
    assert "best model" not in report.lower()


def test_duplicate_priority_fairness_and_policy_metrics_are_collected(tree):
    source, output = tree
    files = {
        "duplicates/P2/test/metrics.json": {"f1": 0.62, "pr_auc": 0.67, "by_pair_type": {"hard_negative": {"n": 4}}},
        "ranking/metrics.json": {"experiment_kind": "expert_priority_ranking", "ranker_status": "fallback_rules", "n_training_queries": 1,
                                 "comparisons": {"test": {"ltr": {"ndcg_at_5": None, "spearman": None}}}},
        "fairness/fairness.json": {"n": 5, "groups": [{"dimension": "language", "group": "banglish", "error_rate": None}]},
        "policy/selection.json": {"kind": "priority_review_policy", "threshold": 0.83, "n_validation_original_groups": 8},
    }
    for name, payload in files.items():
        write(source / name, json.dumps(payload))
    make_report(source, output)
    report = output.read_text(encoding="utf-8")
    for value in ("0.62", "0.67", "0.83", "hard\\_negative", "fallback\\_rules", "banglish"):
        assert value in report
    assert report.count("undefined (null)") >= 3


def test_json_precision_and_csv_strings_are_not_rounded_or_reinterpreted(tree):
    source, output = tree
    write(source / "metrics.json", '{"f1":0.12345678901234567890,"undefined":null}')
    write(source / "results_table.csv", 'model,n,f1,notes\nchar_svm,007,0.1200,\n')
    make_report(source, output)
    report = output.read_text(encoding="utf-8")
    assert "0.12345678901234567890" in report
    assert "007" in report and "0.1200" in report
    assert "Empty cells remain empty, not zero" in report


def test_tables_images_and_other_artifacts_are_all_inventoried(tree):
    source, output = tree
    write(source / "ablation_table.csv", "model,f1\nP1,0.4\n")
    write(source / "groups.tsv", "group\tn\nbn\t3\n")
    write(source / "summary.md", "<script>never inline this</script>")
    write(source / "active.svg", "<svg onload='bad()'></svg>")
    write(source / "notes.txt", "saved report")
    (source / "confusion matrix (bn).png").write_bytes(b"test-image-bytes")
    result = make_report(source, output)
    report = output.read_text(encoding="utf-8")
    assert result["saved_artifacts"] == 6
    assert "confusion%20matrix%20%28bn%29.png" in report
    assert "![Saved figure" in report
    assert "<script>" not in report and "<svg onload" not in report
    assert "unsupported preview format" in report
    for path in source.iterdir():
        assert hashlib.sha256(path.read_bytes()).hexdigest() in report


def test_untrusted_values_cannot_create_html_links_or_extra_table_columns(tree):
    source, output = tree
    write(source / "metrics.json", json.dumps({"warning": "<script>x</script>|![external](https://invalid.example/a)\nrow"}))
    write(source / "table.csv", 'model,notes\nP1,"<b>x</b>|[click](https://invalid.example)"\n')
    make_report(source, output)
    report = output.read_text(encoding="utf-8")
    assert "<script>" not in report and "<b>" not in report
    assert "&lt;script&gt;" in report and "\\|" in report
    assert "!\\[external\\]" in report and "↵" in report


def test_private_row_level_tables_are_linked_but_not_copied(tree):
    source, output = tree
    write(source / "predictions.csv", "id,text,category\n1,PRIVATE_COMPLAINT_TEXT,Water\n")
    write(source / "metrics.json", json.dumps({"contact": "PRIVATE_CONTACT", "nested": {"text_a": "PRIVATE_TEXT"}, "f1": 0.2}))
    make_report(source, output)
    report = output.read_text(encoding="utf-8")
    assert "Row-level/private-data table: 1 saved records" in report
    assert "PRIVATE_COMPLAINT_TEXT" not in report and "PRIVATE_CONTACT" not in report and "PRIVATE_TEXT" not in report
    assert "row-level/private field omitted" in report and "0.2" in report


@pytest.mark.parametrize("metadata, expected", [
    ({"synthetic": True}, "DECLARED SYNTHETIC/FIXTURE"),
    ({"experiment_kind": "synthetic_smoke"}, "DECLARED SYNTHETIC/FIXTURE"),
    ({"label_source": "weak_construction"}, "DECLARED WEAK/NON-GOLD LABELS"),
    ({"is_incident_gold": False}, "DECLARED WEAK/NON-GOLD LABELS"),
    ({"macro_f1": 0.5}, "No top-level provenance declaration"),
])
def test_provenance_is_explicit_and_never_assumed_gold(tree, metadata, expected):
    source, output = tree
    write(source / "metrics.json", json.dumps(metadata))
    make_report(source, output)
    assert expected in output.read_text(encoding="utf-8")


def test_preview_limits_keep_complete_inventory_and_explicit_truncation(tree):
    source, output = tree
    write(source / "table.csv", "model,f1,n\na,0.1,2\nb,0.2,3\nc,0.3,4\n")
    write(source / "metrics.json", json.dumps({"a": 1, "b": 2, "c": 3}))
    make_report(source, output, max_table_rows=1, max_columns=2, max_json_rows=1)
    report = output.read_text(encoding="utf-8")
    assert "1 records and 2 columns" in report and "1 of 3 scalar/terminal fields" in report
    assert "0.3" not in report
    assert all(hashlib.sha256(path.read_bytes()).hexdigest() in report for path in source.iterdir())


def test_oversized_and_malformed_files_remain_linked_without_replacement_scores(tree):
    source, output = tree
    write(source / "large.json", '{"f1": 0.9, "padding": "' + "x" * 200 + '"}')
    write(source / "broken.json", "{broken")
    write(source / "broken.csv", "a,b\n1\n")
    result = make_report(source, output, max_preview_bytes=100)
    report = output.read_text(encoding="utf-8")
    assert "file exceeds preview byte limit" in report
    assert "could not safely parse" in report and len(result["warnings"]) == 2
    assert "0.9" not in report


@pytest.mark.parametrize("payload", ['{"f1":NaN}', '{"f1":Infinity}', '{"n":1,"n":2}'])
def test_invalid_or_duplicate_json_values_are_not_silently_accepted(tree, payload):
    source, output = tree
    write(source / "metrics.json", payload)
    result = make_report(source, output)
    assert result["warnings"] and "could not safely parse" in output.read_text(encoding="utf-8")


def test_existing_report_requires_deliberate_replacement_and_is_deterministic(tree):
    source, output = tree
    write(source / "metrics.json", '{"n":3}')
    make_report(source, output)
    original = output.read_bytes()
    with pytest.raises(ValueError, match="--overwrite"):
        make_report(source, output)
    assert output.read_bytes() == original
    make_report(source, output, overwrite=True)
    assert output.read_bytes() == original


def test_output_cannot_contaminate_or_overwrite_source_evidence(tree):
    source, _ = tree
    path = write(source / "nested" / "result.md", "unchanged evidence")
    with pytest.raises(ValueError, match="outside"):
        make_report(source, path, overwrite=True)
    assert path.read_text(encoding="utf-8") == "unchanged evidence"


def test_symlink_to_external_evidence_is_not_followed(tree, tmp_path, monkeypatch):
    source, output = tree
    target = write(tmp_path / "external.json", '{"secret":"DO_NOT_READ"}')
    link = source / "outside.json"
    try:
        link.symlink_to(target)
    except OSError:
        # Windows accounts may lack symlink privilege. Exercise the same link
        # exclusion boundary with simulated link metadata; Linux CI uses a real
        # symlink above, without requiring host permission changes.
        write(link, "DO_NOT_READ")
        original = collector._linklike
        monkeypatch.setattr(collector, "_linklike", lambda path: path == link or original(path))
    result = make_report(source, output)
    report = output.read_text(encoding="utf-8")
    assert result["excluded_artifacts"] == 1 and "DO_NOT_READ" not in report
    assert "excluded link/outside source root" in report and "No link followed" in report


def test_cli_missing_source_is_a_clear_failure_and_leaves_no_report(tmp_path, capsys):
    output = tmp_path / "report.md"
    assert main(["--results-dir", str(tmp_path / "missing"), "--output", str(output)]) == 2
    assert "Results directory must exist" in capsys.readouterr().err
    assert not output.exists()
