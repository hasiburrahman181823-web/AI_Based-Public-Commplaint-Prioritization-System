"""Gold-label attachment preserves candidates and never performs annotation voting."""

from __future__ import annotations

import csv
import json

import pandas as pd
import pytest

from ml_training.scripts.apply_duplicate_labels import apply_labels, main


PAIR_COLUMNS = ("pair_id", "id_a", "id_b", "original_id_a", "original_id_b", "split", "label", "text_a", "text_b")


def write_csv(path, columns, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    return path


@pytest.fixture
def dataset(tmp_path):
    source, output = tmp_path / "candidate_pairs", tmp_path / "gold_pairs"
    rows, labels = {}, []
    for split in ("train", "val", "test"):
        rows[split] = [
            {"pair_id": f"{split}-pair-{index}", "id_a": f"{split}-a-{index}", "id_b": f"{split}-b-{index}",
             "original_id_a": f"{split}-g{index}", "original_id_b": f"{split}-g{index}",
             "split": split, "label": str(index), "text_a": f"পানি নেই {index}", "text_b": f"No water {index}"}
            for index in (0, 1)
        ]
        write_csv(source / split / "pairs.csv", PAIR_COLUMNS, rows[split])
        labels.extend({"pair_id": row["pair_id"], "gold_label": str(1 - int(row["label"]))} for row in rows[split])
    labels_path = write_csv(tmp_path / "adjudicated.csv", ("pair_id", "gold_label"), labels)
    return {"source": source, "output": output, "labels_path": labels_path, "labels": labels, "rows": rows}


def rewrite_labels(dataset, columns=("pair_id", "gold_label")):
    write_csv(dataset["labels_path"], columns, dataset["labels"])


def test_merge_keeps_candidate_labels_metadata_and_all_splits(dataset):
    summary = apply_labels(dataset["source"], dataset["labels_path"], dataset["output"])
    assert summary["label_status"] == "explicit_adjudicated_input"
    assert summary["pairs"] == 6
    assert summary["splits"] == {"train": 2, "val": 2, "test": 2}
    for split in ("train", "val", "test"):
        source = pd.read_csv(dataset["source"] / split / "pairs.csv", dtype=str, keep_default_na=False)
        gold = pd.read_csv(dataset["output"] / split / "pairs.csv", dtype=str, keep_default_na=False)
        pd.testing.assert_frame_equal(gold.loc[:, list(PAIR_COLUMNS)], source)
        assert gold["gold_label"].tolist() == ["1", "0"]
        assert "gold_label" not in source
    combined = pd.read_csv(dataset["output"] / "pairs.csv", dtype=str, keep_default_na=False)
    assert len(combined) == 6 and combined["pair_id"].is_unique
    assert json.loads((dataset["output"] / "label_manifest.json").read_text()) == summary


def test_incomplete_gold_coverage_fails_without_publishing_any_output(dataset):
    dataset["labels"].pop()
    rewrite_labels(dataset)
    with pytest.raises(ValueError, match="lack adjudicated labels"):
        apply_labels(dataset["source"], dataset["labels_path"], dataset["output"])
    assert not dataset["output"].exists()


def test_unknown_adjudicated_id_fails_without_silent_extra_labels(dataset):
    dataset["labels"].append({"pair_id": "unknown-incident", "gold_label": "1"})
    rewrite_labels(dataset)
    with pytest.raises(ValueError, match="unknown pair IDs"):
        apply_labels(dataset["source"], dataset["labels_path"], dataset["output"])
    assert not dataset["output"].exists()


@pytest.mark.parametrize("decision", ["0", "1"])
def test_repeated_annotation_rows_are_not_automatically_voted(dataset, decision):
    dataset["labels"].append(dict(dataset["labels"][0], gold_label=decision))
    rewrite_labels(dataset)
    with pytest.raises(ValueError, match="resolve conflicts manually"):
        apply_labels(dataset["source"], dataset["labels_path"], dataset["output"])


@pytest.mark.parametrize("decision", ["", "skip", "2", "-1", "true", "1.0"])
def test_gold_decisions_must_be_explicit_binary_values(dataset, decision):
    dataset["labels"][0]["gold_label"] = decision
    rewrite_labels(dataset)
    with pytest.raises(ValueError, match="explicitly 0 or 1"):
        apply_labels(dataset["source"], dataset["labels_path"], dataset["output"])


def test_existing_outputs_require_intentional_overwrite(dataset):
    apply_labels(dataset["source"], dataset["labels_path"], dataset["output"])
    with pytest.raises(ValueError, match="output exists"):
        apply_labels(dataset["source"], dataset["labels_path"], dataset["output"])
    summary = apply_labels(dataset["source"], dataset["labels_path"], dataset["output"], overwrite=True)
    assert summary["pairs"] == 6


@pytest.mark.parametrize("choice", ["same", "nested", "ancestor", "contains_labels"])
def test_input_output_path_isolation(dataset, choice):
    if choice == "same":
        output = dataset["source"]
    elif choice == "nested":
        output = dataset["source"] / "gold"
    elif choice == "ancestor":
        output = dataset["source"].parent
    else:
        output = dataset["labels_path"].parent
        # Place the source outside that ancestor to exercise labels isolation.
        external = dataset["source"].parent / "outside" / "pairs"
        for split in ("train", "val", "test"):
            write_csv(external / split / "pairs.csv", PAIR_COLUMNS, dataset["rows"][split])
        # The label input is now placed in an otherwise unrelated output tree.
        labels = dataset["source"].parent / "outputs" / "adjudicated.csv"
        write_csv(labels, ("pair_id", "gold_label"), dataset["labels"])
        with pytest.raises(ValueError, match="labels input"):
            apply_labels(external, labels, labels.parent)
        return
    with pytest.raises(ValueError, match="non-nested"):
        apply_labels(dataset["source"], dataset["labels_path"], output)


def test_conflicting_source_gold_must_be_reconciled_even_with_overwrite(dataset):
    for row in dataset["rows"]["test"]:
        row["gold_label"] = row["label"]
    write_csv(dataset["source"] / "test" / "pairs.csv", (*PAIR_COLUMNS, "gold_label"), dataset["rows"]["test"])
    with pytest.raises(ValueError, match="differing gold labels"):
        apply_labels(dataset["source"], dataset["labels_path"], dataset["output"], overwrite=True)


def test_duplicate_source_ids_and_wrong_split_are_rejected(dataset):
    dataset["rows"]["test"][0]["pair_id"] = dataset["rows"]["train"][0]["pair_id"]
    write_csv(dataset["source"] / "test" / "pairs.csv", PAIR_COLUMNS, dataset["rows"]["test"])
    with pytest.raises(ValueError, match="globally unique"):
        apply_labels(dataset["source"], dataset["labels_path"], dataset["output"])
    dataset["rows"]["test"][0]["pair_id"] = "test-pair-0"
    dataset["rows"]["test"][0]["split"] = "val"
    write_csv(dataset["source"] / "test" / "pairs.csv", PAIR_COLUMNS, dataset["rows"]["test"])
    with pytest.raises(ValueError, match="split column disagrees"):
        apply_labels(dataset["source"], dataset["labels_path"], dataset["output"])


def test_malformed_labels_csv_rejected_and_cli_works(dataset, capsys):
    assert main(["--pairs-dir", str(dataset["source"]), "--labels", str(dataset["labels_path"]),
                 "--output-dir", str(dataset["output"])]) == 0
    assert json.loads(capsys.readouterr().out)["label_status"] == "explicit_adjudicated_input"
    dataset["labels_path"].write_text('pair_id,gold_label\n"unfinished\n', encoding="utf-8")
    assert main(["--pairs-dir", str(dataset["source"]), "--labels", str(dataset["labels_path"]),
                 "--output-dir", str(dataset["output"]), "--overwrite"]) == 2
    assert "ERROR:" in capsys.readouterr().err
