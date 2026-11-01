"""Independent annotation, statistical agreement, and persistent API behavior."""

from __future__ import annotations

import csv
import io

import pytest
from fastapi.testclient import TestClient

from annotation.agreement import cohen_kappa
from annotation.app import PAIR_COLUMNS, build_items, create_app
from ml_training.config import SCHEMA_COLUMNS


@pytest.mark.parametrize(
    "left,right,expected",
    [
        ([0, 0, 1, 1], [0, 0, 1, 1], 1),
        ([0, 0, 1, 1], [0, 1, 0, 1], 0),
        ([0, 0, 1, 1], [1, 1, 0, 0], -1),
    ],
)
def test_unweighted_kappa_known_values(left, right, expected):
    report = cohen_kappa(left, right)
    assert report["kappa"] == pytest.approx(expected)
    assert report["n_overlap"] == 4
    assert report["weighting"] == "unweighted"
    assert report["undefined_reason"] is None
    assert sum(map(sum, report["confusion_matrix"])) == 4


def test_quadratic_weighting_gives_adjacent_priority_errors_less_penalty():
    left, right = [1, 2, 3, 4, 5], [1, 2, 3, 5, 4]
    weighted = cohen_kappa(left, right, (1, 2, 3, 4, 5), "quadratic")
    unweighted = cohen_kappa(left, right, (1, 2, 3, 4, 5))
    assert weighted["kappa"] == pytest.approx(0.9)
    assert unweighted["kappa"] == pytest.approx(0.5)
    assert weighted["exact_agreement"] == pytest.approx(0.6)


@pytest.mark.parametrize("left,right,fragment", [([], [], "No items"), ([1, 1], [1, 1], "constant")])
def test_undefined_kappa_returns_none_with_explanation(left, right, fragment):
    report = cohen_kappa(left, right)
    assert report["kappa"] is None
    assert fragment in report["undefined_reason"]


@pytest.mark.parametrize(
    "left,right,labels,weighting,fragment",
    [
        ([0], [], (0, 1), "unweighted", "same length"),
        ([2], [0], (0, 1), "unweighted", "belong to labels"),
        ([0], [0], (0, 0), "unweighted", "distinct values"),
        ([0], [0], (0,), "unweighted", "at least two"),
        ([0], [0], (0, 1), "linear", "weighting"),
    ],
)
def test_invalid_agreement_inputs_are_rejected(left, right, labels, weighting, fragment):
    with pytest.raises(ValueError, match=fragment):
        cohen_kappa(left, right, labels, weighting)


def write_csv(path, columns, records):
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(records)
    return path


@pytest.fixture
def dataset(tmp_path):
    pairs = []
    for index in range(4):
        pairs.append({
            "text_a": f"পানি নেই {index}", "text_b": f"No water, incident {index}",
            "loc_a": "Road 7", "loc_b": "Road 7", "time_a": "2026-10-29T08:00:00Z",
            "time_b": "2026-10-29T08:00:00Z", "label": str(index % 2),
            "pair_type": "positive_cross_language", "pair_id": f"p{index}",
            "split": "train", "id_a": f"source-{index}", "id_b": f"translation-{index}",
            "original_id_a": f"g{index}", "original_id_b": f"g{index}",
        })
    complaints = [{
        "id": f"c{index}", "original_id": f"g{index}", "text": f"Individual complaint {index}",
        "lang": "en", "category": "water", "severity": "do-not-anchor-reviewer",
        "source_type": "source", "location": "Road 7", "timestamp": "2026-10-29T08:00:00Z",
    } for index in range(4)]
    pairs_path = write_csv(tmp_path / "pairs.csv", PAIR_COLUMNS, pairs)
    complaints_path = write_csv(tmp_path / "train.csv", SCHEMA_COLUMNS, complaints)
    return {"pairs_path": pairs_path, "complaints_path": complaints_path,
            "db_path": tmp_path / "labels.sqlite3", "pairs": pairs, "complaints": complaints}


@pytest.fixture
def client(dataset):
    with TestClient(create_app(dataset["pairs_path"], [dataset["complaints_path"]], dataset["db_path"])) as test_client:
        yield test_client


def save(client, item_id, annotator="alice", label=1, reason="Same road and event onset", **extra):
    body = {"item_id": item_id, "annotator": annotator, "label": label, "reason": reason}
    body.update(extra)
    return client.post("/api/annotations", json=body)


def test_items_hide_generator_suggestions_source_severity_and_other_annotator_labels(client):
    assert client.get("/api/health").json() == {"status": "ok", "items": 8}
    initial = client.get("/api/items", params={"annotator": "alice"}).json()
    assert initial["total"] == 4
    assert initial["items"][0]["annotation"] is None
    payload = initial["items"][0]["payload"]
    assert "label" not in payload and "pair_type" not in payload
    assert save(client, "pair:p0").status_code == 200
    bob = client.get("/api/items", params={"annotator": "bob", "include_labeled": True}).json()
    assert all(item["annotation"] is None for item in bob["items"])
    own = client.get("/api/items", params={"annotator": "alice", "include_labeled": True}).json()
    assert own["items"][0]["annotation"]["annotator"] == "alice"
    priority = client.get("/api/items", params={"annotator": "alice", "task": "priority"}).json()
    assert all("severity" not in item["payload"] for item in priority["items"])


@pytest.mark.parametrize("label", [-1, 2, "1", 1.0, True, None])
def test_duplicate_labels_require_strict_zero_or_one(client, label):
    assert save(client, "pair:p0", label=label).status_code == 422


@pytest.mark.parametrize("label", [0, 6, -1, "3", 3.0, True, None])
def test_priority_requires_strict_one_through_five(client, label):
    assert save(client, "complaint:c0", label=label).status_code == 422


@pytest.mark.parametrize("label", [1, 2, 3, 4, 5])
def test_every_valid_priority_level_is_accepted(client, label):
    assert save(client, "complaint:c0", label=label, reason="Stated impact supports this level").status_code == 200


@pytest.mark.parametrize("reason", ["", " \n\t "])
def test_label_and_skip_require_a_meaningful_reason(client, reason):
    assert save(client, "pair:p0", reason=reason).status_code == 422
    assert save(client, "pair:p0", label=None, status="skipped", reason=reason).status_code == 422


def test_unknown_item_and_invalid_annotator_are_rejected(client):
    assert save(client, "pair:missing").status_code == 404
    assert save(client, "pair:p0", annotator="alice/bob").status_code == 422
    assert client.get("/api/items", params={"annotator": "alice/bob"}).status_code == 422
    assert client.get("/api/agreement", params={"annotator_a": "alice", "annotator_b": "alice"}).status_code == 422


def test_skips_remain_unlabeled_and_are_excluded_from_overlap(client):
    assert save(client, "pair:p0", label=0, status="skipped").status_code == 422
    assert save(client, "pair:p0", label=None, status="skipped", reason="Missing incident location").status_code == 200
    assert save(client, "pair:p0", annotator="bob", label=0).status_code == 200
    own = client.get("/api/items", params={"annotator": "alice", "include_labeled": True}).json()
    assert own["skipped"] == 1 and own["labeled"] == 0
    assert own["items"][0]["annotation"]["label"] is None
    report = client.get("/api/agreement", params={"annotator_a": "alice", "annotator_b": "bob"}).json()
    assert report["duplicate"]["n_overlap"] == 0
    assert report["duplicate"]["kappa"] is None
    assert save(client, "pair:p0", label=1, expected_revision=1).status_code == 200


def test_agreement_uses_independent_task_specific_overlap(client):
    for annotator in ("alice", "bob"):
        for index, label in enumerate([0, 1]):
            assert save(client, f"pair:p{index}", annotator=annotator, label=label).status_code == 200
        for index, label in enumerate([1, 5]):
            assert save(client, f"complaint:c{index}", annotator=annotator, label=label, reason="Evidence anchor").status_code == 200
    assert save(client, "pair:p2", label=0).status_code == 200
    assert save(client, "pair:p3", annotator="bob", label=None, status="skipped").status_code == 200
    report = client.get("/api/agreement", params={"annotator_a": "alice", "annotator_b": "bob"}).json()
    for task, weighting in (("duplicate", "unweighted"), ("priority", "quadratic")):
        assert report[task]["n_overlap"] == 2
        assert report[task]["kappa"] == pytest.approx(1)
        assert report[task]["weighting"] == weighting


def test_stale_revision_is_rejected_and_does_not_erase_current_judgment(client):
    assert save(client, "pair:p0", label=1).json()["revision"] == 1
    assert save(client, "pair:p0", label=0, expected_revision=0).status_code == 409
    assert save(client, "pair:p0", label=0, expected_revision=1, reason="New location evidence").json()["revision"] == 2
    own = client.get("/api/items", params={"annotator": "alice", "include_labeled": True}).json()
    assert own["items"][0]["annotation"]["label"] == 0
    assert own["items"][0]["annotation"]["revision"] == 2


def test_restart_preserves_labels_and_rejects_changed_existing_items(dataset):
    first = create_app(dataset["pairs_path"], [dataset["complaints_path"]], dataset["db_path"])
    with TestClient(first) as client:
        assert save(client, "pair:p0", reason="Stored original reason").status_code == 200
    restarted = create_app(dataset["pairs_path"], [dataset["complaints_path"]], dataset["db_path"])
    with TestClient(restarted) as client:
        own = client.get("/api/items", params={"annotator": "alice", "include_labeled": True}).json()
        assert own["items"][0]["annotation"]["reason"] == "Stored original reason"
    dataset["pairs"][0]["text_a"] = "Changed incident text"
    write_csv(dataset["pairs_path"], PAIR_COLUMNS, dataset["pairs"])
    with pytest.raises(ValueError, match="changed since this database was created"):
        create_app(dataset["pairs_path"], [dataset["complaints_path"]], dataset["db_path"])


def test_annotation_export_preserves_reason_text_labels_skips_and_provenance(client):
    reason = 'একই ঘটনা, same incident\n"Road 7" evidence'
    assert save(client, "pair:p0", reason=reason).status_code == 200
    assert save(client, "complaint:c0", annotator="bob", label=4, reason="Essential service affected").status_code == 200
    assert save(client, "pair:p1", label=None, status="skipped", reason="Insufficient time context").status_code == 200
    response = client.get("/api/export/annotations.csv")
    assert response.status_code == 200
    assert "attachment" in response.headers["content-disposition"]
    records = list(csv.DictReader(io.StringIO(response.text)))
    assert len(records) == 3
    by_item = {record["item_id"]: record for record in records}
    assert by_item["pair:p0"]["reason"] == reason
    assert by_item["pair:p0"]["original_id_a"] == "g0"
    assert by_item["pair:p0"]["label"] == "1"
    assert by_item["complaint:c0"]["label"] == "4"
    assert by_item["pair:p1"]["status"] == "skipped"
    assert by_item["pair:p1"]["label"] == ""


def test_natural_five_column_collection_loads_directly_with_stable_ids(tmp_path):
    path = write_csv(tmp_path / "natural.csv", ("text", "lang", "category", "location", "timestamp"), [{
        "text": "পানি নেই", "lang": "bn", "category": "water", "location": "Dhaka",
        "timestamp": "2026-10-29T08:00:00+06:00",
    }])
    first = build_items(None, [path])
    assert first == build_items(None, [path])
    identifier, item = next(iter(first.items()))
    assert identifier.startswith("complaint:natural_")
    assert item["payload"]["split"] == "natural_test"
    assert item["payload"]["original_id"] == ""
    with TestClient(create_app(complaints_paths=[path], db_path=tmp_path / "natural.sqlite3")) as client:
        assert save(client, identifier, label=3, reason="Significant reported impact").status_code == 200


def test_duplicate_group_cannot_cross_pair_or_complaint_splits(dataset, tmp_path):
    row = dict(dataset["complaints"][0])
    path = write_csv(tmp_path / "val.csv", SCHEMA_COLUMNS, [row])
    with pytest.raises(ValueError, match="crosses splits"):
        create_app(dataset["pairs_path"], [path], dataset["db_path"])


def test_conflicting_same_id_records_are_rejected(dataset):
    altered = dict(dataset["complaints"][0], text="Different text under the same id")
    write_csv(dataset["complaints_path"], SCHEMA_COLUMNS, [dataset["complaints"][0], altered])
    with pytest.raises(ValueError, match="Conflicting records share item ID"):
        build_items(None, [dataset["complaints_path"]])


def test_review_pagination_reaches_items_after_first_two_hundred(tmp_path):
    records = [{
        "text": f"Synthetic complaint {index}", "lang": "en", "category": "water",
        "location": "", "timestamp": "",
    } for index in range(205)]
    path = write_csv(tmp_path / "natural.csv", ("text", "lang", "category", "location", "timestamp"), records)
    with TestClient(create_app(complaints_paths=[path], db_path=tmp_path / "pagination.sqlite3")) as client:
        params = {"annotator": "alice", "task": "priority", "include_labeled": True, "limit": 200}
        first = client.get("/api/items", params={**params, "offset": 0}).json()
        second = client.get("/api/items", params={**params, "offset": 200}).json()
        assert first["total"] == second["total"] == 205
        assert len(first["items"]) == 200 and len(second["items"]) == 5
        first_ids = {item["item_id"] for item in first["items"]}
        second_ids = {item["item_id"] for item in second["items"]}
        assert not first_ids & second_ids
        assert len(first_ids | second_ids) == 205
        last_id = second["items"][-1]["item_id"]
        assert save(client, last_id, label=2, reason="Localized synthetic case").status_code == 200
        reviewed = client.get("/api/items", params={**params, "offset": 200}).json()
        assert reviewed["available"] == 205
        assert reviewed["labeled"] == 1
        assert reviewed["items"][-1]["annotation"]["label"] == 2
