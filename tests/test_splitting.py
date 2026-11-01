"""Regression tests for atomic groups and approximate category stratification."""

import pandas as pd
import pytest

from ml_training.splitting import split_grouped


def fixture_frame(groups_per_category=40, rows_per_group=2):
    return pd.DataFrame(
        [
            {"id": f"{category}-{group}-{row}", "original_id": f"{category}-{group}", "category": category}
            for category in ("water", "electricity", "roads")
            for group in range(groups_per_category)
            for row in range(rows_per_group)
        ]
    )


def test_groups_never_leak_and_every_row_is_preserved():
    data = fixture_frame()
    splits = split_grouped(data)
    group_sets = {name: set(frame.original_id) for name, frame in splits.items()}
    assert group_sets["train"].isdisjoint(group_sets["val"])
    assert group_sets["train"].isdisjoint(group_sets["test"])
    assert group_sets["val"].isdisjoint(group_sets["test"])
    combined = pd.concat(splits.values()).sort_index()
    pd.testing.assert_frame_equal(combined, data)


def test_balances_rows_and_each_category():
    data = fixture_frame()
    splits = split_grouped(data)
    for name, ratio in (("train", 0.70), ("val", 0.15), ("test", 0.15)):
        assert len(splits[name]) == round(len(data) * ratio)
        assert splits[name].category.value_counts().to_dict() == {category: round(80 * ratio) for category in ("water", "electricity", "roads")}


def test_three_groups_per_category_get_category_coverage():
    splits = split_grouped(fixture_frame(groups_per_category=3))
    assert all(set(frame.category) == {"water", "electricity", "roads"} for frame in splits.values())


def test_deterministic_with_same_seed():
    data = fixture_frame()
    first = split_grouped(data, seed=123)
    second = split_grouped(data, seed=123)
    for name in first:
        pd.testing.assert_frame_equal(first[name], second[name])


def test_seed_changes_assignments_when_equivalent_choices_exist():
    data = fixture_frame()
    first = split_grouped(data, seed=123)
    second = split_grouped(data, seed=456)
    assert set(first["train"].original_id) != set(second["train"].original_id)


def test_mixed_category_groups_remain_atomic():
    data = fixture_frame(groups_per_category=8)
    data.loc[data.index % 3 == 0, "category"] = "mixed"
    with pytest.warns(UserWarning, match="multiple categories"):
        splits = split_grouped(data)
    all_groups = [set(frame.original_id) for frame in splits.values()]
    assert not (all_groups[0] & all_groups[1] or all_groups[0] & all_groups[2] or all_groups[1] & all_groups[2])
    assert sum(map(len, splits.values())) == len(data)


def test_unequal_groups_stay_intact():
    data = pd.DataFrame([
        {"original_id": str(group), "category": "water", "row": row}
        for group, size in enumerate([40, 20, 12, 9, 7, 5, 4, 3])
        for row in range(size)
    ])
    splits = split_grouped(data)
    assert all(not frame.empty for frame in splits.values())
    assert len(splits["train"]) / len(data) == pytest.approx(0.70, abs=0.08)
    assignment = pd.concat([frame.assign(partition=name) for name, frame in splits.items()])
    assert assignment.groupby("original_id").partition.nunique().eq(1).all()


def test_tiny_dataset_warns_and_keeps_group_whole():
    data = pd.DataFrame({"original_id": ["one", "one"], "category": ["water", "water"]})
    with pytest.warns(UserWarning) as caught:
        splits = split_grouped(data)
    assert any("fewer than 3" in str(w.message) for w in caught)
    assert any("empty splits" in str(w.message) for w in caught)
    assert len(splits["train"]) == 2
    assert splits["val"].empty and splits["test"].empty


@pytest.mark.parametrize("ratios", [(0.8, 0.1), (0.7, 0.2, 0.2), (0.7, 0.3, 0), (float("nan"), 0.15, 0.15), ("x", 0.15, 0.15)])
def test_rejects_invalid_ratios(ratios):
    with pytest.raises(ValueError, match="ratios"):
        split_grouped(fixture_frame(), ratios=ratios)


@pytest.mark.parametrize("column", ["original_id", "category"])
@pytest.mark.parametrize("invalid", [None, "", "   "])
def test_rejects_missing_group_or_label_values(column, invalid):
    data = fixture_frame()
    data.loc[0, column] = invalid
    with pytest.raises(ValueError, match=column):
        split_grouped(data)


def test_rejects_empty_and_missing_columns():
    with pytest.raises(ValueError, match="missing required columns"):
        split_grouped(pd.DataFrame({"text": ["hello"]}))
    with pytest.raises(ValueError, match="empty dataset"):
        split_grouped(pd.DataFrame(columns=["original_id", "category"]))


def test_custom_ratios():
    data = fixture_frame(groups_per_category=40)
    splits = split_grouped(data, ratios=(0.5, 0.25, 0.25))
    assert [len(splits[name]) for name in ("train", "val", "test")] == [120, 60, 60]
