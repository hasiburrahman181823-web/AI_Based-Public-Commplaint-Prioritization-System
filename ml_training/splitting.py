"""Deterministic, approximate stratification with indivisible original-ID groups.

Ratios describe *rows*, not numbers of groups. Exact ratios and label proportions
are not generally attainable when a group contains several rows or labels.
"""

from __future__ import annotations

import warnings
from collections.abc import Sequence

import numpy as np
import pandas as pd

SPLIT_NAMES = ("train", "val", "test")


def split_grouped(
    df: pd.DataFrame,
    ratios: Sequence[float] = (0.70, 0.15, 0.15),
    seed: int = 42,
) -> dict[str, pd.DataFrame]:
    """Split rows without allowing an ``original_id`` into multiple partitions.

    A seeded largest-group-first greedy allocation balances row counts and each
    category's row distribution. Coverage constraints try to put a category in
    all three partitions when it occurs in at least three distinct groups. Two
    local improvement passes then reduce ratio error without removing achieved
    coverage. Rare labels, mixed-label groups, and large groups can prevent exact
    stratification; warnings explain these limits. Input order is preserved
    inside each result. The same input, seed, and ratios give the same result.
    """
    try:
        target = np.asarray(ratios, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError("ratios must contain three positive numbers summing to 1") from exc
    if (
        target.shape != (3,)
        or not np.isfinite(target).all()
        or (target <= 0).any()
        or not np.isclose(target.sum(), 1.0, rtol=0, atol=1e-8)
    ):
        raise ValueError("ratios must contain three positive numbers summing to 1")
    missing = {"original_id", "category"} - set(df.columns)
    if missing:
        raise ValueError(f"Split input is missing required columns: {', '.join(sorted(missing))}")
    if df.empty:
        raise ValueError("Cannot split an empty dataset")
    for column in ("original_id", "category"):
        invalid = df[column].isna() | df[column].astype(str).str.strip().eq("")
        if invalid.any():
            raise ValueError(f"Column '{column}' contains {int(invalid.sum())} null or blank values")

    category_codes, categories = pd.factorize(df["category"], sort=False)
    positions = list(df.groupby("original_id", sort=False, observed=True).indices.values())
    group_count = len(positions)
    category_count = len(categories)
    label_rows = np.zeros((group_count, category_count), dtype=np.int64)
    for group, row_positions in enumerate(positions):
        label_rows[group] = np.bincount(category_codes[row_positions], minlength=category_count)
    sizes = label_rows.sum(axis=1)
    present = label_rows > 0
    groups_per_category = present.sum(axis=0)
    eligible = groups_per_category >= 3
    rare = [str(categories[i]) for i in np.flatnonzero(~eligible)]
    if rare:
        warnings.warn(
            "Categories in fewer than 3 original_id groups cannot appear in every split: "
            + ", ".join(rare)
            + ". Keeping groups intact; category stratification is best effort.",
            UserWarning,
            stacklevel=2,
        )
    mixed = int((present.sum(axis=1) > 1).sum())
    if mixed:
        warnings.warn(
            f"{mixed} original_id groups contain multiple categories; "
            "stratification is best effort and groups remain intact.",
            UserWarning,
            stacklevel=2,
        )

    rng = np.random.default_rng(seed)
    tie_break = rng.random(group_count)
    order = sorted(range(group_count), key=lambda g: (-int(sizes[g]), float(tie_break[g])))
    assignments = np.full(group_count, -1, dtype=int)
    row_counts = np.zeros(3, dtype=np.int64)
    category_rows = np.zeros((3, category_count), dtype=np.int64)
    category_groups = np.zeros((3, category_count), dtype=np.int64)
    split_groups = np.zeros(3, dtype=np.int64)
    remaining = groups_per_category.copy()
    total_rows = len(df)
    category_totals = label_rows.sum(axis=0)
    coverage_conflict = False

    def delta(group: int, split: int, direction: int = 1) -> float:
        row_fraction = direction * sizes[group] / total_rows
        label_fraction = direction * label_rows[group] / category_totals
        row_change = 2 * (row_counts[split] / total_rows - target[split]) * row_fraction
        label_change = 2 * (category_rows[split] / category_totals - target[split]) * label_fraction
        return float(row_change + row_fraction**2 + np.mean(label_change + label_fraction**2))

    def update(group: int, split: int, direction: int) -> None:
        row_counts[split] += direction * sizes[group]
        category_rows[split] += direction * label_rows[group]
        category_groups[split] += direction * present[group]
        split_groups[split] += direction

    for step, group in enumerate(order):
        remaining -= present[group]
        candidates = []
        for split in range(3):
            after_groups = category_groups.copy()
            after_groups[split] += present[group]
            missing_coverage = (after_groups == 0).sum(axis=0)
            category_feasible = (remaining[eligible] >= missing_coverage[eligible]).all()
            after_split_groups = split_groups.copy()
            after_split_groups[split] += 1
            global_feasible = group_count < 3 or group_count - step - 1 >= (after_split_groups == 0).sum()
            if category_feasible and global_feasible:
                candidates.append(split)
        if not candidates:
            # Mixed-label groups may have incompatible coverage requirements.
            coverage_conflict = True
            candidates = list(range(3))
            if group_count >= 3:
                feasible = [s for s in candidates if group_count - step - 1 >= (split_groups + (np.arange(3) == s) == 0).sum()]
                candidates = feasible or candidates
        selected = min(candidates, key=lambda s: (delta(group, s), s))
        assignments[group] = selected
        update(group, selected, 1)

    # Improve the balance while preserving groups and every achieved eligible
    # category's coverage. This is bounded O(groups * categories), not pairwise.
    for _ in range(2):
        changed = False
        for group in order:
            source = int(assignments[group])
            if split_groups[source] <= 1:
                continue
            if ((category_groups[source] == 1) & present[group] & eligible).any():
                continue
            removal = delta(group, source, -1)
            destination = min((s for s in range(3) if s != source), key=lambda s: (removal + delta(group, s), s))
            if removal + delta(group, destination) < -1e-12:
                update(group, source, -1)
                update(group, destination, 1)
                assignments[group] = destination
                changed = True
        if not changed:
            break

    if coverage_conflict or ((category_groups[:, eligible] == 0).any()):
        warnings.warn(
            "Group/category constraints prevented full category coverage in all splits; "
            "inspect the per-split category counts before training.",
            UserWarning,
            stacklevel=2,
        )
    empty = [SPLIT_NAMES[s] for s in range(3) if split_groups[s] == 0]
    if empty:
        warnings.warn(
            "Too few original_id groups to populate every split; empty splits: " + ", ".join(empty),
            UserWarning,
            stacklevel=2,
        )

    row_assignments = np.empty(total_rows, dtype=int)
    for group, row_positions in enumerate(positions):
        row_assignments[row_positions] = assignments[group]
    return {name: df.iloc[np.flatnonzero(row_assignments == split)].copy() for split, name in enumerate(SPLIT_NAMES)}
