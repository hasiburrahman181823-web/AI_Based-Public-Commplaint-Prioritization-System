"""Train optional expert pairwise priority ranking, with an explicit rules fallback."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import shutil
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import joblib
import numpy as np
import pandas as pd

from ml.priority_ranker import evaluate_rankings, fit_priority_ranker, load_expert_judgments, ranker_score
from ml_training.scripts.train_classifiers import _publish, resolve_path


def _score_rows(frame, rules):
    results = [rules.score(row) for row in frame.to_dict("records")]
    values = np.asarray([[result["feature_values"][name] for name in rules.feature_names] for result in results], dtype=float)
    return results, values.reshape(len(frame), len(rules.feature_names))


def _validate_paths(judgments: Path, config: Path, output: Path, artifact: Path, overwrite: bool):
    if output.exists() and not output.is_dir():
        raise ValueError("Ranking result output must be a directory")
    if output == artifact or output in artifact.parents or artifact in output.parents:
        raise ValueError("Keep result directory separate from the ranker artifact path")
    if judgments == config:
        raise ValueError("Judgment data and rules configuration must be separate files")
    for source in (judgments, config):
        if source == artifact or source == output or output in source.parents:
            raise ValueError("Ranking outputs must not overwrite or contain input files")
    snapshot = artifact.parent / (artifact.name + ".rules.yaml")
    variants_snapshot = artifact.parent / (artifact.name + ".variants.json")
    if not overwrite and (artifact.exists() or snapshot.exists() or variants_snapshot.exists() or output.exists() and any(output.iterdir())):
        raise ValueError("Priority ranking output exists; use fresh paths or --overwrite")


def run_experiment(args) -> dict:
    from ml.priority_rules import PriorityRules

    judgments, config, output, artifact_path = map(resolve_path, (args.judgments, args.config, args.output_dir, args.artifact))
    _validate_paths(judgments, config, output, artifact_path, args.overwrite)
    rules = PriorityRules(config)
    frame = load_expert_judgments(judgments)
    scored, values = _score_rows(frame, rules)
    train_mask = frame["split"].eq("train").to_numpy()
    config_hash = hashlib.sha256(config.read_bytes()).hexdigest()
    artifact = fit_priority_ranker(
        frame.loc[train_mask].reset_index(drop=True), values[train_mask], rules.feature_names, config_hash,
        seed=args.seed, min_train_queries=args.min_train_queries, min_comparisons=args.min_comparisons,
        min_train_items=args.min_train_items, max_pairs_per_query=args.max_pairs_per_query,
    )
    artifact["rules_preprocessing_fingerprints"] = dict(rules.preprocessing_fingerprints)
    if any(result.get("config_sha256") != config_hash for result in scored):
        raise ValueError("Rule score fingerprint does not match the configured YAML bytes")
    score_sets = {
        "severity_only": np.asarray([result["feature_values"]["severity"] for result in scored], dtype=float),
        "rule_based": np.asarray([result["priority_score"] for result in scored], dtype=float),
        "ltr": np.asarray([ranker_score(artifact, result)["ranking_score"] for result in scored], dtype=float),
    }
    reports, table = {}, []
    for split in ("val", "test"):
        mask = frame["split"].eq(split).to_numpy()
        reports[split] = evaluate_rankings(frame.loc[mask].reset_index(drop=True), {name: scores[mask] for name, scores in score_sets.items()})
        for name, metrics in reports[split].items():
            row = {"split": split, "model": name, "ranker_status": artifact["status"],
                   **{key: value for key, value in metrics.items() if key not in ("queries", "aggregation")}}
            table.append(row)
            print(json.dumps(row, ensure_ascii=False, allow_nan=False), flush=True)
    report = {
        "schema_version": 1, "experiment_kind": "expert_priority_ranking",
        "ranker_status": artifact["status"], "fallback_reason": artifact["reason"],
        "rules_config_sha256": config_hash, "rules_preprocessing_fingerprints": dict(rules.preprocessing_fingerprints),
        "judgments_sha256": hashlib.sha256(judgments.read_bytes()).hexdigest(),
        "seed": args.seed, "feature_names": list(rules.feature_names), "sampling": artifact["sampling"],
        "n_training_items": artifact["n_training_items"], "n_training_queries": artifact["n_training_queries"],
        "comparisons": reports, "warnings": frame.attrs.get("warnings", []),
        "design": {"training": "expert train queries only; no source severity converted to expert labels",
                   "validation": "diagnostic only; fixed learner and sufficiency minima; no model or threshold tuning",
                   "test": "same frozen held-out query items across severity-only, rule-based and LTR",
                   "ndcg": "NDCG@5 uses gain 2**expert_priority-1; predicted-score ties are averaged",
                   "spearman": "rank 1 is most urgent; average tied predicted ranks compared with explicit expert_rank or ranks from priority",
                   "score_semantics": "LTR linear utility is not a probability, priority label, or rule-points scale"},
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".priority-ranking-results-", dir=output.parent) as temporary, tempfile.TemporaryDirectory(prefix=".priority-ranking-artifact-", dir=artifact_path.parent) as artifact_temporary:
        stage, astage = Path(temporary), Path(artifact_temporary)
        pd.DataFrame(table).to_csv(stage / "ranking_comparison.csv", index=False, encoding="utf-8")
        (stage / "metrics.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        columns = ["split", "model", "ranker_status", "n_items", "n_queries", "ndcg_at_5", "spearman"]
        lines = ["# Expert priority ranking comparison", "", "All scorers use identical held-out query items; undefined metrics are shown explicitly.", "",
                 "| " + " | ".join(columns) + " |", "| " + " | ".join(["---"] * len(columns)) + " |"]
        for row in table:
            render = lambda value: "undefined" if value is None else f"{value:.6f}" if isinstance(value, float) else str(value)
            lines.append("| " + " | ".join(render(row.get(name)) for name in columns) + " |")
        if artifact["status"] == "fallback_rules":
            lines.extend(["", "LTR used the explicit rules fallback: " + artifact["reason"]])
        (stage / "ranking_comparison.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
        predictions = frame.loc[:, ["id", "original_id", "query_id", "split", "expert_priority", "expert_rank"]].copy()
        for name, scores in score_sets.items():
            predictions[name + "_score"] = scores
        predictions.to_csv(stage / "ranking_predictions.csv", index=False, encoding="utf-8")
        staged_artifact = astage / artifact_path.name
        artifact["rules_snapshot"] = artifact_path.name + ".rules.yaml"
        artifact["variants_snapshot"] = artifact_path.name + ".variants.json"
        joblib.dump(artifact, staged_artifact, compress=3)
        shutil.copyfile(config, astage / artifact["rules_snapshot"])
        shutil.copyfile(rules.variants_path, astage / artifact["variants_snapshot"])
        _publish(astage, artifact_path.parent)
        _publish(stage, output)
    print(f"Priority ranker status: {artifact['status']}")
    if artifact["reason"]:
        print(f"Rules fallback reason: {artifact['reason']}")
    print(f"Saved priority comparison: {output / 'ranking_comparison.csv'}")
    print(f"Saved priority ranker: {artifact_path}")
    return report


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--judgments", default="templates/expert_priority_judgments.csv")
    parser.add_argument("--config", default="config/priority_rules.yaml")
    parser.add_argument("--output-dir", default="results/priority")
    parser.add_argument("--artifact", default="artifacts/priority_ranker.joblib")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--min-train-queries", type=int, default=3)
    parser.add_argument("--min-comparisons", type=int, default=20)
    parser.add_argument("--min-train-items", type=int, default=10)
    parser.add_argument("--max-pairs-per-query", type=int, default=1000)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main(argv: list[str] | None = None):
    args = build_parser().parse_args(argv)
    try:
        run_experiment(args)
        return 0
    except (ValueError, OSError, RuntimeError, ImportError, csv.Error) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
