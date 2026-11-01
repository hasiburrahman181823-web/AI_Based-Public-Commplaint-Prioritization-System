# Phase 5: priority, explanations, uncertainty and review

Run commands from the repository root in the configured Python environment.
This phase keeps source severity, rule priority, expert relevance, learned
ranking utility and model confidence as separate quantities. The scripts print
measured evaluation results; this guide contains no research scores.

Read [the experiment design and validity limits](priority_experiment_design.md)
before interpreting a ranking comparison, confidence threshold or group gap.

## 1. Review and freeze the transparent priority configuration

```powershell
python -m pip install -r requirements.txt
```

Edit `config/priority_rules.yaml` with the domain experts. The six components
are severity, category criticality, duration, duplicate-report volume,
vulnerable-context keywords and life-risk keywords. All component points,
weights, bands, aliases, negation phrases and output priority-label thresholds
are explicit YAML values. The loader uses `yaml.safe_load`, validates finite
nonnegative weights and records the configuration's SHA-256 fingerprint.

Default settings are research starting values, not empirically validated
dispatch policy. The category/severity maps accept source labels without
case sensitivity, including the supplied BanglaPSG `Low`, `Medium`, `High`
severity labels and six source categories. Unknown labels remain unknown and
contribute zero with review notes; they are never guessed from an expert
priority rating.

Duration uses the multilingual Phase 4 parser and supplied report-time context.
The default strict bands mean more than 24 hours or more than 48 hours; exactly
48 hours does not satisfy the greater-than-48-hour band. Only the highest
eligible duration/report-volume band applies. Unknown/ambiguous duration has
no imputed boost and remains distinguishable from a known zero duration.

The default vulnerable aliases cover hospital, school, child and elderly
contexts in Bangla, English and Banglish. Life-risk aliases cover fire,
electric shock, flood and gas leakage. Whole-phrase matching avoids English
subwords such as `hospitality`; known Bangla inflections are explicit aliases.
Concept caps prevent repeated words or translated aliases from stacking points.
Adjacent clear negation such as `no fire`, `আগুন নেই` or `agun nai` suppresses
that keyword boost and retains the negated evidence. This is a small heuristic,
not a general negation or hazard-understanding system.

`duplicate_group_size` is a **report-volume count**, not the number of affected
people. The analysis service collapses known translations/variants using
available lineage before counting reports. Independent source reports remain
separate even when they share an incident ID. Derived versions with unresolved
lineage are excluded from the volume estimate and flagged for review. Limited
candidate retrieval,
incorrect lineage or duplicate mistakes can still distort that proxy. Do not
rewrite the explanation as a population estimate.

Inspect a rule decision directly:

```python
from ml.priority_rules import PriorityRules

rules = PriorityRules("config/priority_rules.yaml")
decision = rules.score({
    "text": "হাসপাতালে আগুন, ৩ দিন ধরে সমস্যা",
    "lang": "bn",
    "category": "Healthcare",
    "severity": "High",
    "timestamp": "2026-10-30T10:00:00+06:00",
    "duplicate_group_size": 3,
})
print(decision)
```

The output includes `priority_score`, `priority_label`, `rule_breakdown`,
unweighted `feature_values`, matched/negated keyword evidence, missing fields,
duration, warnings and the config fingerprint. Every contribution is
`points × weight`; any configured final cap appears as a separate adjustment.
The example is illustrative and supplies no collected research judgment.

## 2. Prepare the saved calibrated classifiers and duplicate index

Use the Phase 3 category/severity artifacts trained and calibrated on the
appropriate incident-isolated splits. For the supplied source labels, a
lightweight training command is:

```powershell
python -m ml_training.scripts.train_classifiers --input-dir ml_training/data/processed --models char_svm word_lr --severity-order Low Medium High
```

For MiniLM classifier experiments, install `requirements-embeddings.txt` and
use the Phase 3 guide. Retain the complete artifact tree, including encoder
weights and normalization snapshots. Phase 4 duplicate models are separate
artifacts and require their own gold-pair workflow.

No classifier, duplicate model, expert labels or human-review policy is
fabricated when a corresponding artifact is unavailable. Rule cues can still
be returned, but category/severity/confidence are unavailable and the complaint
requires human review. A numerical rule score is not calibrated confidence.

## 3. Select the confidence review threshold on validation data

Export aligned validation predictions from the exact saved classifier pair:

```powershell
python -m ml_training.scripts.export_validation_predictions --input ml_training/data/processed/val.csv --category-model artifacts/category/char_svm.joblib --severity-model artifacts/severity/char_svm.joblib --output results/uncertainty/validation_joint.csv
```

The exporter checks the saved Phase 3 experiment's validation-file fingerprint.
Use `--allow-unverified-validation` only for a deliberate alternative validation
workflow, and retain the resulting provenance limitation. A filename or
`split=val` declaration alone does not prove incident isolation.

Select the empirical operating point:

```powershell
python -m ml_training.scripts.select_review_threshold --predictions results/uncertainty/validation_joint.csv --category-model artifacts/category/char_svm.joblib --severity-model artifacts/severity/char_svm.joblib --output-dir results/uncertainty --policy-output artifacts/review_policy.json --max-error-rate 0.1 --min-accepted 10
```

The joint classification confidence statistic is the minimum of the two
predicted-class calibrated probabilities. Error means category incorrect **or**
severity incorrect on the same row. That minimum is a selection statistic,
not a calibrated joint correctness probability and not priority confidence.

The policy maximizes accepted validation coverage under the requested observed
error ceiling and minimum accepted support. Infeasible constraints produce an
explicit reject-all policy. Confidence ties are accepted/rejected together;
the policy does not split equal-confidence records to improve the curve.
Policies bind to the exact category/severity artifact and preprocessing/encoder
resource fingerprints. Replacing a classifier or its resources requires a new
compatible policy.

The command writes:

```text
results/uncertainty/
├── validation_joint.csv
├── coverage_error_curve.csv
├── coverage_error_curve.json
├── coverage_error_curve.png
└── selection.json
artifacts/review_policy.json
```

Using the same validation set first for probability calibration and then for
review-policy selection can be optimistic. Prefer a separate prespecified
policy-validation cohort for stronger deployment claims, with incidents still
isolated from training and final evaluation. A selected empirical error ceiling
is not a statistical guarantee for future rows or individual areas/languages.
Additional review triggers in the analysis service can reduce actual automatic
coverage beyond the confidence-only curve.

## 4. Analyze one complaint or a batch

CLI for a complete configured analysis:

```powershell
python -m ml_training.scripts.analyze_complaints --text "মিরপুর ১০ এ ৩ দিন ধরে পানি নেই" --category-model artifacts/category/char_svm.joblib --severity-model artifacts/severity/char_svm.joblib --review-policy artifacts/review_policy.json --rules-config config/priority_rules.yaml --ranker artifacts/priority_ranker.joblib --duplicate-model artifacts/duplicates/P2.joblib --existing existing_complaints.csv
```

Omit optional ranker/duplicate arguments when those artifacts are not available.
The engine records missing context and uses the explicit rule fallback. The
default service sends life-risk keyword cases, ambiguous language, conflicting
location evidence, missing category/severity models, missing policy and low
confidence to human review.
These are review requests, not verified emergency diagnoses.

The requested application API is:

```python
from ml.analysis import analyze_complaint, configure_analyzer

configure_analyzer(
    category_model="artifacts/category/char_svm.joblib",
    severity_model="artifacts/severity/char_svm.joblib",
    review_policy="artifacts/review_policy.json",
    rules_config="config/priority_rules.yaml",
    ranker="artifacts/priority_ranker.joblib",
    duplicate_model="artifacts/duplicates/P2.joblib",
    existing_complaints=[],  # Replace with the current complaint store.
)

result = analyze_complaint(
    "মিরপুর ১০ এ ৩ দিন ধরে পানি নেই",
    location="Mirpur 10",
    timestamp="2026-10-30T10:00:00+06:00",
)
print(result)
```

Configure the reusable analyzer with your trusted artifacts at application
startup; see `configure_analyzer` and `PriorityAnalyzer` in `ml/analysis.py`.
The exact three-argument convenience function must infer language. Bangla
script and a small Latin/Banglish cue heuristic are not a language classifier.
For batch or service use with known metadata, use the analyzer's
`analyze_record` method with explicit `lang=bn`, `en` or `banglish`.

The structured output contains the requested category, severity,
`priority_score`, `priority_label`, `confidence`, `explanation`,
`review_required` and duplicate candidates. It also records relevant method,
missingness and review reasons. Unavailable category/severity/confidence are
JSON `null`, not guessed labels or numeric zero. Low-confidence and other
review-triggered decisions are marked `needs_human_review`.

Rule priority stays on the configured points scale. A trained ranker returns a
separate `ranking_score`; its unbounded utility is not converted into the YAML
priority-label bands or a 1–5 expert judgment. Source severity is also a separate
field. Classifier confidence does not certify the priority/ranking decision or
the duplicate score.

Batch output for evaluation/auditing:

```powershell
python -m ml_training.scripts.analyze_complaints --input complaints.csv --output results/analysis/predictions.jsonl --csv-output results/analysis/predictions.csv --category-model artifacts/category/char_svm.joblib --severity-model artifacts/severity/char_svm.joblib --review-policy artifacts/review_policy.json --rules-config config/priority_rules.yaml
```

Supply each input's explicit language and available location/timestamp; retain
stable IDs and provenance. Add the ranker/index arguments above when available.
Use fresh output paths, or `--overwrite` for an intentional replacement.

### Explanation semantics

The linear TF-IDF explanation multiplies each active feature value by the
relevant fitted class coefficient. Its contributions explain the base model's
raw linear margin, not the final sigmoid-calibrated probability. Character
n-grams can overlap and are not independent words or causal evidence.

Embedding classifiers have linear contributions over embedding coordinates;
coordinates are not interpretable lexical terms. Any token-occlusion summary
is explicitly approximate, depends on tokenization/context and is not SHAP or
an exact additive probability decomposition. The six rule contributions and,
when trained, linear ranker contributions are reported separately. Do not add
unrelated explanation scales together or describe them as causal attribution.

## 5. Add expert learning-to-rank judgments when available

`templates/expert_priority_judgments.csv` is header-only:

```text
id,original_id,query_id,split,text,lang,category,severity,location,timestamp,duplicate_group_size,expert_priority,expert_rank
```

Each final row is one complaint. `expert_priority` is an adjudicated integer
1–5, with 5 most urgent. `expert_rank` is optional, positive and uses 1 for most
urgent; supply all ranks or no ranks within each query. A `query_id` describes
an actual ranking queue/context. Preserve that context instead of inventing
query IDs from a desired label order. Keep both query IDs and incident
`original_id` groups isolated across train, validation and test.

The earlier annotation tool can collect independent priority judgments, but
those labels become expert ground truth only through the intended expert
protocol, provenance and adjudication. Do not manufacture expert labels from
source severity, rule scores or generated text. Duplicate report counts are
report-volume observations, not estimated people affected.

```powershell
python -m ml_training.scripts.train_priority_ranker --judgments templates/expert_priority_judgments.csv --config config/priority_rules.yaml --output-dir results/priority --artifact artifacts/priority_ranker.joblib --seed 42
```

The implementation chooses the permitted pairwise-logistic option. It learns a
linear utility from unequal expert-priority comparisons within training
queries, using mirrored feature differences. Tied expert relevance is excluded
from training. Sampling is bounded by `--max-pairs-per-query` without creating
every pair in a large queue. Validation/test judgments do not fit coefficients.

Default sufficiency minima are configurable through `--min-train-queries`,
`--min-comparisons` and `--min-train-items`. Missing/insufficient or uninformative
training judgments produce `fallback_rules` with a reason. The LTR comparison
then explicitly uses rule scores; it is not a learned-model improvement. An
empty template does not cause fabricated metric rows or fabricated labels.
Freeze sufficiency rules before final evaluation rather than relaxing them
after seeing test performance.

Every scorer uses identical held-out query items. Severity-only uses the
configured source-severity points, rule-based uses the full weighted rules,
and LTR uses its utility or the explicit fallback. The comparison reports:

- NDCG@5 with gain `2**expert_priority - 1` and tie-averaged predicted-score DCG.
- Spearman correlation between descending-score ranks and expert ranks, with
  average ranks for ties. Without explicit ranks, expert-priority ordering
  supplies the ranks.
- Per-query support, metric eligibility and equal-weight macro averages over
  queries with defined metrics. Non-discriminatory/empty queries use null
  metrics with reasons; they are not assigned manufactured perfect results.

Outputs are `results/priority/metrics.json`, `ranking_comparison.csv`,
`ranking_comparison.md` and `ranking_predictions.csv`, plus the joblib ranker
and its YAML snapshot. The artifact verifies the rules and preprocessing-resource
fingerprints; after editing rules or normalization resources, deliberately
refit or use the matching saved resources.

## 6. Audit areas and languages on reviewed evaluation data

```powershell
python -m ml_training.scripts.check_priority_fairness --input results/analysis/predictions.csv --output-dir results/fairness --min-group-size 20
```

Required input fields are `priority_score,lang`. The audit prefers
`canonical_area_id`, then `area_id`, then a normalized raw `location` fallback.
Prefer a reviewed canonical area when available; raw text locations are not
automatically one geographic group across languages. Unknown areas remain a
separate group rather than being silently dropped.

The audit reports group score distributions, support, mean-score gaps/ratios
and review rates when available. Classification errors require compatible
`category_true/category_pred` and/or `severity_true/severity_pred`, or an
explicit binary `classification_error`. Without truth, error fields are null.
If explicit aligned integer `priority_true,priority_pred` on the same 1–5 scale
are supplied, it can report priority MAE and under-prioritization rates. Raw
rule points, unbounded LTR utility and YAML priority labels are not silently
converted to those integers.

Small groups retain descriptive statistics but are excluded from disparity
comparisons. Error-rate comparisons additionally need enough labeled rows.
Conditional summaries compare language/area within observed category/severity
contexts. The report writes `fairness.json`, `groups.csv`,
`conditional_groups.csv` and `fairness.md`.

A score gap does not establish that one language or area was discriminated
against: actual need, source selection, reporting frequency, missing metadata,
annotation and classification errors can differ. Inspect conditional support,
coverage and evidence before making a fairness claim.

## 7. Verify and retain the research record

```powershell
python -m pytest -q
```

Retain model/checkpoint fingerprints, rule YAML, expert provenance/query
definitions, incident partitions, review-policy selection data, subgroup
supports, output schemas and all generated results. Tests use synthetic
fixtures and do not establish emergency validity, fairness, ranking quality or
future accepted-error guarantees. No expert dataset, compatible deployed policy
or pretrained checkpoint is invented by this workflow.
