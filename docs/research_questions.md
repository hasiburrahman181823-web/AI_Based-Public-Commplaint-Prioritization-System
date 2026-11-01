# Research questions mapped to experiments

These are prespecified questions and evidence requirements. The scripts produce
measurements only after the reviewed inputs and real artifacts are supplied.
Refer to [saved results](thesis_results.md); an empty evidence folder means the
questions remain empirically unanswered.

| RQ | Comparison and unit | Primary evidence | Controls and qualification |
| --- | --- | --- | --- |
| RQ1: How do lightweight lexical and multilingual representation models categorize grievances and predict ordered severity across languages? | Character `char_wb` TF-IDF 1–5 + LinearSVC, word TF-IDF + logistic regression, MiniLM embeddings + LR/SVM; complaint as measurement unit, incident as independence unit. | Per-language category/severity macro-F1; severity quadratic weighted kappa with declared label order; reports and confusion matrices. | Same incident-isolated splits and label universe. Hyperparameters/normalization fixed before final test. Optional XLM-R is a separate comparison with its actual hardware/training budget. |
| RQ2: How much do language derivation and Banglish spelling noise affect generalization? | Each frozen classifier on Bangla, derived English, derived Banglish, independent natural complaints and aligned synthetic-noise Banglish. | Cohort support and macro-F1; paired clean/noisy F1 drop and changed-text counts. | No mixing derived/natural cohorts. A negative F1 drop is reported as observed, not forced to zero. Noise transformations/seeds are fixed. Natural severity requires independently supplied severity labels. |
| RQ3: Does explicit location/time evidence improve same-incident detection beyond semantic similarity? | Same pair test set for B1 TF-IDF cosine, B2 MiniLM cosine, C0 learned context control, P1 + location, P2 + time. | Pair precision/recall/F1 and PR-AUC, hard/easy-negative and language-pair breakdowns, matched error exports. | Same learner for nested C0/P1/P2; thresholds from validation only; same test IDs/fingerprint. P1−C0 isolates location and P2−P1 isolates time. B2−P1 changes learner/context too and is not a location-only causal effect. |
| RQ4: Do ranking rules and available expert-trained LTR align with expert queue judgments? | Severity-only, rule score and pairwise LTR on identical held-out queue items/query IDs. | Equal-query NDCG@5 and Spearman with eligible item/query counts, ties and undefined cases. | Train only on independent expert training queries. Report qualifications, experts, independent ratings, queries and comparisons separately. If sufficiency fails, identify `fallback_rules`; identical fallback outputs are not a learned-model result. |
| RQ5: Can calibrated classification confidence support a useful human-review operating point? | Saved calibrated category/severity probabilities, validation-selected review policy and actual deployed review rules. | Top-label ECE/reliability; coverage-versus-error curve, accepted support and actual review workload. | Separate calibration/policy validation where feasible; no final-test threshold selection. Empirical accepted-error constraints do not guarantee future or subgroup risk. Confidence is not a calibrated priority/duplicate probability. |
| RQ6: Are outcomes distributed differently across observed areas/languages, and what limits interpretation? | Descriptive priority-score/error summaries and conditional context strata on the same held-out records. | Group support, means/distributions, compatible-truth error rates, disparities and unknown groups. | Keep small/unknown groups visible; apply stated support minima. Do not compare rule points to expert 1–5 grades as if the scales match. Observational disparities are not causal discrimination findings or a fairness certificate. |

## Commands and saved outputs

| Evidence family | Runner | Typical saved outputs |
| --- | --- | --- |
| Classification/severity, robustness and ECE | `python -m ml_training.scripts.train_classifiers` | `results/metrics.json`, `results/results_table.csv`, target/model/cohort classification reports, `confusion_matrix.png`, `reliability_diagram.png`, `reliability_bins.csv`, predictions. |
| Duplicate ablations | `python -m ml_training.scripts.train_duplicates` | `results/duplicates/metrics.json`, `ablation_table.csv/.md`, per-model validation/test metrics, PR evidence, pair-type/language strata and predictions. |
| Duplicate errors/clusters | `python -m ml_training.scripts.analyze_duplicate_errors` and `cluster_duplicates` | Explicitly selected FP/FN exports and predicted cluster membership. These do not alone evaluate cluster quality. |
| Expert ranking | `python -m ml_training.scripts.train_priority_ranker` | `results/priority/metrics.json`, `ranking_comparison.csv/.md`, `ranking_predictions.csv`, sufficiency/fallback status. |
| Review policy | `python -m ml_training.scripts.select_review_threshold` | Saved selection/provenance, `coverage_error_curve.csv/.json/.png`. |
| Descriptive disparities | `python -m ml_training.scripts.check_priority_fairness` | `results/fairness/fairness.json/.md`, `groups.csv`, `conditional_groups.csv`. |
| Evidence collection | `python make_thesis_report.py --overwrite` | `docs/thesis_results.md`; links and exact saved values, never a new experiment. |

Use the full argument examples in [Phase 3](phase3_workflow.md), [Phase 4](phase4_workflow.md)
and [Phase 5](phase5_workflow.md). In a study with several runs, place each run
under a distinct results directory. Do not overwrite one experiment's outputs
while preserving another experiment's headline table. The collector inventories
all saved runs without selecting the most favorable one.

## Support table required before conclusions

Record these from the actual collection/evaluation manifests: original source
rows; verified incidents; derived variants by language; independent natural
reports by language/category/severity; annotated and adjudicated pair counts by
split/type/language; annotator and expert counts; expert-rated items, queries,
comparisons and disagreements; known/missing locations/times; eligible calibration
and ranking cases; and unknown/excluded rows with reasons. Keep each counting
unit explicit. Unknown and unavailable are valid entries; they are not zero
observations unless the collection records establish that zero.

## Outcome language

The questions ask whether and under what conditions a method helps, not whether
the proposed model can be made to win. A null, mixed or worse result answers the
question if the design and support are credible. If evidence is absent, state
that the implementation supports the experiment but that the research question
has not yet been evaluated. See [honest conclusion examples](thesis_validity_ethics.md).
