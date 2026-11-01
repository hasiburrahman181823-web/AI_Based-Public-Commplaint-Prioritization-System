# Thesis chapter outline and evidence checklist

This outline describes the implemented system and planned experiments. It is
not a claim that the final research dataset, expert labels or neural-model
evaluations have already been completed. Use [the saved-results report](thesis_results.md)
for actual available outputs and [workspace readiness](current_workspace.md)
for data still requiring collection/review.

## 1. Introduction

- Define multilingual public-service grievance triage in Bangladesh and the
  practical tasks: category, ordered severity, same-incident detection and
  reviewable queue ranking.
- Distinguish a submitted report, a real-world incident, a derived language
  version, a duplicate pair and an estimated report count. Report count is not
  the number of people affected.
- State the research questions in [the experiment map](research_questions.md),
  the intended users, scope and measurable contributions. A reproducible
  evaluation or a documented negative finding can be a contribution.
- Give the study boundaries: local gazetteer coverage, missing incident/time
  metadata, availability of natural-language complaints, expert judgments and
  deployment infrastructure.

## 2. Related work

Review verified primary literature on Bangla/Banglish NLP, multilingual
sentence representations, grievance categorization, duplicate-event detection,
ordinal evaluation, calibration/selective prediction, learning to rank and
human oversight. Separate reported findings in prior work from hypotheses in
this thesis. Record the exact checkpoint/model revision and dataset version.
Do not invent citations, claim the checkpoint supports a language from its
name alone, or equate multilingual pretraining with validated Banglish quality.

## 3. Dataset construction, annotation and ethics

Describe the imported source datasets and their provenance/license constraints,
the original source label meanings, mapping, Unicode/digit/text cleaning and
extensible Banglish spelling dictionary. Cite [dataset sources](dataset_sources.md).

Explain incident-aware splitting, exact-cleaned-text fallback groups and their
limits. Document which rows are original, translated, transliterated or manual
paraphrases, retaining lineage and derivation flags. Report source records,
independent incidents, variants, pairs and queue items separately.

Document the translation/review process, volunteers' consent, natural collection,
annotator qualifications, duplicate/priority instructions, independent labels,
Cohen's kappa support and adjudication. Count actual annotators and expert
judgments; do not derive those counts from the number of generated pairs.
The natural five-column template does not supply severity automatically.

Include cohort tables by split/language/category/severity, missing location/time
rates, expert-query sizes, disagreements and exclusions. Explain how contact
details, raw complaints and error examples are protected or redacted.

## 4. Methods and prespecified experiment design

1. Classification/severity: character TF-IDF + LinearSVC, word TF-IDF +
   logistic regression, frozen MiniLM embeddings + LR/SVM; optional XLM-R is a
   separately resourced experiment. Specify exact severity order.
2. Calibration/uncertainty: validation-only calibration, equal-width top-label
   ECE, reliability diagrams and separate review-policy operating-point
   selection where feasible. State additional life-risk/context review rules.
3. Duplicates: B1/B2 and nested C0/P1/P2 feature ablations, fixed learner,
   validation-only thresholds, missingness/conflict features, per-pair strata,
   retrieval and clustering. Explain that location flags are not distances.
4. Priority: severity-only, configured rules and expert-trained pairwise LTR or
   explicit rule fallback. Distinguish points, utility, relevance grades,
   queue rank and probabilities.
5. Prespecify evaluation units, incident/query split isolation, seeds,
   hyperparameters, cohorts, support and any statistical analysis. Register
   protocol deviations before interpreting final test outcomes.

The detailed controls are in [duplicate design](duplicate_experiment_design.md)
and [priority design](priority_experiment_design.md).

## 5. System architecture and implementation

Describe the data/ML pipeline, trusted artifact snapshots, cached CPU inference,
location/time guards, relational records, transactional grouping, versioned
admin feedback and bilingual frontend. Reference the README architecture diagram
and backend/frontend/deployment guides.

Separate implementation evidence (unit/API/UI checks, migrations and builds)
from research evaluation. Report observed deployment hardware, versions,
cold-start behavior, memory/latency/throughput measurements and failure tests
only if actually measured. A structurally tested 50k vector index is not a
50k-complaint latency or recall benchmark.

## 6. Results and discussion

Start with [the generated evidence inventory](thesis_results.md). Present
classification and severity by language and independent natural cohort, then
robustness/calibration, duplicate ablations and error strata, priority ranking,
review coverage and descriptive subgroup disparities.

For every comparison, give cohort/sample support, label provenance, the same
test fingerprint, chosen validation settings, missing/undefined values and
uncertainty when a valid analysis was performed. Do not mix validation and test
rows, derived and natural cohorts, or weak-label and independently adjudicated
results in one headline score.

Discuss improvements, ties and regressions. Relate error examples to the
prespecified hypothesis, inspect both false positives and false negatives, and
consider model complexity/resource cost alongside accuracy. Do not select
favorable seeds or rewrite the research question after seeing the test set.

## 7. Limitations, conclusion and future work

Use [validity and ethics guidance](thesis_validity_ethics.md) to scope the claims.
Answer each research question at the strength supported by its evidence.
State when a result is unavailable, inconclusive, restricted to derived data,
or does not favor the proposed model. List concrete future work that would
resolve observed limitations rather than promising unmeasured benefits.

## Appendices and reproducibility package

- Data schema, source/license provenance and collection consent procedure.
- Annotation guideline, individual/adjudicated label protocols and agreement
  supports; publish only material authorized for release.
- Configuration, seeds, dependency/model/resource fingerprints and commands.
- Complete per-cohort tables, confusion matrices, reliability bins/diagrams,
  pair strata, ranking-query results and documented undefined cases.
- Privacy notice, deployment/security checklist, artifact hashes and code/test
  verification record. A demo video/GIF must use synthetic or consented data.

## Generate the saved-results chapter material

From the repository root:

```powershell
python make_thesis_report.py
# After adding/replacing experiment outputs:
python make_thesis_report.py --results-dir results --output docs/thesis_results.md --overwrite
```

The first command protects an existing generated report. The collector records
every file except `.gitkeep`, hashes regular artifacts, previews saved JSON/CSV/
TSV values and links figures. Preview limits and unsafe/unreadable formats are
reported explicitly. Row-level text/contact tables are linked without copying
their data. It does not calculate missing metrics or choose a model winner.
Keep result files unchanged while collecting. Review the report before sharing.
