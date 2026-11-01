# Validity, limitations, ethics and honest conclusions

This guidance complements [duplicate design](duplicate_experiment_design.md),
[priority design](priority_experiment_design.md) and [annotation rules](annotation_guideline.md).
It does not supply research outcomes or a completed ethics approval.

## Current evidence boundary

The workspace readiness record reports an imported source corpus and a small
derived multilingual annotation starter. Exact normalized-text grouping is a
leakage-control fallback, not verified incident identity. English translations,
natural collection, independent duplicate gold labels and expert queue judgments
still require the documented work. See [current workspace](current_workspace.md)
for the recorded import/starter counts; do not recast those counts as final
research sample sizes.

The natural test template and expert-judgment template are not evidence of
completed collection. At this release the results directory has no saved
research experiment outputs. No natural-test accuracy, expert-label count,
proposed-method gain, fairness gap or real neural-checkpoint latency is inferred
from implementation tests. When collection changes, report the measured counts
and date/provenance rather than retaining this status paragraph as a result.

## Threats to validity

| Threat | Practical consequence | Mitigation and remaining limit |
| --- | --- | --- |
| Construct: severity versus operational urgency | Source severity, expert priority, rule points and ranking utility measure different things. | Declare each scale and its source. Never manufacture expert 1–5 labels from source severity or score bands. |
| Construct: duplicate text versus one real incident | Translations and similar wording may hide repeated but distinct service failures. | Independent incident annotation with area/time/context; audit both difficult positives and negatives. IDs and timestamps alone are insufficient. |
| Internal: incident leakage | Related variants split across partitions inflate generalization estimates. | Group by verified incident and query lineage; preserve derivations and fingerprints. Exact-text fallback cannot discover every paraphrase. |
| Internal: construction shortcuts | Positive translation pairs and negative category/location rules can make labels predictable without incident reasoning. | Adjudicate gold labels; include natural and same-language examples, same-category hard negatives and diverse real metadata patterns; hold pair sets fixed across ablations. |
| Internal: category oracle/context confounding | Duplicate features using provided category may assume information unavailable in production. | State category source and repeat without it; report predicted-category behavior separately. Use C0/P1/P2 to isolate feature additions within the same learner. |
| Internal: test-set adaptation | Repeated rule/alias/threshold tuning on final errors invalidates a final holdout. | Prespecify controls; use training/validation only; retire and replace a holdout if it becomes development data. Record protocol changes. |
| Measurement: small or unrepresentative annotation | Disagreement, rare classes and tiny natural/expert cohorts make estimates unstable. | Report supports, qualifications, disagreement/adjudication and eligible cases; use valid uncertainty analysis when available. Kappa and correlation can be undefined. |
| Measurement: calibration reuse | Policy selection on the same validation rows used for calibration can be optimistic. | Prefer separate policy validation or nested design; disclose reuse. A validation error ceiling is not a future risk bound. |
| Measurement: missing location/time | Missingness may correlate with language, area or complainant resources. | Explicit missing/ambiguous/conflict features; retain the same pairs; report support/missingness and sensitivity analyses. Do not impute real event dates from the current clock. |
| External: derived language | Deterministic transliteration/translated complaints differ from spontaneous English/Banglish. | Keep derived flags, assess translation quality and evaluate an independent natural cohort separately. Derived success alone does not establish natural-language success. |
| External: geographic/domain coverage | A Dhaka-focused gazetteer and one source corpus may not cover other districts, dialects or service systems. | Declare actual area/variant coverage, unknown rates and source domain. Treat expansion as new evaluation, not assumed transfer. |
| External: deployment resources and drift | Free-host sleep, CPU memory, connectivity, new vocabulary and changing services affect production. | Measure on actual hardware/cohort, report cold/warm states and cache behavior, monitor drift and review workload. Structural tests are not a throughput or recall benchmark. |
| Conclusion: many analyses/seeds | Selecting one favorable slice can create a misleading superiority claim. | Preserve complete results/supports and all prespecified runs; state primary endpoints and multiple-comparison choices before interpretation. |
| Conclusion: clustering/retrieval | Pair-test scores do not directly measure candidate recall or final cluster correctness. | Evaluate retrieval/cluster outputs against separately verified incident groups if claiming those outcomes; discuss top-k misses and connected-component chaining. |

## Limits that must be reported with the final dataset

### Derived and natural language data

Give the exact number of source incidents and derived variants in every split.
Document deterministic transliteration and human/translation-tool provenance,
quality review and failures. Keep the independent natural set out of dictionary,
rule and model selection. Report each natural language/class support and the
number with severity labels. A small natural set supports a limited estimate,
not a broad population or district-level generalization claim. If it is empty,
say that natural generalization was not evaluated.

### Expert and annotator judgments

State how many people independently annotated which items, their expertise and
any overlap with authors/volunteers. Count experts, ratings, unique items,
incidents, queues/queries and derived pairwise comparisons separately. Report
agreement on shared items before adjudication and the final adjudication rule.
Pairwise comparisons generated from a few expert queues do not increase the
number of independent experts or incidents. If no qualifying judgments are
available, LTR remains untrained or falls back to rules; that is a limitation,
not evidence that a learned ranking model works.

### Uncertainty, fairness and explanations

Give denominators and undefined metrics explicitly. Preserve low-support and
unknown groups instead of replacing missing errors with zero. Observed score
differences may reflect need, composition, missingness, annotation or the model;
conditional summaries do not identify causal unfairness. Equal aggregate scores
do not guarantee equitable service.

Rule contributions explain configured points. Linear coefficient contributions
explain a base margin, and approximate token deletion explains context-dependent
sensitivity. Neither establishes causality, truth of a complaint, expert
correctness or a calibrated probability contribution. Classification confidence
does not certify priority or same-incident identity.

## Ethical considerations and operational boundaries

- Obtain an appropriate institutional review/ethics determination before real
  volunteer/expert collection or release. Preserve consent, withdrawal scope,
  compensation and recruitment details; repository code is not approval.
- Minimize personal information. Do not request government IDs, detailed medical
  histories or unnecessary contact information. Voluntary contact must remain
  private to authorized staff and outside ML/context/export paths unless there
  is a separately justified, consented purpose.
- Use the public privacy notice and a deployment-specific retention/deletion
  procedure. Consent to annotation does not automatically authorize public
  release, cloud transfer, model training or indefinite retention.
- Redact or paraphrase identifying locations/contact details before publishing
  error examples, figures, demo GIFs, screenshots or annotation exports. Review
  reasons and explanations may contain private text even when the separate
  `text` column is omitted.
- Preserve disagreement and allow correction. Admin overrides need reasons and
  remain feedback candidates for adjudication, not unquestioned gold labels.
- Avoid claiming demographic attributes from school/hospital/elderly keywords.
  Such words are triage cues, not verified identity or harm estimates.
- Keep accountable human review for life-risk/context uncertainty and important
  operational decisions. The portal is not a verified emergency dispatch system,
  and automatic priority must not promise a response time or decide entitlement
  to essential services without an authorized process.
- State database/model hosting regions, processors, access roles and incident
  response responsibilities for the actual deployment. Free hosting does not
  remove privacy, security or institutional obligations.
- Reassess model performance, language coverage and review workload after data
  or deployment changes. A recorded checkpoint and a test suite do not eliminate
  future drift or a risk of harmful false merges/under-prioritization.

## Honest wording when the proposed method does not beat a baseline

Use these as wording patterns after inserting only saved measurements and
supports into the surrounding results table. These sentences deliberately
contain no fictional numerical result:

> On the prespecified held-out cohort, the proposed method did not outperform
> the baseline on the primary metric. The study therefore does not establish
> overall superiority under this evaluation design.

> The location/time additions changed the observed error pattern. Any claimed
> reduction in a particular error type is restricted to the reported stratum
> and support; it does not establish a general improvement or causal effect.

> The derived-language cohort and the natural-language cohort gave different
> outcomes. The findings should not be generalized from translated examples
> to naturally written Banglish without the independent natural-set evidence.

> The available expert data did not meet the training-sufficiency conditions.
> The deployed ranking path used its documented rules fallback, so this run
> does not evaluate a successfully trained LTR model.

> The confidence policy met the stated empirical criterion on its validation
> cohort. This result does not guarantee future accepted-error rates, and the
> additional operational review rules changed actual workflow coverage.

> The contribution is a reproducible multilingual triage pipeline, documented
> annotations/design and evidence about the tested trade-offs. The available
> results do not support a claim of a universally better priority or duplicate
> detector.

If uncertainty intervals overlap or a significance test was not prespecified
or performed, do not say the methods are equivalent or significantly different.
Say the evidence is inconclusive at the reported support and describe the
observed measurements. Do not rename an unmeasured convenience, deployment or
explanation advantage as an accuracy improvement.
