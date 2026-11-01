# Priority-ranking experiment design and validity limits

This is a prespecified research design for Phase 5. It provides no research
scores or production guarantees. Record departures before interpreting a final
comparison, and do not revise rules or labels repeatedly against final-test
errors while retaining a held-out-test claim.

## Separate constructs and decisions

| Quantity | Meaning | What it does not establish |
| --- | --- | --- |
| Source/predicted severity | Dataset severity label on its documented ordered scale. | An expert 1–5 priority rating, a queue rank, or an emergency decision. |
| Rule priority score | Explicit YAML component points times configured weights. | A probability, a validated urgency estimate, or unique people affected. |
| Rule priority label | Configured band over rule points. | A relabeling of severity or a learned expert grade. |
| Expert priority | Independently reviewed/adjudicated relevance grade 1–5 in a specified queue context. | A label generated from source severity or from the proposed rules. |
| LTR ranking score | Pairwise-logistic linear utility over transparent component features. | A calibrated probability, a rule-points score, or a 1–5 rating. |
| Classification confidence | Calibrated category/severity predicted-class probabilities and their minimum selection statistic. | Calibrated joint correctness, priority correctness, duplicate correctness or future error control. |
| Human-review status | Workflow flag for low confidence, missing support or explicit risk/context triggers. | Verification of an incident, hazard, demographic fact, or fairness outcome. |

Preserve these names in tables and explanations. Missing models/labels remain
unavailable; there is no fallback that manufactures a confident category or
severity prediction. Rule cues may still be shown with explicit review status.

## Units, ground truth and split design

The ranking unit is a complaint within an expert-defined query/queue context.
The independence unit is the real incident; known variants and translations
must remain together. Query contexts also remain in one partition. Negative
evidence about unrelated incidents, source IDs or exact-text hashes alone does
not prove semantic independence.

Experts provide a final relevance grade and optionally a complete explicit
ranking within each query. Preserve individual judgments, reasons, expertise
criteria, adjudication and agreement records. Volunteered labels are not
automatically expert labels. Do not create grades or queries from the proposed
score to make the target agree with the model. Source severity and the earlier
annotation tool's independent priority judgments remain distinct records until
the stated expert protocol establishes how they are used.

Fit pairwise-logistic coefficients only on training-query comparisons. Keep the
rule configuration and model hyperparameters fixed for the primary comparison.
The validation/test ranking reports are diagnostics of those fixed scorers;
final test judgments do not select weights, aliases, features or sufficiency
minima. If data are insufficient, report the explicit rule fallback and its
reason. An LTR row identical to rules under fallback is not evidence that a
learned method was trained or improved ranking.

## Primary ranking comparison

Compare severity-only, full rules and LTR on exactly the same ordered held-out
items and query identities. Severity-only uses configured severity points;
rules use all configured components; LTR uses its learned utility or explicit
fallback. Retain the cohort fingerprint, supports and per-query outcomes.

NDCG@5 uses exponential relevance gain `2**expert_priority - 1`, the available
prefix when fewer than five items are present, and averaged DCG across tied
predicted scores. It uses expert grades even when a separate explicit expert
rank is supplied. Spearman uses average descending-score ranks and explicit
expert ranks, or relevance-derived ranks when explicit ranks are absent; rank
1 is most urgent in both. Explicit ranks and grades can disagree, so document
the separate meaning and annotation of each rather than silently reconciling
them in the test set.

Report equal-query macro averages only across queries for which a metric is
defined, together with eligible-query counts. Constant relevance has no
discriminatory NDCG evidence; constant predicted/gold ranks have undefined
Spearman. Small, empty or non-discriminatory queries must remain visible instead
of receiving fabricated perfect measurements or being silently removed.

## Human-review operating point

Use the exact saved calibrated classifiers to generate validation predictions.
The exporter verifies their recorded validation input where available and
retains artifact fingerprints. Accepting a CSV's asserted split or disabling
provenance verification cannot independently prove that it was held out.

Define joint classification error as either category or severity incorrect.
Use the minimum of their predicted-class probabilities as a selection score,
then choose maximum observed validation coverage satisfying the prespecified
empirical error ceiling and minimum accepted support. Equal-confidence rows
move together. An infeasible ceiling/support constraint produces reject-all,
not a handpicked threshold that violates the stated policy.

This confidence-only curve is not automatically the deployed review curve.
Life-risk keywords, language ambiguity, extraction conflicts, absent models or
policy and other missing context can cause additional reviews. Report actual
workflow coverage separately. A review policy is not a calibrator for ranking
utility, priority labels or duplicate scores.

The Phase 3 validation set was already used for probability calibration.
Reusing it for review-policy selection may overfit the observed operating
point. Prefer a separate prespecified policy-validation set or an explicitly
reported nested design, keeping incidents isolated and the final test outside
selection. The empirical constraint is not a finite-sample risk bound, future
error guarantee or a guarantee for each area/language subgroup.

## Explainability interpretation

Rule contributions are exact on their configured additive points scale,
including any separately recorded final cap. They expose assumptions but do
not establish that those assumptions are empirically correct or just.

For a linear TF-IDF classifier, active feature value times class coefficient
explains its base raw margin. Intercepts, binary coefficient direction,
multiclass comparison and overlapping character features matter. A subsequent
sigmoid calibration transforms margins; margin contributions are not additive
probability contributions or a causal decomposition of the complaint.

Linear embedding-coordinate contributions can explain the fitted numerical
utility, but coordinates have no guaranteed lexical interpretation. Approximate
token removal/occlusion changes contextual embeddings and is not an exact
coefficient term or SHAP attribution. Ranker component contributions are a
separate linear-utility decomposition. Display each scale and method explicitly.

## Fairness audit design

Use reviewed evaluation rows with explicit languages and canonical areas where
available. Report score distributions, review rates and supports first. Error
disparities require compatible ground truth and predicted labels on the same
rows and scale. Missing error labels yield unavailable measurements, not zero
error. Do not compare arbitrary rule scores with expert integers as if they
shared a scale.

Compare enough-supported groups and retain small/unknown groups descriptively.
Inspect category/severity-conditional summaries as well as marginal gaps.
Those summaries can expose composition differences, but they neither remove
all confounding nor establish a causal effect of language or area. Avoid
asserting systematic discrimination, equal treatment or a fairness guarantee
solely from a mean-score difference or its absence.

## Main threats to validity

| Threat | Consequence | Reporting/design response |
| --- | --- | --- |
| Unvalidated rule weights and label bands | Plausible priorities may be inappropriate for the actual agency or expert policy. | Elicit/freeze expert policy, audit component effects and evaluate independently rather than treating readable rules as validated decisions. |
| Keyword coverage and context | Negation, idioms, quoted text, historical hazards, spelling variants or dialects can create missed/false boosts. | Retain matches and negated evidence, audit extraction by language, and use review triggers. Adjacent negation is not general semantic hazard detection. |
| Language identification | The three-argument API cannot always distinguish English from romanized Bangla; normalization and keyword coverage can differ. | Prefer explicit reviewed language in evaluation/batch inputs; report heuristic/ambiguous cases and subgroup support. |
| Classifier error propagation | Category/severity mistakes change rule features and learned ranking input, even with correct keyword evidence. | State whether ranking experiments use gold/source or predicted category/severity; evaluate the actual upstream pipeline separately and retain classification errors. |
| Report volume mistaken for population | Translations, repeated submissions, spam or coordinated reporting may inflate apparent group size; underserved areas may report less. | Deduplicate known lineage, label the count as reports, cap its contribution, and do not claim people affected. Audit accessibility/underreporting and adversarial reporting separately. |
| Retrieval/cluster uncertainty | Top-k truncation misses reports; false duplicate bridges merge incidents; exact-text IDs miss variants. | Expose retrieval limits and lineage assumptions. Pair tests and structural 50k checks do not establish counting or cluster validity. |
| Missing durations and event-time ambiguity | Absence is not zero need; reported duration may refer to something other than the service failure. | Keep missingness/ambiguity visible, never use the current clock to invent onset, and audit duration coverage by group. |
| Expert disagreement and query construction | Inconsistent grades, ranks, expertise or artificial queues can create a convenient but invalid target. | Preserve reasons/protocol and adjudication, define actual queue context, report rank/grade disagreement and test independent queues. |
| Correlated variants and paired comparisons | Many rows/comparisons share incidents, giving an inflated impression of independent evidence. | Isolate incidents/queries; resample at an appropriate incident/query level for any uncertainty analysis. This implementation does not fabricate independent-row confidence intervals. |
| Sufficiency fallback selection | Changing minima after seeing results can disguise a weak learned model or trivial rule equality. | Prespecify minima, report fallback status and reason, and do not claim LTR learning when fallback is active. |
| Scale conflation | Rule points, utility, class confidence and expert ratings can be presented as one quantity. | Keep fields/scales and explanation methods separate. No arbitrary raw-score-to-expert-grade conversion is assumed. |
| Calibration and policy-validation reuse | A small reused validation set can make accepted-error estimates optimistic. | Prefer an independent policy-validation cohort, preserve model/input fingerprints and evaluate the frozen policy once on final held-out data. |
| Subgroup prevalence and label coverage | Mean-score/error gaps can reflect differing need or missing labels rather than model bias alone. | Report supports, truth coverage, conditional summaries and selection effects; retain unknown/small groups. |
| Small-group instability and multiple comparisons | Noisy gaps or selective subgroup reports can create unsupported fairness claims. | Prespecify dimensions/support thresholds and report all eligible/unsupported groups; descriptive gaps are not causal proof. |
| Review capacity and human behavior | Escalation may exceed staffing, and human review itself can be delayed or biased. | Measure queue load, actual coverage and reviewer outcomes under a stated operational policy; confidence-only curves omit these effects. |
| Emergency-policy overreach | A research classifier may be interpreted as reliable emergency dispatch or clinical triage. | Keep explicit human review and institution-approved policy boundaries; this research code supplies no validated emergency outcome claim. |
| Shift and drift | Different districts, language styles, disasters, agency priorities or new model versions change score/error behavior. | Version/freeze resources and expert policy, monitor later reviewed data and revalidate compatible confidence policies after model changes. |

The final thesis record should retain exact models/checkpoints, input hashes,
rule/keyword resources, expert query definitions, incident partitions,
calibration and policy-validation cohorts, sufficiency rules, subgroup supports,
review reasons and output manifests. Synthetic implementation tests and
unavailable-data fallbacks do not establish empirical ranking quality or
fairness.
