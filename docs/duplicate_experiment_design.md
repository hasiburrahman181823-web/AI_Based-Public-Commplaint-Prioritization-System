# Duplicate-detection experiment design and threats to validity

This document specifies the intended comparison before interpreting results.
It contains no measured performance claims. Record departures from this plan
with the experiment outputs instead of selecting a favorable protocol after
inspecting the final test set.

## Research question and unit of analysis

The task is binary same-incident detection: two complaints can be duplicates
when they describe one real-world incident, despite different languages or
wording. Sharing a category, neighborhood, or generic phrase is insufficient.
Distinct incidents can have similar descriptions and nearby report times.

The measurement unit is a complaint pair. The independence unit for partitioning
is the incident identified by `original_id`: all known translations,
transliterations, paraphrases and other variants stay in one partition. Negative
pairs connect different incident groups within that same partition. A true
incident discovered to span multiple source IDs must be audited and grouped
upstream before relying on split isolation. String IDs cannot establish semantic
independence by themselves.

Use separate train, validation and final test pair sets. Resolve language and
category from complaint metadata by stable row ID; neither comes from the
duplicate label, candidate pair type, nor a cross-lingual naming convention.
Features do not include IDs, `original_id`, the candidate pair type, or any human
duplicate decision. Preserve those fields only for split checks and analysis.

## Labels and comparison cohort

Final research labels should be independently written/adjudicated human
same-incident decisions under the annotation guideline. Keep the original two
annotators' records and reasons, agreement statistics, and the separate final
gold table. A `gold_label` column is a provenance declaration, not an automatic
quality guarantee: humans still need to verify grouping and translation fidelity.

Phase 2's generated `label` proposes positives and negatives from construction
rules. Running with `--allow-weak-labels` is an explicitly weak-label/synthetic
experiment and must be reported as such. It cannot substantiate an independently
annotated cross-language detection claim.

All selected ablations use exactly the same frozen test pair IDs and gold
labels. Do not filter a model's difficult cases, discard missing-metadata pairs,
or construct different test sets for each feature configuration. Missingness is
represented explicitly, and support is reported for each analysis slice. Record
the pair-set fingerprint and category-source mode with the measured outputs.

## Prespecified models and feature semantics

| Model | Learning inputs | Validation use |
| --- | --- | --- |
| B1 | Character `char_wb` TF-IDF cosine, n-grams 1–5; vocabulary and IDF fit once per complaint ID appearing in training pairs. | Choose a scalar decision threshold by pair F1. |
| B2 | Cosine of L2-normalized frozen multilingual MiniLM sentence embeddings. | Choose a scalar decision threshold by pair F1. |
| C0 | Learned classifier over embedding cosine, category match/missingness, and unordered language-pair indicators. | Choose the classifier-score threshold by pair F1. |
| P1 | The same learned classifier and C0 features plus location match/difference/missingness/conflict. | Same threshold rule. |
| P2 | The same learned classifier and P1 features plus timestamp gap, duration gap, tentative interval overlap and missingness. | Same threshold rule. |

Use a fixed learner for nested C0/P1/P2 comparisons. The default is a scaler and
class-balanced logistic regression; gradient boosting with balanced training
sample weights is a separate prespecified learner comparison. Fit all learned
parameters on training pairs only. Frozen pretrained encoders do not learn from
the evaluation pairs. Fix the model checkpoint/revision and preprocessing
resources across the embedding models.

`loc_match` means an exact common canonical area ID from the gazetteer.
`loc_distance_flag` is a categorical flag for known distinct non-ancestor areas;
it is **not** a kilometre distance, geographic proximity estimate, or proof that
incidents differ. An ancestor/descendant location relationship is not equivalent
to two precisely equal locations. Missing/ambiguous extraction is distinct from
observed disagreement; explicit metadata/text conflicts are represented.

`time_gap` is an absolute difference between supplied timestamp anchors in
hours. `duration_gap` compares supported extracted duration estimates. Ongoing
phrases can form tentative intervals ending at a supplied timestamp; overlap is
unknown when intervals are not supported. Missing values use explicit indicators
rather than interpreting an absent duration or time as zero elapsed time. No
current-time value is used to manufacture an event start.

Category match uses available complaint category metadata when
`--category-source provided`. Repeat with `--category-source none` as a
sensitivity analysis. The unordered language-pair indicators distinguish all
cross-language and same-language combinations; they are available context, but
can also encode dataset-construction shortcuts.

The added C0 control is necessary for attribution. P1 versus C0 isolates the
location feature addition, and P2 versus P1 isolates time features, conditional
on the chosen learner and context inputs. B2 versus P1 changes more than location:
it introduces a learned classifier, category context, and language-pair context.
Do not label that entire difference a location-only gain.

## Thresholds and metrics

Choose thresholds on validation pairs only, maximizing positive-class pair F1
with ties favoring the higher threshold. Predictions use `score >= threshold`.
Save the threshold, policy, validation support and actual model parameters.
The thresholded test measurements are an evaluation of that frozen operating
point; test labels do not select the threshold, learner or resource rules.

Report overall precision, recall and F1, together with support and positive
prevalence. Here `pr_auc` is average precision, the step-integral of precision
over recall; the independently named trapezoidal PR area is a different
interpolation convention. Use one stated primary definition consistently.
Cosines and learned ranking scores are not asserted to be calibrated duplicate
probabilities. A high score is not an independently verified confidence level.

Report the same metrics/support by candidate pair type and by `bn-en`,
`bn-banglish`, `en-banglish`, and observed same-language pairs. Empty strata have
null measurements. Single-true-label slices cannot estimate ranking
discrimination; their PR-AUC is explicitly undefined. For hard/easy negative-
only slices, false-positive rate and negative support are more informative than
positive-class F1. Do not interpret a zero-division convention as observed
negative-detection failure or success without inspecting the counts.

## Main threats to validity

| Threat | Consequence | Design response and reporting limit |
| --- | --- | --- |
| Small Dhaka-centered gazetteer | Place coverage and alias accuracy may not generalize across Bangladesh, rural areas, dialects or informal landmarks. | Describe coverage, inspect unknown/ambiguous rates, and evaluate on geographically broader human data before claiming national validity. Freeze resources before the final run. |
| Fuzzy aliases and ambiguous place names | Similar spellings or generic parent areas can create false matches; neighboring numbered areas need not be one incident. | Preserve ambiguity/conflict and missingness; audit numeric aliases and extraction errors. Similarity scores are heuristic, not calibrated extraction confidence. |
| Metadata availability and quality | A populated source location field can make a laboratory task easier than production free text; incorrect fields may override correct textual evidence. | State whether features use provided metadata, extracted text, or both. Report missingness/conflicts and evaluate a realistic availability scenario separately. |
| Report time versus event time | Two reports on different dates can describe a continuous outage, while matching report dates can conceal separate incidents. | Treat timestamp gaps and inferred intervals as uncertain features. Validate temporal rules against human incident context; never declare a date difference sufficient ground truth by itself. |
| Duration parsing limits | Numeric ranges, multiple durations, unsupported number words and references such as “three days ago” can be misread or unresolved. | Keep unresolved values unknown, review extraction outputs, and report coverage. A deterministic rule is reproducible but not inherently accurate. |
| Hard-negative construction uses location/date | A model may rediscover the same rule used to assign negative suggestions, creating circular evidence for location/time benefits. | Use independent human same-incident labels and naturally occurring difficult cases. Clearly label weak-label experiments; do not equate rule recovery with real-world novelty. |
| Language-pair construction shortcuts | Generated positives are often cross-language, while inferred-style hard negatives often share a language. Language indicators can predict the construction rule rather than incident identity. | Inspect label prevalence within every language pair, collect human cross-language hard negatives and same-language positives, and use C0 to expose context-only gains. Missing negative support prevents a full cross-language discrimination claim. |
| Derived translations/transliterations | Literal shared wording and copied metadata can inflate cross-language similarity; transliteration mistakes can also corrupt proposed positives. | Mark derivations, verify fidelity, and distinguish derived test results from independently authored natural-language complaints. Natural collection must not be recycled source translations. |
| Gold/oracle category context | Human category labels may exceed the accuracy of an upstream production classifier and share structure with pair-generation rules. | Declare the category source. Compare `provided` versus `none`; a no-category run does not measure errors from a predicted category pipeline. Evaluate such propagation separately if production categories are predicted. |
| Pair sampling and class prevalence | Precision, PR-AUC and an F1-selected threshold depend on the sampled positive/negative mixture, which may differ greatly from a large complaint store. | Report construction, prevalence, pair types and operating thresholds. Validate the intended production decision policy on representative data rather than transferring a balanced-sample precision claim. |
| Correlated pairs and copied incident variants | Many pairs share complaints or incidents, so treating rows as independent exaggerates precision of statistical conclusions. | Keep incidents isolated across partitions. Any uncertainty analysis should resample at incident/block level under a specified dependency design, not naively bootstrap individual pair rows. This implementation does not claim independent-pair confidence intervals. |
| Validation reuse and small support | Comparing many learners, gazetteer settings and thresholds on one small validation set can overfit it. | Prespecify comparisons, keep hyperparameters fixed where practical, report the selection process, and use new calibration/validation data for substantial revisions. Final test labels must remain outside selection. |
| Final-test error-driven edits | Correcting aliases or rules after seeing test errors turns the test set into development data. | Use error analysis descriptively for the frozen run. Evaluate revisions on a new holdout or explicitly report the loss of final-test status. |
| Incorrect `original_id` provenance | Exact-text hashes do not discover paraphrases, translations or distinct IDs for one real incident, allowing semantic leakage despite syntactic checks. | Audit incident group identities before splitting and revise upstream group assignments when adjudication reveals related records. |
| Reused classification/auxiliary datasets lack incident metadata | A classification dataset may contain complaint labels without verified incident identity, location, timestamp or duplicate decisions; an auxiliary language resource may have a different domain. | Use only actual source fields, keep missing metadata unknown, and obtain real duplicate annotation/provenance. Text translations and auxiliary spelling resources cannot establish incident gold or fill unavailable locations/dates. |

## Retrieval and clustering are separate evaluation questions

The production index uses a matrix–vector dot product over stored normalized
embeddings, with storage proportional to the number of complaints times the
embedding dimension. It avoids a corpus-by-corpus matrix and targets a corpus
on the order of 50k complaints. This is an architectural scale target, not a
measured latency, memory, or recall result. Hardware, encoder/device, query
batching, cache state, corpus changes and metadata processing affect deployment
cost. Benchmark them on the intended system.

For learned models, the retrieval stage selects top embedding-cosine candidates
before location/time reranking. A missed candidate is never recovered by the
reranker. Pair-test scores evaluate the supplied fixed pairs; they do not measure
end-to-end candidate recall from a large complaint store. Measure retrieval
recall and final retrieval-plus-reranking quality separately, with a fixed
candidate budget selected independently of final test labels.

Connected components use thresholded predicted edges. One erroneous bridge can
merge different incidents, and a transitive A–B–C chain can merge endpoints
without direct pair support. Conversely, a sparse sampled graph can miss the
edges required to join genuine duplicates. Therefore pairwise precision/F1
does not prove cluster precision, incident recovery, or consistency. Retain
singletons, inspect bridges and large components, and evaluate against an
independently adjudicated incident clustering before making cluster-quality
claims. No such cluster metric is fabricated by this workflow.

## Minimum thesis reporting record

Retain the input fingerprints and frozen pair IDs, incident split definitions,
label/adjudication provenance, natural/derived composition, source-language
distribution, per-stratum support/prevalence, metadata availability, gazetteer
and normalization snapshots, model checkpoint/revision, feature sets, learner
parameters, seeds, threshold-selection policy, dependency versions and output
manifests. Report measured ablation tables with these conditions and limits.
Implementation tests, including synthetic encoders and structural index checks,
are not empirical model results or evidence of production throughput.
