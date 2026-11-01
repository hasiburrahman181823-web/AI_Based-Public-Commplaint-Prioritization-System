# Complaint annotation guideline

Use two separate tasks: decide whether a pair describes the same real-world
incident, and rate the priority of each individual complaint. A duplicate label
does not establish priority. A high priority does not establish duplication.
Source `severity` labels remain the dataset's original labels; the human
`priority` rating below is a separate field and scale.

## Duplicate decision: 0 or 1

- **1 — duplicate:** both reports describe the same identifiable real-world
  incident or the continuation of that incident. Wording and language may differ.
- **0 — different incident:** the reports describe distinct incidents, including
  distinct events within the same category.
- **Skip:** available evidence is insufficient, contradictory, or too ambiguous
  to decide. Do not guess a label to finish a task. Leave a note explaining what
  evidence would resolve it.

Category agreement, lexical similarity, or a shared neighborhood alone is not
enough for a positive label. Compare the affected service or asset, location,
event dates, duration, and any distinctive details. Do not infer that two groups
must be different incidents just because they have different `original_id`
values; those values are provenance information, not independent ground truth.

An explicit conflict in locations or event dates is evidence of distinct
incidents. Distinguish event time from report time: a report submitted tomorrow
may describe today's continuing outage. An explicit account of a continuous
three-day outage can support duplication with an earlier report of that same
outage. Two separate outages after a stated restoration are separate incidents.
Conflicting details that could be a reporting error require review or a skip;
do not invent an explanation.

These are illustrative examples, not collected test records:

| Pair | Decision and reason |
| --- | --- |
| “ধানমন্ডি ৭ নম্বর রোডে সকাল ৮টা থেকে পানি নেই” and “No water on Dhanmondi Road 7 since 8 this morning,” with matching event date | 1: same service, road, and onset. |
| “Mirpur 10 transformer caught fire today” and “Uttara Sector 4 transformer caught fire today” | 0: separate explicitly located assets, despite the same category. |
| “No electricity since Monday on Road 3” and a Wednesday report explicitly describing that Road 3 outage continuing since Monday | 1: continued event; different reporting dates are consistent. |
| “Water is unavailable” and “pani nai,” with no location, dates, or event linkage | Skip: a generic similarity does not identify one incident. |

Write a short reason citing evidence, such as “same road and Monday onset” or
“different named transformers and neighborhoods.” Avoid reasons that merely
repeat the label, such as “duplicate because similar.”

## Individual complaint priority: 1–5

Rate the complaint itself, once per complaint ID. Consider stated human safety,
duration, affected people, and disruption of essential services. Use evidence
actually present in the complaint and its approved context. Do not invent an
outage duration or population count, or apply a mechanical formula to estimates
that the reporter did not supply.

| Priority | Anchor | Example of sufficient stated evidence |
| --- | --- | --- |
| 1 | Routine, minor issue | Minor cosmetic damage without an indicated safety risk or service disruption. |
| 2 | Small, localized disruption | A localized service problem affecting a small stated area, without evidence of immediate danger or severe impact. |
| 3 | Significant disruption, without immediate danger | A substantial stated interruption or access problem; no evidence of imminent injury. |
| 4 | Severe, prolonged disruption affecting many people or an essential service | An explicitly prolonged widespread water outage, or a severe essential-service interruption with major stated effects. |
| 5 | Imminent human safety threat | Exposed live wiring endangering pedestrians, active fire, or another explicit immediate threat to people. |

Use the anchors with judgment; an immediate safety threat can warrant 5 even
when only one person is at risk. An unspecified duration or impact is unknown,
not zero. If the text does not support a defensible rating, skip and state what
is missing. Do not convert a source `severity` value directly into this scale.

Every saved priority needs a short reason that links the chosen level to stated
evidence, for example “5: exposed live wire beside an occupied footpath” or
“3: the report states an access road is blocked; no immediate injury risk is
described.” Annotation is a research judgment, not an emergency dispatch action.

## Independent annotation and review

1. Use a stable local pseudonym in the app's annotator field. It is self-asserted
   identity for recordkeeping, not an authenticated account.
2. Annotators work independently on an overlapping sample in two passes: pair
   duplication first, then individual priority. Do not coordinate labels during
   these passes. The app hides automatic pair labels, pair types, provenance
   cues, and other annotators' answers from the labeling view.
3. Save each decision with its reason. Use the skip option for uncertainty;
   a skipped decision is unlabeled, not a negative or a low priority.
4. Compare agreement only after both independent passes. Have a reviewer resolve
   disagreements using the evidence and this guideline. Preserve the original
   individual labels and reasons; maintain any adjudicated gold decisions
   separately rather than rewriting independent annotations to improve agreement.

Translations, rule-based transliterations, and manually written paraphrases are
**derived candidates**. Their common provenance proposes a duplicate label but
does not prove faithful meaning preservation. Human reviewers must verify that
translation or paraphrasing has preserved the incident, location, time, and
important qualifiers before treating a positive pair as gold. Automatically
generated hard negatives can be false negatives: different source groups or
metadata do not guarantee distinct incidents. Review negative candidates too.

## Agreement calculation

For an annotator pair, compare only overlapping records that both annotators
actually labeled for that task. Exclude skips, missing labels, and records seen
by only one annotator. Do not mix pair-level duplicate decisions with
complaint-level priority ratings.

- Duplicate labels use standard **unweighted Cohen's kappa** on 0/1 decisions:
  `kappa = (observed_agreement - expected_agreement) / (1 - expected_agreement)`.
- Priority uses **quadratic weighted Cohen's kappa** on levels 1–5. Disagreement
  weights are `((i - j) / 4)^2`, with expected counts calculated from each
  annotator's marginal distribution. Adjacent disagreements receive less weight
  than widely separated ratings.

Report the overlap count alongside each statistic. With no eligible overlap,
or when the expected-disagreement denominator is zero (for example, both
annotators give every overlapping item the same constant label), kappa is
undefined. Return JSON `null` with an explanation rather than reporting perfect
reliability. Review the disagreement cases and class distributions as well as
the kappa values; a single statistic is not a substitute for adjudication.

## Independent natural test collection

`templates/natural_test_set.csv` is an empty collection template with columns
`text,lang,category,location,timestamp`. Collect real, independently authored
complaints for a separate held-out evaluation set. Do not fill it with generated
examples, translations, transliterations, or paraphrases of training records.
Keep it outside candidate generation and training. Check for overlapping
incidents with existing data before using it as an independent evaluation set.

Use permission or consent appropriate to the collection source. Collect only
what is needed: coarse incident location and event/report time when available;
omit names, phone numbers, precise home addresses, and other unnecessary
identifiers. Redact personal details before sharing annotation files. Missing
location or time remains empty. Preserve ISO-8601 timezone offsets when known.
Have humans review language, category, and evaluation labels rather than
inventing them. The annotation app accepts this five-column collection format
directly for priority review, generating stable IDs when they are absent.
