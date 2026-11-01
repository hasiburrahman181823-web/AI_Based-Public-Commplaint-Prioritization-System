# Phase 2: multilingual candidates, pairs, and human labels

Run these commands from the repository root after Phase 1 preparation. The
existing train/val/test assignments are authoritative. Derived rows inherit their
source `original_id` and split; do not split generated versions again.

## 1. Export a multilingual work set

```powershell
python -m ml_training.scripts.build_multilingual_set export --input-dir ml_training/data/processed --output-dir ml_training/data/multilingual --sample-size 100 --seed 42
```

The seeded sample selects one Bangla representative per selected `original_id`.
Choose a sample size no larger than the number of available Bangla groups;
the export reports an error if there are too few. The work set contains those
sampled representative source rows and derived rule-based Banglish
transliterations in the same splits. Other Phase 1 rows remain in the original
processed files. Rule-based transliteration is a
script conversion candidate, not phonetic modeling or English translation.
Review its quality before using it for evaluation.

The output `train.csv`, `val.csv`, and `test.csv` contain the standardized nine
complaint columns plus `is_derived`, `derivation`, and `parent_id`. Source rows
have `is_derived=false`, `derivation=source`, and an empty `parent_id`;
transliterated rows have `is_derived=true`,
`derivation=rule_transliteration`, and their source Bangla row ID as `parent_id`.

`translation_queue.csv` contains:

```text
translation_id,original_id,source_id,split,source_text,category,severity,location,timestamp,english_text
```

Copy the queue to a working file such as `filled.csv`. A human translator fills
**only `english_text`**, preserving meaning, location, time, and qualifiers. Keep
the IDs and all other queue metadata unchanged; import validates them against
the export snapshot. Do not add invented details or translate category labels.

```powershell
python -m ml_training.scripts.build_multilingual_set import --work-dir ml_training/data/multilingual --translations filled.csv
```

Import adds derived English rows (`derivation=manual_translation`) to the source
splits. Full queue coverage is required by default. Use `--allow-partial` only
when intentionally importing a subset; missing or blank translations are skipped.
Duplicate/unknown translation IDs or changed immutable metadata are errors.
Use `--overwrite` to deliberately replace an existing export or translation
import, and retain a reviewed copy of translator work before doing so.

## 2. Add optional manual paraphrases

`templates/manual_paraphrases.csv` is header-only:

```text
id,original_id,text,lang,category,severity,source_type,location,timestamp,split
```

Add a new globally unique `id` per paraphrase. Its `original_id` must already
exist, and `split` must match that group's existing `train`, `val`, or `test`
assignment. Use `lang` values `bn`, `en`, or `banglish`. Keep category, severity,
and known incident metadata consistent with the source; use
`source_type=manual_paraphrase`. Do not change the incident to make the wording
more varied. Pair generation marks imported rows as derived manual paraphrases.

For example, an illustrative row for an already existing group might be:

```csv
id,original_id,text,lang,category,severity,source_type,location,timestamp,split
para-demo-001,existing-case-001,No water on Road 7 since 8 this morning,en,water,high,manual_paraphrase,Dhanmondi Road 7,2026-10-29T08:00:00+06:00,train
```

This example is not supplied as collected data; replace every value with a real
source-linked paraphrase. An empty template adds no records.

## 3. Generate reviewable pair candidates

```powershell
python -m ml_training.scripts.generate_pairs --input-dir ml_training/data/multilingual --output-dir ml_training/data/pairs --paraphrases templates/manual_paraphrases.csv --max-pairs-per-type 1000 --seed 42
```

Pair generation runs independently within each established split. It emits:

- `pairs.csv`: all pair candidates.
- `train/pairs.csv`, `val/pairs.csv`, `test/pairs.csv`: split-specific candidates.
- `summary.json`: counts, sampling settings, and any sampling limitations.

The eight pair payload columns are:

```text
text_a,text_b,loc_a,loc_b,time_a,time_b,label,pair_type
```

Provenance columns include `pair_id`, `split`, `id_a`, `id_b`, `original_id_a`,
`original_id_b`, plus derivation/parent metadata for each side. `label` is the
generator's proposed 0/1 label; it is not an adjudicated human label.

| `pair_type` | Candidate rule | Required human check |
| --- | --- | --- |
| `positive_cross_language` | Same source group, different languages | Verify meaning and incident details survived translation/transliteration. |
| `positive_paraphrase` | Same source group, with a manually imported paraphrase | Verify it describes the original incident without changing facts. |
| `hard_negative` | Same category, different source groups, similar text style; explicit location or UTC event-date difference | Distinguish separate incidents from continued events or metadata/report-time differences. These can be false negatives. |
| `easy_negative` | Different source groups and categories | Verify the categories and incident separation are correct. |

Style uses supplied text-style information when available, otherwise language
and a whitespace-token-length bucket (short: at most 12; medium: 13–30; long:
more than 30). Use the optional `text_style` column on input split files to
provide your own style labels; explicit styles match other explicit styles only.
Missing location/time never counts as evidence of a
difference. Bounded seeded sampling may produce fewer pairs than the cap when
eligible combinations are scarce; review `summary.json`. Candidate counts do
not create additional real incidents.

If review reveals a real incident crossing existing splits, flag it for an
upstream grouping correction and regenerated data. Do not independently move
one translation or paraphrase into a different split to patch the symptom.

## 4. Start local independent annotation

```powershell
python -m annotation.app --pairs ml_training/data/pairs/pairs.csv --complaints ml_training/data/multilingual/train.csv --complaints ml_training/data/multilingual/val.csv --complaints ml_training/data/multilingual/test.csv --db annotation/annotations.sqlite3 --port 8000
```

Open <http://127.0.0.1:8000>. The FastAPI app serves its HTML interface locally;
keep the server bound to the local computer. Enter a stable local annotator
pseudonym. This is a self-asserted name, not authenticated access control.

Use [annotation_guideline.md](annotation_guideline.md). Each annotator labels
pair duplication (`0` or `1`) with a reason, then independently rates individual
complaints (`1`–`5`) with a reason. Skip uncertain items. The labeling view hides
generator suggestions and other annotators' labels. Arrange meaningful overlap
between two annotators before comparing their results; priority is one decision
per individual complaint, not one decision for a pair.

SQLite stores the local annotations using a write-ahead log. For a database
backup, stop the server cleanly and close its connections before copying the
database and input data; copying only a live `.sqlite3` file can omit recent
annotations still in its WAL file. An online backup should use SQLite's backup
API. The annotation export is available at
<http://127.0.0.1:8000/api/export/annotations.csv>.

For pseudonyms `alice` and `bob`, agreement is available at:

```text
http://127.0.0.1:8000/api/agreement?annotator_a=alice&annotator_b=bob
```

The result reports unweighted Cohen's kappa for duplicate decisions and
quadratic weighted Cohen's kappa for priority. Each calculation uses only
overlapping non-skipped labels for that task. Undefined kappa returns `null`
with an explanation, including no-overlap and constant-label denominator-zero
cases. Review disagreements with a human adjudicator before finalizing gold
labels; preserve independent annotations for the agreement audit.

The interface includes unweighted priority kappa as an additional comparison.
It never substitutes generator suggestions for missing human judgments.
Use the review checkbox and Previous/Next controls to revisit saved or skipped
items. The server pages large queues, so review can continue beyond 200 items.
An item edited in another browser tab returns a conflict; refresh before editing
it again. Changed item content requires a new database to avoid reusing labels
for a different complaint.

## Expected files and counts

For `N` sampled groups, export writes `2N` complaint rows: one selected Bangla
source and one derived Banglish version per group. A fully filled translation
import produces `3N` rows. Actual train/val/test group counts follow the sampled
groups' existing assignments; a small random sample need not have exact 70/15/15
ratios or cover every category. The manifest reports achieved language counts
and completed/pending translations.

`pairs.csv` contains the eight requested payload columns followed by identity,
split and derivation metadata. Per-split copies have the same schema, including
headers when empty. `summary.json` reports candidate counts by type, bounded
sampling attempts and warnings. Do not interpret missing pair types as complete
coverage of all possible incident relationships.

The annotation CSV is one row per item and annotator, with task, human label,
reason, skip status, latest revision, save time, and complaint provenance.
Export it before adjudication. Keep the two independent judgments, compare
their overlap with kappa, and create a separate adjudicated gold-label table
keyed by `item_id` after discussing disagreements. A skip needs follow-up or
exclusion from scored evaluation; it is never automatically a negative.

## 5. Collect an independent natural test set

Use `templates/natural_test_set.csv`, whose exact header is:

```text
text,lang,category,location,timestamp
```

The template has no fabricated observations. Collect independently authored
real complaints with appropriate consent/permission and minimal personal data.
This is a separate held-out set, not a translation or paraphrase of the source
dataset. Keep it out of training, translation sampling, and pair generation.
Audit overlapping incidents and have humans validate labels before evaluation.
The app accepts the five-column natural CSV directly and creates stable
complaint IDs when absent. After collecting and reviewing real records, start
priority-only annotation with:

```powershell
python -m annotation.app --complaints templates/natural_test_set.csv --db annotation/natural_annotations.sqlite3 --port 8000
```

The empty template intentionally yields no annotation items; populate a working
copy with real collected complaints first. Use that working-copy path in the
command. A separate database keeps natural-test annotations distinct. Missing
source severity is allowed for this collection format; do not invent severity
or priority merely to fill a schema.

Identical five-field submissions share a generated annotation ID. If distinct
collection records must be tracked separately, add an optional unique `id`
column before starting annotation. Keep IDs and complaint content stable for
the duration of an annotation round.

## Sampling limits

`--max-pairs-per-type 1000` is a ceiling **per pair type per split**, not a
guarantee of 1000 available pairs and not a global 1000-pair limit. The four types
across three splits allow up to 12 times that value in total when eligible
candidates exist. Bounded search can underfill any type even before all possible
combinations have been enumerated. Read the printed warnings and `summary.json`
for achieved counts; never generate fabricated incidents to reach a target.
