# Dataset schema

The raw dataset belongs at `ml_training/data/raw/banglapsg.csv`. The preparation
script accepts UTF-8 CSV and produces the same standardized schema in every
train, validation, and test file. The source file is never modified.

## Columns

All identifiers and labels are strings. In particular, identifiers such as
`000123` and severity values such as `01` must retain their leading zeros.

| Column | Type | Required in source? | Meaning and preparation rules |
| --- | --- | --- | --- |
| `id` | string | No | Unique row identifier. Supplied nonempty values must be unique. Missing or blank values receive a deterministic generated identifier. Keep source row order stable when reproducing generated row IDs. |
| `original_id` | string | No, but strongly recommended | Identifier shared by an original record and all its translations, paraphrases, augmentations, or other variants. Every row in a group stays in one split. Missing or blank values receive a SHA-256 group identifier from cleaned source text, without category or severity, so exact cleaned-text duplicates stay together. |
| `text` | string | Yes | Nonempty complaint or report text after cleaning. Text that becomes empty after cleaning is rejected. Banglish normalization is applied when `lang` is `banglish`. |
| `lang` | string enum | No | One of `bn`, `en`, or `banglish`. If absent, text containing Bangla Unicode characters is assigned `bn`; other text is assigned `en`. This is a basic heuristic, not a language classifier. Provide `banglish` explicitly for Latin-script Bangla. |
| `category` | string | Yes | Nonempty category label supplied by the dataset. Labels are trimmed and otherwise retained. This phase does not impose a category taxonomy. |
| `severity` | string | Yes | Nonempty severity label supplied by the dataset. Labels are trimmed and otherwise retained. This phase does not impose an ordinal scale or infer severity from the text. |
| `source_type` | string | No | Dataset or collection-source label. Defaults to `banglapsg` when no source value is supplied. |
| `location` | string | No | Optional location metadata as supplied by the dataset. Empty when unavailable; no geocoding or location inference is performed. |
| `timestamp` | string | No | Optional ISO 8601 timestamp. Valid supplied timestamps are normalized to UTC; unavailable values remain empty. Invalid nonempty timestamps are rejected. Naive timestamps are interpreted as UTC. Prefer an explicit timezone, for example `2026-10-29T16:30:00+06:00`. |

The prepared column order is:

```text
id,original_id,text,lang,category,severity,source_type,location,timestamp
```

## Source column mapping

`COLUMN_MAPPING` in `ml_training/config.py` maps standardized names to source
CSV header names. For a source file with `complaint`, `label`, and `priority`,
use this mapping:

```python
COLUMN_MAPPING = {
    "text": "complaint",
    "category": "label",
    "severity": "priority",
    "id": "id",
    "original_id": "original_id",
    "lang": "lang",
    "source_type": "source_type",
    "location": "location",
    "timestamp": "timestamp",
}
```

The three mandatory source columns are the mapped `text`, `category`, and
`severity` columns. Missing mandatory columns produce a clear error naming the
missing standardized field, its expected source header, and the available CSV
headers. Optional columns may be omitted to use the defaults described above.
Header names are matched according to the mapping; edit the dictionary to match
the actual source rather than relying on guessed aliases.

## Cleaning and Banglish normalization

General text cleaning applies Unicode NFKC normalization, translates Bangla
digits `০১২৩৪৫৬৭৮৯` to `0123456789`, removes HTTP(S) and `www.` URLs, collapses
whitespace, trims the result, and lowercases Latin text. It does not transliterate
Bangla script or translate between languages.

For rows explicitly labeled `banglish`, `ml/normalize.py` also reduces runs of
three or more identical ASCII letters to a single letter and resolves common
spelling variants through `ml/banglish_variants.json`. For example, `paniiii`
becomes `pani`, `pany` becomes `pani`, `nei` becomes `nai`, and `biddut` becomes
`bidyut`. Ordinary double letters are retained unless the dictionary maps their
whole word. The JSON file maps an alias to its canonical spelling and can be
extended without editing the normalization code.

The inferred `en` label cannot distinguish English from Banglish. A Latin-script
row without an explicit `banglish` label receives general cleaning only. Supply
language labels before preparation when Banglish normalization is needed.

## Grouping and split guarantees

The target row ratios are 70% train, 15% validation, and 15% test. Splitting is
deterministic for a fixed input and random seed. The allocation uses category
information to preserve category proportions where feasible while keeping each
`original_id` group intact. Groups may contain more than one category.

Group sizes, rare categories, or very small datasets can make exact row ratios
and full category coverage in all three splits impossible. The pipeline uses a
best-effort allocation and reports limitations rather than dividing a group.
Review the split manifest for actual row counts, group counts, category counts,
and achieved proportions.

These are required invariants:

- Every accepted input row appears in exactly one output split.
- Every output `id` is unique across all splits.
- The sets of `original_id` values in train, validation, and test are pairwise
  disjoint.
- No row is dropped silently to improve the split proportions.

Generated group identifiers detect exact duplicates after general text cleaning.
They cannot reliably identify paraphrases, translations, distinct Banglish
spellings, or augmentations. Supply the same `original_id` for all known related
records before preparation to prevent those records from crossing split
boundaries. Supplied group identifiers take precedence over generated text
hashes.

## Output representation

The processed data directory contains `train`, `val`, and `test` files in the
requested CSV and/or Parquet formats, plus a JSON split manifest. CSV files use
UTF-8 with a header and no dataframe index. Parquet files preserve string values
without relying on CSV type inference. When reading CSV outputs, request string
dtypes so numeric-looking IDs and labels retain their original representation.

For example, a prepared row can look like this:

```csv
id,original_id,text,lang,category,severity,source_type,location,timestamp
report-001,case-001,pani nai 2 din,banglish,water,high,banglapsg,Dhaka,2026-10-29T10:30:00Z
```

The identifiers in this example are supplied source identifiers; generated
identifiers have a pipeline-defined deterministic representation. Category and
severity values shown here are examples, not a required label vocabulary.

## Phase 2 extensions

Multilingual CSVs retain the nine columns above and add:

| Column | Representation | Meaning |
| --- | --- | --- |
| `is_derived` | `true` / `false` strings | `true` for every created transliteration, translation, or imported manual paraphrase; `false` for selected source originals. |
| `derivation` | string | `source`, `rule_transliteration`, `manual_translation`, or `manual_paraphrase`. These are provenance values, not quality ratings. |
| `parent_id` | string | Source row `id` for generated language versions; empty for source originals. Manually imported paraphrases are linked by required `original_id`; `parent_id` may be empty. |

`train.csv`, `val.csv`, and `test.csv` determine split membership; the split is
not an extra column in those complaint files. Generated versions always inherit
their source group and file assignment. The multilingual manifest snapshots
selected source rows and base versions, and the translation queue carries an
explicit `split` for validation. Never edit its grouping metadata to rebalance
language rows.

Pair CSVs start with the required payload:

```text
text_a,text_b,loc_a,loc_b,time_a,time_b,label,pair_type
```

They also include `pair_id`, `split`, `id_a`, `id_b`, `original_id_a`,
`original_id_b`, `is_derived_a`, `is_derived_b`, `derivation_a`, `derivation_b`,
`parent_id_a`, and `parent_id_b`. A pair may involve two different groups for
negative candidates, but both groups belong to the same existing split.
`label` is a proposed integer 0/1, subject to independent human review.

Annotation exports are long-form: one row per `(item_id, annotator)`, including
`task_type`, `label`, `status`, `reason`, `revision`, `updated_at` and original
text/incident provenance. Duplicate labels use 0/1; priority labels use 1–5.
Skipped items have empty labels and a reason, and do not contribute to kappa.
The latest saved judgment is exported; revisions prevent conflicting writes
but do not constitute a complete edit-history archive.
