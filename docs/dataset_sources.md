# Supplied dataset imports and provenance

The importer reads the supplied local archives as data. It never executes their
Python files, follows packaged instructions, or extracts member paths onto the
filesystem. Local source values, import manifests, and auxiliary samples stay
under ignored data directories and are excluded from the source-code ZIP.

## BanglaPSG workbook

The inspected local archive is named
`BanglaPSG A Bangla Public Service Grievance Datase (2).zip`. Its single workbook
member is:

```text
BanglaPSG A Bangla Public Service Grievance Datase/BanglaPSG-Grievance-Dataset.xlsx
```

The observed `Sheet1` headers are `No.`, `Grievance Text`, `Category`, and
`Severity`, followed by empty formatted columns. The import mapping is explicit
in `SOURCE_COLUMN_MAPPING` in `import_research_datasets.py`:

| Canonical field | Observed source column |
| --- | --- |
| `text` | `Grievance Text` |
| `category` | `Category` |
| `severity` | `Severity` |
| source row-number provenance | `No.` |

The local workbook contains 6,105 nonblank source records. It also contains
44,392 fully blank formatted rows; the importer records their exclusion rather
than treating them as grievance observations. These are observed source counts,
not model performance results.

| Category value, retained verbatim | Source records |
| --- | ---: |
| Electricity | 1,007 |
| Food | 1,036 |
| Healthcare | 1,015 |
| Road & Transport | 1,061 |
| Waste | 1,015 |
| Water | 971 |

| Severity value, retained verbatim | Source records |
| --- | ---: |
| High | 1,653 |
| Low | 1,988 |
| Medium | 2,464 |

No source text, category, or severity value was missing in this inspected copy.
The intended severity order for commands using these exact labels is
`--severity-order Low Medium High`; confirm the published annotation scale
before interpreting it as an ordinal construct.

### Identifiers, formulas, and unavailable metadata

The `No.` column includes formula cells with cached numeric values. The importer
reads those stored values without recalculating any formula. It preserves
formula strings in the inert `source_formulas_json` provenance column. It does
not use the formulas or cached row numbers to infer incident relationships.
The cached `No.` values also have three duplicate occurrences beyond their
first occurrence, so they are not used as globally unique row identifiers.

When an explicit source `id` is absent, each imported row receives a stable ID
derived from the workbook fingerprint, sheet index, and physical spreadsheet
row. The original source columns are preserved as `source_<column number>_<header>`
fields alongside member, sheet, and spreadsheet-row provenance.

This workbook does not supply a verified incident-parent ID, structured
location, timestamp, or language column. Those fields remain blank in raw
imports. Phase 1 applies its documented script-based language heuristic and
generates exact-normalized-text grouping IDs. Such grouping cannot identify
different descriptions of one event. An equal or different generated ID is
not independent evidence of duplicate incidents.

The local workbook also has repeated exact text. These observations are
preserved; the preparation pipeline's group-aware splitting keeps equal
normalized text together. No rows are silently removed to balance categories.

### Source release and licensing

Two candidate official catalogue records have related BanglaPSG titles:
[the kcr8zfmtvb record](https://data.mendeley.com/datasets/kcr8zfmtvb/1) and
[the w6c4zx5t2s record](https://data.mendeley.com/datasets/w6c4zx5t2s/1).
The local filename, schema, and fingerprint do not by themselves establish
which catalogue release produced these exact bytes. The import manifest
therefore records the archive and workbook SHA-256 values without claiming a
verified release binding. Confirm the applicable release, licence, attribution,
and annotation protocol before redistributing source records or writing a
definitive dataset citation. The code distribution contains no supplied rows.

## BanglaDual auxiliary language resource

The supplied second archive is named
`BanglaDual A 1.24-Million Pair Multi-Source Datase.zip`. Its inventory contains
large merged/task-specific Parquet data, model-training splits, processing
scripts, and small packaged CSV examples. The importer hashes and inventories
the archive, then reads only the bounded `Sample Data` CSV members. It never
opens the large Parquet members or runs the packaged processing scripts.

Observed sample schemas are:

| Packaged member | Columns |
| --- | --- |
| `english_samples.csv` | `Banglish`, `English`, `Source` |
| `bangla_samples.csv` | `Banglish`, `Bangla`, `Source` |
| `sample_merged_data.csv` | `Banglish`, `Bangla`, `English`, `Source_bangla`, `Source_english` |

The importer exports at most the requested number of language pairs, using a
deterministic round-robin across sample files. A merged source row may provide
both a Bangla and an English target; blank targets are not invented. The sample
can contain repeated content across packaged members, which is recorded in
the manifest. It is neither a random nor a representative sample of the full
corpus, and the importer does not claim that it loaded the title's full size.

The completed bounded import of the supplied sample files produced 201 language
pairs: 80 English translation examples and 121 Bangla transliteration examples.
These are imported sample counts, not a count of incident duplicates or a claim
about the full archive's record count.

Each output pair retains `source_member`, `source_csv_row`, and upstream source
attribution. It is marked `source_type=auxiliary_translation` or
`auxiliary_transliteration`, and `is_incident_gold=false`. The exported schema
deliberately omits incident categories, severity, and incident-group IDs.

These are general-language translation/transliteration examples drawn from
multiple domains. They can support language-normalization inspection or a
separately designed language-adaptation experiment. They cannot substitute for
reviewed cross-language versions of a specific grievance, independent natural
complaints, or human incident-duplicate labels. They are never merged into the
incident train/validation/test files by this importer.

The candidate official [BanglaDual catalogue record](https://data.mendeley.com/datasets/pxw2bb4wyt/1)
states CC BY 4.0. Preserve the dataset citation and per-row upstream attribution,
and verify how that release corresponds to the supplied archive before
redistribution. The local byte-level release binding remains unverified.

## Exact commands

From the repository directory, install the core requirements and import both
archives. Paths containing spaces must remain quoted:

```powershell
python -m pip install -r requirements.txt
python -m ml_training.scripts.import_research_datasets --banglapsg-zip "C:/Users/hasib/Downloads/BanglaPSG A Bangla Public Service Grievance Datase (2).zip" --bangladual-zip "C:/Users/hasib/Downloads/BanglaDual A 1.24-Million Pair Multi-Source Datase.zip" --output-dir ml_training/data/raw --auxiliary-output ml_training/data/auxiliary --auxiliary-sample-size 2000
```

Use `--overwrite` only when intentionally replacing the importer outputs. The
source archives remain unchanged. Normal output files are:

```text
ml_training/data/raw/
├── banglapsg.csv
└── dataset_import_report.json

ml_training/data/auxiliary/
├── bangladual_language_sample.csv
└── bangladual_sample_manifest.json
```

For the inspected workbook's canonical CSV, no additional source-column
mapping is needed in Phase 1:

```powershell
python -m ml_training.scripts.prepare_data --input ml_training/data/raw/banglapsg.csv
```

If using a different workbook, `--workbook-member`, `--sheet`, and an optional
`--column-map` JSON can select and map its source data explicitly. Missing or
blank required labels produce `banglapsg_source_staging.csv` and issues in the
report, with exit code 3. They do not produce a usable canonical dataset or
imputed severity labels. Successful canonical imports return 0; invalid input
or write failures return 2.

The import reports include the archive inventory and hashes, effective mapping,
source and export counts, source-label distributions, formula handling,
metadata gaps, sample bounds, and output-file fingerprints. Preserve them with
the upstream source citation and the preparation/model manifests. Incident
duplicate experiments still require reviewed pairs with preserved incident
groups; category/severity labels alone do not supply that ground truth.
