# Phase 4: cross-language, location-aware duplicate detection

Run commands from the repository root. This phase uses the existing
incident-isolated train, validation and test assignments; it never re-splits
pairs at random. Read [the experiment design and validity limits](duplicate_experiment_design.md)
before treating generated candidates as research ground truth. The scripts
print their measured results; no performance scores are bundled.

Classification source data, including BanglaPSG, can seed complaint text and
the labels actually present in the source. It may not contain verified real
incident IDs, locations, or timestamps. Leave absent metadata unknown and
collect/adjudicate duplicate incidents before making a location-aware research
claim. An auxiliary language or romanization resource can support spelling
review, but does not supply same-incident gold labels or turn unrelated texts
into complaint-pair evidence.

## 1. Finish pair adjudication and prepare metadata

Phase 2 `pairs.csv` files contain proposed `label` values, not independent
human truth. The default Phase 4 loader requires an explicit `gold_label`
column, or a `human_label` column when `gold_label` is absent. Every accepted
pair must have a final `0` or `1`; skips and disagreements need adjudication
before they enter the fixed evaluation dataset. The loader does not silently
drop unlabeled pairs or take a majority vote from the annotation export.

Retain the two independent annotators' records for the agreement audit. Create
a separate adjudicated table keyed by `pair_id`, with this header:

```text
pair_id,gold_label
```

In the annotation export, duplicate `item_id` values have a `pair:` prefix.
Remove that prefix when forming `pair_id`, and include only adjudicated
duplicate decisions; priority judgments belong to a different task. Keep the
reason and adjudicator provenance in your annotation records.

Attach those final decisions to separate frozen pair copies:

```powershell
python -m ml_training.scripts.apply_duplicate_labels --pairs-dir ml_training/data/pairs --labels adjudicated_pairs.csv --output-dir ml_training/data/pairs_gold
```

The helper requires complete coverage of the matching source pair set and
rejects unknown/duplicate IDs, nonbinary decisions, and conflicting existing
gold labels. It preserves the generated candidate `label`, adds `gold_label`,
and writes split files, a combined `pairs.csv`, and `label_manifest.json`.
It performs no voting, incident inference, or automatic conflict resolution.
Use separate, non-nested source/output directories and preserve the completed
adjudication table with the input manifest.

The command examples below use this `ml_training/data/pairs_gold` directory.
A combined CSV with an explicit `split` column is also accepted by `--pairs`.
Do not copy a record into another split to fill a missing class. Resolve actual
incident grouping errors upstream and regenerate the affected dataset.

Supply the complaint CSVs used to create the pairs. Phase 4 resolves language
and category through the complaint IDs, and checks text, location, timestamps,
and original incident identity against those records. If you imported manual
paraphrases, supply that CSV as an additional `--complaints` argument. Its
records are not necessarily present in the multilingual split files.

The loader rejects repeated/reversed pairs, missing complaint records,
conflicting metadata, duplicate IDs, and incident groups crossing splits.
`gold_label` takes precedence over `human_label`. The original candidate label
is retained as provenance rather than used as an extra feature.

For implementation smoke checks only, `--allow-weak-labels` explicitly permits
the generated `label` suggestions. Such output is marked weak-label evaluation
and must not be described as an independently annotated thesis result.

## 2. Install dependencies and review the gazetteer

```powershell
python -m pip install -r requirements.txt
```

RapidFuzz supports the fuzzy location matcher. The starter gazetteer is
`ml/location_gazetteer.json`; it includes Dhaka areas and specific Mirpur and
Uttara subdivisions with Bangla, English and Banglish aliases. Extend the JSON
before freezing the experiment. An entry has this form:

```json
{
  "id": "dhaka_mirpur_10",
  "name": "Mirpur 10",
  "parent_id": "dhaka_mirpur",
  "aliases": ["মিরপুর ১০", "মিরপুর দশ", "mirpur 10", "mirpur ten", "mirpur dosh"]
}
```

IDs must be unique and stable. Parents must exist, hierarchies must be acyclic,
and an alias cannot normalize to two different areas. Normalization handles
Bangla digits and separators, so `মিরপুর ১০`, `Mirpur 10`, and `mirpur-10`
resolve to the same canonical ID. Fuzzy matching is conservative about numeric
subdivisions and ambiguity. An unknown location is not evidence of agreement
or disagreement. Explicit location metadata takes precedence, while conflicting
text is preserved as a conflict feature for review.

```python
from ml.location import LocationGazetteer
from ml.time_features import extract_time_features

gazetteer = LocationGazetteer("ml/location_gazetteer.json", fuzzy_threshold=90)
print(gazetteer.extract("মিরপুর ১০ এ পানি নেই"))
print(gazetteer.extract("water unavailable", explicit_location="mirpur-10"))
print(extract_time_features("৩ দিন ধরে পানি নেই", "2026-10-30T10:00:00+06:00"))
print(extract_time_features("3 din dhore pani nai"))
print(extract_time_features("No water for three days"))
```

These are illustrative inputs, not collected research observations. Duration
features use explicit supported phrases in hours/days. An ongoing duration
plus a supplied timestamp can define a tentative report-anchored interval.
Missing values remain unknown; the code does not use the current clock to
invent an onset. Naive timestamps are interpreted as UTC; prefer explicit
ISO-8601 offsets. Ambiguous ranges or multiple distinct durations remain
unresolved. `time_gap` compares timestamp anchors in hours, which may be report
times rather than event times.

## 3. Run the lightweight TF-IDF cosine baseline

```powershell
python -m ml_training.scripts.train_duplicates --pairs ml_training/data/pairs_gold --complaints ml_training/data/multilingual/train.csv --complaints ml_training/data/multilingual/val.csv --complaints ml_training/data/multilingual/test.csv --models B1 --results-dir results/duplicates_b1 --artifacts-dir artifacts/duplicates_b1 --cache-dir .cache/duplicate_training --seed 42
```

If the pairs include manual paraphrases, append
`--complaints path/to/manual_paraphrases.csv`. Keep the actual file used during
pair generation; the header-only template adds no metadata records.

B1 fits character TF-IDF (`char_wb`, n-grams 1–5) on unique training complaint
IDs appearing in the training pairs only; repeated pair use does not refit the
same complaint ID multiple times. Distinct IDs with identical text remain
distinct records for this vocabulary fit. It scores pairs with cosine similarity. The decision threshold is
chosen using validation pair F1, with ties favoring the higher threshold. Test
labels never enter vocabulary fitting or threshold choice.

## 4. Run the embedding ablations on the same fixed pairs

Install the optional embedding environment:

```powershell
python -m pip install -r requirements-embeddings.txt
```

Use a complete local/cached encoder to remain offline:

```powershell
python -m ml_training.scripts.train_duplicates --pairs ml_training/data/pairs_gold --complaints ml_training/data/multilingual/train.csv --complaints ml_training/data/multilingual/val.csv --complaints ml_training/data/multilingual/test.csv --models B1 B2 C0 P1 P2 --classifier lr --embedding-model path/to/local/minilm --category-source provided --gazetteer ml/location_gazetteer.json --fuzzy-threshold 90 --results-dir results/duplicates --artifacts-dir artifacts/duplicates --cache-dir .cache/duplicate_training --seed 42
```

Replace the local model path with your actual saved MiniLM directory. The
default model ID, when no local path is supplied, is
`sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`. An initial online
retrieval requires `--allow-download` explicitly. `--embedding-revision` pins a
cached/downloaded model revision. Pretrained weights are frozen features,
shared across the selected embedding ablations; they are not fine-tuned here.

| Model | Pair score or feature set |
| --- | --- |
| B1 | Training-only character TF-IDF cosine. |
| B2 | Frozen multilingual MiniLM cosine only. |
| C0 | Learned MiniLM cosine plus category match/missing indicators and unordered language-pair indicators. |
| P1 | C0 plus location match, categorical location-difference flag, location missingness, and location conflict. |
| P2 | P1 plus timestamp gap, duration gap, interval overlap, and their missingness indicators. |

C0 is an extra diagnostic control. P1 minus C0 isolates the added location
features within the same learned classifier; P2 minus P1 isolates time features.
B2 versus P1 also changes the learner and context covariates, so it does not by
itself estimate a location-only benefit. See the exact definitions and threats
in the design document.

The default learned classifier is a StandardScaler plus class-balanced logistic
regression. To evaluate gradient boosting as a separate, prespecified learner,
repeat the same command with `--classifier gradient_boosting` and fresh result
and artifact directories. Gradient boosting uses balanced training sample
weights. All learned parameters fit on training pairs; validation selects the
decision threshold. These output scores are not claimed to be calibrated
probabilities.

`--category-source provided` uses the supplied complaint category metadata.
That may be a gold/oracle input relative to production classification. Repeat
with `--category-source none` and new output directories to measure sensitivity
to removing category context. It does not simulate a noisy category classifier.

Use fresh run directories for comparisons. `--overwrite` is for deliberate
replacement, and should not be used to discard the earlier experiment needed
for a reproducibility audit. Freeze the gazetteer, normalization dictionary,
pair files and labels before the final run.

## 5. Read the ablation outputs

The experiment writes these core outputs:

```text
results/duplicates/
├── metrics.json
├── ablation_table.csv
├── ablation_table.md
├── feature_audits/
│   ├── train.csv
│   ├── val.csv
│   └── test.csv
└── <model>/
    ├── validation/
    │   ├── metrics.json
    │   ├── predictions.csv
    │   ├── precision_recall_curve.csv
    │   └── precision_recall_curve.png
    └── test/
        ├── metrics.json
        ├── predictions.csv
        ├── precision_recall_curve.csv
        └── precision_recall_curve.png

artifacts/duplicates/
├── experiment.json
├── B1.joblib
├── B2.joblib
├── C0.joblib
├── P1.joblib
├── P2.joblib
├── resources/
│   ├── location_gazetteer.json
│   └── banglish_variants.json
└── minilm_encoder/                # present when embedding models are selected
```

Only selected models are created; feature audits are emitted when a learned
C0/P1/P2 model is selected. The measured output/manifest is authoritative
for the complete file list. Empty or single-label slices do not get fabricated
PR curves. Feature audits contain the full P2 feature vector and `pair_id`/`label`
for auditing. Those identifier/label audit columns are **not** passed to the
classifier: each model selects only its declared numerical feature names.
Original IDs and pair type are also excluded from model features.

Overall metrics include pairwise precision, recall, F1 and `pr_auc` defined as
**average precision**, a step-integral of precision over recall. The separately
named `trapezoidal_pr_auc` uses trapezoidal interpolation and is not the same
statistic. Keep the definition fixed in the thesis and across model comparisons.

Breakdowns include `pair_type` and the unordered language pairs `bn-en`,
`bn-banglish`, and `en-banglish`, plus any same-language slices present. Each
slice includes support. Empty strata use null metrics. A single-label stratum
has no ranking-discrimination estimate and its PR-AUC is null with a reason.
Hard/easy negatives alone contain no positive examples: inspect their
false-positive rate and counts rather than interpreting positive-class F1 as
negative-detection quality. Precision/recall/F1 use an explicit zero-denominator
convention for nonempty slices.

Every selected model is evaluated on the same test pair IDs and labels. Scores
from different models have different meanings/scales. Validation thresholds
and their selection policy are saved, so report the measured operating point
alongside threshold-independent ranking metrics. No result should be copied
from documentation as a measured score.

## 6. Export the top false positives and false negatives

```powershell
python -m ml_training.scripts.analyze_duplicate_errors --predictions results/duplicates/P2/test/predictions.csv --output-dir results/duplicates/P2/error_analysis --top-k 50
```

This writes `false_positives.csv`, `false_negatives.csv`, and
`error_summary.json`. It preserves text, location/time, language-pair,
pair-type, and incident provenance from the prediction file. False positives
are ordered by highest score above the saved threshold; false negatives by
lowest score below it. Analyze one model/threshold at a time. A score far from
the boundary is not proof of well-calibrated confidence.

Use final-test error analysis descriptively. If it leads to changes in the
gazetteer, rules, model, or threshold, those changes need an independent future
evaluation set rather than repeated selection on the same final test labels.

## 7. Build candidate duplicate groups

```powershell
python -m ml_training.scripts.cluster_duplicates --predictions results/duplicates/P2/test/predictions.csv --all-complaints ml_training/data/multilingual/test.csv --output results/duplicates/P2/test/clusters.csv
```

By default the clustering script uses the single saved validation-selected
threshold from the prediction file. An explicit `--threshold` is accepted when
chosen independently of test labels; it must not be optimized against the test
clusters. `--all-complaints` can be repeated to retain unpaired singleton IDs;
use records belonging to the intended evaluation split. `--model` selects one
model if a prediction file contains several.

Connected components use thresholded scores and never inspect true duplicate
labels or `original_id` as evidence for merging. Output columns are
`id,duplicate_group_id,split,group_size`. Different dataset splits are never
joined. The result is a candidate graph partition: an A–B and B–C chain can
merge A and C even without a positive A–C score. A sampled pair graph also
omits many unscored edges. Pairwise F1 does not establish cluster quality.

## 8. Query a reusable production index

Use only trusted joblib artifacts and keep their complete saved resource tree.
The following example loads an application complaint store, builds the index
once, then reuses it. The new complaint is illustrative, not a collected test
observation:

```python
import pandas as pd
from ml.duplicate_index import configure_duplicate_finder, find_duplicates

existing = pd.read_csv(
    "existing_complaints.csv", dtype=str, keep_default_na=False
).to_dict("records")
finder = configure_duplicate_finder(
    "artifacts/duplicates/P2.joblib",
    cache_dir=".cache/duplicate_production",
    top_k=100,
)
finder.build_index(existing)
new = {
    "id": "new-complaint",
    "text": "মিরপুর ১০ এ ৩ দিন ধরে পানি নেই",
    "lang": "bn",
    "category": "water",
    "location": "Mirpur 10",
    "timestamp": "2026-10-30T10:00:00+06:00",
}
matches = finder.find_duplicates(new, None)
print(matches)
print(finder.last_query_stats)

# Required two-argument convenience API: reuses the configured finder/cache.
same_matches = find_duplicates(new, existing)
```

Each complaint needs a unique nonempty `id`, `text`, and explicit `lang`.
Available category/location/timestamp metadata follows the training feature
semantics; missing values remain missing. The same ID as the query is excluded
from candidates.

The NumPy index stores normalized float32 embeddings and computes a matrix–vector
dot product per query, not a corpus-by-corpus similarity matrix. Disk embeddings
are cached by text and encoder/preprocessing identity. Learned models retrieve
the top embedding-cosine candidates **before** reranking with location/time.
Candidates outside `top_k` cannot be returned, even if metadata would make them
strong duplicate matches. Measure candidate recall before making a production
coverage claim. B1 has a corresponding sparse TF-IDF retrieval path.

The architecture targets a corpus on the order of 50k complaints without an
all-pairs matrix; this is not a measured latency, memory-budget, or retrieval-
recall guarantee. Embedding storage is proportional to `N × embedding_dimension`
float32 values, plus metadata and cache overhead. The exact two-argument wrapper
also checks a signature over the supplied corpus metadata on each call. For a
stable service, use `build_index` once and `finder.find_duplicates(new, None)`;
rebuild/update deliberately when the store changes. Reuse the encoder and finder
for requests rather than reloading weights each time.

## 9. Verify the implementation and retain provenance

```powershell
python -m pytest -q
```

Tests use synthetic fixtures, including deterministic test encoders where real
pretrained weights are unavailable. They check implementation behavior, not
the research performance of MiniLM, extraction accuracy on your corpus,
50k-query latency, or real-world duplicate recall. Preserve input hashes,
adjudication records, split identities, model revisions, resource snapshots,
feature definitions, thresholds, and all measured outputs with the thesis run.
