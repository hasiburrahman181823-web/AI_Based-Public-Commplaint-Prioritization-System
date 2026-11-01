# Phase 3: category and severity experiments

Run these commands from the repository root in your Python environment. The
scripts print the metrics they calculate and write the same measurements to
JSON, CSV, reports, and figures. This repository does not bundle trained weights
or claim research scores.

## 1. Prepare the experiment inputs

Use the Phase 1 processed splits or the Phase 2 multilingual splits. Each input
directory must contain `train.csv`, `val.csv`, and `test.csv` with the documented
complaint schema. The multilingual work set is needed for English and Banglish
evaluation when those languages are absent from the original data. All derived
versions of an incident must retain its `original_id` and original split.

Training never re-splits these files. It checks identifiers, split isolation,
languages, and target labels before fitting. Categories are learned from the
training split; an unseen held-out category is an error rather than a silently
dropped observation. Validation must cover exactly the training target classes
for calibration. If a class is missing, correct the upstream group split or
collect independent calibration examples before proceeding. Never move an
isolated translation across a split boundary.

For severity experiments, provide the dataset's actual severity labels in
**low-to-high order**. Severity is not assumed to be alphabetical or numeric.
For example, `--severity-order low medium high` is correct only if those are
your dataset's labels and their intended order. If severity ordering has not
been established, run category experiments alone with `--targets category`.

The independently collected natural test set remains a separate file. Its
five-column template supports category evaluation after human label review:

```text
text,lang,category,location,timestamp
```

To evaluate severity on natural records, add a separately reviewed `severity`
column using the same target scale as training. The annotation tool's 1–5
priority judgments are a separate construct; they are not automatically
converted into the source severity labels. Missing natural severity labels are
reported explicitly and do not become a fabricated score. Do not use natural
test observations to fit a model, calibrate probabilities, or tune parameters.

## 2. Install the required environment

The lightweight classical experiments need the core requirements:

```powershell
python -m pip install -r requirements.txt
```

The MiniLM experiments additionally need the embedding dependencies:

```powershell
python -m pip install -r requirements-embeddings.txt
```

The core requirements include scikit-learn, SciPy, matplotlib, and joblib.
Embedding dependencies are optional so TF-IDF baselines can run without Torch
or Sentence Transformers. For a GPU environment, choose the Torch build
appropriate to that environment before installing the optional requirements.

## 3. Train the two TF-IDF baselines

Run both targets using your true severity order:

```powershell
python -m ml_training.scripts.train_classifiers --input-dir ml_training/data/multilingual --models char_svm word_lr --targets category severity --severity-order low medium high --results-dir results --artifacts-dir artifacts --seed 42
```

For category alone:

```powershell
python -m ml_training.scripts.train_classifiers --input-dir ml_training/data/multilingual --models char_svm word_lr --targets category --results-dir results_category --artifacts-dir artifacts_category
```

The model names mean:

| CLI name | Features | Classifier |
| --- | --- | --- |
| `char_svm` | Character TF-IDF, `analyzer="char_wb"`, n-grams 1–5 | `LinearSVC` |
| `word_lr` | Word TF-IDF | Logistic regression |
| `minilm_lr` | Frozen multilingual MiniLM sentence embeddings | Logistic regression |
| `minilm_svm` | Frozen multilingual MiniLM sentence embeddings | Linear SVM |

All four classifiers use `class_weight="balanced"`. The word baseline uses
word **unigrams** and whitespace tokenization (`token_pattern=r"(?u)\S+"`)
because the usual word-character regex can split Bangla combining marks or
discard short Bangla words. Punctuation attached to a whitespace token stays
with that token. TF-IDF features have no hidden vocabulary-size cap. The
character baseline uses the requested `char_wb` n-grams 1–5.

The same feature/classifier choices are fitted independently for category and
severity. These are nominal classifiers with **ordinal-aware severity
evaluation**; the severity learner itself is not an ordinal-regression model.
Text receives the shared cleaning pipeline, and explicitly tagged Banglish
also receives dictionary normalization. Feature vocabularies and IDF weights
are fitted on training text only.

Hyperparameters are fixed in the implementation. The validation split is used
for held-out probability calibration, not fitting vocabularies or selecting a
best test result. `CalibratedClassifierCV(FrozenEstimator(fitted_model),
method="sigmoid")` fits calibration on validation predictions while keeping
the training estimator frozen. The deployed prediction rule is the argmax of
these calibrated probabilities; reports and calibration diagnostics evaluate
that same rule. No test predictions are used to fit the calibrator.

## 4. Add the MiniLM embedding experiments

The default encoder is
`sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`.
The scripts use cached/local weights by default. To use a complete encoder
directory already present on disk:

```powershell
python -m ml_training.scripts.train_classifiers --input-dir ml_training/data/multilingual --models minilm_lr minilm_svm --targets category severity --severity-order low medium high --embedding-model path/to/local/minilm --embedding-device cpu --embedding-batch-size 64 --results-dir results_minilm --artifacts-dir artifacts_minilm
```

To explicitly permit an initial download instead:

```powershell
python -m ml_training.scripts.train_classifiers --input-dir ml_training/data/multilingual --models char_svm word_lr minilm_lr minilm_svm --targets category severity --severity-order low medium high --allow-download --results-dir results --artifacts-dir artifacts
```

The default `--models` includes all four models and the default `--targets`
includes both targets. Select the TF-IDF models explicitly if you have not
installed/cached the optional encoder. `--embedding-revision` can pin a model
revision or commit. Record that choice with your experiment; use the same
encoder for fitting and inference. `--embedding-device cuda` requires an
appropriate GPU/Torch environment; CPU remains supported.

Embeddings are frozen features. Fitting a logistic regression or SVM on them
does not fine-tune MiniLM. The external encoder is saved once under
`artifacts/minilm_encoder/`; each joblib classifier references this resource
instead of embedding a live GPU object in a pickle.

Existing output protection avoids accidental replacement. Use a new pair of
results/artifact directories for a new experiment, or pass `--overwrite` when
you deliberately intend to replace this run's generated files. `--overwrite`
leaves old files for unselected models and cohorts in place. Prefer fresh run
directories; use the current `metrics.json` and its recorded file list to
identify the outputs belonging to the current run.

## 5. Evaluate independent language and natural-test cohorts

Include the natural test CSV when launching training/evaluation:

```powershell
python -m ml_training.scripts.train_classifiers --input-dir ml_training/data/multilingual --natural-test ml_training/data/natural_test/collected.csv --models char_svm word_lr --targets category severity --severity-order low medium high --noise-rate 0.15 --ece-bins 10 --results-dir results --artifacts-dir artifacts
```

Each fitted model/target is evaluated on the available test cohorts separately:

- Bangla test rows (`lang=bn`).
- English test rows (`lang=en`).
- Banglish test rows (`lang=banglish`).
- The separately supplied natural test file.
- A paired noisy copy of the Banglish test rows.

Combined-test reporting, when present, is additional to these separate cohorts.
A missing language slice or unavailable natural input is recorded with a skip
status and null metrics, not a numerical zero. Read cohort statuses and support
counts before comparing models. Natural records lacking a target label are
identified as unlabeled for that target; their label is not inferred.

Accuracy, macro-F1, and weighted-F1 use the fixed target label universe.
Therefore, a class absent from one language slice still contributes zero F1 to
that slice's macro average. This makes the averaging convention consistent
across slices; always report the label universe and support counts alongside
the resulting score. Severity additionally reports quadratic weighted kappa
using the supplied low-to-high order. QWK is undefined for certain constant
distributions, and is written as JSON `null` with a reason rather than `NaN`.

Top-label expected calibration error uses equal-width confidence bins:

```text
ECE = sum over bins [(bin_count / sample_count) * abs(bin_accuracy - bin_mean_confidence)]
```

Bins are left-closed/right-open except the final bin, which includes confidence
1. Empty bins have no observed accuracy/confidence. The reliability diagram
plots observed correctness against calibrated confidence, with sample counts
below it. ECE depends on the chosen bin count, so keep `--ece-bins` consistent
when comparing experiments. Calibration is assessed on held-out test data;
low validation calibration loss alone does not establish test calibration.

## 6. Interpret the paired spelling-noise robustness result

`--noise-rate` controls the probability of selecting an eligible ASCII word
for one spelling perturbation. The noise function uses a seeded local random
generator, preserves Bangla script, numeric/alphanumeric tokens, URLs, and
spacing, and applies letter elongation or a common phonetic spelling change.
Only the complaint text is perturbed; incident identity, category, severity,
location, and timestamp are retained.

The noisy text receives exactly the normal preprocessing used for clean
inference. Some perturbations may be removed by normalization. The output
therefore reports the changed-text count before and after preprocessing as
well as:

```text
macro_f1_drop = clean_banglish_macro_f1 - noisy_banglish_macro_f1
```

The drop is signed. It can be zero or negative; the script does not force a
decline or invent a robustness score. Compare the same original Banglish rows
with the same labels in both conditions. This synthetic stress test does not
replace evaluation on naturally written Banglish.

## 7. Read reports and use the saved models

The classical experiment writes:

```text
results/
├── metrics.json
├── results_table.csv
└── <target>/<model>/<cohort>/
    ├── classification_report.json
    ├── classification_report.txt
    ├── confusion_matrix.csv
    ├── confusion_matrix.png
    ├── reliability_bins.csv
    ├── reliability_diagram.png
    └── predictions.csv

artifacts/
├── experiment.json
├── <target>/<model>.joblib
├── resources/banglish_variants.json
└── minilm_encoder/                 # present when embedding models are fitted
```

Empty/skipped cohorts do not receive fabricated report plots. The JSON records
experiment settings, cohort outcomes, calibration provenance, and model
metadata; the table provides one model/target/cohort row for thesis comparisons.
Use actual generated rows and reports when writing your thesis. Console output
prints measured values, with a schematic shape such as:

```text
<JSON line with target, model, cohort, status, and metrics calculated from your data>
<The noisy-cohort JSON line additionally contains the signed macro_f1_drop>
Saved measured results: <results directory>/metrics.json
Saved thesis table: <results directory>/results_table.csv
Saved joblib models and inference resources: <artifact directory>
```

Run a saved classical classifier with the shared preprocessing:

```powershell
python -m ml_training.scripts.predict_classifiers --model artifacts/category/char_svm.joblib --text "pany nei" --lang banglish
```

Figure generation checks matplotlib's installed font inventory. It prefers
Nirmala UI on Windows or an available Noto Sans Bengali/Bangla font, with
DejaVu Sans as a fallback for Latin text. No font is downloaded automatically.
If Bangla labels appear as boxes or matplotlib reports missing glyphs, install
a Bangla-capable font locally and restart Python. If font discovery is stale,
set `MPLCONFIGDIR` to a new writable cache directory before launching the
script so matplotlib rebuilds its font inventory. CSV/JSON labels retain their
Unicode text regardless of the plotting fonts available on the machine.

Keep the artifact directory's relative resource layout intact when copying it
to another machine. Embedding classifiers require their saved encoder folder;
the normalization snapshot preserves the training dictionary. Joblib files use
Python pickle: load only artifacts you created or otherwise trust. Do not load
arbitrary downloaded model pickles to inspect them.

## 8. Optional XLM-R fine-tuning on GPU/Colab

This is a separate, optional experiment with Hugging Face weights and a Torch
training loop. It is not required for the TF-IDF or MiniLM experiments. Install
the core dependencies and optional XLM-R requirements in the chosen runtime:

```powershell
python -m pip install -r requirements.txt
python -m pip install -r requirements-xlmr.txt
python -m ml_training.scripts.finetune_xlmr --help
```

For a cached/local pretrained checkpoint:

```powershell
python -m ml_training.scripts.finetune_xlmr --target category --input-dir ml_training/data/multilingual --natural-test ml_training/data/natural_test/collected.csv --model path/to/local/xlm-roberta-base --device cuda --epochs 3 --batch-size 8 --max-length 128 --learning-rate 2e-5 --seed 42
```

For severity, the order remains explicit:

```powershell
python -m ml_training.scripts.finetune_xlmr --target severity --severity-order low medium high --input-dir ml_training/data/multilingual --model FacebookAI/xlm-roberta-base --allow-download --device auto --epochs 3 --n-bins 10 --noise-rate 0.15
```

`--allow-download` is required for online checkpoint retrieval; otherwise local
or cached files must be complete. `--revision` pins the optional pretrained
checkpoint. `--device auto` chooses an available accelerator or CPU; GPU/Colab
is optional, although fine-tuning can be expensive on CPU.

XLM-R fits for a fixed epoch count on training records. A scalar temperature
is fitted using validation logits, then used for test probabilities. Validation
is reserved for temperature calibration; no test-based checkpoint selection is
performed. The script uses the same cohort evaluation functions and ordinal
severity convention as the classical models. Its bin flag is `--n-bins`.

Defaults keep these results separate as `results/xlmr_<target>/` and
`artifacts/xlmr_<target>/`. Override them with `--results-dir` and
`--artifacts-dir` if needed. Hugging Face `save_pretrained` writes weights and
tokenizer under the artifact's `model/` directory; `model.joblib` is a descriptor
containing label order, relative model path, and calibration temperature, not a
pickle of the neural weights. Preserve the entire directory when copying it.

## 9. Keep the comparison reproducible

Retain input manifests, split IDs, annotation provenance, model/encoder
revisions, severity order, preprocessing resources, dependency versions, random
seed, and complete output directories for each experiment. Use the same fixed
test cohorts and averaging convention across models. Resolve unknown labels,
group leakage, and translation-quality problems upstream rather than filtering
them after inspecting test scores.

Run the unit tests with:

```powershell
python -m pytest -q
```

The tests use synthetic fixtures for implementation checks. They do not provide
research metrics, establish transliteration/translation quality, or substitute
for your reviewed, held-out dataset.
