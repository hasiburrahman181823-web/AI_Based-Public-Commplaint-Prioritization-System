# Verification record

Frontend refinement (1 November 2026): removed the visible required-field star
and decorative status dots from the public/tracking pages; the complaint field
remains required and keeps its accessible description. Simplified bilingual
copy, card styling and motion while preserving the logo green. All 35 frontend
tests and the TypeScript/Vite build passed. Browser checks confirmed the star
and dots are absent, visible text remains `rgb(7, 102, 87)`, and the English
desktop and 390-pixel Bangla public/tracking layouts have no horizontal overflow.
The temporary browser language/viewport settings were restored. Reduced-motion
rules remain in the source; an operating-system preference change was not tested.

Phase 8 wrap-up recheck (1 November 2026): all 21 thesis-report tests and all
15 deployment-helper tests passed. The saved-results report was regenerated
from the currently empty `results/` directory and states that no research
measurements are available. A Dockerfile copy was narrowed to avoid storing
staged private models twice; standalone Docker environment-file instructions
now distinguish literal values from Python dotenv quoting. Docker execution,
image size and cloud deployment remain unverified.

The repository was checked using Python 3.14.7, pandas 3.0.6, NumPy 2.5.3,
PyArrow 25.0.1, pytest 9.2.0, FastAPI 0.141.1, Uvicorn 0.53.0, Pydantic 2.13.5,
and HTTPX 0.28.1. These are observed test-environment versions, not a lockfile.

## Unit and integration tests

```bash
python -m pytest -q
```

Latest full Python run (1 November 2026): **1,022 passed**, with no failures or skips.
The installed Starlette and SlowAPI dependencies emitted 249 deprecation
warnings under Python 3.14; the warnings do not represent failed tests.
The frontend separately passes **35 Vitest tests** and its TypeScript/production
build. These are implementation checks, not research performance. The earlier
Phase 6 release passed 911 Python tests; its checks below remain a historical
record of that release.

The suite covers normalization, strict CSV parsing, mapping and validation,
deterministic group-safe splits, transliteration, sampling, translation queue
integrity, derived metadata, manual paraphrases, candidate pair rules, bounded
generation, SQLite persistence, independent labels, revision conflicts,
pagination, exports, and known/undefined Cohen's kappa cases.

## Synthetic end-to-end run

The complete CLI workflow was exercised on 240 synthetic Bangla rows from
120 incidents and three categories. The run produced:

| Stage | Observed result |
| --- | --- |
| Phase 1 | 168 train / 36 validation / 36 test rows; exact 70/15/15 for this fixture |
| Multilingual export | 120 source originals + 120 derived Banglish versions |
| Filled-queue import | 120 additional derived English test records; 360 total |
| Pair generation | 279 pairs, including manual paraphrases and all three other candidate types |
| Split isolation | No original incident crossed train, validation, and test |
| Annotation | Two synthetic annotators saved duplicate and priority judgments; CSV export and shared-item agreement endpoints succeeded |

English test strings were synthetic placeholders used to verify the import
mechanics. This test does **not** validate translation quality, transliteration
accuracy, incident labels, or the final research dataset.

## Browser check

The local FastAPI interface was opened in the in-app browser with synthetic
data. Loading a queue, saving a duplicate judgment, advancing to the next item,
and the updated saved-count indicator were checked. The final interface was
reloaded after the last code change. The preview database and synthetic input
files are not included in this repository.

## Phase 3 implementation checks

The Phase 3 checks use synthetic fixtures only. The TF-IDF/SVM and TF-IDF/logistic
regression experiments were exercised end to end for category and severity,
including validation-only calibration, per-language and natural cohorts,
noise/clean alignment, real report PNGs, measured JSON/CSV exports, and portable
joblib inference with a saved normalization dictionary. Generated confusion
matrices and reliability diagrams were also visually inspected.

Embedding integration tests use a deterministic test encoder, including encoder
reuse, saved-resource references, and LR/SVM model round trips. They do not
verify a real MiniLM checkpoint. Optional XLM-R tests cover numerical softmax
and temperature-loss helpers, offline defaults, explicit class coverage and
output-path safety without importing the neural training dependencies.

No pretrained weights were downloaded, and no real MiniLM encoding or XLM-R
fine-tuning was run in this workspace. Those execution paths require the
optional dependencies and a cached/local checkpoint, or an explicit download
flag in the user's environment. No research scores or trained research models
are supplied. Running the experiment scripts prints the measurements from the
provided data and writes them to the selected result directory.

## Phase 4 and supplied-data checks

Location and time rules, label-independent symmetric pair features, strict
gold-label input checks, validation-only threshold selection, per-stratum
metrics, error exports, connected components, and cached retrieval were tested
with synthetic observations. The real TF-IDF cosine baseline and both learned
feature-classifier choices were exercised end to end. Tests verify that all
selected ablations retain the same ordered test pairs and fingerprint, and that
changing test labels cannot change the selected threshold or raw scores.

Real feature extraction, saved sklearn pipelines and inference-resource
snapshots were used for production-inference integration tests. MiniLM vectors
were supplied by a deterministic test encoder; no real checkpoint performance
is inferred. A structural 50k-record test checks that retrieval stores an N×D
matrix, caches repeated text and avoids N×N allocation. This does not measure
real checkpoint latency, memory budget or end-to-end candidate recall.

The supplied BanglaPSG workbook was imported read-only, preserving source
labels and source provenance without executing formulas or packaged code.
The full source CSV passed preparation and group isolation checks. BanglaDual's
packaged CSV examples were imported as a bounded auxiliary language sample,
separate from incident data. The importer has miniature archive/workbook tests
for missing labels, formulas, malformed schemas and provenance. See
[dataset sources](dataset_sources.md) and [current workspace state](current_workspace.md).

No final gold duplicate labels, real MiniLM experiment, research ablation
scores, independently verified incident clusters or production throughput
measurements are bundled. Those require the user's reviewed data, completed
translation/annotation workflow and actual model checkpoint.

## Phase 5 implementation checks

Synthetic fixtures exercise configurable multilingual priority rules and simple
negation, exact rule contributions, expert-query/incident isolation, bounded
pairwise ranking, explicit insufficient-data fallbacks, and tie-aware held-out
NDCG/Spearman calculations. No expert priority labels have been invented.

The calibrated-classifier runtime is tested with real fitted sklearn models.
Tests check selected-class raw-margin explanations, binary coefficient direction,
multiclass contributions, saved-resource integrity, batched inference, and
separation of rule points from learned ranking utility. Embedding tests use a
deterministic test encoder with real fitted/calibrated LR/SVM classifiers;
dimension contributions and approximate token-deletion sensitivity are labeled
separately. No real neural checkpoint performance is claimed.

Validation export, confidence-policy selection/loading, JSON/batch analysis,
and fairness export are exercised end to end. Tests check both model and resource
fingerprints, validation-only provenance, reject-all behavior, confidence ties,
unknown/missing model fallback, review flags, source-report parent lineage,
and descriptive subgroup metrics with unavailable truth represented explicitly.
The coverage/error PNG is generated from synthetic validation fixtures.

No real expert-ranker, deployment review threshold, priority accuracy or fairness
gap is supplied as a research result. Real evaluation requires actual model
artifacts, independent judgments and reviewed validation/evaluation cohorts.

## Phase 6 backend checks

The full run includes 146 backend tests: relational storage and migration
upgrade/downgrade/ORM parity (20), authentication/configuration/validation (53),
cached ML service and exact-area/time guards (25), feedback export (12), endpoint
integration (30), and startup/dependency failure behavior (6).

TestClient exercises every endpoint, admin-only reads and mutations, bounded
anonymous submission, private candidate redaction, all four filters,
pagination, priority/status overrides, preserved automatic analysis, feedback
and audit records, stale-version conflicts, group confirmation/splitting,
preview without persistence, health and aggregate statistics. Two independent
application/ML instances concurrently submit to a shared SQLite database and
retain a single eligible incident group. An admin split remains protected from
subsequent automatic similarity matches and unrelated membership moves.

The Mirpur 10 cross-language regression uses a deterministic semantic analyzer
that deliberately proposes high similarity for different areas. Real backend
guards group the Bangla/Banglish/English same-area reports and veto the
different-area report. This verifies backend behavior, not real MiniLM accuracy.
Other tests exercise the actual rule-only analyzer: missing classifier outputs
stay null, human review is required, and the analyzer loads once per worker.

Strict readiness rejects missing artifacts; startup requires migrations and
available rate-limit storage. A runtime rate-store failure returns a sanitized
503 without saving a complaint. Database health failures are also sanitized.
The exported OpenAPI contains typed responses and JWT security definitions.
All 15 Postman requests and their variable-substituted JSON bodies were checked;
the environment contains no password or bearer token.

Observed additional versions: SQLAlchemy 2.0.52, Alembic 1.20.1, SlowAPI 0.1.10,
and PyJWT 2.13.0. No live PostgreSQL or Redis server, real neural checkpoint,
reverse proxy or production load test was run. Production rollout still needs
those integration checks and the user's trained, validated artifacts. Source
packages exclude operational databases, feedback exports, private credentials,
research data, model weights and caches.

## Phase 7 frontend checks

The frontend has 31 focused automated checks using Vitest, Testing Library and
jsdom. The API boundary tests cover anonymous optional-contact submission,
cookie omission, explicit admin bearer headers, false-valued review filters,
redacted tracking routes, stale-version errors without mutation retries,
expired-session signaling, malformed proxy responses and network failures.

Interaction tests cover local language detection and manual correction,
English/Bangla preference persistence and document language, memory-only admin
sessions, a receipt only after confirmed submission, preserved drafts after
errors, limited public tracking, malformed/missing IDs and cancellation when a
pending tracking route is cleared. The cancellation test exposed a stale busy
flag; the page now resets it and ignores aborted results.

Administrator tests cover the server-side review flag, reason/version-bound
priority overrides, stale-review errors without false success, actual submitted
language versions and a selected-subset split payload. Explanation/chart checks
verify a label-only human override, unavailable evidence without invented terms,
readable Bangla review reasons, actual TF-IDF/token-deletion explanations,
default `moderate` priority labels, and an accessible time-series data table
with current-state/UTC semantics. Unrelated JSON success responses cannot be
mistaken for a confirmed submission receipt.
These checks use synthetic fixtures and mocked network boundaries. They do not
measure real-model language or duplicate accuracy.

TypeScript checking and the Vite production build passed. All 19 Postman request definitions, variables and
substituted JSON bodies were checked; the added admin requests explicitly use
bearer authentication and public tracking does not. Password/token environment
values remain empty. The new private contact field is never embedded in public
tracking, analysis context or ordinary feedback exports.

The latest Python run includes 156 backend checks. Added coverage verifies
private contacts, anonymous tracking redaction and limits, review filtering,
admin-only whole-system analytics and an incremental migration preserving
existing Phase 6 complaints and linked feedback. No live PostgreSQL/Redis
server was used.

Local browser checks used a disposable SQLite database and synthetic complaints.
The real form submitted a Bangla complaint, received a UUID tracking receipt,
and opened a status page without complaint/contact content. Admin sign-in,
priority queue, rule explanations, expanded three-language incident groups and
the actual-data charts were inspected. The group fixture was explicitly merged
by an administrator for this UI check; no untrained semantic accuracy is implied.
English desktop and Bangla 390×844 layouts were visually checked; the mobile
form fits without horizontal overflow. Temporary viewport settings were reset.
Saved previews contain synthetic data only and are separate from the source ZIP.

The compiled preview returned HTTP 200 for the public page and a direct tracking
route. Its `/api/health` proxy reached the real local backend, which correctly
reported degraded ML readiness because no trained artifacts were configured.
Observed tooling: Node 22.22.0, React 19.3.1, Vite 7.3.7, TypeScript 5.9.3,
Tailwind 4.3.3 and Vitest 3.2.7. The public JS is about 94 kB gzip plus about
9 kB gzip CSS; the separately loaded admin JS is about 19 kB gzip in this build.
These are generated build sizes, not a slow-network latency benchmark.

Automated DOM checks do not establish visual layout, real screen-reader
behavior, device performance or slow-network latency. Production database,
Redis, proxy and real-model integration retain the Phase 6 verification limits.

## Single-green theme and motion update

The requested visual revision uses the logo green `#076657` for all interface
text, including placeholders, buttons, status labels, admin panels and SVG chart
labels. White and pale green surfaces keep the green text readable. Priority
charts use green/white patterns with their labels and accessible data tables.

Added short page/card entrances, staggered instructions, menu underlines,
pointer-hover/press feedback, focus highlights, a receipt check animation,
sliding detail panels and chart entrances. The status-dot pulse stops after two
cycles; no decorative animation runs indefinitely. Loading spinners remain
active only while work is pending. Reduced-motion CSS disables animations,
transitions and decorative hover/press transforms while preserving visible
content and functional selection indicators.

All 31 frontend tests and the TypeScript/Vite build passed after this update.
Browser checks found only `rgb(7, 102, 87)` in visible public/admin text samples,
and verified chart pattern fills and entrance animations. The 390-pixel Bangla
layout had no horizontal overflow. Desktop/public/admin and mobile previews
were captured with synthetic data; viewport and language preferences were
restored. Reduced-motion behavior was reviewed in source; an operating-system
preference change was not performed. Backend code was unchanged, so the prior
921-test Python result remains the recorded backend/research verification.

## Single-colour frontend follow-up

The visual follow-up uses the logo green (`#076657`) for interface text and
monochrome green/white patterns for chart priorities. Source review checked the
shared text classes and SVG axis labels. Explicit priority/status labels and
the accessible chart table remain available alongside the patterned marks.

Entrance effects use short transform/opacity animations, with visible base
styles so disabling animation does not hide content. Hover motion is limited
to devices with a fine pointer; the stylesheet includes reduced-motion
handling. The drawer overlay remains outside animated content containers,
preserving its viewport positioning. The earlier test and browser evidence
above describes its recorded run; the release check records any subsequent
verification of this styling update separately.
