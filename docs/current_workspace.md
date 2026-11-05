# Current workspace data readiness

Current release verification (1 November 2026): **1,022 Python tests** and
**35 frontend tests** passed, with a successful frontend build and correctness
lint. See the [security evidence](security_verification.md),
[filled release checklist](security_checklist.md),
[actual dependency audits](security/dependency_audit.md) and
[deployment guide](deployment.md). Docker, hosted services, real neural model
compatibility and production configuration remain unverified. Phase-specific
counts later in this document describe their earlier releases.

This record describes the local work completed from the two user-supplied ZIP
archives. The downloadable source-code ZIP excludes source rows, generated
splits, translation work files, caches and trained models. A fresh checkout
must run the documented importer against the user's archives.

## Completed locally

- Imported 6,105 BanglaPSG source rows with their actual category and severity
  values. Source text, workbook row identity and formula/cache provenance are
  retained in the private raw import. No packaged instructions or scripts ran.
- Prepared all source rows into 4,274 training, 916 validation and 915 test
  rows. The 5,997 generally normalized-text groups are disjoint across splits.
  These groups are a duplicate-control fallback, not verified incident IDs.
- Preserved repeated source text. One cleaned-text group contains multiple
  category labels; it stays atomic and the preparation manifest records the
  limitation. Review conflicting labels before a definitive research run.
- Exported an annotation starter sample of 100 source-text groups, each with
  one Bangla source and one derived rule-based Banglish version. Existing split
  assignment is retained. This is a small workflow starter, not the final
  multilingual training/evaluation cohort.
- Created `ml_training/data/multilingual/translation_queue.csv` with 100 pending
  English translations. No English grievance translations were invented.
- Generated Phase 2 pair suggestions from that starter sample. These are
  annotation candidates with construction labels; no gold decisions were added.
- Imported 201 packaged BanglaDual example language pairs to a separate
  auxiliary CSV. The large original archive is inventoried and fingerprinted;
  its full Parquet corpus was not loaded or replicated. The auxiliary examples
  have `is_incident_gold=false` and are not incident training/test observations.

## Still required for a valid Phase 4 thesis experiment

1. Establish actual incident lineage and review source label conflicts. Exact
   normalized text grouping cannot identify every related paraphrase or event.
2. Fill and review the English translation queue, then import it with the
   Phase 2 script. Increase the multilingual sample deliberately if needed;
   keep related variants in their original split.
3. Collect/review locations and timestamps where available. The supplied
   workbook has no structured location or timestamp fields. Missing values
   remain unknown; text extraction is evidence requiring validation, not a
   guarantee of event identity. Missing metadata can prevent the Phase 2
   generator from finding hard-negative candidates; do not fabricate values
   to fill a pair quota.
4. Include naturally occurring same-language positives and cross-language hard
   negatives, not only translated positives and construction-rule negatives.
5. Complete independent duplicate annotation and adjudication. Preserve the
   original annotator records, then apply the final `pair_id,gold_label` table
   with `apply_duplicate_labels` before the default Phase 4 runner.
6. Supply a real cached MiniLM checkpoint or explicitly allow its initial
   download. Implementation tests use a deterministic test encoder for neural
   integration; no real MiniLM experiment or production latency is claimed.

The independent natural collection template remains empty. No final research
duplicate scores, gold incident clusters or trained research models have been
produced. See [the Phase 4 workflow](phase4_workflow.md) and
[experiment design](duplicate_experiment_design.md) for exact commands and
limitations.

## Phase 5 readiness

Priority rules, an empty expert-judgment template, optional pairwise-logistic
ranking, explanation/inference tools, validation-threshold selection and fairness
audits are implemented. There are no trained category/severity artifacts, expert
queue judgments or selected production review policy in this workspace. The
default analysis function therefore uses rule cues with null classifier outputs
and `needs_human_review`. Follow [the Phase 5 guide](phase5_workflow.md) after
training classifiers and collecting the necessary reviewed expert data.

## Phase 6 readiness

The FastAPI backend, four SQLAlchemy domain tables, Alembic migration, admin JWT
helpers, rate limiting, validation, cached ML adapter, versioned human review,
feedback export, typed OpenAPI and Postman collection are implemented. The full
repository suite passes 911 tests, including 146 backend checks. Endpoint
grouping checks use controlled semantic inference with the real area/time and
transaction guards; they are not measurements of a trained duplicate model.

No administrator password/signing secret or operational complaint database is
bundled. Configure your private `.env`, apply the migration and follow
[the backend guide](phase6_backend.md). Development supports rule-only review
when artifacts are missing. Production additionally requires PostgreSQL, shared
Redis, ready model artifacts and deployment integration/load validation.

## Phase 7 interface

Verified release: 921 Python tests, 31 frontend tests and a successful production
build. Public submission/tracking, admin login/explanations/groups/charts and
desktop/mobile bilingual layouts were checked locally with synthetic data.

A React/TypeScript/Tailwind interface is provided in `frontend/`, connecting to
the real FastAPI API through an environment-configured base URL. It includes
public submission/tracking, bilingual controls, administrator review and
duplicate-group workflows, explanations and operational charts. The new
migration stores optional private contact details; public tracking returns a
limited summary only. The frontend does not create translations, collected
complaints, expert labels or trained model artifacts.

See [the frontend guide](phase7_frontend.md) for installation, build and workflow
commands. The verification record distinguishes implementation checks from
real-model accuracy and deployment testing.

## Citizen account follow-up — 2026-11-05

Citizen registration/sign-in and account-only complaint history have been added
after the release described above. Anonymous submission and tracking remain
available; staff access remains separate. Profile fields live in a dedicated
account table and only signed-in submissions receive an ownership link.

For this follow-up, all 37 frontend tests pass and the production frontend build
passes locally. Python source compilation also passes. Backend pytest and the
database migration comparison were not run locally because the available
Windows Python package installation was interrupted by package-download and
temporary-directory access failures. The checked-in GitHub CI workflow remains
the required backend verification; the feature is not considered deployed
until that check and the hosting deploys succeed.
