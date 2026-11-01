# Source change inventory

Comparison: the existing Phase 7 green-motion source ZIP against this release.
This checkout has no Git metadata. This is a file-content comparison, not a Git diff.
It includes the Phase 8 deployment/report work already underway as well as security hardening.
Private files, datasets, installed dependencies and generated frontend builds are excluded.

The core security changes are backend configuration/authentication/input boundaries,
retention/deletion, privacy rendering, dependency pins, tests and release evidence.

## Modified

- `.env.example`
- `.gitignore`
- `README.md`
- `backend/app.py`
- `backend/config.py`
- `backend/openapi.json`
- `backend/postman_collection.json`
- `backend/schemas.py`
- `backend/scripts/create_admin.py`
- `backend/security.py`
- `docs/current_workspace.md`
- `docs/phase6_backend.md`
- `docs/verification.md`
- `frontend/.env.example`
- `frontend/package-lock.json`
- `frontend/package.json`
- `frontend/src/App.tsx`
- `frontend/src/i18n.tsx`
- `frontend/src/pages/PublicPage.tsx`
- `frontend/src/pages/TrackingPage.tsx`
- `frontend/src/styles.css`
- `frontend/src/test/admin.test.tsx`
- `frontend/src/test/public.test.tsx`
- `requirements-backend.txt`
- `requirements-embeddings.txt`
- `requirements-xlmr.txt`
- `requirements.txt`
- `tests/test_backend_security.py`

## Added

- `.dockerignore`
- `.github/workflows/ci.yml`
- `Dockerfile`
- `backend/retention.py`
- `backend/scripts/delete_data.py`
- `compose.yaml`
- `deploy/__init__.py`
- `deploy/artifacts/.gitkeep`
- `deploy/configure_local.py`
- `deploy/entrypoint.py`
- `deploy/healthcheck.py`
- `deploy/lock_runtime.py`
- `docs/containers.md`
- `docs/data_retention.md`
- `docs/deployment.md`
- `docs/research_questions.md`
- `docs/security/dependency_audit.md`
- `docs/security/npm-audit-after.json`
- `docs/security/npm-audit-before.json`
- `docs/security/npm-installed-before.json`
- `docs/security/pip-audit-direct.json`
- `docs/security/pip-audit-research.json`
- `docs/security/pip-audit-runtime-before.json`
- `docs/security/pip-audit-runtime.json`
- `docs/security/verification_results.json`
- `docs/security_changes.md`
- `docs/security_checklist.md`
- `docs/security_verification.md`
- `docs/thesis_outline.md`
- `docs/thesis_results.md`
- `docs/thesis_validity_ethics.md`
- `frontend/src/pages/PrivacyPage.tsx`
- `frontend/src/privacy.ts`
- `frontend/src/test/privacy.test.tsx`
- `frontend/vercel.json`
- `make_thesis_report.py`
- `requirements-ci.txt`
- `requirements-research-audit.txt`
- `requirements-runtime-audit.txt`
- `requirements-runtime-embeddings.in`
- `requirements-runtime-embeddings.lock`
- `requirements-runtime.in`
- `requirements-runtime.lock`
- `ruff.toml`
- `scripts/create_admin.py`
- `scripts/delete_data.py`
- `tests/test_deployment_helpers.py`
- `tests/test_retention.py`
- `tests/test_security_hardening.py`
- `tests/test_thesis_report.py`

## Removed

None.
