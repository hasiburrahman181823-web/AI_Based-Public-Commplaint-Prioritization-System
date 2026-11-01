# Security hardening verification

Verified locally **1 November 2026** with disposable synthetic fixtures, not
operational complaints or production credentials. No hosted deployment, model
accuracy or general security certification is claimed.

## Evidence

The final full Python suite passed **1,022 tests**, with no failures or skips,
in 149.67 seconds. It reported 249 third-party deprecation warnings. Focused backend
checks passed 154 tests. Subsequent administrator-file provisioning checks passed
82 tests, including no generated credential output and no overwrite. Frontend
verification passed **35 tests** across 7 files, TypeScript checking and the Vite
production build. Correctness lint passed using the repository's configured rules.

Backend tests exercise real TestClient requests, SQLite and in-memory SlowAPI.
ML grouping uses controlled semantic fixtures, not a real pretrained checkpoint.
Production docs route checks inject SQLite and omit lifespan, so do not establish
PostgreSQL/Redis startup. A separate production factory check rejected missing,
short and default JWT keys before services were contacted.

A disposable Git repository's `git check-ignore --no-index` confirmed private
environment/database/journal/raw-data/artifact/Compose exclusions. This source
checkout has no Git metadata/history; historical exposure could not be assessed.
No real private environment files were printed.

Observed: Python 3.14.7, pytest 9.2.0, FastAPI 0.141.1, Starlette 1.6.0,
SQLAlchemy 2.0.52, SlowAPI 0.1.10, PyJWT 2.15.1, Argon2-cffi 25.1.0.
Local Ruff: 0.14.2; CI pins 0.17.3. Audits: pip-audit 2.10.1.
Frontend: Node 22.22.0, npm 10.9.4, Vite 7.3.7, Vitest 5.0.5.
Linux Python 3.12.16 container/CI execution is unverified. Third-party Python
deprecation warnings were reported, not hidden.

See [actual dependency reports](security/dependency_audit.md) for advisory counts,
before/after JSON, commands and audit scope limits.

## Reproduce

From the repository root in a fresh Python 3.12 virtual environment:

```powershell
python -m pip install -r requirements-backend.txt -r requirements-ci.txt
python -m ruff check .
python -m pytest -q --junitxml=test-results.xml
python -m pip_audit --no-deps --disable-pip -r requirements-runtime-audit.txt
python -m pip_audit --no-deps --disable-pip -r requirements-research-audit.txt
python scripts/create_admin.py --help
python scripts/delete_data.py --help
```

Security-focused checks:

```powershell
python -m pytest -q tests/test_security_hardening.py tests/test_backend_security.py tests/test_retention.py tests/test_backend_api.py tests/test_backend_lifecycle.py
```

From `frontend/`:

```powershell
npm ci --ignore-scripts
npm test
npm run build
npm audit --audit-level=low
```

Tests need no real `.env`, dataset, database credentials or model weights.
Optional heavyweight neural training is unnecessary for these checks. See
[container instructions](containers.md) and the separate research audit commands
in the findings document. With `--disable-pip`, pip-audit does not expand nested
requirements automatically; use the prepared explicit audit input.

## Intended compatibility changes

New complaint/preview text is capped at 1,000 characters; longer stored rows are
preserved. Legacy PBKDF2 admin hashes need regeneration. Admin setup now writes
a new ignored `.env.admin` instead of printing credentials. Production API docs
are disabled. Vitest 5.0.5 requires supported Node 22.12+; the project pins Node
22.22.0. Actual private model compatibility after ML-library updates remains a
deployment check.

## Unverified deployment items

Docker/Compose, live PostgreSQL/Redis, hosted HTTPS/CORS/CSP, trusted proxy peers,
real-model compatibility/performance, backup erasure and remote CI execution are
untested. The [filled release checklist](security_checklist.md) keeps these gates
unchecked. Deletion covers live rows, known linked snapshots and the configured
cache, not arbitrary exports, other workers' memory, backups or machine unlearning.
