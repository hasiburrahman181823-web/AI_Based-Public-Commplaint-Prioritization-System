# Phase 6: persistent complaint API

The FastAPI backend connects the earlier analysis tools to durable complaint,
duplicate-group, admin-feedback and audit records. It includes an Alembic
migration, administrator JWT authentication, bounded anonymous intake, and
endpoint tests. It is a deployment foundation; the repository does not claim
an audited production deployment, real-model duplicate accuracy, a throughput
benchmark or an operational availability guarantee.

## 1. Install and configure a local environment

Run commands from the repository root. Install the backend requirements into
your existing virtual environment:

```powershell
python -m pip install -r requirements-backend.txt
```

For a fresh checkout, copy the example configuration. If `.env` already exists,
edit it instead of replacing your earlier configuration:

```powershell
Copy-Item .env.example .env
python scripts/create_admin.py --username admin
```

The helper prompts twice for the password and writes the administrator name,
salted Argon2id hash and a random JWT signing secret to the ignored private
`.env.admin` file. It refuses to overwrite an existing file and never prints
credentials. Copy the assignments privately into `.env` or your provider's
secret settings. `--output` selects another ignored private environment file.
It does not store the plaintext password or create a public registration endpoint.
For noninteractive local provisioning, `--password-env VARIABLE_NAME` reads a
password from an already populated environment variable; avoid plaintext
passwords in command history or shared deployment files.

The required values are:

```dotenv
BACKEND_ENVIRONMENT=development
BACKEND_DATABASE_URL=sqlite:///./backend/data/backend.sqlite3
BACKEND_ADMIN_USERNAME=admin
BACKEND_ADMIN_PASSWORD_HASH=<paste the generated Argon2id hash>
BACKEND_JWT_SECRET=<paste the generated random secret>
BACKEND_REQUIRE_ML_READY=false
BACKEND_RATE_LIMIT_STORAGE_URI=memory://
BACKEND_ALLOW_PUBLIC_READS=false
```

The angle-bracket values above are explanatory placeholders, not usable
credentials. There is no default administrator password or usable signing
secret. Startup rejects a missing signing secret, fewer than 32 characters,
known example/default values, and an invalid password hash. The actual signing
configuration name is `BACKEND_JWT_SECRET`; there is no `JWT_SECRET` alias.
Existing process environment values take precedence over the root `.env` file.

Security-hardening compatibility changes: new intake accepts at most 1,000
characters, legacy PBKDF2 hashes must be regenerated as Argon2id with the
helper, and `/docs`, `/redoc` and `/openapi.json` are disabled in production.
Existing complaint rows and the historical database length constraint are
retained; the smaller limit applies to new submission/preview requests.

## 2. Apply the migration and start the server

```powershell
python -m alembic upgrade head
python -m uvicorn backend.app:create_app --factory --host 127.0.0.1 --port 8000 --no-proxy-headers
```

Run migration commands as a separate rollout step, rather than asking each API
worker to race to create tables. The application does not silently run
`create_all`. The four domain tables are `complaints`, `duplicate_groups`,
`admin_reviews`, and `audit_log`; Alembic also maintains its migration-version
bookkeeping table. Complaint/group versions support optimistic update checks.

The default database file is local and ignored by Git. SQLite is intended for
development and tests; production configuration requires PostgreSQL and shared
Redis rate-limit storage. Back up data before applying a new migration and
verify the migration against the deployment's database version before rollout.
Do not downgrade a live database merely to reset a demonstration.

Open the generated API documentation after local development startup:

- [Swagger UI](http://127.0.0.1:8000/docs)
- [ReDoc](http://127.0.0.1:8000/redoc)
- [OpenAPI JSON](http://127.0.0.1:8000/openapi.json)
- [Health](http://127.0.0.1:8000/health)

The repository also includes `backend/openapi.json`. Regenerate the typed
specification after changing endpoint contracts:

```powershell
python -m backend.scripts.export_openapi
```

This command inspects route definitions without reading `.env`, starting ML
inference, connecting to the deployment database, or using real admin
credentials. It requires the backend Python dependencies. Private optional
fields remain absent in redacted responses instead of being inserted as nulls.

Clients should handle HTTP 401 for missing, invalid or expired administrator
tokens, HTTP 404 for unknown records, HTTP 409 for stale admin versions, HTTP
413 for oversized bodies, HTTP 422 for invalid fields or spam-like payloads,
HTTP 429 for exceeded rate limits, and HTTP 503 for unavailable dependencies.
Refresh and review a record after a version conflict before retrying a mutation.
An invalid login also returns HTTP 401. Citizen submission and preview do not
require a token.

## 3. Understand startup and model readiness

The ML service loads trusted saved artifacts once per worker, retains its
encoder and embedding cache, and serializes access to mutable inference/index
state. Existing complaint rows are refreshed into the worker's candidate index
before new analysis, so another worker's committed submissions are visible.
Synchronous endpoints run in FastAPI's worker thread pool instead of doing
blocking model/database work directly on an async event loop.

The artifact settings point to the Phase 3 classifier bundles, Phase 5 review
policy/ranker and Phase 4 duplicate bundle. Keep their entire resource trees,
including the normalization snapshots and pretrained encoder files. For real
MiniLM operation also install the optional dependencies:

```powershell
python -m pip install -r requirements-embeddings.txt
```

Development can run in explicitly degraded mode without trained artifacts.
It returns the real deterministic rule evidence, null unavailable classifier
outputs and a human-review requirement; it does not invent category/severity
predictions or promise cross-language duplicate grouping. Strict production
startup requires the configured ML pipeline to be ready. Refer to the previous
phase guides to train and validate the artifacts before enabling that mode.

`GET /health` is a small public readiness response without credentials,
filesystem paths or SQL exception details. A database outage returns HTTP 503.
Development degradation is distinct from a fully ready ML pipeline.

## 4. Endpoint and access contract

| Method and path | Default access | Behavior |
| --- | --- | --- |
| `POST /auth/token` | Public, login-rate limited | JSON administrator name/password; returns short-lived bearer token. |
| `POST /complaints` | Anonymous, submission-rate limited | Validates and analyzes one complaint, stores it transactionally, and proposes group membership. |
| `POST /analyze` | Anonymous, analysis-rate limited | Same analysis preview without inserting a complaint, group or review. |
| `GET /tracking/{id}` | Anonymous, rate limited | Minimal status summary for a random tracking ID; no complaint text, contact, group IDs or analysis. |
| `GET /complaints` | Administrator | Paginated filtering by `category`, `priority`, `area`, and `status`. |
| `GET /complaints/{id}` | Administrator | Current complaint fields, version and available stored analysis. |
| `PATCH /complaints/{id}` | Administrator | Reviewed priority/category/severity/status override with reason and expected version. |
| `GET /duplicate-groups` | Administrator | Paginated candidate/reviewed group information. |
| `GET /duplicate-groups/{id}` | Administrator | A current group's membership/version before a reviewed action. |
| `POST /duplicate-groups/{id}/confirm` | Administrator | Confirms a reviewed group; optional complaint IDs support an explicit reviewed merge. |
| `POST /duplicate-groups/{id}/split` | Administrator | Moves a proper nonempty subset into a new group and protects the reviewed split. |
| `GET /stats` | Public | Aggregate counts without complaint text or raw locations. |
| `GET /analytics` | Administrator | Full-corpus chart aggregates, UTC daily current-priority counts and review workload. |
| `POST /admin/data-deletion` | Administrator | Previews reviewed complaint IDs by default; explicit boolean `apply: true` erases them and known linked copies. |
| `GET /health` | Public | Database/service readiness. |

`BACKEND_ALLOW_PUBLIC_READS=true` optionally exposes redacted read responses.
Public read/submit/preview responses do not grant access to other complaints'
text, raw locations, internal analysis or audit/review history. Authentication
does not create citizen accounts: only the configured administrator uses JWTs.

### Anonymous submission or preview

```json
{
  "text": "মিরপুর ১০ এ ৩ দিন ধরে পানি নেই",
  "lang": "bn",
  "location": "Mirpur 10",
  "timestamp": "2026-10-30T10:00:00+06:00"
}
```

`lang` may be omitted to use the documented heuristic; explicit language is
preferable for evaluation. `location` and `timestamp` are optional. Supplied
timestamps require an ISO-8601 timezone offset and are normalized to UTC.
Phase 7 adds optional `contact` (up to 256 characters, without control characters).
Contact is stored as private operational data and only authenticated admin
complaint reads expose it. It is excluded from public receipts/tracking,
ML analysis/index records and feedback exports. Providing contact does not
enrol the citizen in an automatic notification service.
Citizens cannot supply category, severity, priority, status or group IDs as
trusted administrative fields. The example is illustrative, not collected data.

Input is bounded to 1,000 characters after NFKC normalization and a default
32,768-byte JSON body. Control characters are replaced with spaces before
validation; empty/whitespace-only, URL-only, more than two URLs and excessive
repeated characters/words are rejected. Frontend complaint, contact and
explanation values are rendered as React text, without HTML interpretation.
The filters do not
censor grievances because of political, critical or hazard-related vocabulary.
This small filter and IP limits are not a complete anti-bot or abuse system.

Default per-IP SlowAPI limits are `5/minute` on submissions, `10/minute` on
analysis (and a separate tracking counter), and the stricter `3/minute` on
login attempts. Configure `BACKEND_SUBMISSION_RATE_LIMIT`,
`BACKEND_ANALYSIS_RATE_LIMIT` and `BACKEND_LOGIN_RATE_LIMIT` deliberately;
shared Redis is mandatory in production. The JWT lifetime defaults to 30
minutes, with `BACKEND_JWT_EXPIRE_MINUTES` restricted to 1–60 minutes.

### Anonymous status tracking and chart data

The server-issued complaint UUID doubles as the tracking ID. Keep it private:
possession permits the minimal `GET /tracking/{id}` summary, and there is no
public directory/search. The response includes ID, status, language, category,
severity, priority score/label, review requirement, canonical area and creation/
update timestamps. Full complaint reads remain administrator-only by default.
Tracking shares the configured analysis-rate specification, with its own
endpoint limit. A missing ID yields 404 and a malformed UUID yields 422.

`GET /complaints?review_required=true` supports the dashboard review tab, and
`GET /duplicate-groups?group_id=<uuid>` can select a specific group. Admin-only
`GET /analytics` returns the existing aggregate fields plus
`priority_over_time: [{date, priority, count}]` and
`review_workload: [{status, review_required, count}]`. Dates are UTC report
creation dates. Daily priority charts use each record's current priority, not
historical model decisions. Aggregates cover the entire corpus, rather than the
currently loaded queue page.

Existing Phase 6 databases need `python -m alembic upgrade head` for migration
`0002_private_contact`. Existing rows are retained with null contact. Startup
checks the contact column; production additionally checks all Alembic heads
against the migration files shipped with the application.

### Administrator token

Send JSON to `POST /auth/token`:

```json
{"username": "admin", "password": "<your locally configured password>"}
```

Use the returned `access_token` in the `Authorization: Bearer ...` header for
administrator operations. The token has an expiry, fixed issuer/audience,
administrator role, subject and token identifier. The server accepts a fixed
HS256 algorithm rather than trusting the incoming token's algorithm choice.
Login failures use a generic response, and password verification is still
performed for a wrong username.

### Versioned overrides and group review

Fetch the current record before updating it. A priority/status review resembles:

```json
{
  "expected_version": 1,
  "reason": "Reviewed the reported essential-service impact",
  "priority_score": 11,
  "priority_label": "high",
  "status": "in_review"
}
```

The numbers are example input values, not model performance. Use the record's
actual returned version. A stale version returns HTTP 409; refresh and review
again rather than blindly replaying the write. Null or empty overrides are
rejected. Category/severity corrections also become human feedback.

Group confirmation takes `expected_version` and `reason`. An optional
`complaint_ids` array identifies existing reports the administrator has reviewed
for inclusion; this is an explicit human merge, not an automatic semantic
inference. Group splitting also requires `complaint_ids`, which must be a
strict subset of the current group. Moving every member or an unrelated ID is
an error. A split produces a new group while keeping remaining members in the
original group; reports and prior review evidence are not deleted.

## 5. Duplicate grouping and concurrent submissions

Semantic scores only propose candidates. Automatic grouping additionally
requires exact known canonical-area agreement, without location ambiguity or
metadata/text conflict. Supported disjoint ongoing intervals veto automatic
joining; different report timestamps alone do not prove separate incidents.
Thus a high text-similarity score cannot override known area disagreement.

A new report can join one eligible candidate group. Existing groups are never
automatically merged. Suggestions spanning several groups create a separate
reviewable proposal. Groups protected by a reviewed split are not silently
recombined; an explicit administrator review is needed.

The in-process reentrant lock protects shared model/index state. PostgreSQL
uses a transaction-level advisory coordination lock for the grouping write;
SQLite development uses its write transaction plus local synchronization.
These mechanisms avoid independent workers assigning from inconsistent
pre-commit views. They also serialize part of intake: benchmark the intended
deployment rather than inferring throughput from the earlier 50k structural
index test. Retrieval top-k, model accuracy and reviewed incident lineage still
limit end-to-end duplicate recall.

## 6. Human feedback and audit evidence

Each administrator correction records the actor, action, reason, previous
values, new values and time in `admin_reviews`, alongside an `audit_log` entry.
Automatic submitted analysis remains separate from the current override
fields. Group confirmations/splits preserve corresponding review evidence.
Reviewed privacy deletion can redact related audit narratives and identifiers
while preserving actor/action/time evidence. There is no general-purpose
audit-delete endpoint. This is application-level audit retention, not a claim
that an administrator with unrestricted SQL access cannot alter the database.

Use the operator feedback-export helper described in this repository to write
the corrections to a private training-feedback directory. Preserve review IDs,
before/after values, reasons and original model provenance. Human corrections
are candidates for reviewed retraining data; do not silently map arbitrary
rule scores to the expert 1–5 ranking scale or train on unadjudicated conflicts.
Keep feedback from the final evaluation cohort out of a training update unless
you deliberately retire and replace that holdout.

```powershell
python -m backend.scripts.export_feedback --output backend/exports/admin_feedback.jsonl
python -m backend.scripts.export_feedback --format csv --output backend/exports/admin_feedback.csv --since "2026-10-01T00:00:00Z"
# Explicitly include submitted complaint text only in an approved private export:
python -m backend.scripts.export_feedback --include-text --output backend/exports/admin_feedback_with_text.jsonl
```

Existing export files are protected unless `--overwrite` is supplied. Each
export also writes a `.manifest.json` with row count, format, SHA-256 and cutoff.
Database URLs are read from the private environment; they are not included in
the manifest. The default omits the separate complaint `text` column, but
reasons, original analysis and explanation terms can still contain sensitive
content. Treat the entire export as private. Group review rows remain events;
they are not silently expanded into duplicate-pair gold labels or expert ranks.

## 7. Run the Postman collection

Import `backend/postman_collection.json` and `backend/postman_environment.json`
into Postman. Select the imported environment and set `base_url` and the
administrator credentials locally. The supplied password/token values are
empty; no usable credentials are bundled. Avoid syncing filled credential
values into a shared workspace or exporting them with the code.

Run health, login, preview and the illustrative submissions. Collection scripts
capture the bearer token, report IDs and returned versions. Refresh the
administrator complaint/group reads before PATCH, confirmation or split so
the optimistic versions are current. The collection includes Bangla,
Banglish and English Mirpur reports and a same-text/different-area submission.
The explicit confirmation/merge request is a human action for reviewed sample
reports; it is not evidence that an untrained development model automatically
recognized them. The split request needs the sample reports to be in one group.

Use the collection on a disposable local database: confirmation and splitting
are real administrator writes. The collection's confirmation explicitly adds
the three illustrative Mirpur reports after review, including in degraded mode.
The scripts refresh stored versions from GET responses, but an unrelated write
between the read and mutation can still yield 409. Refresh rather than disabling
the version check. Responses are not assertions of real-model accuracy.

The collection also includes a deletion preview and a separately guarded apply
request. Applying requires an explicit local `deletion_apply_approved=true`
variable and reviewed complaint UUID; do not run destructive requests against
live data as an automatic collection test. Default submission limits also mean
rapidly repeating the collection can produce 429; wait for the limit window.

### Reviewed retention and deletion

First verify the requester's authority and scope through the published privacy
contact. `POST /admin/data-deletion` requires an administrator JWT and this
payload (replace the explanatory UUID placeholder with a reviewed actual ID):

```json
{
  "complaint_ids": ["<verified tracking UUID>"],
  "reason": "verified_privacy_request",
  "apply": false
}
```

Preview is the default and does not change domain rows or caches. After reviewing
the preview, use the literal JSON boolean `true` to apply; strings such as
`"true"` or `"false"` are rejected. The endpoint does not require an
`expected_version`. Deletion is irreversible: it removes selected complaint
rows/contact, direct reviews, known linked narrative/analysis snapshots and
the configured embedding cache, then refreshes the active worker's index.
A new minimal audit event records the administrator and counts without copying
the free-text reason or erased IDs into its JSON. Missing IDs are idempotent.

The body-size limit still applies: use batches of at most 500 UUIDs and a short
reason; any encoded request above 32,768 bytes receives 413 even though the
schema permits up to 1,000 IDs. The same body limit applies to merge/split.

The operator CLI supports default dry-run selection of currently
resolved/rejected cases last updated over 90 days ago, or specific verified IDs:

```powershell
python scripts/delete_data.py
python scripts/delete_data.py --complaint-id "<verified tracking UUID>"
python scripts/delete_data.py --complaint-id "<verified tracking UUID>" --apply --actor "authorised-steward" --reason "verified_privacy_request"
```

Stop/restart other API workers and follow the [complete retention runbook](data_retention.md)
for cache copies, holds, exports and backup expiry. No automatic retention job
or machine-unlearning guarantee is supplied. The schema has no immutable
closure timestamp; age selection uses `updated_at`, not claimed closure age.

## 8. Production configuration and verification limits

Production requires `BACKEND_ENVIRONMENT=production`, PostgreSQL, shared Redis
rate-limit storage and ready artifacts. A typical `.env` uses a
`postgresql+psycopg://...` URL and a `redis://...` or `rediss://...` storage URL;
keep credentials in a secret manager or private environment, not source code.
Use Redis authentication/TLS and database transport/access controls appropriate
to the deployment. The rate limiter fails closed with a service error if its
configured storage becomes unavailable; it must not silently become unlimited.

Terminate HTTPS at a configured reverse proxy/load balancer, restrict database
and Redis access, set exact permitted CORS origins, size worker/review capacity,
and exercise backup/restore and migration rollback plans before rollout.
The local run command disables proxy-header trust. Behind an approved proxy,
explicitly allowlist trusted forwarding peers; do not accept arbitrary client
`X-Forwarded-For` values to choose a rate-limit identity. Avoid logging passwords,
tokens or raw complaint bodies; the domain audit does not persist raw IPs.

`BACKEND_ALLOWED_ORIGINS` is a comma-separated list of exact HTTP(S) origins,
for example `https://<your-frontend-host>`. Wildcards and origins with paths or
credentials are rejected. When an allowlist is configured, unknown browser
origins receive 403 before persistence; production also enforces an empty list
by rejecting all requests carrying an Origin header. Set the actual frontend
origin before rollout. Origin checks are not authentication and do not prevent
non-browser clients from calling public endpoints.

Every API response receives `X-Content-Type-Options: nosniff`,
`X-Frame-Options: DENY`, `Referrer-Policy: no-referrer`, a restrictive basic
Content-Security-Policy and `Cache-Control: no-store`. Local documentation has
a narrow development-only CSP exception; all three documentation routes are
disabled in production. Static frontend headers are configured separately in
`frontend/vercel.json` and must be verified on the actual host.

For the supplied Docker launcher, set `FORWARDED_ALLOW_IPS` to explicit trusted
proxy IPs/CIDRs when needed. It defaults to no proxy-header trust. The legacy
`BACKEND_TRUST_PROXY_HEADERS` field does not configure Uvicorn's trusted peers;
do not rely on it to establish rate-limit identities. Apply and verify these
controls against the actual hosting provider's proxy behavior.

There is one environment-configured administrator account. The implementation
does not claim a multi-user permission system, refresh-token rotation or a
per-token revocation table. Use short token lifetimes and deliberate signing-key
rotation; revocation/identity-provider integration is a later operational
extension, not something a JWT alone supplies.

Password hashes use Argon2id with random salts, 64 MiB memory, time cost 3 and
parallelism 1. Legacy PBKDF2 values are rejected; regenerate the hash with the
interactive helper before upgrading an existing deployment. No security
certification is claimed. See the
[FastAPI bearer/JWT tutorial](https://fastapi.tiangolo.com/tutorial/security/oauth2-jwt/)
for the underlying authentication pattern and the
[PyJWT API reference](https://pyjwt.readthedocs.io/en/stable/api.html)
for fixed-algorithm, issuer and audience validation.

Run tests from the repository root:

```powershell
python -m pytest -q
```

The backend suite covers authentication, payload rejection, persistence,
overrides/feedback, version conflicts, duplicate confirmation/splitting,
analysis preview and health/statistics. The cross-language regression uses
a deterministic semantic test double and real grouping/area guards. It establishes
implementation behavior, not measured MiniLM accuracy. SQLite/TestClient
coverage of the portable SQLAlchemy schema does not replace running the
production database, Redis, model checkpoints, proxy, concurrency and load
tests in the intended deployment.

Use the filled [security checklist](security_checklist.md), actual
[dependency audit findings](security/dependency_audit.md),
[container instructions](containers.md) and [deployment guide](deployment.md)
before rollout. Unchecked provider controls remain operator work; passing local
tests does not establish that the deployed service has those controls.
