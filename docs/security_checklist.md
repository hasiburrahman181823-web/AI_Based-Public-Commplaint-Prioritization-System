# Pre-deployment security checklist

Verified **1 November 2026**. DONE means the stated local check passed, not a
security certification. NOT DONE requires operator action or further testing.
See [verification evidence](security_verification.md), [dependency findings](security/dependency_audit.md)
and the [deletion runbook](data_retention.md). Use synthetic data until release gates are completed.

## Application controls: done locally

- [x] **DONE — Environment configuration.** Database/Redis credentials, admin hash
  and signing key come from private environment settings. Credential examples
  are blank. Tests reject usable default credentials. Every `VITE_*` value is
  public; backend credentials are not passed to the frontend.
- [x] **DONE — Private-file exclusions.** An isolated Git ignore check verified
  `.env`, `.env.admin`, nested environment files, database/journal files, raw
  data, model/artifact directories and private Compose environment files.
  `.env.example` remains distributable. Historical tracked files are a separate
  check: the source checkout has no Git history to inspect.
- [x] **DONE — Startup key checks.** `BACKEND_JWT_SECRET` is required, must contain
  at least 32 characters and cannot equal known/default placeholders. A production
  factory check rejected missing, short and default keys before contacting live
  services. Configuration tests also reject wildcard CORS origins.
- [x] **DONE — Administrator provisioning.** `scripts/create_admin.py` prompts
  twice and hashes with Argon2id (64 MiB, time cost 3, parallelism 1). It writes
  a new ignored `.env.admin`, never prints credentials, and preserves existing
  files. Tests verify hashing, wrong-password rejection, prompts, no secret
  output and no overwrite. Legacy PBKDF2 deployments must regenerate their hash.
- [x] **DONE — Short JWT lifetime and authorization.** Default expiry is 30 minutes;
  maximum is 60. Tests reject unsigned, invalid, expired, wrong-role/issuer/audience
  and missing-claim tokens. Missing/invalid JWTs return 401 on admin reads,
  overrides, group actions and deletion. Frontend tokens remain in memory.
- [x] **DONE — SlowAPI limits.** Defaults per client IP: submission 5/minute,
  preview 10/minute, login 3/minute. Actual TestClient requests trigger 429 on
  all three POST endpoints. Storage-failure tests fail closed. The tests use
  memory storage; production Redis/proxy behavior remains a release gate.
- [x] **DONE — Validated input.** Submission and preview enforce 1,000 normalized
  characters, strip controls, reject blank input, excessive repetition and more
  than two URLs. Tests cover valid multilingual complaints, oversize input,
  controls, spam, timestamps, extra fields and the 32 KiB request boundary.
  Existing longer stored rows remain readable; their database constraint is
  preserved to avoid breaking historical records.
- [x] **DONE — Stored-text rendering.** React displays complaint, location,
  contact and explanation strings as text. A DOM regression test injects HTML
  and verifies no executable image/script/event-handler or JavaScript link is
  created. This is a rendering-path check, not a full browser penetration test.
- [x] **DONE — Explicit CORS.** `BACKEND_ALLOWED_ORIGINS` supplies exact origins.
  Tests verify an allowed preflight and unknown-origin preflight/POST rejection
  with 403, no allow-origin header and no database write. CORS does not replace
  authentication for non-browser callers.
- [x] **DONE — API headers and docs.** Success, validation and origin-error
  responses are tested for nosniff, frame denial, no-referrer, basic CSP and
  no-store. Production `/docs`, `/redoc` and `/openapi.json` return 404 in route
  tests. Development documentation stays available.
- [x] **DONE — Optional private contact.** No name is collected. Optional phone/email
  is stored in a separate private contact column. Tests verify contact never
  reaches ML input, public receipts/tracking or feedback export; frontend tests
  verify optionality and a separate payload field.
- [x] **DONE — Privacy notice and disclaimer.** Bilingual `/privacy` and public
  links are tested. The public form displays exactly: “This is an AI
  decision-support demo, not an official government system.” Missing operator
  details are visibly identified as demonstration configuration.
- [x] **DONE — Reviewed retention/deletion.** CLI and JWT-protected
  `POST /admin/data-deletion` preview by default. Applying requires explicit
  confirmation; the API requires strict boolean `apply: true`. Tests verify
  row deletion, review/audit/survivor-analysis redaction, cache purge and failure
  rollback. Closed-record eligibility normally uses 90 days since `updated_at`;
  age selection excludes open reports. No automatic schedule is installed.
- [x] **DONE — Admin audit events.** Integration tests cover priority/status
  changes, group confirmation/merge and split with authenticated actor, timestamp
  and changed values. Deletion adds a count-only event. Application error logs
  omit exception messages that could expose connection credentials.
- [x] **DONE — Dependency pins and audits.** Direct Python/npm dependencies are
  pinned; Linux runtime locks include hashes and CPU-only Torch. Before/final
  audit JSON is retained. Final scoped Python and npm audits report no known
  advisories; [the findings document](security/dependency_audit.md) gives counts,
  commands and scope limits. An advisory lookup is not proof of no vulnerabilities.
- [x] **DONE — Local verification.** Pytest, frontend tests, TypeScript/build and
  correctness lint passed; counts and tooling are in the verification record.
  CI includes lint, tests, audits and a Docker build. Hosted CI execution is unverified.

## Deployment release gates: not done here

- [ ] **NOT DONE — Set and rotate production secrets.** Set
  `BACKEND_ENVIRONMENT=production`, new random `BACKEND_JWT_SECRET`, unique admin
  identity/Argon2id hash, private database/Redis URLs and trusted model/policy paths.
  Keep `BACKEND_ALLOW_PUBLIC_READS=false`; require ready artifacts. Rotate any
  exposed keys/passwords. Signing-key rotation invalidates existing tokens.
  Restrict credential-file access, including inherited Windows permissions.
- [ ] **NOT DONE — Audit historical/public copies.** Inspect actual Git history,
  remotes, ZIPs, screenshots, build logs and images for exposed secrets. Ignore
  rules do not remove old commits. Rotate suspected compromised credentials.
- [ ] **NOT DONE — Restrict database/Redis dashboards.** Configure TLS/certificate
  verification, private network/IP restrictions, authenticated Redis and least-privilege
  runtime roles separate from migration credentials. Keep their ports private.
  Disable or secure Supabase Data API/anonymous grants before creating tables.
- [ ] **NOT DONE — Verify hosted HTTPS/CORS/proxy identity.** Set exact origins.
  Configure `FORWARDED_ALLOW_IPS` only for real trusted proxy IPs/CIDRs, never `*`.
  This is the launcher's operative setting; `BACKEND_TRUST_PROXY_HEADERS` alone
  does not configure Uvicorn peers. Test shared Redis limits through the real
  proxy, across workers and restarts, plus public unknown-origin rejection.
- [ ] **NOT DONE — Verify hosting headers/docs.** Deploy frontend headers, narrow
  CSP `connect-src` to the actual API, inspect hosted responses, and verify API
  docs are inaccessible publicly. Local tests do not establish provider behavior.
- [ ] **NOT DONE — Run real services/containers.** Docker is unavailable locally.
  No image/Compose execution, live PostgreSQL/Redis integration, production
  multiworker/load test or remote CI run was performed. Check migrations,
  permissions, readiness and dependency failures on the deployment.
- [ ] **NOT DONE — Validate actual ML artifacts/resources.** Accept only trusted
  joblib resources: loading them can execute code. Verify real model/version
  compatibility, calibration, memory limits and cold starts. Synthetic semantic
  tests do not establish MiniLM accuracy or fit on a small free instance.
- [ ] **NOT DONE — Publish an operator policy.** Set `VITE_PRIVACY_OPERATOR` and
  genuine `VITE_PRIVACY_CONTACT_URL`, rebuild, approve a feasible retention
  schedule and assign an owner/request channel. Review small-count/location
  aggregate disclosure before collecting personal data.
- [ ] **NOT DONE — Erase all authorised copies.** Schedule reviewed retention
  runs; verify ownership/holds; stop/restart all workers for maintenance. Handle
  exports, separate caches/replicas, backups, restored deletions and any model
  withdrawal/retraining separately. Live SQL deletion is not certified media erasure.
- [ ] **NOT DONE — Rehearse recovery/incidents.** Test backup restore and replay
  the deletion ledger before reopening. Restrict logs and set retention. Record
  release/image/model versions and approved domains. This checklist does not
  certify legal compliance, an uptime SLA or a completed penetration assessment.
