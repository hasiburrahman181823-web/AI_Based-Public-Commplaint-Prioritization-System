# Deployment: local containers, a free research preview, and a strict service

Provider documentation checked **1 November 2026**. Plans, quotas and account
eligibility can change. This repository supplies deployment files and a guide;
it does not contain a running cloud deployment, account credentials, trained
research artifacts or a production availability guarantee.

## Choose the deployment you can support

| Component | Research-preview option | Material constraint |
| --- | --- | --- |
| Database | Neon Free | Currently 1 GB/project and 100 CU-hours/project/month; compute can suspend when quota is exhausted and sleeps after inactivity. [Neon Free allowances](https://github.com/neondatabase/website/blob/main/content/faqs/free-plan-limits-and-quotas.md). |
| Alternative database | Supabase Free | Currently two active free projects and a 500 MB database allowance per project. Low-activity projects can pause after seven days. [Pricing allowances](https://supabase.com/pricing), [billing](https://supabase.com/docs/guides/platform/billing-on-supabase), [pausing](https://supabase.com/docs/guides/platform/free-project-pausing). |
| Backend | Render Free, for a bounded demo | Current free web-service compute is 0.1 CPU/512 MB. Fit must be measured; a CPU PyTorch/MiniLM pipeline is not promised to fit. [Compute plans](https://render.com/docs/compute-plans). |
| Rate-limit store | Render Key Value in the backend's region | Use its Redis-protocol connection URL. Check current free-instance eligibility and limits in your account. [Key Value connections](https://render.com/docs/key-value). |
| Frontend | Vercel Hobby, if eligible | Hobby is for personal, noncommercial use; institutional/public operational use may require a different plan. [Hobby terms](https://vercel.com/docs/plans/hobby). |
| Alternative backend | Hugging Face Docker Space, conditionally | CPU Basic hardware has no hourly charge, but current documentation requires a paid account plan to create Docker/Gradio compute Spaces. Standard Postgres/Redis egress is another blocker below. [Spaces overview](https://huggingface.co/docs/hub/spaces-overview). |

The free Render service sleeps after 15 idle minutes and may take about a minute
to wake. Its disk is ephemeral, free-instance allowances are limited, and Render
expressly positions free instances for previews rather than production. Do not
use Render's expiring free Postgres as the only copy of collected grievances.
[Render free-service constraints](https://render.com/docs/free).

**A free static frontend plus a free database does not establish a free, always-on
multilingual ML service.** Use only synthetic examples until the operator has
adopted the [security checklist](security_checklist.md), retention policy and
privacy notice. Upgrade resources when actual memory, startup or reliability
requirements exceed a free allowance.

The strict backend checks are deliberate:

- `BACKEND_ENVIRONMENT=production` requires PostgreSQL, a shared Redis-protocol
  rate-limit store and `BACKEND_REQUIRE_ML_READY=true`.
- Ready category/severity classifiers, a duplicate bundle and a compatible
  validation-selected review policy must actually be present.
- Missing artifacts can run in explicitly degraded **development/demo** mode.
  That mode returns unavailable classifier confidence and human-review flags;
  it is not the proposed multilingual system operating with validated models.

## 1. Prepare the repository and artifacts

Run from the repository root. A fresh local checkout needs private credentials:

```powershell
Copy-Item .env.example .env
python scripts/create_admin.py --username admin
```

The helper writes its administrator/hash/signing-secret assignments to a new,
ignored `.env.admin` file and prints no credentials. Merge those assignments
privately into the root `.env` or the hosting secret settings. Existing output
files are never overwritten; use `--output .env.admin-new` for reviewed rotation.
On Windows, restrict inherited file access to the operator account. Preserve an
existing `.env` rather than replacing it.
The helper prompts for a password and confirmation and generates an Argon2id
hash. Older PBKDF2 values are no longer accepted: regenerate the hash before
starting this hardened version. There is no default password. Signing secrets
must contain at least 32 characters and must not be copied example/default
strings; the helper generates a random value.
Never paste these values into frontend configuration, screenshots, a public
Docker build argument, a Git commit or a Docker image layer.
Hosting secret fields take the raw value, not the whole `KEY=value` assignment.
Omit dotenv syntax quotes around the hash when copying it into such a field;
preserve every literal `$` character in the Argon2id value.

The default `runtime` Docker target installs pinned CPU-only neural dependencies;
`lightweight` omits them for rule/TF-IDF previews. The image never downloads a
model at startup. The source archive contains no trained artifacts.

For a strict deployment, finish the Phase 3–5 training/validation workflows first.
Train and verify trusted joblib bundles in an environment compatible with the
pinned runtime. Keep each bundle's full relative resource tree, including its
saved normalizer dictionary, encoder directory and policy fingerprints. A
joblib file can execute code when loaded: accept only your own reviewed builds.

Trusted deployment weights can be staged in ignored `deploy/artifacts/` before
building. Only those explicitly staged artifacts are copied into `/app/artifacts`.
Review the staging directory before every build. Do not stage complaints,
annotation databases, feedback exports, `.env` files or keys. An image containing
research weights/resources should use a private registry or protected source
workflow unless their public release has been reviewed and is permitted.

```powershell
docker build --platform linux/amd64 --target runtime -t banglapsg-api:cpu .
# A smaller dependency variant for an explicitly limited preview:
docker build --platform linux/amd64 --target lightweight -t banglapsg-api:lightweight .
```

Local Compose runs PostgreSQL and Redis, applies migrations through a one-shot
service, then starts the API. Follow the exact secret-file and Compose-version
instructions in [the container guide](containers.md). Database and cache volumes
are separate from the code; do not use a volume-deleting shutdown to reset a live
database. Keep the deployed version, artifact fingerprints and migration revision
in the operator's release record.

```powershell
python -m deploy.configure_local
docker compose up --build --wait api
```

The local helper provisions separate ignored environment files. The default
Compose image is the lighter rule/TF-IDF preview; set
`BACKEND_DOCKER_TARGET=runtime` in the process environment before Compose when
testing the full CPU image with your staged/mounted trusted artifacts.

## 2. Create hosted PostgreSQL

### Neon

1. Create a project in a region near the backend. Use a dedicated database/role
   for this application and record its owner privately.
2. In **Connect**, copy the provider-generated direct connection URI. Keep its
   TLS parameters and percent-encoding. The backend accepts `postgresql://` and
   `postgresql+psycopg://`; it normalizes the former to the installed psycopg
   driver.
3. For the initial one-worker service, a direct URL is the simplest choice.
   Neon also offers a `-pooler` endpoint using transaction pooling. Use the direct
   connection for migrations; do not invent a hostname by editing a pooled one.
4. Save the service URI as `BACKEND_DATABASE_URL` in the backend host's secret
   environment. Keep any migration-owner URI only in the operator's private
   environment.
5. Check quota and wake-up behavior before each collection session. A sleeping
   database and a sleeping backend are separate sources of startup delay.

[Neon connection instructions](https://neon.com/docs/connect/connect-from-any-app),
[direct/pooled connection guidance](https://github.com/neondatabase/website/blob/main/content/faqs/connect-application-using-connection-string.md).

For verified TLS in the Debian-based container, use `sslmode=verify-full` and
`sslrootcert=/etc/ssl/certs/ca-certificates.crt` in the database URI's query
parameters; preserve applicable channel-binding settings. The runtime includes
the public CA bundle. On an operator machine, use that machine's correct trusted
root path instead. Test with the actual driver and hostname; do not solve a
certificate error by disabling verification.
[Neon verified-TLS instructions](https://neon.com/docs/connect/connect-securely).

### Supabase alternative

1. Create a project and save the database password privately.
2. **Before migration, disable the Data API** in the Data API integration
   overview. This app uses direct SQL plus its own JWT-protected FastAPI routes;
   it does not use Supabase's browser SDK, REST or GraphQL data endpoints.
   Alembic-created tables do not acquire application-specific RLS policies.
   Existing Supabase defaults can otherwise grant Data API roles access to new
   `public` tables. Re-enable the Data API only after a separate grants/RLS review.
   [Supabase Data API security](https://supabase.com/docs/guides/api/securing-your-api).
3. Copy the exact connection string from **Connect**. Direct connections normally
   need IPv6. An IPv4-only persistent backend can use the shared **session** pooler
   on port 5432; copy its different username as well as its hostname. Use a direct
   connection from a compatible operator machine for migration/backup work.
4. Require encrypted transport. Supabase documents that `sslmode=require` encrypts
   but does not verify server identity. For verification, download its root
   certificate and configure `sslmode=verify-full` plus `sslrootcert` pointing to
   that certificate inside the actual runtime. Do not merely download the file
   without telling the driver to use it. Follow your chosen provider's equivalent
   verified-TLS setup.
5. Resume a paused project before opening collection. Do not treat a free service
   as an independent long-term backup.

[Supabase connections, networking and TLS](https://supabase.com/docs/guides/database/connecting-to-postgres).

## 3. Prepare a shared Redis-protocol store

Create a Render Key Value instance in the same region/workspace as the Render
backend, and copy its internal connection URL. Prefer authenticated internal
connections and keep external access disabled unless specifically needed. Other
providers can be used when they supply a reachable `redis://` or `rediss://` URL.
External connections should use authenticated TLS and restricted access.
[Render Key Value access controls](https://render.com/docs/key-value).

Internal authentication is opt-in on Render: enable it for the new instance,
then use its authenticated internal URL before connecting the backend. Follow
the provider's migration procedure if an existing service already uses an
unauthenticated URL. Free Key Value is memory-only and can restart or become
temporarily unavailable during maintenance. **Inference:** a restart clears
the application's rate-limit counters; do not claim that those counters survive
restarts. A persistent paid store is needed if that operational property is
required. [Free Key Value limits](https://render.com/docs/free#free-key-value),
[internal authentication](https://render.com/docs/key-value#requiring-auth-for-internal-connections).

Set `BACKEND_RATE_LIMIT_STORAGE_URI` to that URL. An Upstash-style HTTPS REST URL
and REST token are **not** substitutes for the Redis protocol used by SlowAPI in
this implementation. Never put the Redis URI in a `VITE_*` variable. In-memory
rate limits are for a single local/demo process and do not satisfy production
configuration. A store outage fails closed; investigate a 503 rather than
removing the limiter.

## 4. Configure backend environment and migrate

Use the hosting dashboard's private environment/secret fields. These names match
the actual application; the value column describes what the operator supplies.

| Variable | Strict deployment value |
| --- | --- |
| `BACKEND_ENVIRONMENT` | `production` |
| `BACKEND_REQUIRE_ML_READY` | `true` |
| `BACKEND_DATABASE_URL` | The verified hosted PostgreSQL URI |
| `BACKEND_RATE_LIMIT_STORAGE_URI` | The authenticated Redis-protocol URI |
| `BACKEND_ADMIN_USERNAME` | The administrator name generated locally |
| `BACKEND_ADMIN_PASSWORD_HASH` | The helper's complete salted Argon2id hash |
| `BACKEND_JWT_SECRET` | The helper's random signing secret |
| `BACKEND_JWT_EXPIRE_MINUTES` | A deliberate short session lifetime; default `30` |
| `BACKEND_ALLOW_PUBLIC_READS` | `false` |
| `BACKEND_ALLOWED_ORIGINS` | Exact frontend origins, comma-separated, without paths |
| `BACKEND_CATEGORY_MODEL` | `/app/artifacts/category/char_svm.joblib`, or your verified bundle path |
| `BACKEND_SEVERITY_MODEL` | `/app/artifacts/severity/char_svm.joblib`, or your verified bundle path |
| `BACKEND_REVIEW_POLICY` | `/app/artifacts/review_policy.json` |
| `BACKEND_DUPLICATE_MODEL` | `/app/artifacts/duplicates/P2.joblib`, or the selected validated bundle |
| `BACKEND_PRIORITY_RANKER` | Your expert-trained ranker path; blank when the explicit rules fallback is intended |
| `BACKEND_EMBEDDING_CACHE_DIR` | `/home/user/cache/backend` |
| `HF_HUB_OFFLINE`, `TRANSFORMERS_OFFLINE` | `1` |
| `RUNTIME_PROFILE` | `cpu` for a source-built full CPU image; a build-time profile, not a credential |

Use your actual selected model names. A calibration/review policy bound to
different artifacts or preprocessing resources is not interchangeable. Keep
offline encoder files available at the paths referenced by those bundles.

For a **synthetic rule-only preview**, use `BACKEND_ENVIRONMENT=development`,
`BACKEND_REQUIRE_ML_READY=false`, and blank category/severity/duplicate/review-policy/
ranker paths. Keep hosted Postgres, private admin credentials and the Redis store.
Label it a demo and collect synthetic examples only; do not silently present
degraded outputs as evaluated model predictions. Do not configure the privacy
operator/contact fields for public collection until the policy is actually adopted.

Migrate from a trusted operator shell before rolling out the service. The
private operator environment supplies the direct migration PostgreSQL URL:

```powershell
python -m alembic upgrade head
python -m alembic current
```

The command reads `BACKEND_DATABASE_URL`; temporarily select the direct
migration-owner URI in that private environment, then restore the service URI
for serving. Do not put credentials in command arguments or CI logs. Use a
separate rollout step instead of running Alembic in every web worker. Back up
first, review migration output, and keep a tested restore plan. A container
equivalent is:

```powershell
docker run --rm --env-file .env.docker banglapsg-api:cpu python -m alembic upgrade head
```

The ignored `.env.docker` uses literal, unquoted values as explained in the
[container guide](containers.md); Python dotenv syntax quotes must not become
part of the Docker environment value. It must point to the intended hosted
database; never reuse an unexamined local development file against production. Render Free does not
provide a one-off-job/shell workflow, so perform the migration from the operator
machine/container rather than assuming such a dashboard feature exists.

### Reverse proxy and rate-limit identity

The default container rejects forwarded-header trust. This is safe against
arbitrary `X-Forwarded-For`, but behind a proxy it can rate-limit all visitors by
the proxy's address. Supply `FORWARDED_ALLOW_IPS` only after verifying the host's
actual trusted proxy IP/CIDR ranges and that direct untrusted access is blocked.
The entrypoint then enables Uvicorn proxy handling for that allowlist. Do not use
`*`, guess a range, or assume `BACKEND_TRUST_PROXY_HEADERS` configures Uvicorn.
If the host does not provide a verifiable trust boundary, retain the safe default
and treat its aggregate limit as a known preview limitation; do not claim
per-citizen limits. Test forged forwarding headers and concurrent visitors before
public collection.

## 5. Render backend walkthrough

1. Push reviewed source to your own Git repository, or build and publish the
   reviewed image to a private registry. Keep data, exports and secrets out.
2. In Render, create a **Web Service** and select **Docker**. For source builds,
   use the repository root as context and `Dockerfile` as its file. The source
   package itself has no trained weights. To include reviewed weights, use the
   explicit ignored staging directory and a private image workflow; Git-ignored
   local files will not appear in a remote source checkout.
3. The default image target supports CPU neural inference. For a rule-only source
   build, set the non-secret `RUNTIME_PROFILE=runtime-lite` in Render's environment;
   the Dockerfile uses that build argument to select its final profile. Render
   translates environment values to Docker build arguments, so the Dockerfile
   deliberately consumes only this non-secret selector and never password/key
   build arguments. Alternatively, deploy your prebuilt `lightweight` image.
4. Choose Free only for a bounded demonstration. If actual startup/load exceeds
   its memory or CPU allowance, choose sufficient compute instead of increasing
   worker count or hiding readiness errors. One worker avoids duplicate model
   copies; it is not a throughput guarantee.
5. Enter the private backend environment above. Keep the supplied `PORT` value;
   the entrypoint listens on `0.0.0.0` at that port. Leave the Docker command
   override empty so the validated entrypoint runs.
6. Apply the migration through the independent operator step, then deploy. Set
   the HTTP health-check path to `/health`.
7. Inspect startup logs without printing environment values or complaint bodies.
   Verify the public HTTPS service URL and its readiness response before connecting
   the frontend.

[Render Docker setup](https://render.com/docs/docker),
[HTTP health-check behavior](https://render.com/docs/health-checks).

Use Neon/Supabase for complaint persistence. The embedding cache may be rebuilt
after an ephemeral restart; raw complaint history must not depend on container
disk. Provision backup/restore separately. Free-instance sleep, quota suspension,
restores and cold starts remain operational limitations.

## 6. Hugging Face Docker Spaces alternative: check blockers first

This is a platform-specific configuration walkthrough, **not a claim that the
unchanged stack is freely deployable there**. Current official docs distinguish
zero-hourly-cost CPU Basic hardware from the paid account plan needed to create
a Docker compute Space. Public Spaces expose their source; protected/private
visibility and app access have different audiences. Verify the chosen account
and visibility before publishing code or any trained resources.
[Spaces account and visibility rules](https://huggingface.co/docs/hub/spaces-overview).

The documented outbound network ports are 80, 443 and 8080. Ordinary hosted
Postgres and Redis URLs typically use other TCP ports. **Inference:** this
repository's psycopg/Redis clients cannot be assumed to connect to Neon/Supabase
and Redis from a standard Space. An HTTPS REST database URL does not change
those clients into HTTP clients. Do not disable PostgreSQL/Redis requirements or
move real complaint persistence to ephemeral SQLite to make a demo start.
Use Render/a host supporting the required connections, or separately design,
secure and test an allowed-port gateway before considering Spaces.
[Spaces networking restriction](https://huggingface.co/docs/hub/spaces-overview#networking).

If your account and verified networking solution support this deployment:

1. Create a Docker Space with the intended reviewed audience. Copy only the
   reviewed deployable source/image workflow; exclude all operational data.
2. In the Space's root `README.md` YAML metadata, select Docker and the port:

   ```yaml
   ---
   title: Nagarik research service
   sdk: docker
   app_port: 7860
   ---
   ```

3. Set runtime `PORT=7860`. The application runs as the image's non-root user;
   use its writable cache directory rather than assuming root-owned paths work.
4. Enter database/Redis/admin values using **Secrets**, not public Variables.
   Set non-secret model/cache settings as Variables. Do not put a private model
   access token in `ARG` or `ENV` build instructions. Bundle reviewed offline
   resources in a protected/private artifact workflow instead.
5. Migrate independently, deploy, and test `/health`, admin access and persistence
   through a restart. Sleeping hardware is not an always-on service. No callback,
   emergency response or uptime promise follows from this deployment.

[Docker Space port, permissions and secret guidance](https://huggingface.co/docs/hub/spaces-sdks-docker),
[hardware lifecycle](https://huggingface.co/docs/hub/spaces-gpus).

## 7. Vercel frontend walkthrough

1. Create a Vercel project from the reviewed repository. Set **Root Directory** to
   `frontend`, framework to Vite, install command `npm ci`, build command
   `npm run build`, and output directory `dist`.
2. Set `VITE_API_BASE_URL` to the backend's actual HTTPS origin, without a trailing
   `/api` unless your backend proxy really mounts that prefix. The hosted
   frontend does not have Vite's development `/api` proxy.
3. Before real collection, set `VITE_PRIVACY_OPERATOR` to the actual responsible
   organization/person and `VITE_PRIVACY_CONTACT_URL` to its adopted HTTPS privacy
   request page or `mailto:` contact. Review `/privacy` and the retention policy.
   These public fields are not secrets; never use them for database passwords or
   private contact submissions.
4. Deploy, then copy the actual frontend origin into the backend's
   `BACKEND_ALLOWED_ORIGINS` and restart/redeploy the backend. Add a deliberate
   staging origin separately if needed. Do not allow every `*.vercel.app` domain.
5. The included `frontend/vercel.json` rewrites only known browser routes to
   `index.html`: tracking, administration and privacy. Assets and unknown API
   paths are not rewritten into HTML. Verify a direct tracking URL, `/admin/login`,
   `/privacy`, the favicon and a hashed JavaScript asset.
6. Rebuild/redeploy after changing `VITE_*` values: Vite embeds them in the public
   build. `DEV_API_TARGET` is for local development and does not redirect a hosted
   build's API calls.

[Vite deployment and SPA routes on Vercel](https://vercel.com/docs/frameworks/frontend/vite).

Do not place the CPU backend in a frontend serverless function. The chosen
service architecture retains a process-wide model/index and requires database
transaction coordination. The frontend uses its memory-only admin bearer token;
it does not send browser cookies to the API. Treat a privacy/tracking ID as a
capability and avoid sharing screenshots containing real IDs.

## 8. Health, CORS and cold-start checks

Read the actual `/health` body, not only the provider's green deployment badge:

| Observation | Interpretation/action |
| --- | --- |
| HTTP 200, `status=ready` | Database/store checks pass and the configured ML service reports ready. Still perform application smoke checks. |
| HTTP 200, `status=degraded` | Allowed demo configuration; trained inference is incomplete. Do not call it a ready production ML system. |
| HTTP 503 or failed startup | Required database, rate store, schema or artifacts are unavailable. Fix the cause. |
| HTML instead of JSON | A proxy/host loading page or incorrect base URL; not a confirmed submission. |
| CORS error in browser but direct health works | Check the exact frontend origin, API URL and preflight response; CORS is separate from authentication. |

Before a demonstration, open the backend `/health` and wait for its intended
state, then open the frontend. Backend and database wake-up times add to model
startup time. Keep all required encoder resources in the release so cold starts
do not download weights. Respect provider sleep/usage terms; do not promise that
synthetic keep-alive traffic gives a free uptime guarantee.

The frontend's network timeout is currently 60 seconds. A cold service can exceed
it. Readiness reads may be retried with a bounded delay by the operator, but a
complaint POST must not be retried automatically: a lost response can leave a
saved report without a received tracking receipt. This implementation has no
client idempotency-key contract. Follow the visible uncertain-receipt error and
investigate before asking volunteers to resubmit.

From the deployed frontend, verify with synthetic examples:

1. `/privacy` names the actual operator and working privacy contact; demo-only
   warnings remain when public collection is not configured.
2. A complaint is saved once, returns its UUID receipt, and tracking shows only
   the limited summary. Optional contact never appears in public tracking.
3. Unauthenticated admin reads/updates return 401; admin sign-in and a versioned
   review succeed; a stale version returns 409.
4. Disallowed origins do not receive an allow-origin header. An allowed origin's
   preflight supports JSON POST/PATCH and the Authorization header.
5. Input bounds/rate limits remain effective, including behind the real proxy.
6. A restart retains hosted complaint records; it may legitimately rebuild the
   cache. Verify backup restore, secret rotation and database-owner access too.

Record the deployment date, exact provider plans/limits, container digest,
artifact/resource hashes, migration revision, observed health state and smoke
results. Do not copy credentials, raw complaints or volunteered contacts into
the public thesis repository or demo GIF.

## Rollout boundaries

Cloud accounts, production network paths, quotas, actual model memory/latency,
backup recovery and a hosted end-to-end deployment have not been verified by
the source-code delivery alone. Review [the verification record](verification.md)
for executed local checks. Keep a research preview visibly distinct from an
operational grievance service, and move to suitable paid/managed infrastructure
when real users depend on availability or privacy controls.
