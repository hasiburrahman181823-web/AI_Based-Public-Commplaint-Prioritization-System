# Containers, pinned dependencies and local services

The default Docker image is a CPU-only neural inference image. A smaller
`runtime-lite` target excludes Torch, Transformers and sentence-transformers
for intentional rule/TF-IDF deployments. Both omit training-only Parquet,
Excel, plotting, frontend and test dependencies. CPU Torch remains substantial;
no exact image size or free-host memory fit is claimed without an actual build.

This workspace has no Docker executable, so the image and Compose stack were
not run locally. The lockfiles were resolved for CPython 3.12/Linux x86_64,
Linux dependency markers were checked, and the helper tests/lint were run.
The CI includes an actual full-CPU image build/import check for a Docker runner.

## Local Docker Compose

Install Docker with Compose **2.30.0 or newer**, Python, and the backend
requirements. This minimum Compose version supports `env_file: format: raw`,
which preserves the `$` characters in Argon2 hashes. Run from the repository:

```powershell
python -m pip install -r requirements-backend.txt
python -m deploy.configure_local
docker compose version
docker compose up --build --wait api
docker compose logs -f api
```

The helper prompts twice, calls the same Argon2id password hasher as the API,
and creates `deploy/local.backend.env`, `deploy/local.postgres.env` and
`deploy/local.redis.env`. Each service receives only its own environment file.
Random database/Redis credentials and the JWT signing key are generated locally;
no password, hash or signing secret is printed. Existing files are never
overwritten. The files are private operational state and must not be committed,
attached to bug reports, or included in a source archive. Windows file ACLs and
container-host access still need the operator's protection.

Compose starts PostgreSQL and authenticated Redis on its private network, waits
for their health checks, runs one migration container, then starts the API at
`http://127.0.0.1:8000`. Database and Redis ports are not published. The API is
non-root with a read-only root filesystem, a cache volume and dropped Linux
capabilities. The Redis password is local-development environment configuration;
Docker administrators can inspect environment/process configuration. Use managed
secret/TLS controls for a real production service.

The default Compose target is `runtime-lite`. Missing artifacts use the honest
development review fallback. For the full CPU image:

```powershell
$env:BACKEND_DOCKER_TARGET="runtime"
docker compose up --build --wait api
```

Compose mounts your existing `./artifacts/` read-only and persists PostgreSQL,
Redis and the embedding cache in named volumes. Do not delete volumes as a
routine restart. Stop services while retaining data with:

```powershell
docker compose down
```

Keep the Vite frontend in its separate terminal, using the Phase 7 instructions.

## Build one image

The Python base image, PostgreSQL and Redis use exact release tags plus verified
registry digests. Python packages have exact versions and distribution hashes.
The supported build platform is Linux x86_64; other platforms require their own
CPU wheel locks. On an ARM laptop, use Docker's explicit amd64 emulation and
expect different performance.

```powershell
docker build --platform linux/amd64 --target runtime -t banglapsg-api:cpu .
docker build --platform linux/amd64 --target runtime-lite -t banglapsg-api:lite .
```

Hosts exposing Docker build arguments can build the default final stage with
`RUNTIME_PROFILE=runtime-lite`; the default profile is `cpu`. `lightweight` is
also an alias for `runtime-lite`. Never call a lite deployment MiniLM-capable.

The Docker context is an allowlist. Datasets, existing root artifacts, research
results, private environment files, frontend dependencies and test files do not
enter the image. To intentionally bake trusted models into a **private image**,
copy the complete reviewed artifact/resource tree into `deploy/artifacts/`
before building. Its contents are gitignored and excluded from source ZIPs;
only its `.gitkeep` is provided. For example, the staged tree may contain
`category/`, `severity/`, `duplicates/`, normalization snapshots, encoder files,
and `review_policy.json` exactly as referenced by the saved bundles. Preserve
relative paths and fingerprints. Do not include raw datasets, `.env` files,
credentials or operator exports. Joblib can execute code: load only trusted,
self-created, reviewed artifacts, never user uploads.

No pretrained model is downloaded during build/startup. Offline Hugging Face
flags default to enabled. Without real artifacts strict production startup
must fail; do not turn off readiness and then present a rule-only demo as a
validated production model. Artifact dependency versions must match training;
retrain or validate compatibility rather than assuming every older joblib
will load under this pinned runtime.

## Port, proxy, migration and readiness

`PORT` defaults to `8000`; set it to the hosting provider's assigned port (e.g.
`7860` for a configured Space). The launcher binds `0.0.0.0`, validates the port
and uses one worker so each worker does not duplicate model RAM. It disables
access logging of URLs and disables forwarded-header trust by default.

Only after verifying that your ingress overwrites forwarding headers should
you set `FORWARDED_ALLOW_IPS` to explicit comma-separated IP addresses/CIDRs.
Wildcards and hostnames are rejected. Without a verified allowlist, rate limits
may aggregate requests under the proxy's IP; this is safer than accepting
arbitrary client-spoofed identities. `BACKEND_TRUST_PROXY_HEADERS` alone does
not configure Uvicorn. Test client identity through the real production proxy.

Migrations remain an explicit release step. The entrypoint forwards an operator
command unchanged, allowing:

```powershell
docker run --rm --env-file .env.docker banglapsg-api:cpu python -m alembic upgrade head
docker run --rm --env-file .env.docker -p 127.0.0.1:8000:8000 banglapsg-api:cpu
```

Prepare the ignored `.env.docker` privately using Docker's literal `KEY=value`
format. Do not pass the Python `.env`/`.env.admin` unchanged when values have
surrounding quotes: Docker keeps those quote characters, so a quoted Argon2
hash becomes invalid. Remove only syntax quotes when copying each reviewed
value; preserve its literal `$` characters. Do not use `export`, substitutions,
inline comments, or trailing spaces. Docker's [environment-file parser](https://github.com/docker/cli/blob/master/pkg/kvfile/kvfile.go)
documents that values are used literally. Compose's generated raw environment
files already use the correct unquoted values; no conversion is needed there.

That `.env.docker` must contain external database/Redis URLs reachable from this
container. URLs containing Compose service names work only on the Compose
network, not from an unrelated `docker run`. Back up and test migrations against
the intended production database before release. Serve PostgreSQL with TLS and
proper certificate verification; `ca-certificates` is installed in the image.

`GET /health` drives the dependency-free Docker health check. Development may
return HTTP 200 with `status=degraded`; production requires ready artifacts and
returns 503 on unavailable dependencies. The container health check has a
180-second startup grace period, not a promise that every hosted platform will
allow that duration. Measure cold startup with the actual corpus/model and
adjust the provider's startup limits where supported. Readiness does not create
an SLA or warm a sleeping free service.

Writable state belongs in `/home/user/cache` and `/home/user/data`; code and
artifact files are read-only in local Compose. Hosted ephemeral caches may
disappear on sleep/redeploy. The database is the source of truth. Never use an
ephemeral SQLite file as hosted complaint storage.

## Locks, audits and CI

`requirements-runtime.in` declares direct serving dependencies;
`requirements-runtime.lock` includes every resolved version/hash. The embeddings
input and lock add a direct official CPU Torch wheel and its dependencies.
The lock generator checks Linux markers, optional psycopg binary dependencies,
all version constraints and absence of NVIDIA/Triton packages. It deliberately
caught a Linux-only `hf-xet` dependency omitted by a Windows-host resolution.

Regenerate on a Linux CPython 3.12 environment, then check and audit the diff:

```bash
python -m pip install --dry-run --ignore-installed --only-binary=:all: --report core-resolution.json -r requirements-runtime.in
python -m pip install --dry-run --ignore-installed --only-binary=:all: --report cpu-resolution.json -r requirements-runtime-embeddings.in
python -m deploy.lock_runtime --core-report core-resolution.json --cpu-report cpu-resolution.json
python -m pip_audit --no-deps --disable-pip -r requirements-runtime-audit.txt
```

The advisory lookup file is **not an installation input**: it maps Torch's
`+cpu` build suffix to its upstream release version for database lookup. Install
only the CPU hash lock to avoid accidentally selecting a CUDA wheel. An audit
is a time-specific advisory lookup, not proof that a package or application has
no vulnerabilities. Review new findings, update pins/hashes and rerun tests;
CI does not ignore audit failures.

CI uses commit-SHA-pinned official GitHub actions, read-only repository token
permissions, correctness lint, Python tests, dependency audit, frontend tests/
type checking/build, and a CPU-image build/import check. It requires no real
database credentials, pretrained download, model weights or production secrets.
