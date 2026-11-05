# Phase 7: bilingual public and administrator interface

The React/TypeScript frontend connects directly to the Phase 6 FastAPI service.
It provides complaint submission, private tracking by an unguessable tracking
ID, administrator review, duplicate-group decisions and operational charts.
Tailwind styles are compiled locally. The application does not require a
hosted UI service, external translation service or third-party chart service.

## Start the backend

Use Python 3.10+ and a virtual environment. From the repository root:

```powershell
python -m pip install -r requirements-backend.txt
# Fresh checkout only; preserve an existing private .env:
Copy-Item .env.example .env
python -m backend.scripts.create_admin --username admin
```

Copy the helper's generated assignments into the root private `.env`, then run:

```powershell
python -m alembic upgrade head
python -m uvicorn backend.app:create_app --factory --host 127.0.0.1 --port 8000 --no-proxy-headers
```

The latest migrations add optional private contact information, citizen profiles
and separate ownership links. Apply them to an existing database before starting
this version of the application. Existing complaints remain anonymous and are
not linked by matching contact details; earlier complaints, reviews and audit
records are retained.

The [backend guide](phase6_backend.md) explains authentication, model artifacts,
database settings, rate limits, feedback export and production requirements.
Development still supports rule-only analysis when trained artifacts are absent.
The frontend shows missing confidence and human-review requirements explicitly.

## Start the frontend

Use Node.js 22.12+ and npm. In a second terminal, from the repository root:

```powershell
Set-Location frontend
npm ci
# Fresh checkout only; preserve an existing frontend .env:
Copy-Item .env.example .env
npm run dev
```

Open the local address printed by Vite. Keep the backend terminal running.
The default frontend API base is `/api`; the development server forwards that
prefix to `http://127.0.0.1:8000` and removes the prefix before forwarding.
No production secrets belong in frontend environment variables: Vite embeds
`VITE_*` settings in the public JavaScript bundle.

`VITE_API_BASE_URL` configures the client URL. For the bundled development proxy
and a production reverse proxy, keep:

```dotenv
VITE_API_BASE_URL=/api
```

To call a backend hosted at a separate origin, set its full HTTPS base URL and
restart/rebuild the frontend. Configure `BACKEND_ALLOWED_ORIGINS` on the backend
with the exact frontend origin. Do not add `/api` to a direct FastAPI URL unless
the deployment actually mounts the API at that prefix. CORS origin configuration
does not provide authentication; administrator endpoints still require JWTs.

## Build and check

Run from `frontend/`:

```powershell
npm test
npm run build
npm run preview
```

The build checks TypeScript and creates `frontend/dist/`. The preview command
serves the compiled assets for local inspection; it is not a replacement for
the production reverse proxy. The backend tests run separately from the
repository root:

```powershell
python -m pytest -q
```

Compiled assets, dependency folders, private `.env` files, browser-test output,
databases and feedback exports are excluded from the source archive. The npm
lockfile is included so `npm ci` reproduces the declared dependency tree.

## Public workflow

1. Select English or বাংলা. The UI language choice is remembered locally.
2. Write the complaint and optionally supply an area and contact information.
   The live language badge uses a lightweight local heuristic for Bangla,
   English and Banglish. It is an estimate; mixed-script and short text can be
   ambiguous, and it is not a research language-classification result.
3. Submit once. A successful receipt appears after the backend confirms that
   the complaint was saved. It includes the tracking ID and a tracking link.
   A network/server error preserves the form and does not show a false receipt.
4. Keep the tracking ID and open the tracking page to check the current status.
   The public response contains only the tracking summary: it omits complaint
   wording, raw location, contact, internal analysis and other reports' IDs.

At `/account`, a citizen can register with an email and password or sign in.
Name and phone are optional profile details stored separately from complaint
text. If signed in when submitting, the report appears in `/account/complaints`
with its own text, location, optional contact and current status. If not signed
in, the report remains anonymous and is tracked by its receipt ID. Existing
anonymous reports are not automatically attached to a later account. Password
recovery, email verification and self-service account deletion are not available
in this demo; use a unique password and follow the privacy notice before entering
real profile information.

Anyone holding the tracking ID can view that limited summary; treat a copied
tracking link as something you are deliberately sharing. Public tracking does
not expose the administrator's complaint detail view. Contact is optional,
stored for authorized administrators, and excluded from inference and ordinary
feedback export. This phase does not send email/SMS or promise a callback.

Text and contact drafts remain in memory and are not placed in browser storage.
There is no offline queue that silently submits later. The UI reports network,
validation and rate-limit errors and provides retry controls. Avoid repeated
submissions after an uncertain network interruption: the backend does not
currently implement client idempotency keys.

## Administrator workflow

Sign in with the administrator account configured by the backend helper. The
JWT is held in application memory and sent only to the configured API; reloading
the page or closing the tab requires signing in again. Passwords and tokens are
not stored in local storage. Sign out when finished on a shared device.

The separate staff login remains at `/admin/login`. The dashboard includes:

- A priority queue with category, priority, area and status filters and pagination.
- A Needs review tab using the persisted `review_required` flag.
- Complaint details with calibrated classification confidence when available,
  rule contributions, model explanation terms and review reasons.
- Versioned status/priority/category/severity corrections with a required reason.
- Duplicate groups with the actual Bangla, English and Banglish reports shown
  together. Missing languages remain empty; the UI does not invent translations.
- Explicit confirmation, reviewed merge and proper-subset split actions.
- Category, area, priority-over-time and review-workload charts from the API,
  including accessible text/table values and empty-data states.

Fetch current details before a mutation. Every review sends the current
`expected_version` and a reason. HTTP 409 means that another review changed the
record: refresh, inspect the new information, and make the decision again.
The UI must not silently overwrite a newer decision. Merge/split choices are
human decisions and become feedback/audit evidence; they do not establish
measured duplicate-model accuracy. An automatic match cannot undo a reviewed
split.

The analytics endpoint is administrator-only. Its charts show current labels
and statuses. Priority-over-time counts current priority labels by each
complaint's original UTC creation date; it is not a log of historical priority
changes. Counts are operational observations, not estimates of unique people
affected. A duplicate group can contain multiple reports of one incident.
The recorded-group total also includes groups left empty by reviewed merges;
these records remain available to preserve audit references.

## Colour and motion

The interface uses the logo's green (`#076657`) for headings, ordinary text,
labels and controls. White and pale green surfaces keep that text readable.
Priority and review states retain explicit labels; chart series also use
patterns so their meaning does not depend on different colours.

Short entrance animations and hover/focus effects provide visual feedback.
The operating system's reduced-motion preference disables decorative movement.
Content and controls remain usable when those effects are disabled; no extra
animation library or remote asset is needed.

## Deploy the compiled frontend

Serve the contents of `frontend/dist/` as static files. Browser routes need an
HTML fallback to `index.html`; otherwise opening a tracking/admin route directly
may return the static server's 404. Proxy `/api/` to the existing FastAPI service
when using the default base URL. For example, an Nginx location configuration is:

```nginx
location /api/ {
    proxy_pass http://127.0.0.1:8000/;
    proxy_set_header Host $host;
    proxy_set_header X-Forwarded-For $remote_addr;
    proxy_set_header X-Forwarded-Proto $scheme;
}

location / {
    try_files $uri $uri/ /index.html;
}
```

Set the server's document root to the deployed `dist` directory and terminate
HTTPS at the proxy. The example replaces forwarded client-address headers.
When FastAPI is reachable only through this local trusted proxy, launch Uvicorn
with `--proxy-headers --forwarded-allow-ips=127.0.0.1` instead of the local
development command's `--no-proxy-headers`. Keep the backend inaccessible to
untrusted forwarding peers so rate limits use the intended client identity.

Use the Phase 6 production database, Redis and model-readiness settings for a
production backend. This source delivery does not create a hosted deployment.
Actual device/screen-reader testing, slow-network measurements, production
browser support, live PostgreSQL/Redis and real-model validation remain
deployment checks; see [the verification record](verification.md) for the
checks actually run in this workspace.
