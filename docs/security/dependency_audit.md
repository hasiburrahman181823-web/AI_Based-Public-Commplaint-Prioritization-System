# Dependency audit record

Checked on **2026-11-01**. These are actual advisory lookups, not an assurance
that the application, packages or deployment have no vulnerabilities.

## Python

The final runs used **pip-audit 2.10.1**, its PyPI vulnerability service, no
ignored advisory IDs and no automatic fixes.

| Scope | Releases checked | Affected releases | Advisory records | Skipped releases | Exit |
| --- | ---: | ---: | ---: | ---: | ---: |
| Initial runtime candidate | 66 | 2 | 24 | 0 | Not recorded here |
| Final CPU inference runtime | 71 | 0 | 0 | 0 | 0 |
| Final direct research/backend/CI/neural declarations | 32 | 0 | 0 | 0 | 0 |
| Final combined runtime/research/CI metadata graph | 108 | 0 | 0 | 0 | 0 |

The initial findings affected **PyJWT 2.13.0 (15 advisory records)** and
**Transformers 4.57.6 (9 advisory records)**. The final pins are PyJWT 2.15.1
and Transformers 5.22.1. The raw initial findings remain in
[pip-audit-runtime-before.json](pip-audit-runtime-before.json). The actual final
tool output was `No known vulnerabilities found`; raw machine-readable results
are [pip-audit-runtime.json](pip-audit-runtime.json),
[pip-audit-direct.json](pip-audit-direct.json), and
[pip-audit-research.json](pip-audit-research.json).

All direct Python requirement declarations are now exact versions. The two
deployment lockfiles additionally pin every selected distribution and SHA-256
hash for CPython 3.12/Linux x86_64. Regenerating them from the saved resolution
reports produced byte-identical files: **45 core distributions** and **71 CPU
inference distributions**. The generator checked Linux dependency markers,
version constraints, psycopg's binary extra and absence of NVIDIA/Triton
dependencies.

`requirements-runtime-audit.txt` lists every resolved CPU runtime release. It
maps Torch's `+cpu` build suffix to the upstream `2.15.0` release for advisory
lookup. This file is **not an installation input**; install the CPU hash lock
to retain the official CPU wheel. `--no-deps --disable-pip` performs lookup of
the supplied exact versions without downloading or installing neural wheels.
It does not silently stand in for dependency resolution: the complete runtime
closure was checked separately. The direct-declaration run explicitly supplied
`requirements.txt`, because nested `-r` inputs are not expanded by this lookup
mode.

Repeat from the repository root in an environment containing pip-audit:

```powershell
python -m pip_audit --no-deps --disable-pip -r requirements-runtime-audit.txt -f json -o docs/security/pip-audit-runtime.json --progress-spinner off
python -m pip_audit --no-deps --disable-pip -r requirements.txt -r requirements-backend.txt -r requirements-ci.txt -r requirements-embeddings.txt -r requirements-xlmr.txt -f json -o docs/security/pip-audit-direct.json --progress-spinner off
python -m pip_audit --no-deps --disable-pip -r requirements-research-audit.txt -f json -o docs/security/pip-audit-research.json --progress-spinner off
```

The final combined graph includes the serving, research and CI transitives,
plus optional SentencePiece. A binary-only metadata resolution of
`requirements-backend.txt` and `requirements-ci.txt` supplied the research/CI
packages. Its overlapping runtime versions were replaced with the exact CPU
runtime resolution entries. The lock generator then checked every Linux
dependency constraint in the combined graph; every final direct requirement
was also checked against the selected version. SentencePiece's published
metadata was checked separately: its additional dependencies require optional
extras that this project does not request. All 108 selected releases appear
in `requirements-research-audit.txt` and the corresponding raw JSON lookup.
That audit input freezes this checked graph for advisory lookup; it is not an
installation lockfile and does not establish compatibility on other platforms.

The research direct pins match the installed local packages used for the test
run. Optional neural packages were resolved/audited, not loaded with real
trained artifacts. Older serialized models and Transformers training scripts
still need compatibility checks against their actual artifacts/data. No
pretrained model was downloaded for these audits. Actual Windows/Python 3.14
test results and Linux/Python 3.12 metadata validation are separate checks;
the complete Linux research/CI environment was not installed or executed here.

## JavaScript and deployment limits

The whole-project npm audit initially reported **3 affected packages: 2
critical and 1 moderate**, in the Vitest development toolchain (`vitest`,
`tinypool`, `@vitest/mocker`). Vitest was updated from 3.2.7 to the exact
5.0.5 pin and the lockfile was regenerated. The final audit report lists **0
vulnerabilities**, including development dependencies, across its reported
242 dependencies. See [npm-audit-before.json](npm-audit-before.json) and
[npm-audit-after.json](npm-audit-after.json) for the actual results, and the
[security checklist](../security_checklist.md) for test/build verification.

Repeat the npm audit from `frontend/`:

```powershell
npm ci --ignore-scripts
npm audit --json
```

The recorded scan includes development dependencies. Do not present a clean
production-only scan as a clean whole-project scan. An npm audit finding is a
dependency/advisory classification; its severity does not by itself establish
that this public website exposes the affected development tool.

No Docker engine was available locally, so container image construction, OS
package scanning, container runtime imports and hosted deployment behavior
were **not tested**. These Python lookups do not scan a container base image,
an operating system, bundled model contents, JavaScript bundles or hosting
configuration. Rerun both ecosystems' audits before each release; new advisory
records can change these results without a source-code change.
