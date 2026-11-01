"""Export the API specification without credentials, a database or ML startup.

The application factory is built with ephemeral random test credentials and an
in-memory engine, but its lifespan is never entered. No model is downloaded or
loaded, and no private .env file is read. The credentials are neither printed
nor included in the exported specification.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import secrets
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def export_openapi(output: str | Path = ROOT / "backend" / "openapi.json") -> Path:
    from backend.app import create_app
    from backend.config import Settings
    from backend.db import make_engine
    from backend.security import hash_password

    settings = Settings(
        environment="test", database_url="sqlite:///:memory:",
        jwt_secret=secrets.token_urlsafe(48),
        admin_password_hash=hash_password(secrets.token_urlsafe(40)),
        category_model=None, severity_model=None, review_policy=None,
        priority_ranker=None, duplicate_model=None, require_ml_ready=False,
    )
    engine = make_engine(settings.database_url)
    try:
        # openapi() inspects route definitions only. Do not enter app lifespan.
        specification = create_app(settings=settings, engine=engine).openapi()
    finally:
        engine.dispose()
    destination = Path(output).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(specification, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return destination


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "backend" / "openapi.json")
    args = parser.parse_args(argv)
    print("Exported OpenAPI specification: " + str(export_openapi(args.output)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
