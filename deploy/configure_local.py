"""Create private local Compose credentials; never print passwords or hashes."""
from __future__ import annotations
import argparse
import getpass
import os
from pathlib import Path
import re
import secrets


def configure(directory: Path, username: str, password: str) -> list[Path]:
    from backend.security import hash_password
    if not re.fullmatch(r"[A-Za-z0-9_.@-]{1,128}", username):
        raise ValueError("Use a valid administrator username")
    encoded = hash_password(password)
    directory = directory.resolve()
    directory.mkdir(parents=True, exist_ok=True)
    targets = [directory / f"local.{name}.env" for name in ("backend", "postgres", "redis")]
    if any(path.exists() for path in targets):
        raise ValueError("Local credential files already exist; edit them privately or choose a new directory")
    database_password, redis_password = secrets.token_urlsafe(36), secrets.token_urlsafe(36)
    payloads = [
        "\n".join(["BACKEND_ENVIRONMENT=development", "BACKEND_REQUIRE_ML_READY=false",
                    f"BACKEND_ADMIN_USERNAME={username}", f"BACKEND_ADMIN_PASSWORD_HASH={encoded}",
                    f"BACKEND_JWT_SECRET={secrets.token_urlsafe(48)}",
                    f"BACKEND_DATABASE_URL=postgresql+psycopg://banglapsg:{database_password}@db:5432/banglapsg",
                    f"BACKEND_RATE_LIMIT_STORAGE_URI=redis://:{redis_password}@redis:6379/0",
                    "BACKEND_ALLOW_PUBLIC_READS=false", "BACKEND_ALLOWED_ORIGINS=http://127.0.0.1:5173,http://localhost:5173",
                    "BACKEND_EMBEDDING_CACHE_DIR=/home/user/cache/backend", "PORT=8000", ""]),
        f"POSTGRES_DB=banglapsg\nPOSTGRES_USER=banglapsg\nPOSTGRES_PASSWORD={database_password}\n",
        f"REDIS_PASSWORD={redis_password}\n",
    ]
    created = []
    try:
        for path, payload in zip(targets, payloads):
            descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
            with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
                handle.write(payload)
            created.append(path)
    except OSError:
        # Undo only files created by this operation, never existing operator data.
        for path in created:
            path.unlink()
        raise
    return created


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--username", default="admin")
    args = parser.parse_args(argv)
    password = getpass.getpass("Local administrator password: ")
    if password != getpass.getpass("Confirm password: "):
        parser.error("Passwords do not match")
    try:
        created = configure(args.directory, args.username.strip(), password)
    except (ValueError, OSError, ImportError):
        parser.error("Could not create credentials. Install backend requirements, use a valid password, and keep existing files safe.")
    print(f"Created {len(created)} private environment files; no secrets printed. Run docker compose up --build --wait api")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
