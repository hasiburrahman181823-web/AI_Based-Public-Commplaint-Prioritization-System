"""Generate Argon2id .env admin credentials without plaintext password arguments."""

from __future__ import annotations

import argparse
import getpass
import os
from pathlib import Path
import re
import secrets

from backend.security import hash_password


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--username", default="admin")
    parser.add_argument("--password-env", help="Read the password from this environment variable instead of prompting")
    parser.add_argument("--output", type=Path, default=Path(".env.admin"),
                        help="New private credential file (default: .env.admin); never overwritten")
    args = parser.parse_args(argv)
    username = args.username.strip()
    if not re.fullmatch(r"[A-Za-z0-9_.@-]{1,128}", username):
        parser.error("username must use 1–128 letters, digits, '.', '_', '@' or '-'")
    if args.password_env:
        password = os.getenv(args.password_env)
        if password is None:
            parser.error("the named password environment variable is not set")
    else:
        password = getpass.getpass("Admin password (8–256 characters): ")
        confirmation = getpass.getpass("Confirm admin password: ")
        if password != confirmation:
            parser.error("passwords do not match")
    try:
        encoded = hash_password(password)
    except ValueError as exc:
        parser.error(str(exc))
    destination = args.output.expanduser().absolute()
    if not destination.name.startswith(".env.") or destination.name == ".env.example":
        parser.error("output must be a private .env.* file, never .env.example")
    content = ("# Private administrator credentials. Never commit or share this file.\n"
               f"BACKEND_ADMIN_USERNAME={username}\n"
               f"BACKEND_ADMIN_PASSWORD_HASH='{encoded}'\n"
               f"BACKEND_JWT_SECRET={secrets.token_urlsafe(48)}\n")
    try:
        # O_EXCL preserves existing credentials and refuses a symlink target.
        descriptor = os.open(destination, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(content)
    except OSError:
        parser.error("could not create private credential file; existing files are preserved")
    print("Created private administrator credential file. No credentials were printed.")
    print("Merge its assignments into your private .env or hosting secret settings; keep the file private.")
    print("On Windows, review the file's inherited access permissions before deployment.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
