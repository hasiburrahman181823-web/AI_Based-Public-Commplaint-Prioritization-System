"""Small dependency-free Docker readiness check; never print response secrets."""
import json
import os
import urllib.error
import urllib.request


def main():
    raw = os.getenv("PORT", "8000")
    if not raw.isascii() or not raw.isdecimal() or not 1 <= int(raw) <= 65535:
        return 1
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{int(raw)}/health", timeout=5) as response:
            body = json.loads(response.read(16384))
            return 0 if response.status == 200 and body.get("status") in {"ready", "degraded"} else 1
    except (OSError, ValueError, urllib.error.HTTPError):
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
