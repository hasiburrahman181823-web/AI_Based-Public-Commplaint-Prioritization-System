"""Validated single-worker launcher; explicit operator commands pass through."""
from __future__ import annotations
import ipaddress
import os
import sys


def server_command(environment=None):
    environment = os.environ if environment is None else environment
    raw = environment.get("PORT", "8000")
    if not isinstance(raw, str) or not raw.isascii() or not raw.isdecimal() or not 1 <= int(raw) <= 65535:
        raise ValueError("PORT must be an integer from 1 to 65535")
    command = [sys.executable, "-m", "uvicorn", "backend.app:create_app", "--factory",
               "--host", "0.0.0.0", "--port", str(int(raw)), "--workers", "1", "--no-access-log"]
    trusted = environment.get("FORWARDED_ALLOW_IPS", "").strip()
    if trusted:
        peers = [item.strip() for item in trusted.split(",")]
        if not all(peers):
            raise ValueError("FORWARDED_ALLOW_IPS requires explicit IP addresses/CIDRs")
        for peer in peers:
            ipaddress.ip_network(peer, strict=False)
        command += ["--proxy-headers", "--forwarded-allow-ips", ",".join(peers)]
    else:
        command.append("--no-proxy-headers")
    return command


def main(argv=None):
    arguments = list(sys.argv[1:] if argv is None else argv)
    try:
        command = arguments or server_command()
    except ValueError:
        print("Invalid server port or trusted proxy allowlist; see docs/containers.md", file=sys.stderr)
        return 2
    os.execvp(command[0], command)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
