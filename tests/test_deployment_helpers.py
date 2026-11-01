"""Deployment argument, privacy, and cross-platform lock regressions."""
import io
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from deploy.configure_local import configure
from deploy.entrypoint import main, server_command
from deploy import healthcheck
from deploy.lock_runtime import lock_report


@pytest.mark.parametrize("port", ["0", "65536", "abc", "８０００", "8000 --reload", None])
def test_launcher_rejects_invalid_or_injected_ports(port):
    with pytest.raises(ValueError):
        server_command({"PORT": port})


def test_launcher_port_and_trusted_proxy_allowlist():
    default = server_command({})
    assert default[default.index("--port") + 1] == "8000"
    assert "--no-proxy-headers" in default and "--no-access-log" in default
    hosted = server_command({"PORT": "7860", "FORWARDED_ALLOW_IPS": "127.0.0.1,10.20.0.0/16"})
    assert hosted[hosted.index("--port") + 1] == "7860"
    assert "--proxy-headers" in hosted and "--no-proxy-headers" not in hosted
    for value in ("*", "proxy.example.org", "127.0.0.1,", "127.0.0.1 --reload"):
        with pytest.raises(ValueError):
            server_command({"FORWARDED_ALLOW_IPS": value})


def test_migration_command_passes_through_without_launching_uvicorn(monkeypatch):
    received = []
    monkeypatch.setattr("deploy.entrypoint.os.execvp", lambda executable, arguments: received.append((executable, arguments)))
    assert main(["python", "-m", "alembic", "upgrade", "head"]) == 0
    assert received == [("python", ["python", "-m", "alembic", "upgrade", "head"])]


@pytest.mark.parametrize("status", ["ready", "degraded", "unavailable"])
def test_healthcheck_accepts_only_api_readiness_contract(monkeypatch, status):
    def response(*args, **kwargs):
        handle = io.BytesIO(json.dumps({"status": status}).encode())
        handle.status = 200
        return handle
    monkeypatch.setattr(healthcheck.urllib.request, "urlopen", response)
    assert healthcheck.main() == (0 if status in {"ready", "degraded"} else 1)


def test_healthcheck_invalid_port_does_not_make_a_request(monkeypatch):
    monkeypatch.setenv("PORT", "0")
    monkeypatch.setattr(healthcheck.urllib.request, "urlopen", lambda *args, **kwargs: pytest.fail("network request should not occur"))
    assert healthcheck.main() == 1


def test_local_credentials_are_private_separated_and_never_overwritten(tmp_path, monkeypatch, capsys):
    # Password hashing itself is covered by auth tests. Here exercise storage,
    # service-secret separation and literal Argon2 '$' preservation.
    monkeypatch.setattr("backend.security.hash_password", lambda password: "$argon2id$synthetic-fixture")
    created = configure(tmp_path, "local-admin", "synthetic-password")
    assert len(created) == 3
    backend, database, redis = [path.read_text(encoding="utf-8") for path in created]
    assert "BACKEND_ADMIN_PASSWORD_HASH=$argon2id$synthetic-fixture\n" in backend
    assert "synthetic-password" not in backend + database + redis
    assert "BACKEND_JWT_SECRET" not in database + redis
    assert "POSTGRES_PASSWORD" not in backend + redis
    assert "REDIS_PASSWORD" not in backend + database
    assert not capsys.readouterr().out
    before = [path.read_bytes() for path in created]
    with pytest.raises(ValueError, match="already exist"):
        configure(tmp_path, "local-admin", "different-synthetic-password")
    assert before == [path.read_bytes() for path in created]


def test_cross_host_lock_generation_checks_linux_only_dependencies(tmp_path):
    dependency = 'linux-only>=1; sys_platform == "linux"'
    package = {"metadata": {"name": "example", "version": "1", "requires_dist": [dependency]},
               "download_info": {"url": "https://files.pythonhosted.org/example.whl", "archive_info": {"hashes": {"sha256": "a" * 64}}}}
    source, target = tmp_path / "resolution.json", tmp_path / "lock.txt"
    source.write_text(json.dumps({"install": [package]}), encoding="utf-8")
    with pytest.raises(ValueError, match="Linux dependency missing"):
        lock_report(source, target)
    extra = {"metadata": {"name": "linux-only", "version": "1.2"},
             "download_info": {"url": "https://files.pythonhosted.org/linux.whl", "archive_info": {"hashes": {"sha256": "b" * 64}}}}
    source.write_text(json.dumps({"install": [package, extra]}), encoding="utf-8")
    assert lock_report(source, target) == {"example": "1", "linux-only": "1.2"}
    assert "linux-only==1.2 --hash=sha256:" in target.read_text(encoding="utf-8")


def test_committed_runtime_locks_are_hash_pinned_and_exclude_gpu_distributions():
    root = Path(__file__).resolve().parents[1]
    for name in ("requirements-runtime.lock", "requirements-runtime-embeddings.lock"):
        records = [line for line in (root / name).read_text(encoding="utf-8").splitlines() if line and not line.startswith(("#", "--"))]
        assert records and all(" --hash=sha256:" in line for line in records)
        assert not any(line.startswith(("nvidia-", "triton==")) for line in records)
    cpu = (root / "requirements-runtime-embeddings.lock").read_text(encoding="utf-8")
    assert "torch @ https://download.pytorch.org/whl/cpu/" in cpu
    assert "%2Bcpu-cp312-cp312-manylinux_2_28_x86_64.whl" in cpu
