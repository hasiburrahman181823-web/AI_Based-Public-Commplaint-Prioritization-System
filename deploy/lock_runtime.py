"""Convert pip --dry-run JSON reports into hash-locked Linux inference inputs.

Resolve the .in files for CPython 3.12, Linux x86_64 with binary-only wheels.
Reports may be generated on another host; dependency markers are rechecked for
the actual Linux target so a host-specific marker cannot omit a required wheel.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
from packaging.markers import default_environment
from packaging.requirements import Requirement
from packaging.utils import canonicalize_name


def lock_report(report: Path, output: Path) -> dict[str, str]:
    packages = json.loads(report.read_text(encoding="utf-8"))["install"]
    indexed = {canonicalize_name(item["metadata"]["name"]): item for item in packages}
    target = {**default_environment(), "python_version": "3.12", "python_full_version": "3.12.16",
              "sys_platform": "linux", "os_name": "posix", "platform_system": "Linux",
              "platform_machine": "x86_64", "implementation_name": "cpython"}
    extras = {"psycopg": {"binary"}}
    changed = True
    while changed:
        changed = False
        for name, item in indexed.items():
            for raw in item["metadata"].get("requires_dist", []):
                requirement = Requirement(raw)
                if requirement.marker and not any(requirement.marker.evaluate({**target, "extra": extra})
                                                   for extra in {"", *extras.get(name, set())}):
                    continue
                dependency = canonicalize_name(requirement.name)
                if dependency not in indexed:
                    raise ValueError(f"Linux dependency missing from resolution: {name} -> {requirement}")
                version = indexed[dependency]["metadata"]["version"]
                if requirement.specifier and not requirement.specifier.contains(version, prereleases=True):
                    raise ValueError(f"Dependency conflict: {name} requires {requirement}, resolved {version}")
                previous = set(extras.get(dependency, set()))
                extras.setdefault(dependency, set()).update(requirement.extras)
                changed |= previous != extras[dependency]
    lines = ["# Generated from pip resolution; CPython 3.12 / Linux x86_64 only.",
             "# Every selected distribution is version/hash pinned. Refresh and audit deliberately.",
             "--index-url https://pypi.org/simple", "--only-binary=:all:", ""]
    versions = {}
    for name, item in sorted(indexed.items()):
        version = item["metadata"]["version"]
        versions[name] = version
        info = item["download_info"]
        digest = info["archive_info"]["hashes"]["sha256"]
        requirement = f"{name}=={version}"
        if name == "torch":
            if not version.endswith("+cpu") or not info["url"].startswith("https://download.pytorch.org/whl/cpu/"):
                raise ValueError("Torch must resolve to the official CPU-only wheel")
            requirement = "torch @ " + info["url"].split("#", 1)[0]
        if name.startswith("nvidia-") or name == "triton":
            raise ValueError("GPU dependency found in CPU runtime")
        lines.append(requirement + " --hash=sha256:" + digest)
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return versions


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--core-report", required=True, type=Path)
    parser.add_argument("--cpu-report", required=True, type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args(argv)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    core = lock_report(args.core_report, args.output_dir / "requirements-runtime.lock")
    cpu = lock_report(args.cpu_report, args.output_dir / "requirements-runtime-embeddings.lock")
    # Advisory lookup uses upstream torch release 2.x; +cpu is a build suffix.
    audit = ["# Advisory lookup of the exact runtime releases (torch +cpu suffix mapped to upstream version).",
             "# DO NOT install this lookup file; install the hash-locked runtime files instead.",
             "# No wheel installation is needed: pip-audit --no-deps --disable-pip -r this-file."]
    audit.extend(f"{name}=={version.split('+', 1)[0] if name == 'torch' else version}" for name, version in sorted(cpu.items()))
    (args.output_dir / "requirements-runtime-audit.txt").write_text("\n".join(audit) + "\n", encoding="utf-8")
    print(f"Locked {len(core)} core and {len(cpu)} CPU inference distributions; no GPU packages")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
