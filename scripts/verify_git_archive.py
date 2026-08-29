"""Verify a metadata-free Git archive with the exact locked Python environment."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import shlex
import subprocess
import tarfile
import tempfile
from pathlib import Path

EXPECTED_PYTHON_VERSION = "3.13.15"
EVIDENCE_DIRECTORY = Path("artifacts/stage1-fixture/2026-08-27")
IMMUTABLE_DIRECTORIES = (
    EVIDENCE_DIRECTORY,
    Path("examples"),
    Path("schemas"),
)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def _run(command: list[str], *, cwd: Path, env: dict[str, str] | None = None) -> None:
    print(f"$ {shlex.join(command)}", flush=True)
    subprocess.run(command, cwd=cwd, env=env, check=True)


def _capture(command: list[str], *, cwd: Path) -> str:
    result = subprocess.run(command, cwd=cwd, check=True, capture_output=True, text=True)
    return result.stdout.strip()


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _immutable_snapshot(root: Path) -> dict[str, str]:
    paths = [
        root / "uv.lock",
        root / "execution-lock/stage2-execution-lock.json",
        *(path for directory in IMMUTABLE_DIRECTORIES for path in (root / directory).rglob("*")),
    ]
    snapshot: dict[str, str] = {}
    for path in sorted(paths):
        if path.is_symlink():
            raise AssertionError(f"immutable input is a symlink: {path.relative_to(root)}")
        if path.is_file():
            snapshot[path.relative_to(root).as_posix()] = _sha256(path)
    _require("uv.lock" in snapshot, "uv.lock is missing")
    _require(
        "execution-lock/stage2-execution-lock.json" in snapshot,
        "Stage 2 execution lock is missing",
    )
    _require(
        f"{EVIDENCE_DIRECTORY.as_posix()}/evidence-manifest.json" in snapshot,
        "checked evidence manifest is missing",
    )
    return snapshot


def _archive(treeish: str, *, root: Path, destination: Path) -> None:
    print(f"$ git archive --format=tar {treeish}", flush=True)
    with destination.open("wb") as archive_file:
        subprocess.run(
            ["git", "archive", "--format=tar", treeish],
            cwd=root,
            check=True,
            stdout=archive_file,
        )


def _verification_environment() -> dict[str, str]:
    environment = os.environ.copy()
    for name in ("PYTHONHOME", "PYTHONPATH", "UV_PROJECT_ENVIRONMENT", "VIRTUAL_ENV"):
        environment.pop(name, None)
    environment["UV_NO_PROGRESS"] = "1"
    return environment


def _verify_export(export_root: Path, expected_snapshot: dict[str, str]) -> None:
    _require(not (export_root / ".git").exists(), "archive unexpectedly contains .git")
    _require(not (export_root / ".venv").exists(), "archive unexpectedly contains .venv")
    _require(
        _immutable_snapshot(export_root) == expected_snapshot,
        "archive immutable inputs differ from the source tree",
    )

    environment = _verification_environment()
    _run(
        ["uv", "sync", "--python", EXPECTED_PYTHON_VERSION, "--frozen", "--group", "dev"],
        cwd=export_root,
        env=environment,
    )
    commands = [
        ["uv", "lock", "--check"],
        [
            "uv",
            "run",
            "--no-sync",
            "python",
            "-c",
            (f"import platform; assert platform.python_version() == '{EXPECTED_PYTHON_VERSION}'"),
        ],
        ["uv", "run", "--no-sync", "ruff", "check", "."],
        ["uv", "run", "--no-sync", "ruff", "format", "--check", "."],
        ["uv", "run", "--no-sync", "mypy", "src", "tests", "scripts"],
        ["uv", "run", "--no-sync", "pytest", "-q"],
        ["uv", "run", "--no-sync", "python", "scripts/check_schema_sync.py"],
        ["uv", "run", "--no-sync", "python", "scripts/check_public_safety.py"],
        ["uv", "run", "--no-sync", "python", "scripts/verify_stage0.py"],
        ["uv", "run", "--no-sync", "python", "scripts/verify_stage1.py"],
        ["uv", "run", "--no-sync", "python", "scripts/verify_stage2a.py"],
        [
            "uv",
            "run",
            "--no-sync",
            "python",
            "scripts/verify_checked_stage1_evidence.py",
            EVIDENCE_DIRECTORY.as_posix(),
        ],
        [
            "uv",
            "run",
            "--no-sync",
            "llm-inference",
            "validate-config",
            "examples/configs/stage2a-protocol-fixture-v1.json",
        ],
        [
            "uv",
            "run",
            "--no-sync",
            "llm-inference",
            "validate-stage2-request",
            "examples/fixtures/stage2a-completion-request-v1.json",
        ],
        [
            "uv",
            "run",
            "--no-sync",
            "llm-inference",
            "validate-stage2-execution-lock",
            "execution-lock/stage2-execution-lock.json",
        ],
        ["uv", "run", "--no-sync", "llm-inference", "schema-check"],
    ]
    for command in commands:
        _run(command, cwd=export_root, env=environment)

    _require(
        _immutable_snapshot(export_root) == expected_snapshot,
        "archive verification changed uv.lock or checked evidence",
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--treeish",
        default="HEAD",
        help="Git tree-ish to archive; CI and release verification use the default HEAD",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    root = Path(__file__).resolve().parents[1]
    _require(
        platform.python_version() == EXPECTED_PYTHON_VERSION,
        f"archive controller requires Python {EXPECTED_PYTHON_VERSION}",
    )
    treeish = str(args.treeish)
    tested_tree = _capture(["git", "rev-parse", f"{treeish}^{{tree}}"], cwd=root)
    expected_snapshot = _immutable_snapshot(root)

    temporary_path: Path | None = None
    with tempfile.TemporaryDirectory(prefix="lis-git-archive-") as temporary:
        temporary_path = Path(temporary)
        archive_path = temporary_path / "source.tar"
        export_root = temporary_path / "source"
        export_root.mkdir()
        _archive(treeish, root=root, destination=archive_path)
        with tarfile.open(archive_path, mode="r") as archive:
            archive.extractall(export_root, filter="data")
        _verify_export(export_root, expected_snapshot)

    _require(temporary_path is not None and not temporary_path.exists(), "temporary tree remains")
    print(
        json.dumps(
            {
                "immutable_inputs_byte_identical": True,
                "python_version": EXPECTED_PYTHON_VERSION,
                "status": "verified",
                "temporary_directory_removed": True,
                "tested_tree": tested_tree,
                "treeish": treeish,
            },
            sort_keys=True,
            separators=(",", ":"),
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
