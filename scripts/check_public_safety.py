"""Scan repository candidate files without reading anything outside the repository.

Git worktrees use Git's tracked/untracked/ignore semantics. Extracted archives use a
conservative filesystem walk that never follows symlinks and excludes only known generated junk.
"""

from __future__ import annotations

import getpass
import os
import re
import socket
import subprocess
from pathlib import Path

from llm_inference_systems.canonical import canonical_json

_ARCHIVE_EXCLUDED_DIRECTORIES = frozenset(
    {
        ".coverage",
        ".git",
        ".mypy_cache",
        ".pytest_cache",
        ".ruff_cache",
        ".venv",
        "__pycache__",
        "build",
        "dist",
        "htmlcov",
    }
)
_ARCHIVE_EXCLUDED_FILES = frozenset({".coverage", ".DS_Store"})
_ARCHIVE_EXCLUDED_SUFFIXES = frozenset({".pyc", ".pyd", ".pyo"})
_SYMLINK_RULE = "repository-symlink"


def _sort_key(path: Path) -> bytes:
    return os.fsencode(path.as_posix())


def _validated_relative_path(name: str) -> Path:
    relative = Path(name)
    if relative.is_absolute() or not relative.parts or ".." in relative.parts:
        raise ValueError("candidate path escaped repository root")
    return relative


def _git_metadata_exists(root: Path) -> bool:
    metadata = root / ".git"
    return metadata.exists() or metadata.is_symlink()


def _path_has_symlink_component(root: Path, relative: Path) -> bool:
    current = root
    for part in relative.parts:
        current /= part
        if current.is_symlink():
            return True
    return False


def _git_candidate_files(root: Path) -> tuple[Path, ...]:
    result = subprocess.run(
        ["git", "ls-files", "--cached", "--others", "--exclude-standard", "-z"],
        cwd=root,
        check=True,
        capture_output=True,
    )
    relative_paths = sorted(
        (
            _validated_relative_path(os.fsdecode(name))
            for name in result.stdout.split(b"\0")
            if name
        ),
        key=_sort_key,
    )
    files: list[Path] = []
    for relative in relative_paths:
        path = root / relative
        if _path_has_symlink_component(root, relative) or path.is_file():
            files.append(relative)
    return tuple(files)


def _excluded_archive_directory(name: str) -> bool:
    return name in _ARCHIVE_EXCLUDED_DIRECTORIES or name.endswith(".egg-info")


def _excluded_archive_file(name: str) -> bool:
    return name in _ARCHIVE_EXCLUDED_FILES or Path(name).suffix in _ARCHIVE_EXCLUDED_SUFFIXES


def _archive_candidate_files(root: Path) -> tuple[Path, ...]:
    files: list[Path] = []

    def visit(directory: Path, relative_directory: Path) -> None:
        for path in sorted(directory.iterdir(), key=lambda item: os.fsencode(item.name)):
            relative = relative_directory / path.name
            if path.is_symlink():
                if not _excluded_archive_directory(path.name) and not _excluded_archive_file(
                    path.name
                ):
                    files.append(relative)
            elif path.is_dir():
                if not _excluded_archive_directory(path.name):
                    visit(path, relative)
            elif path.is_file() and not _excluded_archive_file(path.name):
                files.append(relative)

    visit(root, Path())
    return tuple(sorted(files, key=_sort_key))


def discover_candidate_files(root: Path) -> tuple[Path, ...]:
    """Return deterministic repository-relative candidate paths for either supported mode."""

    resolved_root = root.resolve(strict=True)
    if _git_metadata_exists(resolved_root):
        return _git_candidate_files(resolved_root)
    return _archive_candidate_files(resolved_root)


def _patterns() -> tuple[tuple[str, re.Pattern[str]], ...]:
    username = getpass.getuser()
    hostname = socket.gethostname()
    fragments = {
        "absolute-apple-home": "/" + "Users" + "/",
        "private-key-header": "-----BEGIN " + "PRIVATE KEY-----",
        "submitted-project-file": "Resume" + "Projects_Submitted.md",
        "internal-ledger-file": "claim" + "-ledger",
        "private-credit-plan": "Cre" + "dila",
        "private-migration-source": "immig" + "ration",
        "canonical-private-project": "canonical " + "Project",
    }
    literal_patterns = tuple(
        (name, re.compile(re.escape(value), re.IGNORECASE)) for name, value in fragments.items()
    )
    environment_patterns = tuple(
        (name, re.compile(re.escape(value), re.IGNORECASE))
        for name, value in (("local-username", username), ("local-hostname", hostname))
        if len(value) >= 3
    )
    token_patterns = (
        ("aws-access-key", re.compile(r"\b(?:AKIA|ASIA)[0-9A-Z]{16}\b")),
        ("github-token", re.compile(r"\bgh[pousr]_[A-Za-z0-9]{30,}\b")),
        ("generic-api-token", re.compile(r"\bsk-[A-Za-z0-9_-]{20,}\b")),
        (
            "credential-file-content",
            re.compile(r"(?im)^\s*(?:aws_secret_access_key|api_key|client_secret)\s*="),
        ),
    )
    return (*literal_patterns, *environment_patterns, *token_patterns)


def scan_repository(root: Path) -> tuple[tuple[str, str], ...]:
    resolved_root = root.resolve(strict=True)
    findings: list[tuple[str, str]] = []
    for relative_path in discover_candidate_files(resolved_root):
        path = resolved_root / relative_path
        relative = relative_path.as_posix()
        if _path_has_symlink_component(resolved_root, relative_path):
            findings.append((relative, _SYMLINK_RULE))
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        for rule, pattern in _patterns():
            if pattern.search(text):
                findings.append((relative, rule))
    return tuple(findings)


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    findings = scan_repository(root)
    if findings:
        print(
            canonical_json(
                {
                    "finding_count": len(findings),
                    "findings": [{"path": path, "rule": rule} for path, rule in findings],
                    "status": "failed",
                }
            )
        )
        return 1
    print(canonical_json({"finding_count": 0, "status": "passed"}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
