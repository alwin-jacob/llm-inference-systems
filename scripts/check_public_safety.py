"""Scan repository candidate files without reading anything outside the repository."""

from __future__ import annotations

import getpass
import re
import socket
import subprocess
from pathlib import Path

from llm_inference_systems.canonical import canonical_json


def _candidate_files(root: Path) -> tuple[Path, ...]:
    result = subprocess.run(
        ["git", "ls-files", "--cached", "--others", "--exclude-standard", "-z"],
        cwd=root,
        check=True,
        capture_output=True,
    )
    relative_names = sorted(name for name in result.stdout.decode().split("\0") if name)
    resolved_root = root.resolve()
    files: list[Path] = []
    for name in relative_names:
        path = (root / name).resolve()
        if not path.is_relative_to(resolved_root):
            raise ValueError("candidate path escaped repository root")
        if path.is_file():
            files.append(path)
    return tuple(files)


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
    findings: list[tuple[str, str]] = []
    for path in _candidate_files(root):
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        relative = path.relative_to(root.resolve()).as_posix()
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
