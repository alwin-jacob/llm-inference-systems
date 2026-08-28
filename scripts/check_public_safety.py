"""Scan repository candidate files without reading anything outside the repository.

Git worktrees use Git's tracked/untracked/ignore semantics. Extracted archives use a
conservative filesystem walk that never follows symlinks and excludes only known generated junk.
"""

from __future__ import annotations

import ntpath
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
_PROHIBITED_GENERATED_SUFFIXES = frozenset(
    {".bin", ".ckpt", ".npy", ".npz", ".nsys-rep", ".onnx", ".pt", ".qdrep", ".safetensors"}
)
_EXECUTABLE_URL_PATTERN = re.compile(r"https?://[^\s\"']+", re.IGNORECASE)
_HOSTNAME_BOUNDARY_CHARACTERS = r"A-Za-z0-9._-"
_HOSTNAME_SEGMENT_PATTERN = r"[A-Za-z0-9][A-Za-z0-9._-]{0,252}"
_HOME_USERNAME_PATTERN = r"[A-Za-z0-9][A-Za-z0-9._-]{0,63}"


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


def _bounded_literal(value: str) -> re.Pattern[str]:
    trailing = rf"(?![{_HOSTNAME_BOUNDARY_CHARACTERS}])"
    leading = rf"(?<![{_HOSTNAME_BOUNDARY_CHARACTERS}])"
    return re.compile(leading + re.escape(value) + trailing, re.IGNORECASE)


def _absolute_home_directories() -> tuple[str, ...]:
    candidates = {str(Path.home())}
    for name in ("HOME", "USERPROFILE"):
        if value := os.environ.get(name):
            candidates.add(value)
    if (drive := os.environ.get("HOMEDRIVE")) and (path := os.environ.get("HOMEPATH")):
        candidates.add(drive + path)

    absolute = {
        value.rstrip("\\/")
        for value in candidates
        if value and (Path(value).is_absolute() or ntpath.isabs(value))
    }
    return tuple(sorted(absolute, key=lambda value: (value.casefold(), value)))


def _private_environment_patterns() -> tuple[tuple[str, re.Pattern[str]], ...]:
    patterns: list[tuple[str, re.Pattern[str]]] = [
        ("absolute-home-directory", _bounded_literal(home)) for home in _absolute_home_directories()
    ]
    patterns.extend(
        (
            (
                "apple-user-home-path",
                re.compile(
                    rf"(?<![A-Za-z0-9._:/\\-])/Users/{_HOME_USERNAME_PATTERN}"
                    rf"(?![A-Za-z0-9._-])",
                    re.IGNORECASE,
                ),
            ),
            (
                "linux-user-home-path",
                re.compile(
                    rf"(?<![A-Za-z0-9._-])/home/{_HOME_USERNAME_PATTERN}"
                    rf"(?![A-Za-z0-9._-])",
                    re.IGNORECASE,
                ),
            ),
            (
                "windows-user-home-path",
                re.compile(
                    rf"(?<![A-Za-z0-9._-])[A-Za-z]:[\\/]"
                    rf"(?:Users|Documents and Settings)[\\/]{_HOME_USERNAME_PATTERN}"
                    rf"(?![A-Za-z0-9._-])",
                    re.IGNORECASE,
                ),
            ),
            (
                "windows-unc-user-home-path",
                re.compile(
                    rf"(?<![\\])\\\\{_HOSTNAME_SEGMENT_PATTERN}[\\/]"
                    rf"(?:Users[\\/])?{_HOME_USERNAME_PATTERN}(?![A-Za-z0-9._-])",
                    re.IGNORECASE,
                ),
            ),
        )
    )

    hostname = socket.gethostname().strip().rstrip(".")
    normalized_hostname = hostname.casefold()
    hostname_is_specific = (
        len(hostname) >= 8
        and normalized_hostname not in {"localhost", "localhost.localdomain"}
        and (
            any(character in hostname for character in ".-_")
            or any(character.isdigit() for character in hostname)
        )
    )
    if hostname_is_specific:
        patterns.append(("local-hostname", _bounded_literal(hostname)))

    return tuple(patterns)


def _patterns() -> tuple[tuple[str, re.Pattern[str]], ...]:
    fragments = {
        "private-key-header": "-----BEGIN " + "PRIVATE KEY-----",
        "submitted-project-file": "Resume" + "Projects_Submitted.md",
        "internal-ledger-file": "claim" + "-ledger",
        "private-credit-plan": "Cre" + "dila",
        "private-migration-source": "immig" + "ration",
        "canonical-private-project": "canonical " + "Project",
        "sensitive-health-file": "medical" + "-record",
        "sensitive-relative-file": "family" + "-record",
        "sensitive-monetary-file": "finance" + "-record",
        "sensitive-credit-file": "loan" + "-application",
        "sensitive-cv-file": "resume" + "_private",
    }
    literal_patterns = tuple(
        (name, re.compile(re.escape(value), re.IGNORECASE)) for name, value in fragments.items()
    )
    token_patterns = (
        ("aws-access-key", re.compile(r"\b(?:AKIA|ASIA)[0-9A-Z]{16}\b")),
        ("github-token", re.compile(r"\bgh[pousr]_[A-Za-z0-9]{30,}\b")),
        ("generic-api-token", re.compile(r"\bsk-[A-Za-z0-9_-]{20,}\b")),
        ("hugging-face-token", re.compile(r"\bhf_[A-Za-z0-9]{20,}\b")),
        (
            "credential-file-content",
            re.compile(
                r"(?im)(?:^|[\"'])"
                r"(?:aws_secret_access_key|api_key|client_secret|proxy_password|"
                r"access_token|refresh_token|HF_TOKEN|HUGGING_FACE_HUB_TOKEN|"
                r"HUGGINGFACE_HUB_TOKEN|HUGGINGFACEHUB_API_TOKEN)"
                r"[\"']?\s*(?:=|:)\s*[\"']?\S+"
            ),
        ),
        (
            "authorization-header",
            re.compile(r"(?im)(?:^|[,{])\s*[\"']?(?:proxy-)?authorization[\"']?\s*:\s*[\"']?\S+"),
        ),
        (
            "cookie-header",
            re.compile(r"(?im)(?:^|[,{])\s*[\"']?(?:cookie|set-cookie)[\"']?\s*:\s*[\"']?\S+"),
        ),
        (
            "proxy-credential-url",
            re.compile(r"(?i)\bhttps?://[^\s/:@]+:[^\s/@]+@[^\s/]+"),
        ),
        (
            "gpu-uuid",
            re.compile(r"(?i)\bGPU-[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}\b"),
        ),
        (
            "model-cache-path",
            re.compile(
                r"(?i)(?:^|[\s=:\"'])(?:~|/[^\s\"']+)?/\.cache/"
                r"(?:huggingface|torch|vllm)(?:/|\b)"
            ),
        ),
        (
            "notebook-account-identifier",
            re.compile(
                r"(?i)\b(?:kaggle\.com/code/[^\s/]+/[^\s]+|"
                r"colab\.research\.google\.com/drive/[^\s/?#]+)"
            ),
        ),
        (
            "private-hostname-suffix",
            re.compile(r"(?i)\b[A-Za-z0-9][A-Za-z0-9.-]*\.(?:corp|internal|lan|local)\b"),
        ),
        (
            "private-account-identifier",
            re.compile(r"(?i)\b(?:account|notebook)[_-]?id\s*(?:=|:)\s*[\"']?\S+"),
        ),
        (
            "employer-control-plane-file",
            re.compile(
                ("sam" + "sung") + r"[^\n]{0,80}(?:claim|ledger|control[-_ ]?plane)",
                re.IGNORECASE,
            ),
        ),
    )
    return (*literal_patterns, *_private_environment_patterns(), *token_patterns)


def scan_repository(root: Path) -> tuple[tuple[str, str], ...]:
    resolved_root = root.resolve(strict=True)
    findings: list[tuple[str, str]] = []
    for relative_path in discover_candidate_files(resolved_root):
        path = resolved_root / relative_path
        relative = relative_path.as_posix()
        if _path_has_symlink_component(resolved_root, relative_path):
            findings.append((relative, _SYMLINK_RULE))
            continue
        if path.suffix.casefold() in _PROHIBITED_GENERATED_SUFFIXES:
            findings.append((relative, "generated-binary-or-profiler-artifact"))
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            findings.append((relative, "unreviewed-binary-content"))
            continue
        if relative.startswith("examples/configs/"):
            for match in _EXECUTABLE_URL_PATTERN.finditer(text):
                if not match.group(0).startswith("http://127.0.0.1"):
                    findings.append((relative, "arbitrary-executable-runtime-url"))
                    break
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
