"""Focused tests for Git-worktree and extracted-archive public-safety discovery."""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from scripts.check_public_safety import discover_candidate_files, scan_repository

ROOT = Path(__file__).resolve().parents[1]
FORBIDDEN_SENTINEL = "-----BEGIN " + "PRIVATE KEY-----"


def _run_git(root: Path, *arguments: str) -> None:
    subprocess.run(
        ["git", *arguments],
        cwd=root,
        check=True,
        capture_output=True,
    )


def _initialize_git_worktree(root: Path) -> None:
    root.mkdir()
    _run_git(root, "init", "--quiet")


def test_git_discovery_includes_tracked_and_untracked_nonignored_files(tmp_path: Path) -> None:
    root = tmp_path / "repository"
    _initialize_git_worktree(root)
    (root / ".gitignore").write_text(".venv/\n*.ignored\n", encoding="utf-8")
    tracked = root / "src" / "tracked.py"
    tracked.parent.mkdir()
    tracked.write_text("tracked source\n", encoding="utf-8")
    _run_git(root, "add", ".gitignore", "src/tracked.py")
    (root / "docs").mkdir()
    (root / "docs" / "untracked.md").write_text("untracked source\n", encoding="utf-8")
    (root / ".venv").mkdir()
    (root / ".venv" / "ignored.txt").write_text("ignored\n", encoding="utf-8")
    (root / "other.ignored").write_text("ignored\n", encoding="utf-8")

    candidates = discover_candidate_files(root)

    assert Path("src/tracked.py") in candidates
    assert Path("docs/untracked.md") in candidates
    assert Path(".venv/ignored.txt") not in candidates
    assert Path("other.ignored") not in candidates


def test_archive_discovery_without_git_finds_nested_source_docs_and_artifacts(
    tmp_path: Path,
) -> None:
    root = tmp_path / "archive"
    for relative in ("src/module.py", "docs/notes.md", "artifacts/future/result.json"):
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("synthetic fixture\n", encoding="utf-8")

    assert not (root / ".git").exists()
    assert discover_candidate_files(root) == (
        Path("artifacts/future/result.json"),
        Path("docs/notes.md"),
        Path("src/module.py"),
    )


def test_archive_discovery_never_invokes_git(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "archive"
    root.mkdir()
    (root / "source.py").write_text("synthetic fixture\n", encoding="utf-8")

    def fail_if_called(*args: object, **kwargs: object) -> None:
        raise AssertionError(f"Git must not be called in archive mode: {args!r} {kwargs!r}")

    monkeypatch.setattr("scripts.check_public_safety.subprocess.run", fail_if_called)

    assert discover_candidate_files(root) == (Path("source.py"),)


def test_archive_discovery_excludes_only_documented_generated_junk(tmp_path: Path) -> None:
    root = tmp_path / "archive"
    root.mkdir()
    excluded_directories = (
        ".git",
        ".venv",
        "__pycache__",
        ".pytest_cache",
        ".mypy_cache",
        ".ruff_cache",
        ".coverage",
        "htmlcov",
        "build",
        "dist",
        "package.egg-info",
    )
    for name in excluded_directories:
        directory = root / "nested" / name
        directory.mkdir(parents=True)
        (directory / "ignored.txt").write_text(FORBIDDEN_SENTINEL, encoding="utf-8")
    (root / ".DS_Store").write_text(FORBIDDEN_SENTINEL, encoding="utf-8")
    (root / "generated.pyc").write_bytes(FORBIDDEN_SENTINEL.encode())
    kept = root / "unfamiliar-source" / "kept.txt"
    kept.parent.mkdir()
    kept.write_text("synthetic fixture\n", encoding="utf-8")

    assert discover_candidate_files(root) == (Path("unfamiliar-source/kept.txt"),)
    assert scan_repository(root) == ()


def test_archive_discovery_order_is_deterministic(tmp_path: Path) -> None:
    root = tmp_path / "archive"
    for relative in ("z-last.txt", "nested/c.txt", "a-first.txt", "nested/a.txt"):
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("synthetic fixture\n", encoding="utf-8")

    expected = (
        Path("a-first.txt"),
        Path("nested/a.txt"),
        Path("nested/c.txt"),
        Path("z-last.txt"),
    )
    assert discover_candidate_files(root) == expected
    assert discover_candidate_files(root) == expected


def test_archive_scan_detects_forbidden_synthetic_pattern(tmp_path: Path) -> None:
    root = tmp_path / "archive"
    root.mkdir()
    (root / "unsafe.txt").write_text(FORBIDDEN_SENTINEL, encoding="utf-8")

    assert scan_repository(root) == (("unsafe.txt", "private-key-header"),)


def test_git_scan_detects_forbidden_synthetic_pattern(tmp_path: Path) -> None:
    root = tmp_path / "repository"
    _initialize_git_worktree(root)
    (root / "unsafe.txt").write_text(FORBIDDEN_SENTINEL, encoding="utf-8")
    _run_git(root, "add", "unsafe.txt")

    assert scan_repository(root) == (("unsafe.txt", "private-key-header"),)


@pytest.mark.parametrize("username", ["root", "runner", "ubuntu", "alwinjacob"])
def test_common_identity_words_are_not_findings(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    username: str,
) -> None:
    root = tmp_path / "archive"
    root.mkdir()
    (root / "source.txt").write_text(
        "The root runner uses ubuntu while alwinjacob reviews the source.\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("USER", username)
    monkeypatch.setenv("LOGNAME", username)

    assert scan_repository(root) == ()


@pytest.mark.parametrize(
    ("username", "home_parts", "expected_rule"),
    [
        ("alwinjacob", ("", "Users", "alwinjacob", "private", "notes.txt"), "apple-user-home-path"),
        ("runner", ("", "home", "runner", "private", "notes.txt"), "linux-user-home-path"),
    ],
)
def test_structured_unix_user_home_paths_are_detected(
    tmp_path: Path,
    username: str,
    home_parts: tuple[str, ...],
    expected_rule: str,
) -> None:
    root = tmp_path / "archive"
    root.mkdir()
    private_path = "/".join(home_parts)
    (root / "unsafe.txt").write_text(f"private source: {private_path}\n", encoding="utf-8")

    assert ("unsafe.txt", expected_rule) in scan_repository(root)


@pytest.mark.parametrize(
    ("private_path", "expected_rule"),
    [
        (
            "C:" + "\\" + "Users" + "\\" + "runner" + "\\" + "private.txt",
            "windows-user-home-path",
        ),
        ("C:/" + "Users" + "/" + "runner" + "/private.txt", "windows-user-home-path"),
        (
            "\\\\build-share" + "\\" + "Users" + "\\" + "runner" + "\\" + "private.txt",
            "windows-unc-user-home-path",
        ),
    ],
)
def test_structured_windows_user_home_paths_are_detected(
    tmp_path: Path,
    private_path: str,
    expected_rule: str,
) -> None:
    root = tmp_path / "archive"
    root.mkdir()
    (root / "unsafe.txt").write_text(f"private source: {private_path}\n", encoding="utf-8")

    assert scan_repository(root) == (("unsafe.txt", expected_rule),)


def test_exact_absolute_home_directory_is_detected(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "archive"
    root.mkdir()
    home = "/" + "opt" + "/private-home/casey"
    (root / "unsafe.txt").write_text(f"home={home}\n", encoding="utf-8")
    monkeypatch.setenv("HOME", home)

    assert scan_repository(root) == (("unsafe.txt", "absolute-home-directory"),)


def test_sufficiently_specific_private_hostname_is_detected(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "archive"
    root.mkdir()
    hostname = "workstation-47." + "internal"
    (root / "unsafe.txt").write_text(f"built-by={hostname}\n", encoding="utf-8")
    monkeypatch.setattr("scripts.check_public_safety.socket.gethostname", lambda: hostname)

    assert ("unsafe.txt", "local-hostname") in scan_repository(root)


@pytest.mark.parametrize(
    ("secret", "expected_rule"),
    [
        ("-----BEGIN " + "PRIVATE KEY-----", "private-key-header"),
        ("AKIA" + "0123456789ABCDEF", "aws-access-key"),
        ("ghp_" + "A" * 30, "github-token"),
        ("sk-" + "synthetic_fixture_token_12345", "generic-api-token"),
        ("client_" + "secret=fixture-only", "credential-file-content"),
    ],
)
def test_existing_secret_like_patterns_remain_detected(
    tmp_path: Path,
    secret: str,
    expected_rule: str,
) -> None:
    root = tmp_path / "archive"
    root.mkdir()
    (root / "unsafe.txt").write_text(secret + "\n", encoding="utf-8")

    assert scan_repository(root) == (("unsafe.txt", expected_rule),)


@pytest.mark.parametrize(
    ("private_value", "expected_rule"),
    [
        ("Author" + "ization: Bearer fixture-secret-value", "authorization-header"),
        ('{"Author' + 'ization":"Bearer fixture-secret-value"}', "authorization-header"),
        ("Proxy-Author" + "ization: Basic fixture-secret-value", "authorization-header"),
        ("Cook" + "ie: session=fixture-secret", "cookie-header"),
        ('{"Cook' + 'ie":"session=fixture-secret"}', "cookie-header"),
        ('{"access_' + 'token":"fixture-secret"}', "credential-file-content"),
        ("https://fixture-user:" + "fixture-pass@proxy.invalid", "proxy-credential-url"),
        (
            "GPU-" + "01234567-89ab-cdef-0123-456789abcdef",
            "gpu-uuid",
        ),
        ("cache=/" + "private/.cache/huggingface/hub", "model-cache-path"),
        (
            "https://kaggle.com/" + "code/private-user/private-notebook",
            "notebook-account-identifier",
        ),
        (
            "https://colab.research.google.com/" + "drive/private-notebook-id",
            "notebook-account-identifier",
        ),
        ("worker-47." + "co" + "rp." + "internal", "private-hostname-suffix"),
        ("account_" + "id=private-account", "private-account-identifier"),
        (("sam" + "sung") + "-internal-control-plane.json", "employer-control-plane-file"),
        ("medical" + "-record.pdf", "sensitive-health-file"),
        ("loan" + "-application.json", "sensitive-credit-file"),
    ],
)
def test_stage2_private_runtime_and_personal_patterns_are_detected(
    tmp_path: Path,
    private_value: str,
    expected_rule: str,
) -> None:
    root = tmp_path / "archive"
    root.mkdir()
    (root / "unsafe.txt").write_text(private_value + "\n", encoding="utf-8")

    assert scan_repository(root) == (("unsafe.txt", expected_rule),)


def test_executable_runtime_config_rejects_arbitrary_remote_url(tmp_path: Path) -> None:
    root = tmp_path / "archive"
    configuration = root / "examples" / "configs" / "unsafe.json"
    configuration.parent.mkdir(parents=True)
    configuration.write_text(
        '{"endpoint":"https://' + 'runtime.invalid/v1/completions"}\n',
        encoding="utf-8",
    )

    assert scan_repository(root) == (
        ("examples/configs/unsafe.json", "arbitrary-executable-runtime-url"),
    )


@pytest.mark.parametrize(
    ("filename", "content", "expected_rule"),
    [
        ("profile.nsys-rep", b"fixture profile", "generated-binary-or-profiler-artifact"),
        ("opaque.dat", b"\xff\xfe", "unreviewed-binary-content"),
    ],
)
def test_unreviewed_generated_or_binary_artifacts_are_rejected(
    tmp_path: Path,
    filename: str,
    content: bytes,
    expected_rule: str,
) -> None:
    root = tmp_path / "archive"
    root.mkdir()
    (root / filename).write_bytes(content)

    assert scan_repository(root) == ((filename, expected_rule),)


def test_archive_symlink_is_reported_without_reading_external_content(tmp_path: Path) -> None:
    root = tmp_path / "archive"
    root.mkdir()
    outside_directory = tmp_path / "synthetic-outside"
    outside_directory.mkdir()
    outside_file = outside_directory / "unsafe.txt"
    outside_file.write_text(FORBIDDEN_SENTINEL, encoding="utf-8")
    (root / "file-link.txt").symlink_to(outside_file)
    (root / "linked-directory").symlink_to(outside_directory, target_is_directory=True)

    assert scan_repository(root) == (
        ("file-link.txt", "repository-symlink"),
        ("linked-directory", "repository-symlink"),
    )


def test_git_candidate_path_cannot_escape_root(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "repository"
    root.mkdir()
    (root / ".git").mkdir()

    def escaping_result(*args: object, **kwargs: object) -> subprocess.CompletedProcess[bytes]:
        return subprocess.CompletedProcess(
            args=[], returncode=0, stdout=b"../outside.txt\0", stderr=b""
        )

    monkeypatch.setattr("scripts.check_public_safety.subprocess.run", escaping_result)

    with pytest.raises(ValueError, match="escaped repository root"):
        discover_candidate_files(root)


def test_git_command_failure_is_not_reclassified_as_archive_mode(tmp_path: Path) -> None:
    root = tmp_path / "repository"
    root.mkdir()
    (root / ".git").mkdir()

    with pytest.raises(subprocess.CalledProcessError):
        discover_candidate_files(root)


def test_current_repository_passes_public_safety_scan() -> None:
    assert scan_repository(ROOT) == ()
