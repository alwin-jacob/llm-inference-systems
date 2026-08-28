from __future__ import annotations

import json
import os
from pathlib import Path

import pytest
from pydantic import ValidationError

from llm_inference_systems.canonical import canonical_json_bytes
from llm_inference_systems.stage2_bundle import (
    Stage2BundleBuilder,
    Stage2BundleError,
    inspect_bundle_state,
    read_committed_summary,
    validate_committed_bundle,
)
from llm_inference_systems.stage2_contracts import (
    BundleState,
    Stage2BundleManifest,
    Stage2EvidenceBoundary,
    Stage2EvidenceScope,
)


def _boundary() -> Stage2EvidenceBoundary:
    return Stage2EvidenceBoundary(
        evidence_scope=Stage2EvidenceScope.TEST_FIXTURE_ONLY,
        stage2a_cpu_fixture_tested=True,
        real_runtime_execution=False,
        model_execution=False,
        tokenizer_execution=False,
        gpu_execution=False,
        cuda_execution=False,
    )


def _reconstruct(raw: dict[str, bytes]) -> dict[str, bytes]:
    requests = json.loads(raw["raw/requests.json"])
    return {"derived/summary.json": canonical_json_bytes({"count": len(requests)}) + b"\n"}


def _builder(tmp_path: Path, name: str = "restart-1") -> Stage2BundleBuilder:
    return Stage2BundleBuilder(
        tmp_path,
        name,
        repetition_index=1,
        boundary=_boundary(),
        source_commit="a" * 40,
    )


def _populate(builder: Stage2BundleBuilder) -> None:
    builder.write_raw("raw/requests.json", b'[{"request_id":"fixture-1"}]\n')
    builder.write_derived("derived/summary.json", b'{"count":1}\n')


def test_incomplete_to_committed_manifest_last_atomic_bundle(tmp_path: Path) -> None:
    builder = _builder(tmp_path)
    assert inspect_bundle_state(builder.staging_path) is BundleState.INCOMPLETE
    _populate(builder)
    manifest = builder.commit(_reconstruct)
    assert manifest.state is BundleState.COMMITTED
    assert not builder.staging_path.exists()
    assert builder.final_path.is_dir()
    assert validate_committed_bundle(builder.final_path, _reconstruct) == manifest
    assert read_committed_summary(builder.final_path, _reconstruct) == {"count": 1}
    assert tuple(entry.path for entry in manifest.files) == (
        "derived/summary.json",
        "raw/requests.json",
    )


def test_crash_before_manifest_remains_inspectably_incomplete(tmp_path: Path) -> None:
    builder = _builder(tmp_path)
    _populate(builder)
    with pytest.raises(Stage2BundleError, match="simulated crash"):
        builder.commit(_reconstruct, crash_before_manifest=True)
    assert inspect_bundle_state(builder.staging_path) is BundleState.INCOMPLETE
    with pytest.raises(Stage2BundleError, match="only committed"):
        read_committed_summary(builder.staging_path, _reconstruct)


def test_terminal_failure_becomes_invalid_with_boundary(tmp_path: Path) -> None:
    builder = _builder(tmp_path)
    builder.write_raw("raw/request.log", b"TEST_FIXTURE_ONLY failure evidence\n")
    builder.invalidate(
        phase="MEASURED_WINDOW",
        reason="PROTOCOL_TERMINAL_MISSING",
        last_valid_boundary="GENERATION_TERMINAL",
    )
    assert inspect_bundle_state(builder.staging_path) is BundleState.INVALID
    with pytest.raises(Stage2BundleError, match="only an incomplete"):
        builder.commit(_reconstruct)
    with pytest.raises(Stage2BundleError, match="only an incomplete"):
        builder.write_raw("raw/later.log", b"late evidence\n")
    with pytest.raises(Stage2BundleError, match="only an incomplete"):
        builder.invalidate(
            phase="SHUTDOWN",
            reason="SECOND_TERMINAL_FAILURE",
            last_valid_boundary="MEASURED_WINDOW",
        )


def test_missing_manifest_and_noncommitted_summary_are_refused(tmp_path: Path) -> None:
    builder = _builder(tmp_path)
    _populate(builder)
    builder.commit(_reconstruct)
    (builder.final_path / "evidence-manifest.json").unlink()
    assert inspect_bundle_state(builder.final_path) is BundleState.INCOMPLETE
    with pytest.raises(Stage2BundleError, match="only committed"):
        read_committed_summary(builder.final_path, _reconstruct)


def test_altered_or_missing_file_is_detected(tmp_path: Path) -> None:
    builder = _builder(tmp_path)
    _populate(builder)
    builder.commit(_reconstruct)
    (builder.final_path / "raw/requests.json").write_bytes(b"[]\n")
    with pytest.raises(Stage2BundleError, match="size or hash"):
        validate_committed_bundle(builder.final_path, _reconstruct)


def test_manifest_written_early_is_detected(tmp_path: Path) -> None:
    builder = _builder(tmp_path)
    _populate(builder)
    builder.commit(_reconstruct)
    manifest = builder.final_path / "evidence-manifest.json"
    derived = builder.final_path / "derived/summary.json"
    newer = manifest.stat().st_mtime_ns + 1_000_000
    os.utime(derived, ns=(newer, newer))
    with pytest.raises(Stage2BundleError, match="written before"):
        validate_committed_bundle(builder.final_path, _reconstruct)


@pytest.mark.parametrize("mutation", ["duplicate", "traversal"])
def test_duplicate_inventory_or_path_traversal_is_rejected(
    tmp_path: Path,
    mutation: str,
) -> None:
    builder = _builder(tmp_path)
    _populate(builder)
    manifest = builder.commit(_reconstruct)
    data = manifest.model_dump(mode="json")
    files = list(data["files"])
    if mutation == "duplicate":
        files.append(files[0])
    else:
        files[0] = {**files[0], "path": "../escape.json"}
    data["files"] = files
    with pytest.raises(ValidationError):
        Stage2BundleManifest.model_validate(data)


def test_symlink_and_generated_binary_are_rejected(tmp_path: Path) -> None:
    builder = _builder(tmp_path)
    with pytest.raises(Stage2BundleError, match="generated binary"):
        builder.write_raw("raw/profile.bin", b"binary")
    raw = builder.staging_path / "raw"
    raw.mkdir(exist_ok=True)
    (raw / "linked.json").symlink_to(tmp_path / "outside.json")
    with pytest.raises(Stage2BundleError, match="symlinks"):
        builder.commit(_reconstruct)


def test_exact_reconstruction_mismatch_is_rejected(tmp_path: Path) -> None:
    builder = _builder(tmp_path)
    builder.write_raw("raw/requests.json", b"[]\n")
    builder.write_derived("derived/summary.json", b'{"count":999}\n')
    with pytest.raises(Stage2BundleError, match="exact raw reconstruction"):
        builder.commit(_reconstruct)


def test_atomic_rename_failure_becomes_invalid(tmp_path: Path) -> None:
    def fail_replace(_source: Path, _target: Path) -> None:
        raise OSError("simulated rename failure")

    builder = Stage2BundleBuilder(
        tmp_path,
        "restart-1",
        repetition_index=1,
        boundary=_boundary(),
        source_commit="a" * 40,
        replace=fail_replace,
    )
    _populate(builder)
    with pytest.raises(Stage2BundleError, match="fsync or atomic rename"):
        builder.commit(_reconstruct)
    assert inspect_bundle_state(builder.staging_path) is BundleState.INVALID


def test_fsync_failure_becomes_invalid(tmp_path: Path) -> None:
    builder = _builder(tmp_path)
    _populate(builder)
    calls = 0

    def fail_second(_path: Path) -> None:
        nonlocal calls
        calls += 1
        if calls == 2:
            raise OSError("simulated fsync failure")

    builder._sync_path = fail_second
    with pytest.raises(Stage2BundleError, match="fsync or atomic rename"):
        builder.commit(_reconstruct)
    assert inspect_bundle_state(builder.staging_path) is BundleState.INVALID


def test_bundle_paths_cannot_escape_or_replace_evidence(tmp_path: Path) -> None:
    builder = _builder(tmp_path)
    with pytest.raises(Stage2BundleError, match="unsafe"):
        builder.write_raw("raw/../escape.json", b"{}\n")
    builder.write_raw("raw/evidence.json", b"{}\n")
    with pytest.raises(Stage2BundleError, match="cannot be replaced"):
        builder.write_raw("raw/evidence.json", b"{}\n")
