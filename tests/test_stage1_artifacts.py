"""Stage 1 raw-bundle integrity, reconstruction, and tamper tests."""

from __future__ import annotations

import base64
import hashlib
import json
import shutil
from pathlib import Path

import pytest

from llm_inference_systems.artifact_io import (
    ValidatedBundle,
    atomic_write,
    validate_execution_bundle,
    with_manifest_content_hash,
)
from llm_inference_systems.canonical import canonical_json_bytes
from llm_inference_systems.stage1_contracts import (
    Stage1ExecutionManifest,
    StreamEvidenceKind,
    StreamEvidenceRecord,
)


def _copy_run(source: Path, target: Path) -> Path:
    shutil.copytree(source, target)
    return target


def _rewrite_stream_events(directory: Path, events: list[StreamEvidenceRecord]) -> None:
    data = b"".join(canonical_json_bytes(event) + b"\n" for event in events)
    atomic_write(directory / "stream-events.jsonl", data)
    manifest = Stage1ExecutionManifest.model_validate_json(
        (directory / "manifest.json").read_bytes()
    )
    raw_hashes = dict(manifest.raw_file_sha256)
    raw_hashes["stream-events.jsonl"] = hashlib.sha256(data).hexdigest()
    updated = with_manifest_content_hash(
        manifest.model_copy(update={"raw_file_sha256": raw_hashes, "content_sha256": None})
    )
    atomic_write(directory / "manifest.json", canonical_json_bytes(updated) + b"\n")


def _load_stream_events(directory: Path) -> list[StreamEvidenceRecord]:
    return [
        StreamEvidenceRecord.model_validate_json(line)
        for line in (directory / "stream-events.jsonl").read_bytes().splitlines()
    ]


def test_valid_bundle_reconstructs_exact_summary(
    stage1_bundle_pair: tuple[Path, ValidatedBundle, Path, ValidatedBundle],
) -> None:
    path, expected, _, _ = stage1_bundle_pair
    validated = validate_execution_bundle(path)
    assert validated.summary == expected.summary
    assert validated.manifest.content_sha256 == expected.manifest.content_sha256


def test_raw_file_hash_tampering_is_rejected(
    stage1_bundle_pair: tuple[Path, ValidatedBundle, Path, ValidatedBundle],
    tmp_path: Path,
) -> None:
    path = _copy_run(stage1_bundle_pair[0], tmp_path / "raw-tamper")
    with (path / "requests.jsonl").open("ab") as handle:
        handle.write(b"{}\n")
    with pytest.raises(ValueError, match="raw file digest"):
        validate_execution_bundle(path)


def test_summary_tampering_is_rejected(
    stage1_bundle_pair: tuple[Path, ValidatedBundle, Path, ValidatedBundle],
    tmp_path: Path,
) -> None:
    path = _copy_run(stage1_bundle_pair[0], tmp_path / "summary-tamper")
    with (path / "summary.json").open("ab") as handle:
        handle.write(b" ")
    with pytest.raises(ValueError, match="summary file digest"):
        validate_execution_bundle(path)


def test_unknown_request_event_reference_is_rejected_even_after_rehash(
    stage1_bundle_pair: tuple[Path, ValidatedBundle, Path, ValidatedBundle],
    tmp_path: Path,
) -> None:
    path = _copy_run(stage1_bundle_pair[0], tmp_path / "unknown-reference")
    events = _load_stream_events(path)
    events[0] = events[0].model_copy(update={"request_id": "unknown-request"})
    _rewrite_stream_events(path, events)
    with pytest.raises(ValueError, match="unknown request"):
        validate_execution_bundle(path)


def test_missing_terminal_is_rejected_even_after_rehash(
    stage1_bundle_pair: tuple[Path, ValidatedBundle, Path, ValidatedBundle],
    tmp_path: Path,
) -> None:
    path = _copy_run(stage1_bundle_pair[0], tmp_path / "missing-terminal")
    events = _load_stream_events(path)
    request_id = stage1_bundle_pair[1].requests[1].request_id
    removed = False
    retained: list[StreamEvidenceRecord] = []
    for event in events:
        if (
            not removed
            and event.request_id == request_id
            and event.kind is StreamEvidenceKind.REQUEST_TERMINAL
        ):
            removed = True
            continue
        retained.append(event.model_copy(update={"sequence": len(retained)}))
    _rewrite_stream_events(path, retained)
    with pytest.raises(ValueError, match="one start, terminal, and end"):
        validate_execution_bundle(path)


def test_duplicate_terminal_is_rejected_even_after_rehash(
    stage1_bundle_pair: tuple[Path, ValidatedBundle, Path, ValidatedBundle],
    tmp_path: Path,
) -> None:
    path = _copy_run(stage1_bundle_pair[0], tmp_path / "duplicate-terminal")
    events = _load_stream_events(path)
    terminal = next(event for event in events if event.kind is StreamEvidenceKind.REQUEST_TERMINAL)
    events.append(terminal)
    events = [event.model_copy(update={"sequence": index}) for index, event in enumerate(events)]
    _rewrite_stream_events(path, events)
    with pytest.raises(ValueError, match="one start, terminal, and end"):
        validate_execution_bundle(path)


def test_impossible_event_chronology_is_rejected_even_after_rehash(
    stage1_bundle_pair: tuple[Path, ValidatedBundle, Path, ValidatedBundle],
    tmp_path: Path,
) -> None:
    path = _copy_run(stage1_bundle_pair[0], tmp_path / "chronology")
    bundle = stage1_bundle_pair[1]
    request = bundle.requests[1]
    events = _load_stream_events(path)
    for index, event in enumerate(events):
        if (
            event.request_id == request.request_id
            and event.kind is StreamEvidenceKind.REQUEST_TERMINAL
        ):
            events[index] = event.model_copy(
                update={"observation_offset_ns": request.timing.dispatch_offset_ns - 1}
            )
            break
    _rewrite_stream_events(path, events)
    with pytest.raises(ValueError, match="chronology"):
        validate_execution_bundle(path)


def test_raw_body_chunks_are_reversibly_retained(
    stage1_bundle_pair: tuple[Path, ValidatedBundle, Path, ValidatedBundle],
) -> None:
    bundle = stage1_bundle_pair[1]
    raw_events = [
        event for event in bundle.stream_events if event.kind is StreamEvidenceKind.RAW_BODY_CHUNK
    ]
    assert raw_events
    for event in raw_events:
        raw = base64.b64decode(event.raw_bytes_base64 or "", validate=True)
        assert len(raw) == event.raw_byte_count
        assert hashlib.sha256(raw).hexdigest() == event.raw_bytes_sha256


def test_repeat_runs_match_semantics_but_not_run_content_identity(
    stage1_bundle_pair: tuple[Path, ValidatedBundle, Path, ValidatedBundle],
) -> None:
    _, first, _, second = stage1_bundle_pair
    assert first.manifest.semantic_fingerprint == second.manifest.semantic_fingerprint
    assert first.manifest.content_sha256 != second.manifest.content_sha256
    assert first.manifest.run_id != second.manifest.run_id


def test_atomic_write_replaces_complete_file_without_temp_residue(tmp_path: Path) -> None:
    path = tmp_path / "atomic.json"
    atomic_write(path, b"first\n")
    atomic_write(path, b"second\n")
    assert path.read_bytes() == b"second\n"
    assert not tuple(tmp_path.glob(".*.tmp-*"))


def test_bundle_file_set_is_closed(
    stage1_bundle_pair: tuple[Path, ValidatedBundle, Path, ValidatedBundle],
    tmp_path: Path,
) -> None:
    path = _copy_run(stage1_bundle_pair[0], tmp_path / "extra-file")
    (path / "unexpected.json").write_text(json.dumps({"synthetic": True}), encoding="utf-8")
    with pytest.raises(ValueError, match="file set"):
        validate_execution_bundle(path)


def test_every_execution_artifact_record_retains_fixture_boundary(
    stage1_bundle_pair: tuple[Path, ValidatedBundle, Path, ValidatedBundle],
) -> None:
    bundle = stage1_bundle_pair[1]
    boundaries = [
        bundle.manifest.boundary,
        bundle.summary.boundary,
        *(request.boundary for request in bundle.requests),
        *(event.boundary for event in bundle.stream_events),
        *(event.boundary for event in bundle.server_events),
    ]
    assert all(boundary.evidence_scope == "TEST_FIXTURE_ONLY" for boundary in boundaries)
    assert all(boundary.synthetic_fixture for boundary in boundaries)
    assert all(not boundary.model_execution for boundary in boundaries)
