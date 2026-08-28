"""Atomic Stage 1 bundle persistence, integrity validation, and reconstruction."""

from __future__ import annotations

import base64
import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path

from pydantic import BaseModel

from llm_inference_systems.canonical import canonical_json_bytes, sha256_identity
from llm_inference_systems.fixture_tokens import parse_input_tokens, parse_output_tokens
from llm_inference_systems.sse import IncrementalSSEParser, SSEProtocolError
from llm_inference_systems.stage1_contracts import (
    FixtureDefinition,
    ServerEventRecord,
    Stage1ExecutionManifest,
    Stage1RequestRecord,
    Stage1RunConfiguration,
    Stage1RunSummary,
    Stage1TerminalClass,
    Stage1WorkloadDefinition,
    StreamEvidenceKind,
    StreamEvidenceRecord,
)
from llm_inference_systems.stage1_metrics import (
    derive_observed_max_client_concurrency,
    derive_stage1_summary,
    semantic_fingerprint,
)

MAX_ARTIFACT_FILE_BYTES = 10 * 1024 * 1024
RUN_FILENAMES = frozenset(
    {
        "manifest.json",
        "requests.jsonl",
        "stream-events.jsonl",
        "server-events.jsonl",
        "summary.json",
    }
)


@dataclass(frozen=True, slots=True)
class ValidatedBundle:
    manifest: Stage1ExecutionManifest
    requests: tuple[Stage1RequestRecord, ...]
    stream_events: tuple[StreamEvidenceRecord, ...]
    server_events: tuple[ServerEventRecord, ...]
    summary: Stage1RunSummary


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def manifest_content_identity(manifest: Stage1ExecutionManifest) -> str:
    return sha256_identity(manifest, omit_fields=frozenset({"content_sha256"}))


def with_manifest_content_hash(manifest: Stage1ExecutionManifest) -> Stage1ExecutionManifest:
    return manifest.model_copy(update={"content_sha256": manifest_content_identity(manifest)})


def verify_manifest_content_hash(manifest: Stage1ExecutionManifest) -> bool:
    return (
        manifest.content_sha256 is not None
        and manifest.content_sha256 == manifest_content_identity(manifest)
    )


def _json_bytes(model: BaseModel) -> bytes:
    return canonical_json_bytes(model) + b"\n"


def _jsonl_bytes(models: tuple[BaseModel, ...]) -> bytes:
    return b"".join(canonical_json_bytes(model) + b"\n" for model in models)


def atomic_write(path: Path, data: bytes) -> None:
    """Atomically replace one final file after flush and fsync."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    try:
        with temporary.open("wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)  # noqa: PTH105 - contract explicitly requires os.replace.
    finally:
        if temporary.exists():
            temporary.unlink()


def write_execution_bundle(
    output_directory: Path,
    *,
    manifest_without_file_hashes: Stage1ExecutionManifest,
    requests: tuple[Stage1RequestRecord, ...],
    stream_events: tuple[StreamEvidenceRecord, ...],
    server_events: tuple[ServerEventRecord, ...],
    summary: Stage1RunSummary,
) -> Stage1ExecutionManifest:
    if output_directory.exists():
        raise ValueError("output directory already exists")
    output_directory.mkdir(parents=True)
    payloads = {
        "requests.jsonl": _jsonl_bytes(requests),
        "stream-events.jsonl": _jsonl_bytes(stream_events),
        "server-events.jsonl": _jsonl_bytes(server_events),
        "summary.json": _json_bytes(summary),
    }
    for filename in (
        "requests.jsonl",
        "stream-events.jsonl",
        "server-events.jsonl",
        "summary.json",
    ):
        atomic_write(output_directory / filename, payloads[filename])
    manifest = manifest_without_file_hashes.model_copy(
        update={
            "raw_file_sha256": {
                filename: _sha256_bytes(payloads[filename])
                for filename in ("requests.jsonl", "stream-events.jsonl", "server-events.jsonl")
            },
            "summary_sha256": _sha256_bytes(payloads["summary.json"]),
            "content_sha256": None,
        }
    )
    manifest = with_manifest_content_hash(manifest)
    atomic_write(output_directory / "manifest.json", _json_bytes(manifest))
    return manifest


def _read_limited(path: Path) -> bytes:
    size = path.stat().st_size
    if size > MAX_ARTIFACT_FILE_BYTES:
        raise ValueError(f"artifact file exceeds size limit: {path.name}")
    return path.read_bytes()


def _parse_jsonl[ModelT: BaseModel](data: bytes, model: type[ModelT]) -> tuple[ModelT, ...]:
    if not data or not data.endswith(b"\n"):
        raise ValueError("JSONL artifact must be nonempty and newline terminated")
    values: list[ModelT] = []
    for line in data.splitlines():
        if not line:
            raise ValueError("JSONL artifact cannot contain blank records")
        values.append(model.model_validate_json(line))
    return tuple(values)


def validate_stage1_inputs(
    workload: Stage1WorkloadDefinition,
    configuration: Stage1RunConfiguration,
    fixture: FixtureDefinition,
) -> None:
    workload_hash = sha256_identity(workload)
    fixture_hash = sha256_identity(fixture)
    if configuration.workload_sha256 != workload_hash:
        raise ValueError("configuration workload identity does not match the workload")
    if configuration.fixture_sha256 != fixture_hash:
        raise ValueError("configuration fixture identity does not match the fixture")
    fixture_cases = {case.case_id: case for case in fixture.cases}
    workload_cases = (workload.warmup_case, *workload.measured_cases)
    if set(fixture_cases) != {case.case_id for case in workload_cases}:
        raise ValueError("workload and fixture case sets differ")
    for case in workload_cases:
        fixture_case = fixture_cases[case.case_id]
        if (
            fixture_case.input_text != case.prompt
            or fixture_case.expected_terminal_class is not case.expected_terminal_class
            or fixture_case.expected_output_token_count != case.expected_output_token_count
        ):
            raise ValueError("workload expectations differ from the fixture definition")
        if not parse_input_tokens(case.prompt):
            raise ValueError("fixture workload must contain exact synthetic input markers")
        token_total = 0
        for action in fixture_case.actions:
            if action.kind.value == "SSE_TOKEN_EVENT":
                token_total += len(parse_output_tokens(action.text or ""))
        if token_total != fixture_case.expected_output_token_count:
            raise ValueError("fixture action token total differs from its declared expectation")
        if token_total > fixture_case.maximum_output_tokens:
            raise ValueError("fixture action token total exceeds maximum_output_tokens")
    measured_outcomes = [case.expected_terminal_class for case in workload.measured_cases]
    if measured_outcomes.count(Stage1TerminalClass.SUCCESS) != 5:
        raise ValueError("Stage 1 measured fixture must declare five successes")
    if measured_outcomes.count(Stage1TerminalClass.FAILED) != 2:
        raise ValueError("Stage 1 measured fixture must declare two non-timeout failures")
    if measured_outcomes.count(Stage1TerminalClass.TIMEOUT) != 1:
        raise ValueError("Stage 1 measured fixture must declare one timeout")


def _validate_raw_chunks(
    request: Stage1RequestRecord,
    events: tuple[StreamEvidenceRecord, ...],
) -> None:
    chunks = tuple(
        sorted(
            (
                event
                for event in events
                if event.request_id == request.request_id
                and event.kind is StreamEvidenceKind.RAW_BODY_CHUNK
            ),
            key=lambda event: (
                event.raw_chunk_sequence if event.raw_chunk_sequence is not None else -1
            ),
        )
    )
    if [event.raw_chunk_sequence for event in chunks] != list(range(len(chunks))):
        raise ValueError("raw chunk sequence must be contiguous for each request")
    decoded: list[bytes] = []
    for event in chunks:
        try:
            raw = base64.b64decode(event.raw_bytes_base64 or "", validate=True)
        except ValueError as error:
            raise ValueError("raw body Base64 is invalid") from error
        if len(raw) != event.raw_byte_count or _sha256_bytes(raw) != event.raw_bytes_sha256:
            raise ValueError("raw body chunk integrity validation failed")
        decoded.append(raw)
    if chunks:
        if request.timing.first_response_body_bytes_offset_ns != chunks[0].observation_offset_ns:
            raise ValueError("first-body timing does not match retained raw evidence")
    elif request.timing.first_response_body_bytes_offset_ns is not None:
        raise ValueError("first-body timing exists without a raw body chunk")

    token_shapes: list[int] = []
    parser = IncrementalSSEParser()
    parser_failed = False
    if request.http_status is not None and 200 <= request.http_status < 300:
        try:
            for raw in decoded:
                for frame in parser.feed(raw):
                    if frame.kind == "data":
                        if frame.data is None:
                            raise ValueError("data frame is missing data")
                        value = json.loads(frame.data)
                        choices = value["choices"]
                        text = choices[0]["text"]
                        if not isinstance(text, str):
                            raise ValueError("fixture token text is not a string")
                        token_shapes.append(len(parse_output_tokens(text)))
            parser.finalize()
        except (
            KeyError,
            IndexError,
            TypeError,
            json.JSONDecodeError,
            SSEProtocolError,
            ValueError,
        ):
            parser_failed = True
    if request.terminal_class is Stage1TerminalClass.SUCCESS:
        if parser_failed or not parser.done:
            raise ValueError("successful request raw body does not reconstruct as successful SSE")
        if tuple(token_shapes) != request.token_event_delta_counts:
            raise ValueError("raw SSE token shape differs from the request record")
    elif (
        request.failure is not None
        and request.failure.kind.value == "PROTOCOL_MALFORMED_STREAM"
        and not parser_failed
    ):
        raise ValueError("malformed-stream failure is not reproduced from raw body evidence")


def _validate_cross_records(
    manifest: Stage1ExecutionManifest,
    requests: tuple[Stage1RequestRecord, ...],
    stream_events: tuple[StreamEvidenceRecord, ...],
    server_events: tuple[ServerEventRecord, ...],
) -> None:
    request_by_id = {request.request_id: request for request in requests}
    if len(request_by_id) != len(requests):
        raise ValueError("request IDs must be unique")
    expected_cases = (
        manifest.workload.warmup_case,
        *manifest.workload.measured_cases,
    )
    if tuple(request.case_id for request in requests) != tuple(
        case.case_id for case in expected_cases
    ):
        raise ValueError("request order and workload order differ")
    if requests[0].phase != "WARMUP" or any(
        request.phase != "MEASURED" for request in requests[1:]
    ):
        raise ValueError("warmup/measured request phases are incorrect")
    if len(requests) != 9:
        raise ValueError("Stage 1 bundle must retain one warmup and eight measured requests")
    for event in stream_events:
        request = request_by_id.get(event.request_id)
        if request is None:
            raise ValueError("stream event references an unknown request")
        if event.case_id != request.case_id or event.phase != request.phase:
            raise ValueError("stream event request identity is inconsistent")
    derive_observed_max_client_concurrency(stream_events)
    for request in requests:
        related = tuple(event for event in stream_events if event.request_id == request.request_id)
        starts = [
            event for event in related if event.kind is StreamEvidenceKind.CLIENT_REQUEST_STARTED
        ]
        ends = [event for event in related if event.kind is StreamEvidenceKind.CLIENT_REQUEST_ENDED]
        terminals = [
            event for event in related if event.kind is StreamEvidenceKind.REQUEST_TERMINAL
        ]
        if len(starts) != 1 or len(ends) != 1 or len(terminals) != 1:
            raise ValueError("each request requires one start, terminal, and end lifecycle event")
        if not (
            starts[0].observation_offset_ns
            <= request.timing.dispatch_offset_ns
            <= request.timing.terminal_offset_ns
            == terminals[0].observation_offset_ns
            <= ends[0].observation_offset_ns
        ):
            raise ValueError("request lifecycle chronology is impossible")
        if terminals[0].terminal_class is not request.terminal_class:
            raise ValueError("terminal event and request terminal class differ")
        token_events = tuple(
            event for event in related if event.kind is StreamEvidenceKind.SSE_TOKEN_EVENT
        )
        if (
            tuple(event.token_delta_count for event in token_events)
            != request.token_event_delta_counts
        ):
            raise ValueError("token evidence and request token deltas differ")
        if tuple(event.observation_offset_ns for event in token_events) != (
            request.token_event_observation_offsets_ns
        ):
            raise ValueError("token evidence and request token observations differ")
        done_count = sum(event.kind is StreamEvidenceKind.SSE_DONE for event in related)
        if done_count != (1 if request.terminal_class is Stage1TerminalClass.SUCCESS else 0):
            raise ValueError("[DONE] evidence does not match terminal semantics")
        _validate_raw_chunks(request, stream_events)
    if [event.sequence for event in server_events] != list(range(len(server_events))):
        raise ValueError("server event sequence must be contiguous")
    for server_event in server_events:
        request = request_by_id.get(server_event.request_id)
        if request is None:
            raise ValueError("server event references an unknown request")
        if server_event.case_id != request.case_id:
            raise ValueError("server event case identity differs from its request")


def validate_execution_bundle(directory: Path) -> ValidatedBundle:
    if not directory.is_dir():
        raise ValueError("run directory does not exist")
    filenames = frozenset(path.name for path in directory.iterdir() if path.is_file())
    if filenames != RUN_FILENAMES:
        raise ValueError("run directory file set is incomplete or unexpected")
    manifest_data = _read_limited(directory / "manifest.json")
    manifest = Stage1ExecutionManifest.model_validate_json(manifest_data)
    if not verify_manifest_content_hash(manifest):
        raise ValueError("manifest content hash is missing or invalid")
    validate_stage1_inputs(manifest.workload, manifest.configuration, manifest.fixture)
    if manifest.workload_sha256 != sha256_identity(manifest.workload):
        raise ValueError("manifest workload identity is invalid")
    if manifest.configuration_sha256 != sha256_identity(manifest.configuration):
        raise ValueError("manifest configuration identity is invalid")
    if manifest.fixture_sha256 != sha256_identity(manifest.fixture):
        raise ValueError("manifest fixture identity is invalid")

    raw_data = {
        filename: _read_limited(directory / filename)
        for filename in ("requests.jsonl", "stream-events.jsonl", "server-events.jsonl")
    }
    for filename, data in raw_data.items():
        if _sha256_bytes(data) != manifest.raw_file_sha256[filename]:
            raise ValueError(f"raw file digest is invalid: {filename}")
    summary_data = _read_limited(directory / "summary.json")
    if _sha256_bytes(summary_data) != manifest.summary_sha256:
        raise ValueError("summary file digest is invalid")
    requests = _parse_jsonl(raw_data["requests.jsonl"], Stage1RequestRecord)
    stream_events = _parse_jsonl(raw_data["stream-events.jsonl"], StreamEvidenceRecord)
    server_events = _parse_jsonl(raw_data["server-events.jsonl"], ServerEventRecord)
    summary = Stage1RunSummary.model_validate_json(summary_data)
    _validate_cross_records(manifest, requests, stream_events, server_events)
    reconstructed = derive_stage1_summary(manifest.configuration, requests, stream_events)
    if reconstructed != summary:
        raise ValueError("stored summary differs from exact raw reconstruction")
    reconstructed_fingerprint = semantic_fingerprint(
        workload_sha256=manifest.workload_sha256,
        configuration_sha256=manifest.configuration_sha256,
        fixture_sha256=manifest.fixture_sha256,
        requests=requests,
        stream_events=stream_events,
        summary=reconstructed,
    )
    if reconstructed_fingerprint != manifest.semantic_fingerprint:
        raise ValueError("semantic fingerprint differs from raw reconstruction")
    return ValidatedBundle(manifest, requests, stream_events, server_events, reconstructed)


def reconstruct_summary(directory: Path) -> Stage1RunSummary:
    """Return only a summary that has been reconstructed and matched exactly."""

    return validate_execution_bundle(directory).summary
