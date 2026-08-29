"""Cardinality-complete Stage 2 experiment attestation and aggregate reconstruction.

Stage 2A exercises these contracts with synthetic CPU fixtures only.  This module
contains no runtime launcher, downloader, model/tokenizer loader, or GPU code.
"""

from __future__ import annotations

import base64
import binascii
import hashlib
import json
import os
import re
from datetime import timedelta
from enum import StrEnum
from pathlib import Path, PurePosixPath
from typing import Annotated, Final, Literal, Self, cast

from pydantic import AwareDatetime, Field, model_validator

from llm_inference_systems.canonical import canonical_json_bytes, sha256_identity
from llm_inference_systems.contracts import (
    Identifier,
    NonNegativeInt,
    PositiveInt,
    Sha256,
    StrictModel,
)
from llm_inference_systems.stage2_attestation import (
    CudaBackedExecutionAttestation,
    LinuxEnvironmentManifest,
    NvidiaT4ResourceAttestation,
    PrometheusMeasurementAttestation,
    PrometheusRawScrapeCapture,
    PublicSafetyAttestation,
    RequestIdentityAttestation,
    RuntimePackageExecutionLockAttestation,
    ServerRestartIdentity,
)
from llm_inference_systems.stage2_bundle import (
    Stage2BundleError,
    decoded_base64_evidence_texts,
    validate_committed_bundle,
)
from llm_inference_systems.stage2_contracts import (
    BundleFileEntry,
    BundleState,
    RawLogCapture,
    Stage2BundleManifest,
    Stage2CancellationRequest,
    Stage2CompletionRequest,
    Stage2EvidenceScope,
    Stage2ManifestBoundFile,
    Stage2PerRequestMetrics,
    Stage2RequestEvidence,
)
from llm_inference_systems.stage2_control import (
    AggregateComparisonState,
    CancellationResult,
    FirstGenerationDeliveryEvidence,
    RestartComparison,
    RestartSemanticRecord,
    Stage2ControlError,
    Stage2RuntimeControlEvidence,
    bundle_manifest_sha256,
    compare_three_restarts,
    validate_aggregate_commit,
)
from llm_inference_systems.stage2_prometheus import (
    PrometheusSnapshot,
    parse_prometheus_snapshot,
)
from llm_inference_systems.stage2_protocol import (
    Stage2CancellationStreamCapture,
    Stage2ProtocolError,
    Stage2StreamValidator,
    correlate_request_logs,
)
from llm_inference_systems.stage2_runtime import (
    LAUNCH_ABSENT_ENVIRONMENT_VARIABLES,
    OFFLINE_RUNTIME_ENVIRONMENT,
    ModelTokenizerSnapshotManifest,
    Stage2LaunchSpec,
)
from llm_inference_systems.stage2_transport import (
    CollectorWireCaptureProvenance,
    FixtureWireCaptureProvenance,
    Stage2HTTPExchangeCapture,
    Stage2OrderedHeadersCapture,
    Stage2WireCaptureProvenance,
)

STAGE2_EXPERIMENT_CASE_IDS: Final = tuple(f"stage2-case-v1-{index:02d}" for index in range(1, 17))
AGGREGATE_MANIFEST_PATH: Final = "aggregate-experiment-manifest.json"

_AGGREGATE_SENSITIVE_PATTERNS: Final = (
    re.compile(re.escape("-----BEGIN " + "PRIVATE KEY-----"), re.IGNORECASE),
    re.compile(r"\b(?:AKIA|ASIA)[0-9A-Z]{16}\b"),
    re.compile(r"\bgh[pousr]_[A-Za-z0-9]{30,}\b"),
    re.compile(r"\bsk-[A-Za-z0-9_-]{20,}\b"),
    re.compile(r"\bhf_[A-Za-z0-9]{20,}\b"),
    re.compile(
        r"(?im)(?:^|[\"'])"
        r"(?:aws_secret_access_key|api_key|client_secret|access_token|refresh_token|"
        r"proxy_password|HF_TOKEN|HUGGING_FACE_HUB_TOKEN|HUGGINGFACE_HUB_TOKEN|"
        r"HUGGINGFACEHUB_API_TOKEN)"
        r"[\"']?\s*(?:=|:)\s*[\"']?\S+"
    ),
    re.compile(r"(?im)(?:^|[,{])\s*[\"']?(?:proxy-)?authorization[\"']?\s*:\s*[\"']?\S+"),
    re.compile(r"(?im)(?:^|[,{])\s*[\"']?(?:cookie|set-cookie)[\"']?\s*:\s*[\"']?\S+"),
    re.compile(r"(?i)\bhttps?://[^\s/:@]+:[^\s/@]+@[^\s/]+"),
    re.compile(r"(?i)(?<![A-Za-z0-9._-])/(?:Users|home)/[A-Za-z0-9._-]+"),
    re.compile(r"(?i)(?:~|/[^\s\"']+)?/\.cache/(?:huggingface|torch|vllm)(?:/|\b)"),
    re.compile(r"(?i)\bGPU-[0-9a-f]{8}(?:-[0-9a-f]{4}){3}-[0-9a-f]{12}\b"),
    re.compile(r"(?i)\b[A-Za-z0-9][A-Za-z0-9.-]*\.(?:corp|internal|lan|local)\b"),
    re.compile(r"(?i)\b(?:account|notebook)[_-]?id\s*(?:=|:)\s*[\"']?\S+"),
    re.compile(("sam" + "sung") + r"[^\n]{0,80}(?:claim|ledger|control[-_ ]?plane)", re.I),
)

_FIXTURE_VALUE_MARKERS: Final = (
    "test_fixture_only",
    "synthetic_protocol_shape_only",
    "synthetic_future_shape_only",
    "synthetic-future-shape",
    "synthetic-shape",
    "<fixture-",
    "stage2-fixture",
)

REQUEST_EVIDENCE_FIELDS: Final = (
    "http_exchange",
    "request_body",
    "request_headers",
    "response_headers",
    "raw_response_body",
    "parsed_sse_events",
    "terminal_boundary",
    "server_logs",
    "server_metrics",
    "lifecycle",
    "token_usage_reconciliation",
)


class Stage2ExperimentError(ValueError):
    """Raised when a complete experiment cannot be reconstructed or committed."""


def _contains_fixture_value(value: object) -> bool:
    if isinstance(value, StrictModel):
        return _contains_fixture_value(value.model_dump(mode="python"))
    if isinstance(value, dict):
        return any(_contains_fixture_value(item) for item in value.values())
    if isinstance(value, (tuple, list)):
        return any(_contains_fixture_value(item) for item in value)
    if isinstance(value, str):
        normalized = value.casefold()
        return any(marker in normalized for marker in _FIXTURE_VALUE_MARKERS)
    return False


def _raw_payload_contains_fixture_value(data: bytes) -> bool:
    try:
        value = json.loads(data)
    except (UnicodeDecodeError, json.JSONDecodeError):
        try:
            value = data.decode("utf-8")
        except UnicodeDecodeError:
            return False
    return _contains_fixture_value(value)


def _safe_relative_path(value: str) -> PurePosixPath:
    path = PurePosixPath(value)
    if (
        value != path.as_posix()
        or path.is_absolute()
        or not path.parts
        or any(part in {"", ".", ".."} for part in path.parts)
    ):
        raise ValueError("experiment evidence path must be normalized and relative")
    return path


ManifestBoundFile = Stage2ManifestBoundFile


def _decode_canonical_base64(value: str, *, label: str) -> bytes:
    try:
        decoded = base64.b64decode(value, validate=True)
    except (binascii.Error, ValueError) as error:
        raise ValueError(f"{label} Base64 is invalid") from error
    if base64.b64encode(decoded).decode("ascii") != value:
        raise ValueError(f"{label} Base64 is not canonical")
    return decoded


def _json_without_duplicate_keys(data: bytes) -> object:
    def pairs_hook(pairs: list[tuple[str, object]]) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, value in pairs:
            if key in result:
                raise Stage2ExperimentError("request JSON contains a duplicate field")
            result[key] = value
        return result

    try:
        return json.loads(data, object_pairs_hook=pairs_hook)
    except UnicodeDecodeError as error:
        raise Stage2ExperimentError("request body is not UTF-8 JSON") from error
    except json.JSONDecodeError as error:
        raise Stage2ExperimentError("request body is not valid JSON") from error


class Stage2ExactRequestBodyCapture(StrictModel):
    exact_bytes_base64: str
    byte_count: NonNegativeInt
    sha256: Sha256
    canonical_request: Stage2CompletionRequest
    canonical_request_sha256: Sha256
    request_identity_sha256: Sha256
    transmission_offset_ns: NonNegativeInt
    identity_sha256: Sha256

    @model_validator(mode="after")
    def validate_body(self) -> Self:
        data = _decode_canonical_base64(self.exact_bytes_base64, label="request body")
        if len(data) != self.byte_count or hashlib.sha256(data).hexdigest() != self.sha256:
            raise ValueError("request body byte count or SHA-256 differs")
        try:
            _json_without_duplicate_keys(data)
            parsed = Stage2CompletionRequest.model_validate_json(data)
        except (Stage2ExperimentError, ValueError) as error:
            raise ValueError("exact request bytes do not parse as the frozen request") from error
        if parsed != self.canonical_request:
            raise ValueError("canonical request differs from exact transmitted bytes")
        if self.canonical_request_sha256 != sha256_identity(parsed):
            raise ValueError("canonical request identity does not reconstruct")
        if self.request_identity_sha256 != sha256_identity(
            {"request": parsed, "request_id": parsed.request_id}
        ):
            raise ValueError("request-body identity does not reconstruct")
        if self.identity_sha256 != sha256_identity(
            self, omit_fields=frozenset({"identity_sha256"})
        ):
            raise ValueError("exact request-body capture identity does not reconstruct")
        return self


class Stage2CancellationExactRequestBodyCapture(StrictModel):
    exact_bytes_base64: str
    byte_count: NonNegativeInt
    sha256: Sha256
    canonical_request: Stage2CancellationRequest
    canonical_request_sha256: Sha256
    request_identity_sha256: Sha256
    transmission_offset_ns: NonNegativeInt
    identity_sha256: Sha256

    @model_validator(mode="after")
    def validate_body(self) -> Self:
        data = _decode_canonical_base64(self.exact_bytes_base64, label="cancellation request body")
        if len(data) != self.byte_count or hashlib.sha256(data).hexdigest() != self.sha256:
            raise ValueError("cancellation request body byte count or SHA-256 differs")
        try:
            _json_without_duplicate_keys(data)
            parsed = Stage2CancellationRequest.model_validate_json(data)
        except (Stage2ExperimentError, ValueError) as error:
            raise ValueError(
                "exact cancellation bytes do not parse as the frozen request"
            ) from error
        if parsed != self.canonical_request:
            raise ValueError("canonical cancellation request differs from exact bytes")
        if self.canonical_request_sha256 != sha256_identity(parsed):
            raise ValueError("canonical cancellation request identity does not reconstruct")
        if self.request_identity_sha256 != sha256_identity(
            {"request": parsed, "request_id": parsed.request_id}
        ):
            raise ValueError("cancellation request-body identity does not reconstruct")
        if self.identity_sha256 != sha256_identity(
            self, omit_fields=frozenset({"identity_sha256"})
        ):
            raise ValueError("exact cancellation request-body identity does not reconstruct")
        return self


class Stage2RawResponseBodyChunk(StrictModel):
    repetition_index: Literal[1, 2, 3]
    case_id: Identifier
    external_request_id: Identifier
    ordinal: NonNegativeInt
    observation_offset_ns: NonNegativeInt
    completed_sse_frame_observation_offsets_ns: tuple[NonNegativeInt, ...]
    exact_bytes_base64: str
    decoded_byte_count: NonNegativeInt
    sha256: Sha256
    source_capture_provenance: Literal[
        "TEST_FIXTURE_ONLY_CPU_SCRIPTED_HTTP",
        "FUTURE_RUNTIME_COLLECTOR_HTTP_BODY",
    ]
    inventory_manifest_path: Literal["raw_response_body.json", "raw/cancellation/client-wire.json"]
    identity_sha256: Sha256

    @model_validator(mode="after")
    def validate_chunk(self) -> Self:
        data = _decode_canonical_base64(self.exact_bytes_base64, label="response-body chunk")
        if not data:
            raise ValueError("retained response-body chunk cannot be empty")
        if len(data) != self.decoded_byte_count or hashlib.sha256(data).hexdigest() != self.sha256:
            raise ValueError("response-body chunk byte count or SHA-256 differs")
        frame_offsets = self.completed_sse_frame_observation_offsets_ns
        if frame_offsets and (
            frame_offsets[0] != self.observation_offset_ns
            or frame_offsets != tuple(sorted(frame_offsets))
        ):
            raise ValueError("completed SSE-frame observations differ from the chunk clock")
        if self.identity_sha256 != sha256_identity(
            self, omit_fields=frozenset({"identity_sha256"})
        ):
            raise ValueError("response-body chunk identity does not reconstruct")
        return self

    def exact_bytes(self) -> bytes:
        return _decode_canonical_base64(self.exact_bytes_base64, label="response-body chunk")


class Stage2TransportCloseCapture(StrictModel):
    external_request_id: Identifier
    close_classification: Literal["CLEAN_EOF", "CLEAN_RESPONSE_CLOSE"]
    close_observation_offset_ns: NonNegativeInt
    response_close_completed: Literal[True]
    post_close_byte_count: Literal[0]
    post_close_event_count: Literal[0]
    raw_response_body_inventory_sha256: Sha256
    request_identity_chain_sha256: Sha256
    identity_sha256: Sha256

    @model_validator(mode="after")
    def validate_close(self) -> Self:
        if self.identity_sha256 != sha256_identity(
            self, omit_fields=frozenset({"identity_sha256"})
        ):
            raise ValueError("transport-close identity does not reconstruct")
        return self


class Stage2RequestWireCapture(StrictModel):
    schema_version: Literal["0.3.0"]
    repetition_index: Literal[1, 2, 3]
    case_id: Identifier
    external_request_id: Identifier
    provenance: Stage2WireCaptureProvenance
    http_exchange: Stage2HTTPExchangeCapture
    request_body: Stage2ExactRequestBodyCapture
    request_headers: Stage2OrderedHeadersCapture
    response_headers: Stage2OrderedHeadersCapture
    response_body_chunks: tuple[Stage2RawResponseBodyChunk, ...] = Field(min_length=1)
    transport_close: Stage2TransportCloseCapture
    identity_sha256: Sha256

    @model_validator(mode="after")
    def validate_capture(self) -> Self:
        if self.request_headers.direction != "TRANSMITTED_REQUEST":
            raise ValueError("request headers use the wrong capture direction")
        if self.response_headers.direction != "RECEIVED_RESPONSE":
            raise ValueError("response headers use the wrong capture direction")
        body_id = self.request_body.canonical_request.request_id
        if not (
            self.external_request_id
            == body_id
            == self.request_headers.effective("x-request-id")
            == self.response_headers.effective("x-request-id")
            == self.transport_close.external_request_id
        ):
            raise ValueError("wire request/header/response/transport identities differ")
        exchange = self.http_exchange
        if (
            exchange.exchange_purpose != "MEASURED_COMPLETION"
            or exchange.repetition_index != self.repetition_index
            or exchange.evidence_unit_id != self.case_id
            or exchange.external_request_id != self.external_request_id
            or exchange.provenance != self.provenance
            or exchange.request_headers != self.request_headers
            or exchange.response_headers != self.response_headers
            or exchange.request_body_byte_count != self.request_body.byte_count
            or exchange.request_body_sha256 != self.request_body.sha256
            or exchange.request_body_transmission_observation_offset_ns
            != self.request_body.transmission_offset_ns
        ):
            raise ValueError("measured HTTP exchange is detached from exact request identity")
        if self.request_body.transmission_offset_ns != self.request_headers.observation_offset_ns:
            raise ValueError("request body and transmitted headers have different dispatch times")
        chunks = self.response_body_chunks
        if tuple(chunk.ordinal for chunk in chunks) != tuple(range(len(chunks))):
            raise ValueError("raw response chunks are missing, duplicated, or reordered")
        if tuple(chunk.observation_offset_ns for chunk in chunks) != tuple(
            sorted(chunk.observation_offset_ns for chunk in chunks)
        ):
            raise ValueError("raw response chunk observation offsets are reordered")
        previous_frame_offset: int | None = None
        for chunk in chunks:
            if (
                previous_frame_offset is not None
                and chunk.observation_offset_ns < previous_frame_offset
            ):
                raise ValueError("raw response chunks overlap prior SSE-frame observations")
            if chunk.completed_sse_frame_observation_offsets_ns:
                previous_frame_offset = chunk.completed_sse_frame_observation_offsets_ns[-1]
        if any(
            (chunk.repetition_index, chunk.case_id, chunk.external_request_id)
            != (self.repetition_index, self.case_id, self.external_request_id)
            for chunk in chunks
        ):
            raise ValueError("raw response chunk identity differs from its request")
        fixture = isinstance(self.provenance, FixtureWireCaptureProvenance)
        expected_chunk_source = (
            "TEST_FIXTURE_ONLY_CPU_SCRIPTED_HTTP"
            if fixture
            else "FUTURE_RUNTIME_COLLECTOR_HTTP_BODY"
        )
        if any(
            chunk.source_capture_provenance != expected_chunk_source
            or chunk.inventory_manifest_path != "raw_response_body.json"
            for chunk in chunks
        ):
            raise ValueError("raw response chunk provenance differs from wire provenance")
        inventory_sha = sha256_identity(chunks)
        close = self.transport_close
        raw_body = b"".join(chunk.exact_bytes() for chunk in chunks)
        body_completion_offset = max(
            offset
            for chunk in chunks
            for offset in (
                chunk.observation_offset_ns,
                *chunk.completed_sse_frame_observation_offsets_ns,
            )
        )
        if (
            close.raw_response_body_inventory_sha256 != inventory_sha
            or close.close_observation_offset_ns <= chunks[-1].observation_offset_ns
            or exchange.response_body_byte_count != len(raw_body)
            or exchange.response_body_sha256 != hashlib.sha256(raw_body).hexdigest()
            or exchange.response_body_inventory_sha256 != inventory_sha
            or exchange.response_body_completion_observation_offset_ns != body_completion_offset
            or exchange.transport_terminal_observation_offset_ns
            != close.close_observation_offset_ns
            or exchange.transport_terminal_classification != close.close_classification
        ):
            raise ValueError("HTTP exchange or transport close is detached from raw body chunks")
        if self.identity_sha256 != sha256_identity(
            self, omit_fields=frozenset({"identity_sha256"})
        ):
            raise ValueError("request wire-capture identity does not reconstruct")
        return self


class Stage2ReplayedSSEEvent(StrictModel):
    ordinal: NonNegativeInt
    observation_offset_ns: NonNegativeInt
    kind: Literal["comment", "data", "done"]
    data: str | None
    comments: tuple[str, ...]
    identity_sha256: Sha256

    @model_validator(mode="after")
    def validate_event(self) -> Self:
        if self.kind == "done" and self.data != "[DONE]":
            raise ValueError("replayed done event differs from [DONE]")
        if self.kind == "data" and self.data is None:
            raise ValueError("replayed data event is empty")
        if self.identity_sha256 != sha256_identity(
            self, omit_fields=frozenset({"identity_sha256"})
        ):
            raise ValueError("replayed SSE-event identity does not reconstruct")
        return self


class Stage2CancellationGenerationEvent(StrictModel):
    sse_event_ordinal: NonNegativeInt
    observation_offset_ns: NonNegativeInt
    output_token_ids: tuple[NonNegativeInt, ...] = Field(min_length=1)
    text: str
    prompt_token_ids: tuple[NonNegativeInt, ...] | None
    identity_sha256: Sha256

    @model_validator(mode="after")
    def validate_event(self) -> Self:
        if self.identity_sha256 != sha256_identity(
            self, omit_fields=frozenset({"identity_sha256"})
        ):
            raise ValueError("cancellation generation-event identity does not reconstruct")
        return self


def replay_stage2_wire_capture(
    capture: Stage2RequestWireCapture,
    identity_chain: object,
) -> tuple[Stage2RequestEvidence, tuple[Stage2ReplayedSSEEvent, ...]]:
    """Replay exact retained chunks through the runtime adapter's incremental SSE parser."""

    from llm_inference_systems.stage2_contracts import RequestIdentityChain

    chain = RequestIdentityChain.model_validate(identity_chain)
    fixture_identity = (
        capture.provenance.fixture_identity_sha256
        if isinstance(capture.provenance, FixtureWireCaptureProvenance)
        else None
    )
    current_frame_offsets: tuple[int, ...] = ()
    current_frame_index = 0

    def frame_clock() -> int:
        nonlocal current_frame_index
        if current_frame_index >= len(current_frame_offsets):
            raise Stage2ProtocolError("raw wire capture lacks a completed SSE-frame observation")
        value = current_frame_offsets[current_frame_index]
        current_frame_index += 1
        return value

    validator = Stage2StreamValidator(
        external_base_id=capture.external_request_id,
        sent_prompt_token_ids=capture.request_body.canonical_request.prompt,
        dispatch_offset_ns=capture.request_body.transmission_offset_ns,
        fixture_identity_sha256=fixture_identity,
        frame_clock=frame_clock,
    )
    try:
        validator.accept_response_headers(
            capture.response_headers.effective("x-request-id"),
            capture.response_headers.observation_offset_ns,
        )
        for chunk in capture.response_body_chunks:
            expected_offsets = chunk.completed_sse_frame_observation_offsets_ns
            current_frame_offsets = expected_offsets[1:]
            current_frame_index = 0
            event_start = len(validator.parsed_sse_events)
            validator.feed(chunk.exact_bytes(), chunk.observation_offset_ns)
            observed_offsets = tuple(
                event.observation_offset_ns for event in validator.parsed_sse_events[event_start:]
            )
            if observed_offsets != expected_offsets or current_frame_index != len(
                current_frame_offsets
            ):
                raise Stage2ProtocolError(
                    "completed SSE-frame observations do not match parser replay"
                )
        evidence = validator.close_transport(
            capture.transport_close.close_observation_offset_ns,
            identity_chain=chain,
        )
    except (Stage2ProtocolError, ValueError) as error:
        raise Stage2ExperimentError("retained HTTP wire evidence failed parser replay") from error
    if capture.transport_close.request_identity_chain_sha256 != sha256_identity(chain):
        raise Stage2ExperimentError("transport close differs from the request identity chain")
    parsed_events: list[Stage2ReplayedSSEEvent] = []
    for event in validator.parsed_sse_events:
        values: dict[str, object] = {
            "ordinal": event.ordinal,
            "observation_offset_ns": event.observation_offset_ns,
            "kind": event.kind,
            "data": event.data,
            "comments": event.comments,
        }
        values["identity_sha256"] = sha256_identity(values)
        parsed_events.append(Stage2ReplayedSSEEvent.model_validate(values))
    return evidence, tuple(parsed_events)


class Stage2CancellationParserReplay(StrictModel):
    external_request_id: Identifier
    response_body_id: Identifier
    replayed_events: tuple[Stage2ReplayedSSEEvent, ...] = Field(min_length=1)
    generation_events: tuple[Stage2CancellationGenerationEvent, ...] = Field(min_length=1)
    all_output_token_ids: tuple[NonNegativeInt, ...] = Field(min_length=1)
    first_generation_delivery: FirstGenerationDeliveryEvidence
    pending_bytes_base64: str
    pending_byte_count: NonNegativeInt
    pending_bytes_sha256: Sha256
    parser_pending_bytes_base64: str
    parser_pending_byte_count: NonNegativeInt
    parser_pending_bytes_sha256: Sha256
    parser_state_at_close: Literal[
        "AT_SSE_FRAME_BOUNDARY",
        "INCOMPLETE_TRAILING_SSE_BYTES",
    ]
    raw_response_body_inventory_sha256: Sha256
    generation_terminal_observed: Literal[False]
    usage_terminal_observed: Literal[False]
    done_terminal_observed: Literal[False]
    clean_transport_eof_observed: Literal[False]
    token_observation_metrics_available: Literal[False]
    token_observation_metrics_unavailable_reason: Literal["CANCELLATION_PROBE_NOT_MEASURED"]
    performance_measurement_eligible: Literal[False]
    identity_sha256: Sha256

    @model_validator(mode="after")
    def validate_replay(self) -> Self:
        raw_pending = _decode_canonical_base64(
            self.pending_bytes_base64, label="cancellation raw pending bytes"
        )
        parser_pending = _decode_canonical_base64(
            self.parser_pending_bytes_base64,
            label="cancellation normalized parser-pending bytes",
        )
        expected_state = (
            "INCOMPLETE_TRAILING_SSE_BYTES" if parser_pending else "AT_SSE_FRAME_BOUNDARY"
        )
        generation_ordinals = tuple(event.sse_event_ordinal for event in self.generation_events)
        replayed_by_ordinal = {event.ordinal: event for event in self.replayed_events}
        flattened_ids = tuple(
            token_id for event in self.generation_events for token_id in event.output_token_ids
        )
        first_event = self.generation_events[0]
        first_delivery = self.first_generation_delivery
        if (
            tuple(event.ordinal for event in self.replayed_events)
            != tuple(range(len(self.replayed_events)))
            or generation_ordinals != tuple(sorted(generation_ordinals))
            or len(generation_ordinals) != len(set(generation_ordinals))
            or any(
                ordinal not in replayed_by_ordinal or replayed_by_ordinal[ordinal].kind != "data"
                for ordinal in generation_ordinals
            )
            or flattened_ids != self.all_output_token_ids
            or first_delivery.external_request_id != self.external_request_id
            or first_delivery.response_body_id != self.response_body_id
            or first_delivery.generation_event_ordinal != first_event.sse_event_ordinal
            or first_delivery.observation_offset_ns > first_event.observation_offset_ns
            or first_delivery.output_token_ids != first_event.output_token_ids
            or len(raw_pending) != self.pending_byte_count
            or hashlib.sha256(raw_pending).hexdigest() != self.pending_bytes_sha256
            or parser_pending != raw_pending.replace(b"\r\n", b"\n")
            or len(parser_pending) != self.parser_pending_byte_count
            or hashlib.sha256(parser_pending).hexdigest() != self.parser_pending_bytes_sha256
            or self.parser_state_at_close != expected_state
        ):
            raise ValueError("cancellation delivery, event, token, or pending-byte replay differs")
        if self.identity_sha256 != sha256_identity(
            self, omit_fields=frozenset({"identity_sha256"})
        ):
            raise ValueError("cancellation parser-replay identity does not reconstruct")
        return self


class Stage2CancellationClientCloseCapture(StrictModel):
    external_request_id: Identifier
    close_classification: Literal["INTENTIONAL_CLIENT_CLOSE_AFTER_FIRST_GENERATION_DELIVERY"]
    close_observation_offset_ns: NonNegativeInt
    response_close_completion_observation_offset_ns: NonNegativeInt
    response_close_completed: Literal[True]
    client_stream_context_exited: Literal[True]
    post_close_byte_count: Literal[0]
    post_close_event_count: Literal[0]
    raw_response_body_inventory_sha256: Sha256
    request_identity_chain_sha256: Sha256
    parser_replay_identity_sha256: Sha256
    identity_sha256: Sha256

    @model_validator(mode="after")
    def validate_close(self) -> Self:
        if self.response_close_completion_observation_offset_ns < self.close_observation_offset_ns:
            raise ValueError("response close completed before its invocation")
        if self.identity_sha256 != sha256_identity(
            self, omit_fields=frozenset({"identity_sha256"})
        ):
            raise ValueError("intentional client-close identity does not reconstruct")
        return self


class Stage2CancellationWireCapture(StrictModel):
    schema_version: Literal["0.3.0"]
    repetition_index: Literal[1, 2, 3]
    external_request_id: Identifier
    provenance: Stage2WireCaptureProvenance
    request_body: Stage2CancellationExactRequestBodyCapture
    http_exchange: Stage2HTTPExchangeCapture
    response_body_chunks: tuple[Stage2RawResponseBodyChunk, ...] = Field(min_length=1)
    parser_replay: Stage2CancellationParserReplay
    intentional_client_close: Stage2CancellationClientCloseCapture
    request_identity: RequestIdentityAttestation
    raw_log_capture: RawLogCapture
    raw_log_capture_sha256: Sha256
    identity_sha256: Sha256

    @model_validator(mode="after")
    def validate_capture(self) -> Self:
        exchange = self.http_exchange
        chain = self.request_identity.identity_chain
        chunks = self.response_body_chunks
        request = self.request_body
        if (
            self.external_request_id != request.canonical_request.request_id
            or self.external_request_id != chain.external_base_id
            or exchange.exchange_purpose != "CANCELLATION"
            or exchange.repetition_index != self.repetition_index
            or exchange.evidence_unit_id != "cancellation-probe"
            or exchange.external_request_id != self.external_request_id
            or exchange.provenance != self.provenance
            or exchange.request_body_byte_count != request.byte_count
            or exchange.request_body_sha256 != request.sha256
            or exchange.request_body_transmission_observation_offset_ns
            != request.transmission_offset_ns
            or request.transmission_offset_ns != exchange.request_headers.observation_offset_ns
            or self.raw_log_capture_sha256 != self.raw_log_capture.raw_bytes_sha256
            or self.raw_log_capture.source_stream_id
            != f"{exchange.server_process_identity}-raw-log"
        ):
            raise ValueError("cancellation HTTP exchange differs from exact request identity")
        try:
            reconstructed_chain = correlate_request_logs(
                self.external_request_id,
                self.raw_log_capture.records,
                cancellation=True,
            )
        except Stage2ProtocolError as error:
            raise ValueError("cancellation raw-log capture does not correlate uniquely") from error
        if reconstructed_chain != chain:
            raise ValueError("cancellation raw-log capture differs from request identity")
        if tuple(chunk.ordinal for chunk in chunks) != tuple(range(len(chunks))):
            raise ValueError("cancellation raw chunks are missing, duplicated, or reordered")
        if tuple(chunk.observation_offset_ns for chunk in chunks) != tuple(
            sorted(chunk.observation_offset_ns for chunk in chunks)
        ):
            raise ValueError("cancellation raw chunk observations are reordered")
        previous_frame_offset: int | None = None
        for chunk in chunks:
            if (
                previous_frame_offset is not None
                and chunk.observation_offset_ns < previous_frame_offset
            ):
                raise ValueError("cancellation raw chunks overlap prior SSE-frame observations")
            if chunk.completed_sse_frame_observation_offsets_ns:
                previous_frame_offset = chunk.completed_sse_frame_observation_offsets_ns[-1]
        if any(
            (chunk.repetition_index, chunk.case_id, chunk.external_request_id)
            != (self.repetition_index, "cancellation-probe", self.external_request_id)
            for chunk in chunks
        ):
            raise ValueError("cancellation raw chunk identity differs")
        fixture = isinstance(self.provenance, FixtureWireCaptureProvenance)
        expected_chunk_source = (
            "TEST_FIXTURE_ONLY_CPU_SCRIPTED_HTTP"
            if fixture
            else "FUTURE_RUNTIME_COLLECTOR_HTTP_BODY"
        )
        if any(
            chunk.source_capture_provenance != expected_chunk_source
            or chunk.inventory_manifest_path != "raw/cancellation/client-wire.json"
            for chunk in chunks
        ):
            raise ValueError("cancellation raw chunk provenance differs")
        raw_body = b"".join(chunk.exact_bytes() for chunk in chunks)
        inventory_sha = sha256_identity(chunks)
        body_completion_offset = max(
            offset
            for chunk in chunks
            for offset in (
                chunk.observation_offset_ns,
                *chunk.completed_sse_frame_observation_offsets_ns,
            )
        )
        close = self.intentional_client_close
        if (
            exchange.response_body_byte_count != len(raw_body)
            or exchange.response_body_sha256 != hashlib.sha256(raw_body).hexdigest()
            or exchange.response_body_inventory_sha256 != inventory_sha
            or exchange.response_body_completion_observation_offset_ns != body_completion_offset
            or exchange.transport_terminal_observation_offset_ns
            != close.close_observation_offset_ns
            or exchange.transport_terminal_classification != close.close_classification
            or self.parser_replay.raw_response_body_inventory_sha256 != inventory_sha
            or close.raw_response_body_inventory_sha256 != inventory_sha
            or close.external_request_id != self.external_request_id
            or close.request_identity_chain_sha256 != self.request_identity.identity_sha256
            or close.parser_replay_identity_sha256 != self.parser_replay.identity_sha256
        ):
            raise ValueError("cancellation exchange, replay, close, and chunk inventory differ")
        if chain.external_abort_log is None or chain.internal_abort_log is None:
            raise ValueError("cancellation wire requires external and internal abort logs")
        first_delivery = self.parser_replay.first_generation_delivery
        first_body = chunks[0].observation_offset_ns
        if (
            not (
                request.transmission_offset_ns
                < exchange.response_header_observation_offset_ns
                < first_body
                <= first_delivery.observation_offset_ns
                <= close.close_observation_offset_ns
                <= chain.internal_abort_log.observation_offset_ns
                <= chain.external_abort_log.observation_offset_ns
            )
            or any(
                offset > close.close_observation_offset_ns
                for chunk in chunks
                for offset in (
                    chunk.observation_offset_ns,
                    *chunk.completed_sse_frame_observation_offsets_ns,
                )
            )
            or first_delivery.body_chunk_ordinal >= len(chunks)
            or chunks[first_delivery.body_chunk_ordinal].observation_offset_ns
            != first_delivery.observation_offset_ns
        ):
            raise ValueError("cancellation HTTP/SSE/close/abort observation order differs")
        reconstructed = replay_stage2_cancellation_wire_capture(self)
        if reconstructed != self.parser_replay:
            raise ValueError("cancellation parser replay does not reconstruct from raw chunks")
        if self.identity_sha256 != sha256_identity(
            self, omit_fields=frozenset({"identity_sha256"})
        ):
            raise ValueError("cancellation wire identity does not reconstruct")
        return self


def replay_stage2_cancellation_wire_capture(
    capture: Stage2CancellationWireCapture,
) -> Stage2CancellationParserReplay:
    """Replay retained partial SSE bytes through the intentional client close."""

    validator = Stage2CancellationStreamCapture(
        external_base_id=capture.external_request_id,
        sent_prompt_token_ids=capture.request_body.canonical_request.prompt,
        dispatch_offset_ns=capture.request_body.transmission_offset_ns,
    )
    try:
        validator.accept_response_headers(
            capture.http_exchange.response_headers.effective("x-request-id"),
            capture.http_exchange.response_headers.observation_offset_ns,
        )
        for chunk in capture.response_body_chunks:
            event_start = len(validator.parsed_sse_events)
            validator.feed(chunk.exact_bytes(), chunk.observation_offset_ns)
            observed_offsets = tuple(
                event.observation_offset_ns for event in validator.parsed_sse_events[event_start:]
            )
            if observed_offsets != chunk.completed_sse_frame_observation_offsets_ns:
                raise Stage2ProtocolError(
                    "cancellation completed-frame inventory differs from parser replay"
                )
        validator.close(capture.intentional_client_close.close_observation_offset_ns)
        validator.complete_transport_close(
            capture.intentional_client_close.response_close_completion_observation_offset_ns
        )
    except Stage2ProtocolError as error:
        raise ValueError("cancellation retained stream failed parser replay") from error
    raw_body = b"".join(chunk.exact_bytes() for chunk in capture.response_body_chunks)
    normalized_body = raw_body.replace(b"\r\n", b"\n")
    normalized_segments = normalized_body.split(b"\n\n")
    if any(segment == b"" for segment in normalized_segments[:-1]):
        raise ValueError("cancellation capture contains an empty or unobserved SSE frame")
    replayed: list[Stage2ReplayedSSEEvent] = []
    for event in validator.parsed_sse_events:
        event_values: dict[str, object] = {
            "ordinal": event.ordinal,
            "observation_offset_ns": event.observation_offset_ns,
            "kind": event.kind,
            "data": event.data,
            "comments": event.comments,
        }
        event_values["identity_sha256"] = sha256_identity(event_values)
        replayed.append(Stage2ReplayedSSEEvent.model_validate(event_values))
    generation_events: list[Stage2CancellationGenerationEvent] = []
    for generation in validator.generation_events:
        generation_values: dict[str, object] = {
            "sse_event_ordinal": generation.sse_event_ordinal,
            "observation_offset_ns": generation.observation_offset_ns,
            "output_token_ids": generation.output_token_ids,
            "text": generation.text,
            "prompt_token_ids": generation.prompt_token_ids,
        }
        generation_values["identity_sha256"] = sha256_identity(generation_values)
        generation_events.append(
            Stage2CancellationGenerationEvent.model_validate(generation_values)
        )
    retained_delivery = validator.first_generation_delivery
    if retained_delivery is None:
        raise ValueError("cancellation replay contains no generation delivery")
    first_delivery = FirstGenerationDeliveryEvidence(
        external_request_id=capture.external_request_id,
        response_body_id=f"cmpl-{capture.external_request_id}",
        generation_event_ordinal=retained_delivery.generation_event_ordinal,
        body_chunk_ordinal=retained_delivery.body_chunk_ordinal,
        observation_offset_ns=retained_delivery.observation_offset_ns,
        output_token_ids=retained_delivery.output_token_ids,
    )
    pending = validator.pending_bytes
    parser_pending = validator.parser_pending_bytes
    values: dict[str, object] = {
        "external_request_id": capture.external_request_id,
        "response_body_id": f"cmpl-{capture.external_request_id}",
        "replayed_events": tuple(replayed),
        "generation_events": tuple(generation_events),
        "all_output_token_ids": tuple(
            token_id for event in generation_events for token_id in event.output_token_ids
        ),
        "first_generation_delivery": first_delivery,
        "pending_bytes_base64": base64.b64encode(pending).decode("ascii"),
        "pending_byte_count": len(pending),
        "pending_bytes_sha256": hashlib.sha256(pending).hexdigest(),
        "parser_pending_bytes_base64": base64.b64encode(parser_pending).decode("ascii"),
        "parser_pending_byte_count": len(parser_pending),
        "parser_pending_bytes_sha256": hashlib.sha256(parser_pending).hexdigest(),
        "parser_state_at_close": (validator.parser_state_at_close),
        "raw_response_body_inventory_sha256": sha256_identity(capture.response_body_chunks),
        "generation_terminal_observed": False,
        "usage_terminal_observed": False,
        "done_terminal_observed": False,
        "clean_transport_eof_observed": False,
        "token_observation_metrics_available": False,
        "token_observation_metrics_unavailable_reason": "CANCELLATION_PROBE_NOT_MEASURED",
        "performance_measurement_eligible": False,
    }
    values["identity_sha256"] = sha256_identity(values)
    return Stage2CancellationParserReplay.model_validate(values)


class Stage2RequestRawEvidence(StrictModel):
    http_exchange: ManifestBoundFile
    request_body: ManifestBoundFile
    request_headers: ManifestBoundFile
    response_headers: ManifestBoundFile
    raw_response_body: ManifestBoundFile
    parsed_sse_events: ManifestBoundFile
    terminal_boundary: ManifestBoundFile
    server_logs: ManifestBoundFile
    server_metrics: ManifestBoundFile
    lifecycle: ManifestBoundFile
    token_usage_reconciliation: ManifestBoundFile

    @model_validator(mode="after")
    def validate_distinct_files(self) -> Self:
        paths = tuple(getattr(self, field).path for field in REQUEST_EVIDENCE_FIELDS)
        if len(paths) != len(set(paths)) or len({path.casefold() for path in paths}) != len(paths):
            raise ValueError("measured-request raw evidence paths must be distinct")
        return self

    def files(self) -> tuple[ManifestBoundFile, ...]:
        return tuple(getattr(self, field) for field in REQUEST_EVIDENCE_FIELDS)


class Stage2RequestLifecycle(StrictModel):
    dispatch_offset_ns: NonNegativeInt
    terminal_offset_ns: PositiveInt
    measurement_phase_start_ns: NonNegativeInt
    measurement_phase_end_ns: PositiveInt
    measurement_phase_identity_sha256: Sha256

    @model_validator(mode="after")
    def validate_interval(self) -> Self:
        if not (
            self.measurement_phase_start_ns
            <= self.dispatch_offset_ns
            < self.terminal_offset_ns
            <= self.measurement_phase_end_ns
        ):
            raise ValueError("measured lifecycle must be a positive interval inside its phase")
        return self


class Stage2MetricAvailability(StrictModel):
    server_ttft_available: bool
    server_generation_time_available: bool
    server_queue_time_available: bool
    server_mean_itl_available: bool
    server_tokens_per_second_available: bool
    client_generation_tpot_available: bool
    token_observation_itl_available: bool
    server_ttft_advancement_allowed: bool
    server_generation_time_advancement_allowed: bool
    server_queue_time_advancement_allowed: bool
    server_mean_itl_advancement_allowed: bool
    server_tokens_per_second_advancement_allowed: bool
    client_generation_tpot_advancement_allowed: bool
    token_observation_itl_advancement_allowed: bool

    @model_validator(mode="after")
    def validate_eligibility(self) -> Self:
        pairs = (
            (self.server_ttft_available, self.server_ttft_advancement_allowed),
            (
                self.server_generation_time_available,
                self.server_generation_time_advancement_allowed,
            ),
            (self.server_queue_time_available, self.server_queue_time_advancement_allowed),
            (self.server_mean_itl_available, self.server_mean_itl_advancement_allowed),
            (
                self.server_tokens_per_second_available,
                self.server_tokens_per_second_advancement_allowed,
            ),
            (
                self.client_generation_tpot_available,
                self.client_generation_tpot_advancement_allowed,
            ),
            (
                self.token_observation_itl_available,
                self.token_observation_itl_advancement_allowed,
            ),
        )
        if any(available != allowed for available, allowed in pairs):
            raise ValueError("metric advancement eligibility must be derived from availability")
        return self


def derive_metric_availability(
    metrics: Stage2PerRequestMetrics,
    evidence: Stage2RequestEvidence,
) -> Stage2MetricAvailability:
    values = (
        metrics.time_to_first_token_ms is not None,
        metrics.generation_time_ms is not None,
        metrics.queue_time_ms is not None,
        metrics.mean_itl_ms is not None,
        metrics.tokens_per_second is not None,
        evidence.client_generation_tpot.value_ns is not None,
        evidence.token_observation_itl.value_ns is not None,
    )
    return Stage2MetricAvailability(
        server_ttft_available=values[0],
        server_generation_time_available=values[1],
        server_queue_time_available=values[2],
        server_mean_itl_available=values[3],
        server_tokens_per_second_available=values[4],
        client_generation_tpot_available=values[5],
        token_observation_itl_available=values[6],
        server_ttft_advancement_allowed=values[0],
        server_generation_time_advancement_allowed=values[1],
        server_queue_time_advancement_allowed=values[2],
        server_mean_itl_advancement_allowed=values[3],
        server_tokens_per_second_advancement_allowed=values[4],
        client_generation_tpot_advancement_allowed=values[5],
        token_observation_itl_advancement_allowed=values[6],
    )


class Stage2MetricAvailabilitySummary(StrictModel):
    total_request_count: Annotated[int, Field(ge=1, le=48)]
    server_ttft_available_count: NonNegativeInt
    server_generation_time_available_count: NonNegativeInt
    server_queue_time_available_count: NonNegativeInt
    server_mean_itl_available_count: NonNegativeInt
    server_tokens_per_second_available_count: NonNegativeInt
    client_generation_tpot_available_count: NonNegativeInt
    token_observation_itl_available_count: NonNegativeInt
    server_ttft_advancement_allowed: bool
    server_generation_time_advancement_allowed: bool
    server_queue_time_advancement_allowed: bool
    server_mean_itl_advancement_allowed: bool
    server_tokens_per_second_advancement_allowed: bool
    client_generation_tpot_advancement_allowed: bool
    token_observation_itl_advancement_allowed: bool

    @model_validator(mode="after")
    def validate_counts(self) -> Self:
        count_fields = (
            "server_ttft_available_count",
            "server_generation_time_available_count",
            "server_queue_time_available_count",
            "server_mean_itl_available_count",
            "server_tokens_per_second_available_count",
            "client_generation_tpot_available_count",
            "token_observation_itl_available_count",
        )
        allowed_fields = (
            "server_ttft_advancement_allowed",
            "server_generation_time_advancement_allowed",
            "server_queue_time_advancement_allowed",
            "server_mean_itl_advancement_allowed",
            "server_tokens_per_second_advancement_allowed",
            "client_generation_tpot_advancement_allowed",
            "token_observation_itl_advancement_allowed",
        )
        for count_field, allowed_field in zip(count_fields, allowed_fields, strict=True):
            count = getattr(self, count_field)
            if count > self.total_request_count:
                raise ValueError("metric-availability count exceeds the request population")
            if getattr(self, allowed_field) != (count == self.total_request_count):
                raise ValueError("metric advancement must require complete population availability")
        return self


def derive_metric_availability_summary(
    requests: tuple[Stage2MeasuredRequestAttestation, ...],
) -> Stage2MetricAvailabilitySummary:
    count = len(requests)
    if count not in {16, 48}:
        raise ValueError("availability summaries apply only to 16 or 48 measured requests")
    fields = (
        "server_ttft_available",
        "server_generation_time_available",
        "server_queue_time_available",
        "server_mean_itl_available",
        "server_tokens_per_second_available",
        "client_generation_tpot_available",
        "token_observation_itl_available",
    )
    counts = {
        field: sum(bool(getattr(request.metric_availability, field)) for request in requests)
        for field in fields
    }
    return Stage2MetricAvailabilitySummary(
        total_request_count=count,
        server_ttft_available_count=counts[fields[0]],
        server_generation_time_available_count=counts[fields[1]],
        server_queue_time_available_count=counts[fields[2]],
        server_mean_itl_available_count=counts[fields[3]],
        server_tokens_per_second_available_count=counts[fields[4]],
        client_generation_tpot_available_count=counts[fields[5]],
        token_observation_itl_available_count=counts[fields[6]],
        server_ttft_advancement_allowed=counts[fields[0]] == count,
        server_generation_time_advancement_allowed=counts[fields[1]] == count,
        server_queue_time_advancement_allowed=counts[fields[2]] == count,
        server_mean_itl_advancement_allowed=counts[fields[3]] == count,
        server_tokens_per_second_advancement_allowed=counts[fields[4]] == count,
        client_generation_tpot_advancement_allowed=counts[fields[5]] == count,
        token_observation_itl_advancement_allowed=counts[fields[6]] == count,
    )


class Stage2MeasuredRequestAttestation(StrictModel):
    schema_version: Literal["0.3.0"]
    repetition_index: Literal[1, 2, 3]
    case_id: Identifier
    external_request_id: Identifier
    request_evidence: Stage2RequestEvidence
    request_identity: RequestIdentityAttestation
    wire_capture: Stage2RequestWireCapture
    lifecycle: Stage2RequestLifecycle
    raw_evidence: Stage2RequestRawEvidence
    metric_availability: Stage2MetricAvailability
    repetition_manifest_sha256: Sha256
    attestation_sha256: Sha256

    @model_validator(mode="after")
    def validate_request(self) -> Self:
        if self.case_id not in STAGE2_EXPERIMENT_CASE_IDS:
            raise ValueError("measured request uses an undeclared Stage 2 case ID")
        evidence = self.request_evidence
        chain = self.request_identity.identity_chain
        if (
            self.external_request_id != evidence.external_request_id
            or chain.external_base_id != self.external_request_id
            or chain.response_body_id != evidence.response_request_id
            or chain.serving_item_id != evidence.serving_item_request_id
            or chain.internal_engine_id != evidence.internal_engine_request_id
            or self.request_identity.identity_sha256 != evidence.request_identity_chain_sha256
        ):
            raise ValueError("measured request identity chain does not reconcile")
        replayed, _ = replay_stage2_wire_capture(self.wire_capture, chain)
        if replayed != evidence:
            raise ValueError("typed request evidence differs from exact wire replay")
        if (
            self.wire_capture.repetition_index != self.repetition_index
            or self.wire_capture.case_id != self.case_id
            or self.wire_capture.external_request_id != self.external_request_id
        ):
            raise ValueError("wire capture is detached from the measured request")
        if (
            self.lifecycle.dispatch_offset_ns != evidence.timing.dispatch_offset_ns
            or self.lifecycle.terminal_offset_ns != evidence.timing.transport_terminal_offset_ns
        ):
            raise ValueError("measured lifecycle differs from parsed terminal evidence")
        expected_availability = derive_metric_availability(
            evidence.server_per_request_metrics,
            evidence,
        )
        if self.metric_availability != expected_availability:
            raise ValueError("metric availability was not derived from request evidence")
        if self.attestation_sha256 != sha256_identity(
            self, omit_fields=frozenset({"attestation_sha256"})
        ):
            raise ValueError("measured-request attestation identity does not reconstruct")
        return self


def build_request_raw_evidence_payloads(
    *,
    wire_capture: Stage2RequestWireCapture,
    request_identity: RequestIdentityAttestation,
    lifecycle: Stage2RequestLifecycle,
) -> dict[str, bytes]:
    """Serialize collector/fixture wire captures and replay-derived evidence.

    This helper accepts lossless wire records, never a caller-supplied parsed request
    object.  ``Stage2RequestEvidence`` is produced only by replay below.
    """

    evidence, replayed_events = replay_stage2_wire_capture(
        wire_capture, request_identity.identity_chain
    )
    repetition_index = wire_capture.repetition_index
    case_id = wire_capture.case_id
    external_request_id = wire_capture.external_request_id
    evidence_scope = wire_capture.provenance.evidence_scope
    if (
        lifecycle.dispatch_offset_ns != evidence.timing.dispatch_offset_ns
        or lifecycle.terminal_offset_ns != evidence.timing.transport_terminal_offset_ns
    ):
        raise Stage2ExperimentError("lifecycle differs from wire-replayed terminal evidence")
    common = {
        "schema_version": "0.3.0",
        "evidence_scope": evidence_scope,
        "wire_provenance": wire_capture.provenance,
        "wire_capture_identity_sha256": wire_capture.identity_sha256,
        "repetition_index": repetition_index,
        "case_id": case_id,
        "external_request_id": external_request_id,
        "measurement_phase": {
            "started_offset_ns": lifecycle.measurement_phase_start_ns,
            "ended_offset_ns": lifecycle.measurement_phase_end_ns,
            "identity_sha256": lifecycle.measurement_phase_identity_sha256,
        },
    }
    content: dict[str, object] = {
        "http_exchange": wire_capture.http_exchange,
        "request_body": wire_capture.request_body,
        "request_headers": wire_capture.request_headers,
        "response_headers": wire_capture.response_headers,
        "raw_response_body": {
            "response_body_chunks": wire_capture.response_body_chunks,
            "transport_close": wire_capture.transport_close,
        },
        "parsed_sse_events": {"events": replayed_events},
        "terminal_boundary": {
            "timing": evidence.timing,
            "transport_close": wire_capture.transport_close,
        },
        "server_logs": request_identity,
        "server_metrics": evidence.server_per_request_metrics,
        "lifecycle": lifecycle,
        "token_usage_reconciliation": {"typed_request_evidence": evidence},
    }
    return {
        field: canonical_json_bytes({**common, "evidence_kind": field, "content": content[field]})
        + b"\n"
        for field in REQUEST_EVIDENCE_FIELDS
    }


def request_raw_evidence_payloads(
    request: Stage2MeasuredRequestAttestation,
    evidence_scope: Stage2EvidenceScope,
) -> dict[str, bytes]:
    if evidence_scope is not Stage2EvidenceScope.TEST_FIXTURE_ONLY or not isinstance(
        request.wire_capture.provenance, FixtureWireCaptureProvenance
    ):
        raise Stage2ExperimentError(
            "typed fixture helper is structurally prohibited from producing live wire evidence"
        )
    return build_request_raw_evidence_payloads(
        wire_capture=request.wire_capture,
        request_identity=request.request_identity,
        lifecycle=request.lifecycle,
    )


_RAW_REQUEST_EVIDENCE_FIELDS: Final = (
    "http_exchange",
    "request_body",
    "request_headers",
    "response_headers",
    "raw_response_body",
    "server_logs",
)


def _derive_request_from_raw_wire(
    raw_payloads: dict[str, bytes],
) -> tuple[
    Literal[1, 2, 3],
    str,
    str,
    Stage2RequestEvidence,
    RequestIdentityAttestation,
    Stage2RequestLifecycle,
    Stage2RequestWireCapture,
    dict[str, bytes],
]:
    if set(raw_payloads) != set(_RAW_REQUEST_EVIDENCE_FIELDS):
        raise Stage2ExperimentError("measured request raw wire set is incomplete")
    parsed: dict[str, dict[str, object]] = {}
    for field, data in raw_payloads.items():
        try:
            value = json.loads(data)
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise Stage2ExperimentError("measured request wire evidence is not JSON") from error
        if not isinstance(value, dict) or data != canonical_json_bytes(value) + b"\n":
            raise Stage2ExperimentError("measured request wire evidence is not canonical JSON")
        if (
            value.get("schema_version") != "0.3.0"
            or value.get("evidence_kind") != field
            or not isinstance(value.get("content"), dict)
            or not isinstance(value.get("measurement_phase"), dict)
        ):
            raise Stage2ExperimentError("measured request wire envelope differs")
        parsed[field] = value
    metadata = tuple(
        (
            value.get("evidence_scope"),
            value.get("wire_provenance"),
            value.get("wire_capture_identity_sha256"),
            value.get("repetition_index"),
            value.get("case_id"),
            value.get("external_request_id"),
            value.get("measurement_phase"),
        )
        for value in parsed.values()
    )
    if any(item != metadata[0] for item in metadata[1:]):
        raise Stage2ExperimentError("measured request wire envelope identities differ")
    (
        evidence_scope,
        provenance_raw,
        wire_identity,
        repetition_index,
        case_id,
        external_request_id,
        phase_raw,
    ) = metadata[0]
    if (
        evidence_scope not in set(Stage2EvidenceScope)
        or repetition_index not in {1, 2, 3}
        or not isinstance(case_id, str)
        or not isinstance(external_request_id, str)
        or not isinstance(wire_identity, str)
        or not isinstance(phase_raw, dict)
    ):
        raise Stage2ExperimentError("measured request wire envelope identity is invalid")

    def content(field: str) -> dict[str, object]:
        return cast(dict[str, object], parsed[field]["content"])

    try:
        provenance = (
            FixtureWireCaptureProvenance.model_validate(provenance_raw)
            if isinstance(provenance_raw, dict)
            and provenance_raw.get("capture_kind") == "FIXTURE_CONSTRUCTOR"
            else CollectorWireCaptureProvenance.model_validate(provenance_raw)
        )
        request_body = Stage2ExactRequestBodyCapture.model_validate_json(
            canonical_json_bytes(content("request_body"))
        )
        http_exchange = Stage2HTTPExchangeCapture.model_validate_json(
            canonical_json_bytes(content("http_exchange"))
        )
        request_headers = Stage2OrderedHeadersCapture.model_validate_json(
            canonical_json_bytes(content("request_headers"))
        )
        response_headers = Stage2OrderedHeadersCapture.model_validate_json(
            canonical_json_bytes(content("response_headers"))
        )
        body_content = content("raw_response_body")
        chunks = tuple(
            Stage2RawResponseBodyChunk.model_validate_json(canonical_json_bytes(item))
            for item in cast(list[object], body_content["response_body_chunks"])
        )
        transport_close = Stage2TransportCloseCapture.model_validate_json(
            canonical_json_bytes(body_content["transport_close"])
        )
        request_identity = RequestIdentityAttestation.model_validate_json(
            canonical_json_bytes(content("server_logs"))
        )
        capture = Stage2RequestWireCapture(
            schema_version="0.3.0",
            repetition_index=repetition_index,
            case_id=case_id,
            external_request_id=external_request_id,
            provenance=provenance,
            http_exchange=http_exchange,
            request_body=request_body,
            request_headers=request_headers,
            response_headers=response_headers,
            response_body_chunks=chunks,
            transport_close=transport_close,
            identity_sha256=wire_identity,
        )
    except (KeyError, TypeError, ValueError) as error:
        raise Stage2ExperimentError("measured request raw wire capture is invalid") from error
    evidence, _ = replay_stage2_wire_capture(capture, request_identity.identity_chain)
    try:
        lifecycle = Stage2RequestLifecycle(
            dispatch_offset_ns=evidence.timing.dispatch_offset_ns,
            terminal_offset_ns=evidence.timing.transport_terminal_offset_ns,
            measurement_phase_start_ns=phase_raw["started_offset_ns"],
            measurement_phase_end_ns=phase_raw["ended_offset_ns"],
            measurement_phase_identity_sha256=phase_raw["identity_sha256"],
        )
    except (KeyError, ValueError) as error:
        raise Stage2ExperimentError("measured request phase provenance is invalid") from error
    expected = build_request_raw_evidence_payloads(
        wire_capture=capture,
        request_identity=request_identity,
        lifecycle=lifecycle,
    )
    return (
        repetition_index,
        case_id,
        external_request_id,
        evidence,
        request_identity,
        lifecycle,
        capture,
        expected,
    )


def reconstruct_request_from_raw_evidence(
    raw_payloads: dict[str, bytes],
    evidence_scope: Stage2EvidenceScope,
) -> tuple[
    Literal[1, 2, 3],
    str,
    str,
    Stage2RequestEvidence,
    RequestIdentityAttestation,
    Stage2RequestLifecycle,
    Stage2RequestWireCapture,
]:
    """Rebuild typed evidence from five raw wire records and verify five derived records."""

    if set(raw_payloads) != set(REQUEST_EVIDENCE_FIELDS):
        raise Stage2ExperimentError("measured request evidence set is incomplete")
    raw_only = {field: raw_payloads[field] for field in _RAW_REQUEST_EVIDENCE_FIELDS}
    (
        repetition_index,
        case_id,
        external_request_id,
        evidence,
        identity,
        lifecycle,
        capture,
        expected,
    ) = _derive_request_from_raw_wire(raw_only)
    if expected["request_body"] != raw_payloads["request_body"]:
        raise Stage2ExperimentError("exact request-body capture does not reconstruct")
    for field in REQUEST_EVIDENCE_FIELDS:
        if raw_payloads[field] != expected[field]:
            raise Stage2ExperimentError(f"stored {field} differs from exact raw-chunk replay")
    if evidence_scope is not capture.provenance.evidence_scope:
        raise Stage2ExperimentError("request evidence scope differs from wire provenance")
    return (
        repetition_index,
        case_id,
        external_request_id,
        evidence,
        identity,
        lifecycle,
        capture,
    )


def scoped_raw_evidence_bytes(
    *,
    evidence_kind: str,
    evidence_scope: Stage2EvidenceScope,
    content: object,
) -> bytes:
    return (
        canonical_json_bytes(
            {
                "schema_version": "0.3.0",
                "evidence_scope": evidence_scope,
                "evidence_kind": evidence_kind,
                "content": content,
            }
        )
        + b"\n"
    )


def environment_raw_evidence_bytes(
    environment: LinuxEnvironmentManifest,
    evidence_scope: Stage2EvidenceScope,
) -> bytes:
    return scoped_raw_evidence_bytes(
        evidence_kind="resource_environment",
        evidence_scope=evidence_scope,
        content=environment.model_dump(
            mode="python",
            exclude={"environment_evidence_sha256", "identity_sha256"},
        ),
    )


def nvidia_raw_evidence_bytes(
    resources: NvidiaT4ResourceAttestation,
    evidence_scope: Stage2EvidenceScope,
) -> bytes:
    return scoped_raw_evidence_bytes(
        evidence_kind="nvidia_isolation",
        evidence_scope=evidence_scope,
        content=resources.model_dump(
            mode="python",
            exclude={"isolation_evidence_sha256", "identity_sha256"},
        ),
    )


def execution_lock_raw_evidence_bytes(
    execution_lock: RuntimePackageExecutionLockAttestation,
    evidence_scope: Stage2EvidenceScope,
) -> tuple[bytes, bytes]:
    shared = execution_lock.model_dump(
        mode="python",
        exclude={
            "resolver_lock_sha256",
            "installed_distribution_inventory_sha256",
            "identity_sha256",
        },
    )
    return (
        scoped_raw_evidence_bytes(
            evidence_kind="runtime_resolver_lock",
            evidence_scope=evidence_scope,
            content=shared,
        ),
        scoped_raw_evidence_bytes(
            evidence_kind="installed_distribution_inventory",
            evidence_scope=evidence_scope,
            content={
                "installed": execution_lock.installed,
                "packages": tuple(
                    {"package": item.package, "version": item.version}
                    for item in execution_lock.artifacts
                ),
            },
        ),
    )


def snapshot_read_only_raw_evidence_bytes(
    snapshot: ModelTokenizerSnapshotManifest,
    evidence_scope: Stage2EvidenceScope,
) -> bytes:
    return scoped_raw_evidence_bytes(
        evidence_kind="snapshot_read_only_verification",
        evidence_scope=evidence_scope,
        content=snapshot.read_only_transition.model_dump(
            mode="python", exclude={"verification_evidence_sha256"}
        ),
    )


def public_safety_raw_evidence_bytes(
    public_safety: PublicSafetyAttestation,
    evidence_scope: Stage2EvidenceScope,
) -> bytes:
    return scoped_raw_evidence_bytes(
        evidence_kind="public_safety_scan",
        evidence_scope=evidence_scope,
        content={
            "finding_count": public_safety.finding_count,
            "passed": public_safety.passed,
            "scan_inventory_sha256": public_safety.scan_inventory_sha256,
        },
    )


def prometheus_raw_scrape_capture_bytes(capture: PrometheusRawScrapeCapture) -> bytes:
    return canonical_json_bytes(capture) + b"\n"


def prometheus_capture_and_snapshot_from_raw(
    data: bytes,
) -> tuple[PrometheusRawScrapeCapture, PrometheusSnapshot]:
    """Reparse a lossless scrape capture while retaining its collector metadata."""

    try:
        capture = PrometheusRawScrapeCapture.model_validate_json(data)
    except ValueError as error:
        raise Stage2ExperimentError("raw Prometheus scrape capture is invalid") from error
    if data != prometheus_raw_scrape_capture_bytes(capture):
        raise Stage2ExperimentError("raw Prometheus scrape capture is not canonical JSON")
    snapshot = parse_prometheus_snapshot(
        capture.raw_exposition(),
        process_start_id=capture.process_start_id,
        scrape_wall_clock_utc=capture.scrape_wall_clock_utc,
        scrape_monotonic_offset_ns=capture.scrape_monotonic_offset_ns,
    )
    return capture, snapshot


def prometheus_snapshot_from_raw_capture(data: bytes) -> PrometheusSnapshot:
    return prometheus_capture_and_snapshot_from_raw(data)[1]


def validate_prometheus_raw_capture_binding(
    data: bytes,
    *,
    measurement: PrometheusMeasurementAttestation,
    evidence_scope: Stage2EvidenceScope,
    repetition_index: int,
    boundary: Literal["baseline", "final"],
) -> None:
    capture, snapshot = prometheus_capture_and_snapshot_from_raw(data)
    expected_snapshot = (
        measurement.baseline_snapshot if boundary == "baseline" else measurement.final_snapshot
    )
    expected_capture = (
        measurement.baseline_capture if boundary == "baseline" else measurement.final_capture
    )
    if (
        capture != expected_capture
        or capture.boundary != boundary
        or capture.evidence_scope is not evidence_scope
        or capture.repetition_index != repetition_index
        or capture.process_start_id != measurement.server_process_identity
        or snapshot != expected_snapshot
    ):
        raise Stage2ExperimentError(
            "raw Prometheus capture metadata is detached from its repetition attestation"
        )


def cancellation_wire_capture_bytes(capture: Stage2CancellationWireCapture) -> bytes:
    return canonical_json_bytes(capture) + b"\n"


def cancellation_wire_capture_from_raw(data: bytes) -> Stage2CancellationWireCapture:
    try:
        capture = Stage2CancellationWireCapture.model_validate_json(data)
    except ValueError as error:
        raise Stage2ExperimentError("raw cancellation HTTP/SSE wire capture is invalid") from error
    if data != cancellation_wire_capture_bytes(capture):
        raise Stage2ExperimentError("raw cancellation HTTP/SSE wire capture is not canonical JSON")
    if replay_stage2_cancellation_wire_capture(capture) != capture.parser_replay:
        raise Stage2ExperimentError("cancellation raw chunks do not reconstruct parser replay")
    return capture


def cancellation_wire_raw_evidence_bytes(
    repetition: Stage2RepetitionAttestation,
    evidence_scope: Stage2EvidenceScope,
) -> bytes:
    if repetition.cancellation_wire.provenance.evidence_scope is not evidence_scope:
        raise Stage2ExperimentError("cancellation wire evidence scope differs")
    return cancellation_wire_capture_bytes(repetition.cancellation_wire)


def build_cuda_raw_evidence_bytes(
    *,
    repetition_index: Literal[1, 2, 3],
    server_process_identity: str,
    runtime_control_sha256: str,
    environment_resource_identity_sha256: str,
    execution: dict[str, object],
    evidence_scope: Stage2EvidenceScope,
    evidence_path: str,
) -> bytes:
    return scoped_raw_evidence_bytes(
        evidence_kind="cuda_runtime_execution",
        evidence_scope=evidence_scope,
        content={
            "evidence_path": evidence_path,
            "environment_resource_identity_sha256": environment_resource_identity_sha256,
            "execution": execution,
            "repetition_index": repetition_index,
            "runtime_control_sha256": runtime_control_sha256,
            "server_process_identity": server_process_identity,
        },
    )


def cuda_raw_evidence_bytes(
    cuda: Stage2RepetitionCudaAttestation,
    evidence_scope: Stage2EvidenceScope,
    evidence_path: str,
) -> bytes:
    return build_cuda_raw_evidence_bytes(
        repetition_index=cuda.repetition_index,
        server_process_identity=cuda.server_process_identity,
        runtime_control_sha256=cuda.runtime_control_sha256,
        environment_resource_identity_sha256=cuda.environment_resource_identity_sha256,
        execution=cuda.execution.model_dump(
            mode="python", exclude={"raw_execution_evidence_sha256", "identity_sha256"}
        ),
        evidence_scope=evidence_scope,
        evidence_path=evidence_path,
    )


class Stage2RepetitionCudaAttestation(StrictModel):
    schema_version: Literal["0.3.0"]
    repetition_index: Literal[1, 2, 3]
    server_process_identity: Identifier
    runtime_control_sha256: Sha256
    repetition_manifest_sha256: Sha256
    environment_resource_identity_sha256: Sha256
    execution: CudaBackedExecutionAttestation
    raw_evidence_files: tuple[ManifestBoundFile, ...] = Field(min_length=1)
    identity_sha256: Sha256

    @model_validator(mode="after")
    def validate_cuda(self) -> Self:
        if self.execution.server_process_identity != self.server_process_identity:
            raise ValueError("CUDA evidence is not bound to its repetition server process")
        paths = tuple(item.path for item in self.raw_evidence_files)
        if len(paths) != len(set(paths)) or len({path.casefold() for path in paths}) != len(paths):
            raise ValueError("CUDA raw-evidence paths must be unique")
        if self.execution.raw_execution_evidence_sha256 not in {
            item.sha256 for item in self.raw_evidence_files
        }:
            raise ValueError("CUDA execution raw hash is not bound to a retained evidence file")
        if self.identity_sha256 != sha256_identity(
            self, omit_fields=frozenset({"identity_sha256"})
        ):
            raise ValueError("repetition CUDA attestation identity does not reconstruct")
        return self


def _manifest_file_map(manifest: Stage2BundleManifest) -> dict[str, BundleFileEntry]:
    paths = tuple(entry.path for entry in manifest.files)
    if len({path.casefold() for path in paths}) != len(paths):
        raise ValueError("repetition manifest contains a case-collision ambiguity")
    return {entry.path: entry for entry in manifest.files}


def reconstruct_experiment_repetition(raw: dict[str, bytes]) -> dict[str, bytes]:
    """Derive the committed repetition summary solely from retained raw files."""

    paths = tuple(sorted(raw))
    if not paths or any(not path.startswith("raw/") for path in paths):
        raise Stage2ExperimentError("repetition reconstruction requires only retained raw files")
    request_paths = tuple(path for path in paths if path.startswith("raw/requests/"))
    request_groups: dict[str, set[str]] = {}
    for path in request_paths:
        parts = PurePosixPath(path).parts
        if len(parts) != 4 or not parts[3].endswith(".json"):
            raise Stage2ExperimentError("request raw path does not match the fixed layout")
        request_groups.setdefault(parts[2], set()).add(parts[3].removesuffix(".json"))
    if len(request_groups) != 16 or any(
        fields != set(_RAW_REQUEST_EVIDENCE_FIELDS) for fields in request_groups.values()
    ):
        raise Stage2ExperimentError("repetition raw inventory lacks the exact 16-by-6 wire set")
    derived: dict[str, bytes] = {}
    repetition_indexes: set[int] = set()
    evidence_scopes: set[Stage2EvidenceScope] = set()
    for external_id in sorted(request_groups):
        raw_group = {
            field: raw[f"raw/requests/{external_id}/{field}.json"]
            for field in _RAW_REQUEST_EVIDENCE_FIELDS
        }
        (
            repetition_index,
            _,
            reconstructed_external_id,
            _,
            _,
            _,
            capture,
            expected,
        ) = _derive_request_from_raw_wire(raw_group)
        repetition_indexes.add(repetition_index)
        evidence_scopes.add(capture.provenance.evidence_scope)
        if reconstructed_external_id != external_id:
            raise Stage2ExperimentError("request raw path differs from its wire identity")
        for field in set(REQUEST_EVIDENCE_FIELDS) - set(_RAW_REQUEST_EVIDENCE_FIELDS):
            derived[f"derived/requests/{external_id}/{field}.json"] = expected[field]
    if len(repetition_indexes) != 1 or len(evidence_scopes) != 1:
        raise Stage2ExperimentError("request wire captures cross a repetition or evidence scope")
    cancellation_path = "raw/cancellation/client-wire.json"
    if cancellation_path not in raw:
        raise Stage2ExperimentError("repetition raw inventory lacks cancellation HTTP/SSE wire")
    cancellation_wire = cancellation_wire_capture_from_raw(raw[cancellation_path])
    prometheus_paths = (
        "raw/prometheus/measured-window-baseline.json",
        "raw/prometheus/measured-window-final.json",
    )
    if any(path not in raw for path in prometheus_paths):
        raise Stage2ExperimentError("repetition raw inventory lacks measured-window scrapes")
    baseline_capture, baseline = prometheus_capture_and_snapshot_from_raw(raw[prometheus_paths[0]])
    final_capture, final = prometheus_capture_and_snapshot_from_raw(raw[prometheus_paths[1]])
    expected_repetition_index = next(iter(repetition_indexes))
    expected_scope = next(iter(evidence_scopes))
    if (
        baseline_capture.repetition_index != expected_repetition_index
        or final_capture.repetition_index != expected_repetition_index
        or cancellation_wire.repetition_index != expected_repetition_index
        or baseline_capture.evidence_scope is not expected_scope
        or final_capture.evidence_scope is not expected_scope
        or cancellation_wire.provenance.evidence_scope is not expected_scope
        or baseline_capture.process_start_id != final_capture.process_start_id
    ):
        raise Stage2ExperimentError(
            "raw Prometheus captures cross a repetition, evidence scope, or process"
        )
    derived["derived/prometheus/measured-window-baseline-snapshot.json"] = (
        canonical_json_bytes(baseline) + b"\n"
    )
    derived["derived/prometheus/measured-window-final-snapshot.json"] = (
        canonical_json_bytes(final) + b"\n"
    )
    summary = {
        "schema_version": "0.3.0",
        "evidence_kind": "repetition_raw_reconstruction",
        "raw_file_count": len(paths),
        "measured_request_count": len(request_groups),
        "request_raw_file_count": len(request_paths),
        "request_derived_file_count": 16
        * (len(REQUEST_EVIDENCE_FIELDS) - len(_RAW_REQUEST_EVIDENCE_FIELDS)),
        "prometheus_raw_scrape_count": 2,
        "measured_http_exchange_count": 16,
        "cancellation_http_exchange_count": 1,
        "prometheus_http_exchange_count": 2,
        "prometheus_parsed_snapshot_count": 2,
        "raw_inventory_sha256": sha256_identity(
            {path: hashlib.sha256(raw[path]).hexdigest() for path in paths}
        ),
    }
    derived["derived/repetition-raw-summary.json"] = canonical_json_bytes(summary) + b"\n"
    return derived


def _reference_is_in_manifest(
    reference: ManifestBoundFile,
    files: dict[str, BundleFileEntry],
) -> bool:
    entry = files.get(reference.path)
    return entry is not None and (entry.sha256, entry.size) == (reference.sha256, reference.size)


def _derive_observed_concurrency(
    requests: tuple[Stage2MeasuredRequestAttestation, ...],
) -> tuple[int, bool]:
    events = sorted(
        (
            *((request.lifecycle.dispatch_offset_ns, 1) for request in requests),
            *((request.lifecycle.terminal_offset_ns, -1) for request in requests),
        ),
        key=lambda event: (event[0], 0 if event[1] == -1 else 1),
    )
    active = 0
    maximum = 0
    overlap = False
    previous_time: int | None = None
    for timestamp, delta in events:
        if previous_time is not None and timestamp > previous_time and active >= 2:
            overlap = True
        active += delta
        if active < 0:
            raise ValueError("measured lifecycle terminal precedes dispatch")
        maximum = max(maximum, active)
        previous_time = timestamp
    if active != 0:
        raise ValueError("measured lifecycle events do not close")
    return maximum, overlap


class Stage2RepetitionAttestation(StrictModel):
    schema_version: Literal["0.3.0"]
    repetition_index: Literal[1, 2, 3]
    runtime_control: Stage2RuntimeControlEvidence
    runtime_control_sha256: Sha256
    server_restart: ServerRestartIdentity
    repetition_manifest: Stage2BundleManifest
    repetition_manifest_sha256: Sha256
    cancellation_result_file: ManifestBoundFile
    cancellation_wire_file: ManifestBoundFile
    cancellation_wire: Stage2CancellationWireCapture
    prometheus_measurement: PrometheusMeasurementAttestation
    cuda_execution: Stage2RepetitionCudaAttestation
    measured_requests: tuple[Stage2MeasuredRequestAttestation, ...] = Field(
        min_length=16, max_length=16
    )
    requested_client_concurrency: Literal[2]
    observed_max_active_concurrency: Literal[2]
    positive_duration_overlap_observed: Literal[True]
    metric_availability_summary: Stage2MetricAvailabilitySummary
    identity_sha256: Sha256

    @model_validator(mode="after")
    def validate_repetition(self) -> Self:
        manifest = self.repetition_manifest
        expected_manifest_sha = bundle_manifest_sha256(manifest)
        if (
            self.runtime_control.repetition_index != self.repetition_index
            or self.server_restart.repetition_index != self.repetition_index
            or manifest.repetition_index != self.repetition_index
            or self.cuda_execution.repetition_index != self.repetition_index
            or self.prometheus_measurement.repetition_index != self.repetition_index
        ):
            raise ValueError("repetition component indexes differ")
        if self.runtime_control_sha256 != sha256_identity(self.runtime_control):
            raise ValueError("runtime-control identity does not reconstruct")
        if self.repetition_manifest_sha256 != expected_manifest_sha:
            raise ValueError("repetition-manifest canonical byte identity differs")
        if self.server_restart.output_bundle_identity_sha256 != expected_manifest_sha:
            raise ValueError("server restart is not bound to its repetition manifest")
        expected_server = self.server_restart.server_process_identity
        runtime_server = self.runtime_control.process_records[self.repetition_index + 1]
        if expected_server != runtime_server.process_identity:
            raise ValueError("server restart differs from the runtime-control server process")
        if (
            self.cuda_execution.server_process_identity != expected_server
            or self.cuda_execution.runtime_control_sha256 != self.runtime_control_sha256
            or self.cuda_execution.repetition_manifest_sha256 != expected_manifest_sha
        ):
            raise ValueError("CUDA attestation is detached from repetition control or manifest")
        requests = self.measured_requests
        if tuple(request.case_id for request in requests) != STAGE2_EXPERIMENT_CASE_IDS:
            raise ValueError("repetition requires the exact ordered 16-case set")
        request_ids = tuple(request.external_request_id for request in requests)
        if len(request_ids) != len(set(request_ids)):
            raise ValueError("measured external request IDs must be unique")
        if set(request_ids) != set(self.runtime_control.measured_request_ids):
            raise ValueError("measured-request attestations and runtime control IDs differ")
        cancellation_id = self.runtime_control.cancellation_probe.identity_chain.external_base_id
        all_ids = (
            *self.runtime_control.stabilization_request_ids,
            *self.runtime_control.workload_shape_warmup_request_ids,
            cancellation_id,
            *request_ids,
        )
        if len(all_ids) != len(set(all_ids)):
            raise ValueError("excluded, cancellation, and measured request IDs must be disjoint")
        probe = self.runtime_control.cancellation_probe
        if (
            self.cancellation_wire.repetition_index != self.repetition_index
            or self.cancellation_wire.external_request_id != cancellation_id
            or self.cancellation_wire.request_identity.identity_chain != probe.identity_chain
            or self.cancellation_wire.raw_log_capture != probe.raw_log_capture
            or self.cancellation_wire.raw_log_capture_sha256 != probe.raw_log_capture_sha256
            or self.cancellation_wire.parser_replay.first_generation_delivery
            != probe.first_generation_delivery
            or self.cancellation_wire.request_body.transmission_offset_ns
            != probe.dispatch_offset_ns
            or self.cancellation_wire.intentional_client_close.close_observation_offset_ns
            != probe.client_close_offset_ns
            or self.cancellation_wire.http_exchange.server_process_identity
            != probe.server_process_identity
        ):
            raise ValueError("cancellation HTTP/SSE wire is detached from abort/drain evidence")
        measured_phase = next(
            phase for phase in self.runtime_control.phases if phase.phase.value == "MEASURED_WINDOW"
        )
        for request in requests:
            if (
                request.repetition_index != self.repetition_index
                or request.repetition_manifest_sha256 != expected_manifest_sha
                or request.lifecycle.measurement_phase_start_ns != measured_phase.started_offset_ns
                or request.lifecycle.measurement_phase_end_ns != measured_phase.ended_offset_ns
                or request.lifecycle.measurement_phase_identity_sha256
                != measured_phase.evidence_identity_sha256
            ):
                raise ValueError("measured request is detached from repetition or measured phase")
        measurement = self.prometheus_measurement
        if (
            measurement.repetition_manifest_sha256 != expected_manifest_sha
            or measurement.server_process_identity != expected_server
            or measurement.measured_phase_identity_sha256 != measured_phase.evidence_identity_sha256
            or measurement.measured_phase_start_offset_ns != measured_phase.started_offset_ns
            or measurement.measured_phase_end_offset_ns != measured_phase.ended_offset_ns
            or measurement.first_measured_request_dispatch_offset_ns
            != min(request.lifecycle.dispatch_offset_ns for request in requests)
            or measurement.last_measured_request_terminal_offset_ns
            != max(request.lifecycle.terminal_offset_ns for request in requests)
            or measurement.final_drain_boundary_offset_ns
            != self.runtime_control.final_drain_completed_offset_ns
            or measurement.final_snapshot != self.runtime_control.final_metric_scrape
        ):
            raise ValueError("Prometheus measured-window evidence is detached from repetition")
        maximum, overlap = _derive_observed_concurrency(requests)
        if maximum != 2 or not overlap:
            raise ValueError("lifecycle evidence must derive exact positive concurrency two")
        if self.metric_availability_summary != derive_metric_availability_summary(requests):
            raise ValueError("repetition metric summary does not reconstruct")
        files = _manifest_file_map(manifest)
        references = [
            self.cancellation_result_file,
            self.cancellation_wire_file,
            measurement.baseline_raw_exposition_file,
            measurement.baseline_parsed_snapshot_file,
            measurement.final_raw_exposition_file,
            measurement.final_parsed_snapshot_file,
            *self.cuda_execution.raw_evidence_files,
            *(reference for request in requests for reference in request.raw_evidence.files()),
        ]
        phase_references = tuple(
            ManifestBoundFile(
                path=phase.evidence_references[0],
                sha256=phase.evidence_identity_sha256,
                size=files[phase.evidence_references[0]].size,
            )
            for phase in self.runtime_control.phases
            if len(phase.evidence_references) == 1 and phase.evidence_references[0] in files
        )
        references.extend(phase_references)
        if len(phase_references) != len(self.runtime_control.phases):
            raise ValueError("every runtime phase requires one manifest-bound evidence file")
        reference_paths = tuple(reference.path for reference in references)
        if len(reference_paths) != len(set(reference_paths)):
            raise ValueError("durable repetition evidence references a path more than once")
        if any(not _reference_is_in_manifest(reference, files) for reference in references):
            raise ValueError("durable evidence reference is absent from the repetition manifest")
        if self.identity_sha256 != sha256_identity(
            self, omit_fields=frozenset({"identity_sha256"})
        ):
            raise ValueError("repetition attestation identity does not reconstruct")
        return self


class Stage2WorkloadCase(StrictModel):
    case_id: Identifier
    sent_prompt_token_ids: tuple[NonNegativeInt, ...] = Field(min_length=64, max_length=64)
    sent_prompt_token_ids_sha256: Sha256

    @model_validator(mode="after")
    def validate_prompt(self) -> Self:
        if self.sent_prompt_token_ids_sha256 != sha256_identity(self.sent_prompt_token_ids):
            raise ValueError("workload prompt-token identity does not reconstruct")
        return self


class Stage2ExperimentWorkload(StrictModel):
    schema_version: Literal["0.3.0"]
    workload_name: Literal["stage2-fixed-16-case-v1"]
    cases: tuple[Stage2WorkloadCase, ...] = Field(min_length=16, max_length=16)
    identity_sha256: Sha256

    @model_validator(mode="after")
    def validate_workload(self) -> Self:
        if tuple(case.case_id for case in self.cases) != STAGE2_EXPERIMENT_CASE_IDS:
            raise ValueError("workload requires the exact ordered versioned 16-case set")
        if self.identity_sha256 != sha256_identity(
            self, omit_fields=frozenset({"identity_sha256"})
        ):
            raise ValueError("workload identity does not reconstruct")
        return self


class Stage2RestartSemanticAttestation(StrictModel):
    repetition_index: Literal[1, 2, 3]
    case_id: Identifier
    measured_request_attestation_sha256: Sha256
    sent_prompt_token_ids: tuple[int, ...] = Field(min_length=64, max_length=64)
    returned_prompt_token_ids: tuple[int, ...] = Field(min_length=64, max_length=64)
    output_token_ids: tuple[int, ...] = Field(min_length=32, max_length=32)
    finish_reason: Literal["length"]
    prompt_tokens: Literal[64]
    completion_tokens: Literal[32]
    total_tokens: Literal[96]
    output_text_sha256: Sha256
    repetition_manifest_sha256: Sha256

    def as_restart_record(self) -> RestartSemanticRecord:
        return RestartSemanticRecord(
            repetition_index=self.repetition_index,
            bundle_manifest_sha256=self.repetition_manifest_sha256,
            case_id=self.case_id,
            sent_prompt_token_ids=self.sent_prompt_token_ids,
            returned_prompt_token_ids=self.returned_prompt_token_ids,
            output_token_ids=self.output_token_ids,
            finish_reason=self.finish_reason,
            prompt_tokens=self.prompt_tokens,
            completion_tokens=self.completion_tokens,
            total_tokens=self.total_tokens,
            output_text_sha256=self.output_text_sha256,
            replacement_run=False,
        )


class Stage2CrossRestartComparison(StrictModel):
    schema_version: Literal["0.3.0"]
    case_id: Identifier
    semantic_records: tuple[
        Stage2RestartSemanticAttestation,
        Stage2RestartSemanticAttestation,
        Stage2RestartSemanticAttestation,
    ]
    comparison: RestartComparison
    identity_sha256: Sha256

    @model_validator(mode="after")
    def validate_comparison(self) -> Self:
        if self.case_id not in STAGE2_EXPERIMENT_CASE_IDS:
            raise ValueError("comparison case ID is outside the declared workload")
        if tuple(record.repetition_index for record in self.semantic_records) != (1, 2, 3):
            raise ValueError("comparison requires exactly one record from each repetition")
        if any(record.case_id != self.case_id for record in self.semantic_records):
            raise ValueError("comparison record case IDs differ")
        reconstructed = compare_three_restarts(
            tuple(record.as_restart_record() for record in self.semantic_records)
        )
        if self.comparison != reconstructed:
            raise ValueError("cross-restart comparison was not derived from semantic records")
        if self.identity_sha256 != sha256_identity(
            self, omit_fields=frozenset({"identity_sha256"})
        ):
            raise ValueError("cross-restart comparison identity does not reconstruct")
        return self


class Stage2AggregateValidationResult(StrictModel):
    state: Literal[
        BundleState.COMMITTED,
        AggregateComparisonState.INVALID_SEMANTIC_NONREPRODUCTION,
    ]
    aggregate_validator: Literal["validate_aggregate_commit"]
    repetition_count: Literal[3]
    measured_request_count: Literal[48]
    comparison_count: Literal[16]
    invalid_case_ids: tuple[Identifier, ...]
    failure_reason: Literal["INVALID_SEMANTIC_NONREPRODUCTION"] | None
    validated_input_sha256: Sha256
    identity_sha256: Sha256

    @model_validator(mode="after")
    def validate_identity(self) -> Self:
        invalid = self.state is AggregateComparisonState.INVALID_SEMANTIC_NONREPRODUCTION
        if invalid != bool(self.invalid_case_ids) or invalid != (self.failure_reason is not None):
            raise ValueError("aggregate terminal state does not match retained semantic failures")
        if self.invalid_case_ids != tuple(sorted(set(self.invalid_case_ids))):
            raise ValueError("invalid semantic case IDs must be sorted and unique")
        if self.identity_sha256 != sha256_identity(
            self, omit_fields=frozenset({"identity_sha256"})
        ):
            raise ValueError("aggregate validation-result identity does not reconstruct")
        return self


def derive_aggregate_validation_result(
    repetitions: tuple[
        Stage2RepetitionAttestation,
        Stage2RepetitionAttestation,
        Stage2RepetitionAttestation,
    ],
    comparisons: tuple[Stage2CrossRestartComparison, ...],
) -> Stage2AggregateValidationResult:
    manifests = tuple(repetition.repetition_manifest for repetition in repetitions)
    case_records = tuple(
        tuple(record.as_restart_record() for record in comparison.semantic_records)
        for comparison in comparisons
    )
    comparison_records = tuple(comparison.comparison for comparison in comparisons)
    invalid_case_ids = tuple(
        comparison.case_id
        for comparison in comparison_records
        if comparison.state is AggregateComparisonState.INVALID_SEMANTIC_NONREPRODUCTION
    )
    if invalid_case_ids:
        try:
            validate_aggregate_commit(
                manifests,
                case_records,
                comparison_records,
                expected_case_ids=STAGE2_EXPERIMENT_CASE_IDS,
            )
        except Stage2ControlError as error:
            if str(error) != "aggregate commit requires passing semantic comparison":
                raise
        else:
            raise ValueError("semantic mismatch unexpectedly passed aggregate commit validation")
        state: BundleState | AggregateComparisonState = (
            AggregateComparisonState.INVALID_SEMANTIC_NONREPRODUCTION
        )
    else:
        state = validate_aggregate_commit(
            manifests,
            case_records,
            comparison_records,
            expected_case_ids=STAGE2_EXPERIMENT_CASE_IDS,
        )
    values: dict[str, object] = {
        "state": state,
        "aggregate_validator": "validate_aggregate_commit",
        "repetition_count": 3,
        "measured_request_count": 48,
        "comparison_count": 16,
        "invalid_case_ids": invalid_case_ids,
        "failure_reason": ("INVALID_SEMANTIC_NONREPRODUCTION" if invalid_case_ids else None),
        "validated_input_sha256": sha256_identity(
            {
                "case_records": case_records,
                "comparisons": comparison_records,
                "expected_case_ids": STAGE2_EXPERIMENT_CASE_IDS,
                "repetition_manifests": manifests,
            }
        ),
    }
    values["identity_sha256"] = sha256_identity(values)
    return Stage2AggregateValidationResult.model_validate(values)


class Stage2ExperimentClassification(StrEnum):
    SYNTHETIC_PROTOCOL_SHAPE_ONLY = "SYNTHETIC_PROTOCOL_SHAPE_ONLY"
    FUTURE_REAL_RUNTIME = "FUTURE_REAL_RUNTIME"


class Stage2ExperimentSummary(StrictModel):
    repetition_count: Literal[3]
    measured_request_count: Literal[48]
    cancellation_probe_count: Literal[3]
    prometheus_measurement_count: Literal[3]
    cuda_attestation_count: Literal[3]
    semantic_comparison_count: Literal[16]
    requested_client_concurrency: Literal[2]
    observed_max_active_concurrency_per_repetition: tuple[Literal[2], Literal[2], Literal[2]]
    fixture_or_protocol_shape_only: bool
    runtime_claim_advancement_allowed: Literal[False]
    performance_claim_advancement_allowed: Literal[False]
    identity_sha256: Sha256

    @model_validator(mode="after")
    def validate_identity(self) -> Self:
        if self.identity_sha256 != sha256_identity(
            self, omit_fields=frozenset({"identity_sha256"})
        ):
            raise ValueError("experiment summary identity does not reconstruct")
        return self


class Stage2ExperimentAttestation(StrictModel):
    schema_version: Literal["0.3.0"]
    measurement_protocol_version: Literal["0.3.0"]
    experiment_id: Identifier
    evidence_scope: Stage2EvidenceScope
    classification: Stage2ExperimentClassification
    workload: Stage2ExperimentWorkload
    launch_spec: Stage2LaunchSpec
    snapshot_manifest: ModelTokenizerSnapshotManifest
    execution_lock: RuntimePackageExecutionLockAttestation
    linux_environment: LinuxEnvironmentManifest
    nvidia_resources: NvidiaT4ResourceAttestation
    public_safety: PublicSafetyAttestation
    repetitions: tuple[
        Stage2RepetitionAttestation,
        Stage2RepetitionAttestation,
        Stage2RepetitionAttestation,
    ]
    comparisons: tuple[Stage2CrossRestartComparison, ...] = Field(min_length=16, max_length=16)
    experiment_metric_availability: Stage2MetricAvailabilitySummary
    aggregate_validation_result: Stage2AggregateValidationResult
    summary: Stage2ExperimentSummary
    identity_sha256: Sha256

    @model_validator(mode="after")
    def validate_experiment(self) -> Self:
        fixture_scope = self.evidence_scope is Stage2EvidenceScope.TEST_FIXTURE_ONLY
        if fixture_scope != (
            self.classification is Stage2ExperimentClassification.SYNTHETIC_PROTOCOL_SHAPE_ONLY
        ):
            raise ValueError("fixture scope can produce only synthetic protocol-shape evidence")
        if tuple(repetition.repetition_index for repetition in self.repetitions) != (1, 2, 3):
            raise ValueError("experiment requires exactly three ordered repetitions")
        server_processes = tuple(
            repetition.server_restart.server_process_identity for repetition in self.repetitions
        )
        if len(set(server_processes)) != 3:
            raise ValueError("experiment requires three distinct server restart processes")
        if (
            tuple(comparison.case_id for comparison in self.comparisons)
            != STAGE2_EXPERIMENT_CASE_IDS
        ):
            raise ValueError("experiment requires exactly 16 ordered semantic comparisons")
        all_requests = tuple(
            request for repetition in self.repetitions for request in repetition.measured_requests
        )
        if len(all_requests) != 48:
            raise ValueError("experiment requires exactly 48 measured-request attestations")
        if any(
            request.wire_capture.provenance.evidence_scope is not self.evidence_scope
            for request in all_requests
        ):
            raise ValueError("request wire provenance differs from experiment scope")
        if any(
            provenance.evidence_scope is not self.evidence_scope
            for repetition in self.repetitions
            for provenance in (
                repetition.cancellation_wire.provenance,
                repetition.prometheus_measurement.baseline_capture.provenance,
                repetition.prometheus_measurement.final_capture.provenance,
            )
        ):
            raise ValueError("cancellation or scrape provenance differs from experiment scope")
        all_external_ids = tuple(
            request_id
            for repetition in self.repetitions
            for request_id in (
                *repetition.runtime_control.stabilization_request_ids,
                *repetition.runtime_control.workload_shape_warmup_request_ids,
                repetition.runtime_control.cancellation_probe.identity_chain.external_base_id,
                *(request.external_request_id for request in repetition.measured_requests),
            )
        )
        if len(all_external_ids) != len(set(all_external_ids)):
            raise ValueError("all external request IDs must be globally unique across restarts")
        workload_by_case = {case.case_id: case for case in self.workload.cases}
        request_by_key = {
            (request.repetition_index, request.case_id): request for request in all_requests
        }
        for request in all_requests:
            if (
                request.request_evidence.sent_prompt_token_ids
                != workload_by_case[request.case_id].sent_prompt_token_ids
            ):
                raise ValueError("measured request prompt differs from shared workload definition")
        for comparison in self.comparisons:
            for record in comparison.semantic_records:
                request = request_by_key[(record.repetition_index, record.case_id)]
                evidence = request.request_evidence
                if (
                    record.measured_request_attestation_sha256 != request.attestation_sha256
                    or record.repetition_manifest_sha256 != request.repetition_manifest_sha256
                    or record.sent_prompt_token_ids != evidence.sent_prompt_token_ids
                    or record.returned_prompt_token_ids != evidence.returned_prompt_token_ids
                    or record.output_token_ids != evidence.final_output_token_ids
                    or record.finish_reason != evidence.finish_reason
                    or record.prompt_tokens != evidence.usage.prompt_tokens
                    or record.completion_tokens != evidence.usage.completion_tokens
                    or record.total_tokens != evidence.usage.total_tokens
                    or record.output_text_sha256 != evidence.output_text_sha256
                ):
                    raise ValueError("semantic comparison is detached from measured request")
        environment_resource_identity = sha256_identity(
            {"linux": self.linux_environment, "nvidia": self.nvidia_resources}
        )
        if any(
            repetition.cuda_execution.environment_resource_identity_sha256
            != environment_resource_identity
            for repetition in self.repetitions
        ):
            raise ValueError("CUDA attestations do not share the declared environment identity")
        launch_identity = sha256_identity(self.launch_spec)
        if any(
            repetition.server_restart.launch_spec_identity_sha256 != launch_identity
            for repetition in self.repetitions
        ):
            raise ValueError("server restarts are not bound to the exact launch specification")
        for repetition in self.repetitions:
            expected_process = repetition.server_restart.server_process_identity
            exchanges = (
                *(request.wire_capture.http_exchange for request in repetition.measured_requests),
                repetition.cancellation_wire.http_exchange,
                repetition.prometheus_measurement.baseline_capture.http_exchange,
                repetition.prometheus_measurement.final_capture.http_exchange,
            )
            try:
                for exchange in exchanges:
                    exchange.require_launch_and_process(
                        self.launch_spec,
                        server_process_identity=expected_process,
                    )
            except ValueError as error:
                raise ValueError(
                    "measured, cancellation, or scrape transport differs from launch/process"
                ) from error
        if (
            self.launch_spec.model_path != self.snapshot_manifest.snapshot_root_path
            or self.launch_spec.tokenizer_path != self.snapshot_manifest.snapshot_root_path
        ):
            raise ValueError("launch paths are not bound to the verified snapshot root")
        process_records = self.repetitions[0].runtime_control.process_records
        if any(
            repetition.runtime_control.process_records != process_records
            for repetition in self.repetitions[1:]
        ):
            raise ValueError("repetition controls disagree on the retained process sequence")
        if (
            self.snapshot_manifest.download_process != process_records[0]
            or self.snapshot_manifest.offline_tokenizer_verification_process != process_records[1]
            or server_processes
            != tuple(process.process_identity for process in process_records[2:])
        ):
            raise ValueError("snapshot and restart identities differ from the process sequence")
        launch_environment = self.launch_spec.environment
        expected_runtime_environment = {
            **OFFLINE_RUNTIME_ENVIRONMENT,
            "HF_HOME": launch_environment.hf_home,
            "VLLM_CONFIG_ROOT": launch_environment.vllm_config_root,
        }
        if (
            any(
                process.environment != expected_runtime_environment
                for process in process_records[2:]
            )
            or launch_environment.absent_variables != LAUNCH_ABSENT_ENVIRONMENT_VARIABLES
        ):
            raise ValueError("runtime processes differ from the exact launch environment")
        if self.experiment_metric_availability != derive_metric_availability_summary(all_requests):
            raise ValueError("experiment metric-availability summary does not reconstruct")
        expected_aggregate = derive_aggregate_validation_result(
            self.repetitions,
            self.comparisons,
        )
        if self.aggregate_validation_result != expected_aggregate:
            raise ValueError("aggregate validation path was omitted, bypassed, or altered")
        expected_summary_values: dict[str, object] = {
            "repetition_count": 3,
            "measured_request_count": 48,
            "cancellation_probe_count": 3,
            "prometheus_measurement_count": 3,
            "cuda_attestation_count": 3,
            "semantic_comparison_count": 16,
            "requested_client_concurrency": 2,
            "observed_max_active_concurrency_per_repetition": (2, 2, 2),
            "fixture_or_protocol_shape_only": fixture_scope,
            "runtime_claim_advancement_allowed": False,
            "performance_claim_advancement_allowed": False,
        }
        expected_summary_values["identity_sha256"] = sha256_identity(expected_summary_values)
        if self.summary != Stage2ExperimentSummary.model_validate(expected_summary_values):
            raise ValueError("experiment summary does not reconstruct")
        if fixture_scope:
            if any(
                request.request_evidence.fixture_identity_sha256 is None for request in all_requests
            ):
                raise ValueError("synthetic protocol-shape evidence requires fixture identities")
        else:
            if _contains_fixture_value(self):
                raise ValueError("synthetic or fixture evidence cannot receive a live boundary")
            environment_identity = sha256_identity(
                {"linux": self.linux_environment, "nvidia": self.nvidia_resources}
            )
            snapshot_identity = sha256_identity(self.snapshot_manifest)
            if any(
                not isinstance(request.wire_capture.provenance, CollectorWireCaptureProvenance)
                or request.wire_capture.provenance.server_process_identity
                != self.repetitions[
                    request.repetition_index - 1
                ].server_restart.server_process_identity
                or request.wire_capture.provenance.model_snapshot_identity_sha256
                != snapshot_identity
                or request.wire_capture.provenance.environment_identity_sha256
                != environment_identity
                for request in all_requests
            ):
                raise ValueError("live wire capture lacks collector/runtime/environment bindings")
            additional_provenances = tuple(
                provenance
                for repetition in self.repetitions
                for provenance in (
                    repetition.cancellation_wire.provenance,
                    repetition.prometheus_measurement.baseline_capture.provenance,
                    repetition.prometheus_measurement.final_capture.provenance,
                )
            )
            if any(
                not isinstance(provenance, CollectorWireCaptureProvenance)
                or provenance.server_process_identity
                != self.repetitions[index // 3].server_restart.server_process_identity
                or provenance.model_snapshot_identity_sha256 != snapshot_identity
                or provenance.environment_identity_sha256 != environment_identity
                for index, provenance in enumerate(additional_provenances)
            ):
                raise ValueError(
                    "live cancellation/scrape capture lacks collector/runtime bindings"
                )
            chains = (
                *(request.request_identity.identity_chain for request in all_requests),
                *(
                    repetition.runtime_control.cancellation_probe.identity_chain
                    for repetition in self.repetitions
                ),
            )
            records = tuple(
                record
                for chain in chains
                for record in (
                    chain.request_received_log,
                    chain.request_add_log,
                    chain.external_abort_log,
                    chain.internal_abort_log,
                )
                if record is not None
            )
            if any(
                request.request_evidence.fixture_identity_sha256 is not None
                or "fixture" in request.external_request_id.casefold()
                or "<fixture-" in request.request_evidence.output_text.casefold()
                for request in all_requests
            ) or any(
                "fixture" in record.source_stream_id.casefold()
                or "TEST_FIXTURE_ONLY" in record.raw_record
                for record in records
            ):
                raise ValueError("fixture evidence cannot receive a future real-runtime boundary")
        if self.identity_sha256 != sha256_identity(
            self, omit_fields=frozenset({"identity_sha256"})
        ):
            raise ValueError("final experiment attestation identity does not reconstruct")
        return self


class AggregateRootState(StrEnum):
    INCOMPLETE = "INCOMPLETE"
    INVALID = "INVALID"
    COMMITTED = "COMMITTED"


class Stage2AggregateExperimentManifest(StrictModel):
    schema_version: Literal["0.3.0"]
    measurement_protocol_version: Literal["0.3.0"]
    experiment_id: Identifier
    state: AggregateRootState
    failure_reason: Identifier | None
    evidence_scope: Stage2EvidenceScope
    created_at_utc: AwareDatetime
    files: tuple[BundleFileEntry, ...] = Field(min_length=1)
    resource_environment_manifest: ManifestBoundFile
    resource_environment_raw_evidence: ManifestBoundFile
    nvidia_isolation_evidence: ManifestBoundFile
    nvidia_isolation_raw_evidence: ManifestBoundFile
    execution_lock_snapshot: ManifestBoundFile
    runtime_resolver_lock_evidence: ManifestBoundFile
    runtime_installed_distribution_inventory: ManifestBoundFile
    reviewed_execution_lock: ManifestBoundFile
    model_tokenizer_snapshot_manifest: ManifestBoundFile
    snapshot_read_only_verification_evidence: ManifestBoundFile
    launch_specification: ManifestBoundFile
    public_safety_result: ManifestBoundFile
    public_safety_raw_scan_evidence: ManifestBoundFile
    shared_workload_definition: ManifestBoundFile
    repetition_manifest_files: tuple[
        ManifestBoundFile,
        ManifestBoundFile,
        ManifestBoundFile,
    ]
    cuda_execution_attestation_files: tuple[
        ManifestBoundFile,
        ManifestBoundFile,
        ManifestBoundFile,
    ]
    cancellation_result_files: tuple[
        ManifestBoundFile,
        ManifestBoundFile,
        ManifestBoundFile,
    ]
    prometheus_measurement_attestation_files: tuple[
        ManifestBoundFile,
        ManifestBoundFile,
        ManifestBoundFile,
    ]
    semantic_comparison_files: tuple[ManifestBoundFile, ...] = Field(min_length=16, max_length=16)
    metric_availability_summary: ManifestBoundFile
    experiment_summary: ManifestBoundFile
    aggregate_validation_result: ManifestBoundFile
    final_attestation: ManifestBoundFile

    @model_validator(mode="after")
    def validate_root(self) -> Self:
        if self.created_at_utc.utcoffset() != timedelta(0):
            raise ValueError("aggregate manifest timestamp must use UTC")
        paths = tuple(entry.path for entry in self.files)
        if (
            paths != tuple(sorted(paths))
            or len(paths) != len(set(paths))
            or len({path.casefold() for path in paths}) != len(paths)
        ):
            raise ValueError("aggregate inventory must be sorted, unique, and collision-free")
        if AGGREGATE_MANIFEST_PATH in paths:
            raise ValueError("aggregate manifest cannot inventory itself")
        if self.state in {AggregateRootState.COMMITTED, AggregateRootState.INVALID}:
            if self.state is AggregateRootState.COMMITTED and self.failure_reason is not None:
                raise ValueError("committed aggregate root cannot retain a failure reason")
            if self.state is AggregateRootState.INVALID and self.failure_reason is None:
                raise ValueError("invalid aggregate root requires a failure reason")
            expected_paths = (
                "shared/resource-environment-manifest.json",
                "shared/raw/resource-environment-evidence.json",
                "shared/nvidia-isolation-evidence.json",
                "shared/raw/nvidia-isolation-evidence.json",
                "shared/execution-lock-snapshot.json",
                "shared/raw/runtime-resolver-lock.json",
                "shared/raw/installed-distribution-inventory.json",
                "shared/raw/reviewed-stage2-execution-lock.json",
                "shared/model-tokenizer-snapshot-manifest.json",
                "shared/raw/snapshot-read-only-verification.json",
                "shared/launch-specification.json",
                "shared/public-safety-result.json",
                "shared/raw/public-safety-scan.json",
                "shared/workload-definition.json",
                *(f"repetition-{index:02d}/evidence-manifest.json" for index in (1, 2, 3)),
                *(f"attestations/cuda-repetition-{index:02d}.json" for index in (1, 2, 3)),
                *(f"repetition-{index:02d}/cancellation-result.json" for index in (1, 2, 3)),
                *(f"attestations/prometheus-repetition-{index:02d}.json" for index in (1, 2, 3)),
                *(f"comparisons/{case_id}.json" for case_id in STAGE2_EXPERIMENT_CASE_IDS),
                "derived/metric-availability-summary.json",
                "derived/experiment-summary.json",
                "derived/aggregate-validation-result.json",
                "derived/final-attestation.json",
            )
            references = (
                self.resource_environment_manifest,
                self.resource_environment_raw_evidence,
                self.nvidia_isolation_evidence,
                self.nvidia_isolation_raw_evidence,
                self.execution_lock_snapshot,
                self.runtime_resolver_lock_evidence,
                self.runtime_installed_distribution_inventory,
                self.reviewed_execution_lock,
                self.model_tokenizer_snapshot_manifest,
                self.snapshot_read_only_verification_evidence,
                self.launch_specification,
                self.public_safety_result,
                self.public_safety_raw_scan_evidence,
                self.shared_workload_definition,
                *self.repetition_manifest_files,
                *self.cuda_execution_attestation_files,
                *self.cancellation_result_files,
                *self.prometheus_measurement_attestation_files,
                *self.semantic_comparison_files,
                self.metric_availability_summary,
                self.experiment_summary,
                self.aggregate_validation_result,
                self.final_attestation,
            )
            if tuple(reference.path for reference in references) != expected_paths:
                raise ValueError("aggregate root omits or substitutes a required durable file")
            file_map = {entry.path: entry for entry in self.files}
            if any(
                reference.path not in file_map
                or (reference.sha256, reference.size)
                != (file_map[reference.path].sha256, file_map[reference.path].size)
                for reference in references
            ):
                raise ValueError("aggregate root reference is not bound to its file inventory")
        elif self.failure_reason is None:
            raise ValueError("incomplete or invalid aggregate root requires an explicit reason")
        return self


class Stage2ReconstructedExperiment(StrictModel):
    attestation: Stage2ExperimentAttestation
    aggregate_manifest: Stage2AggregateExperimentManifest
    aggregate_manifest_sha256: Sha256

    @model_validator(mode="after")
    def validate_boundary(self) -> Self:
        result_state = self.attestation.aggregate_validation_result.state
        committed = result_state is BundleState.COMMITTED
        if committed != (self.aggregate_manifest.state is AggregateRootState.COMMITTED):
            raise ValueError(
                "aggregate root state differs from the reconstructed validation result"
            )
        if not committed and (
            self.aggregate_manifest.state is not AggregateRootState.INVALID
            or self.aggregate_manifest.failure_reason != "INVALID_SEMANTIC_NONREPRODUCTION"
        ):
            raise ValueError("semantic nonreproduction requires a durable invalid aggregate root")
        if (
            self.aggregate_manifest.experiment_id != self.attestation.experiment_id
            or self.aggregate_manifest.evidence_scope is not self.attestation.evidence_scope
        ):
            raise ValueError("aggregate root and final attestation identities differ")
        expected = hashlib.sha256(canonical_json_bytes(self.aggregate_manifest) + b"\n").hexdigest()
        if self.aggregate_manifest_sha256 != expected:
            raise ValueError("aggregate manifest canonical byte identity differs")
        return self


def _path_has_symlink_component(root: Path, relative: PurePosixPath) -> bool:
    current = root
    for part in relative.parts:
        current /= part
        if current.is_symlink():
            return True
    return False


def _path_has_symlink_ancestor(path: Path) -> bool:
    absolute = path.absolute()
    current = Path(absolute.anchor)
    for index, part in enumerate(absolute.parts[1:], start=1):
        current /= part
        if current.is_symlink():
            resolved = current.resolve()
            macos_platform_alias = index == 1 and (
                (current == Path("/var") and resolved == Path("/private/var"))
                or (current == Path("/tmp") and resolved == Path("/private/tmp"))
            )
            if not macos_platform_alias:
                return True
            current = resolved
    return False


def _validate_file(root: Path, entry: BundleFileEntry | ManifestBoundFile) -> bytes:
    relative = _safe_relative_path(entry.path)
    if _path_has_symlink_component(root, relative):
        raise Stage2ExperimentError("aggregate evidence cannot contain symlinks")
    path = root.joinpath(*relative.parts)
    if not path.is_file() or path.is_symlink():
        raise Stage2ExperimentError("aggregate evidence reference is not a regular file")
    data = path.read_bytes()
    if len(data) != entry.size or hashlib.sha256(data).hexdigest() != entry.sha256:
        raise Stage2ExperimentError("aggregate evidence file size or SHA-256 differs")
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError as error:
        raise Stage2ExperimentError("aggregate evidence must be public-safe UTF-8 text") from error
    scan_texts = (text, *decoded_base64_evidence_texts(text))
    if any(
        pattern.search(scan_text)
        for scan_text in scan_texts
        for pattern in _AGGREGATE_SENSITIVE_PATTERNS
    ):
        raise Stage2ExperimentError("aggregate evidence contains prohibited private material")
    return data


def _validate_aggregate_inventory(
    root: Path,
    manifest: Stage2AggregateExperimentManifest,
) -> dict[str, bytes]:
    if not root.is_dir() or root.is_symlink() or _path_has_symlink_ancestor(root):
        raise Stage2ExperimentError("aggregate directory is missing or unsafe")
    all_paths = tuple(root.rglob("*"))
    if any(path.is_symlink() for path in all_paths):
        raise Stage2ExperimentError("aggregate directory cannot contain symlinks")
    actual_files = tuple(
        sorted(path.relative_to(root).as_posix() for path in all_paths if path.is_file())
    )
    expected_files = (*tuple(entry.path for entry in manifest.files), AGGREGATE_MANIFEST_PATH)
    if actual_files != tuple(sorted(expected_files)):
        raise Stage2ExperimentError("aggregate file inventory is incomplete or unexpected")
    if len({path.casefold() for path in actual_files}) != len(actual_files):
        raise Stage2ExperimentError("aggregate directory contains a case-collision ambiguity")
    files = {entry.path: _validate_file(root, entry) for entry in manifest.files}
    if manifest.evidence_scope is Stage2EvidenceScope.FUTURE_REAL_RUNTIME and any(
        _raw_payload_contains_fixture_value(data) for data in files.values()
    ):
        raise Stage2ExperimentError("fixture-marked raw evidence cannot receive live scope")
    return files


def _validate_repetition_bundle_directory(
    root: Path,
    repetition_index: int,
    manifest: Stage2BundleManifest,
) -> None:
    directory = root / f"repetition-{repetition_index:02d}"
    if not directory.is_dir() or directory.is_symlink() or _path_has_symlink_ancestor(directory):
        raise Stage2ExperimentError("repetition bundle directory is missing or unsafe")
    paths = tuple(directory.rglob("*"))
    if any(path.is_symlink() for path in paths):
        raise Stage2ExperimentError("repetition bundle cannot contain symlinks")
    actual = tuple(
        sorted(path.relative_to(directory).as_posix() for path in paths if path.is_file())
    )
    expected = tuple(
        sorted((*tuple(entry.path for entry in manifest.files), "evidence-manifest.json"))
    )
    if actual != expected:
        raise Stage2ExperimentError("repetition manifest inventory is incomplete or unexpected")
    manifest_path = directory / "evidence-manifest.json"
    manifest_mtime = manifest_path.stat().st_mtime_ns
    for entry in manifest.files:
        _validate_file(directory, entry)
        if (directory / entry.path).stat().st_mtime_ns >= manifest_mtime:
            raise Stage2ExperimentError("repetition manifest was not written strictly last")
    try:
        validate_committed_bundle(directory, reconstruct_experiment_repetition)
    except Stage2BundleError as error:
        raise Stage2ExperimentError(
            "repetition manifest does not satisfy committed raw reconstruction"
        ) from error


def write_aggregate_manifest_last(
    root: Path,
    manifest: Stage2AggregateExperimentManifest,
) -> Path:
    if manifest.state is AggregateRootState.INCOMPLETE:
        raise Stage2ExperimentError("an incomplete aggregate manifest cannot be finalized")
    if not root.is_dir() or root.is_symlink() or _path_has_symlink_ancestor(root):
        raise Stage2ExperimentError("aggregate directory is missing or unsafe")
    manifest_path = root / AGGREGATE_MANIFEST_PATH
    if manifest_path.exists() or manifest_path.is_symlink():
        raise Stage2ExperimentError("aggregate manifest already exists")
    all_paths = tuple(root.rglob("*"))
    if any(path.is_symlink() for path in all_paths):
        raise Stage2ExperimentError("aggregate directory cannot contain symlinks")
    actual = tuple(
        sorted(path.relative_to(root).as_posix() for path in all_paths if path.is_file())
    )
    expected = tuple(entry.path for entry in manifest.files)
    if actual != expected:
        raise Stage2ExperimentError("aggregate inventory differs before manifest-last commit")
    files = {entry.path: _validate_file(root, entry) for entry in manifest.files}
    if manifest.evidence_scope is Stage2EvidenceScope.FUTURE_REAL_RUNTIME and any(
        _raw_payload_contains_fixture_value(payload) for payload in files.values()
    ):
        raise Stage2ExperimentError("fixture-marked raw evidence cannot receive live scope")
    attestation = _validate_terminal_graph(root, manifest, files)
    data = canonical_json_bytes(manifest) + b"\n"
    Stage2ReconstructedExperiment(
        attestation=attestation,
        aggregate_manifest=manifest,
        aggregate_manifest_sha256=hashlib.sha256(data).hexdigest(),
    )
    temporary = root / f".{AGGREGATE_MANIFEST_PATH}.tmp-{os.getpid()}"
    try:
        with temporary.open("xb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(manifest_path)
        descriptor = os.open(root, os.O_RDONLY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
    finally:
        if temporary.exists():
            temporary.unlink()
    return manifest_path


def _parse_model[ModelT: StrictModel](
    data: bytes,
    model: type[ModelT],
    *,
    label: str,
) -> ModelT:
    try:
        parsed = model.model_validate_json(data)
    except ValueError as error:
        raise Stage2ExperimentError(f"{label} is invalid") from error
    if data != canonical_json_bytes(parsed) + b"\n":
        raise Stage2ExperimentError(f"{label} is not exact canonical JSON bytes")
    return parsed


def _require_exact_raw(actual: bytes, expected: bytes, *, label: str) -> None:
    if actual != expected:
        raise Stage2ExperimentError(f"{label} does not reconstruct from retained raw evidence")


def _validate_terminal_graph(
    root: Path,
    manifest: Stage2AggregateExperimentManifest,
    files: dict[str, bytes],
) -> Stage2ExperimentAttestation:
    attestation = _parse_model(
        files[manifest.final_attestation.path],
        Stage2ExperimentAttestation,
        label="final experiment attestation",
    )
    if (
        manifest.experiment_id != attestation.experiment_id
        or manifest.evidence_scope is not attestation.evidence_scope
    ):
        raise Stage2ExperimentError("aggregate root differs from the final attestation boundary")
    if (
        _parse_model(
            files[manifest.resource_environment_manifest.path],
            LinuxEnvironmentManifest,
            label="resource/environment manifest",
        )
        != attestation.linux_environment
    ):
        raise Stage2ExperimentError("resource/environment manifest differs from final attestation")
    if (
        manifest.resource_environment_raw_evidence.sha256
        != attestation.linux_environment.environment_evidence_sha256
    ):
        raise Stage2ExperimentError("environment raw hash has no matching aggregate file")
    _require_exact_raw(
        files[manifest.resource_environment_raw_evidence.path],
        environment_raw_evidence_bytes(attestation.linux_environment, attestation.evidence_scope),
        label="resource/environment evidence",
    )
    if (
        _parse_model(
            files[manifest.nvidia_isolation_evidence.path],
            NvidiaT4ResourceAttestation,
            label="NVIDIA isolation evidence",
        )
        != attestation.nvidia_resources
    ):
        raise Stage2ExperimentError("NVIDIA isolation evidence differs from final attestation")
    if (
        manifest.nvidia_isolation_raw_evidence.sha256
        != attestation.nvidia_resources.isolation_evidence_sha256
    ):
        raise Stage2ExperimentError("hardware raw hash has no matching aggregate file")
    _require_exact_raw(
        files[manifest.nvidia_isolation_raw_evidence.path],
        nvidia_raw_evidence_bytes(attestation.nvidia_resources, attestation.evidence_scope),
        label="NVIDIA isolation evidence",
    )
    if (
        manifest.public_safety_raw_scan_evidence.sha256
        != attestation.public_safety.raw_scan_evidence_sha256
    ):
        raise Stage2ExperimentError("public-safety raw hash has no matching aggregate file")
    shared_models: tuple[tuple[ManifestBoundFile, type[StrictModel], StrictModel, str], ...] = (
        (
            manifest.execution_lock_snapshot,
            RuntimePackageExecutionLockAttestation,
            attestation.execution_lock,
            "execution-lock snapshot",
        ),
        (
            manifest.model_tokenizer_snapshot_manifest,
            ModelTokenizerSnapshotManifest,
            attestation.snapshot_manifest,
            "model/tokenizer snapshot manifest",
        ),
        (
            manifest.launch_specification,
            Stage2LaunchSpec,
            attestation.launch_spec,
            "launch specification",
        ),
        (
            manifest.public_safety_result,
            PublicSafetyAttestation,
            attestation.public_safety,
            "public-safety result",
        ),
        (
            manifest.shared_workload_definition,
            Stage2ExperimentWorkload,
            attestation.workload,
            "shared workload definition",
        ),
        (
            manifest.metric_availability_summary,
            Stage2MetricAvailabilitySummary,
            attestation.experiment_metric_availability,
            "metric-availability summary",
        ),
        (
            manifest.experiment_summary,
            Stage2ExperimentSummary,
            attestation.summary,
            "experiment summary",
        ),
        (
            manifest.aggregate_validation_result,
            Stage2AggregateValidationResult,
            attestation.aggregate_validation_result,
            "aggregate validation result",
        ),
    )
    for reference, model, expected, label in shared_models:
        if _parse_model(files[reference.path], model, label=label) != expected:
            raise Stage2ExperimentError(f"{label} differs from final attestation")
    resolver_raw, installed_raw = execution_lock_raw_evidence_bytes(
        attestation.execution_lock, attestation.evidence_scope
    )
    runtime_raw = (
        (
            manifest.runtime_resolver_lock_evidence,
            attestation.execution_lock.resolver_lock_sha256,
            resolver_raw,
            "runtime resolver lock",
        ),
        (
            manifest.runtime_installed_distribution_inventory,
            attestation.execution_lock.installed_distribution_inventory_sha256,
            installed_raw,
            "installed distribution inventory",
        ),
    )
    for reference, expected_sha, expected_bytes, label in runtime_raw:
        if reference.sha256 != expected_sha:
            raise Stage2ExperimentError(f"{label} hash has no matching aggregate file")
        _require_exact_raw(files[reference.path], expected_bytes, label=label)
    if (
        manifest.reviewed_execution_lock.sha256
        != attestation.execution_lock.reviewed_protocol_lock_sha256
    ):
        raise Stage2ExperimentError("reviewed execution-lock hash has no matching aggregate file")
    if (
        manifest.snapshot_read_only_verification_evidence.sha256
        != attestation.snapshot_manifest.read_only_transition.verification_evidence_sha256
    ):
        raise Stage2ExperimentError("snapshot verification hash has no matching aggregate file")
    _require_exact_raw(
        files[manifest.snapshot_read_only_verification_evidence.path],
        snapshot_read_only_raw_evidence_bytes(
            attestation.snapshot_manifest, attestation.evidence_scope
        ),
        label="snapshot read-only verification",
    )
    _require_exact_raw(
        files[manifest.public_safety_raw_scan_evidence.path],
        public_safety_raw_evidence_bytes(attestation.public_safety, attestation.evidence_scope),
        label="public-safety scan",
    )
    if attestation.public_safety.scan_inventory_sha256 != sha256_identity(
        tuple(repetition.repetition_manifest.files for repetition in attestation.repetitions)
    ):
        raise Stage2ExperimentError("public-safety pass is detached from repetition inventories")
    for index, repetition in enumerate(attestation.repetitions):
        prefix = f"repetition-{repetition.repetition_index:02d}/"
        validate_prometheus_raw_capture_binding(
            files[prefix + repetition.prometheus_measurement.baseline_raw_exposition_file.path],
            measurement=repetition.prometheus_measurement,
            evidence_scope=attestation.evidence_scope,
            repetition_index=repetition.repetition_index,
            boundary="baseline",
        )
        validate_prometheus_raw_capture_binding(
            files[prefix + repetition.prometheus_measurement.final_raw_exposition_file.path],
            measurement=repetition.prometheus_measurement,
            evidence_scope=attestation.evidence_scope,
            repetition_index=repetition.repetition_index,
            boundary="final",
        )
        if (
            _parse_model(
                files[manifest.repetition_manifest_files[index].path],
                Stage2BundleManifest,
                label="repetition manifest",
            )
            != repetition.repetition_manifest
        ):
            raise Stage2ExperimentError("repetition manifest differs from final attestation")
        _validate_repetition_bundle_directory(root, index + 1, repetition.repetition_manifest)
        if (
            _parse_model(
                files[manifest.cuda_execution_attestation_files[index].path],
                Stage2RepetitionCudaAttestation,
                label="CUDA execution attestation",
            )
            != repetition.cuda_execution
        ):
            raise Stage2ExperimentError("CUDA execution attestation differs from final attestation")
        if (
            _parse_model(
                files[manifest.prometheus_measurement_attestation_files[index].path],
                PrometheusMeasurementAttestation,
                label="Prometheus measurement attestation",
            )
            != repetition.prometheus_measurement
        ):
            raise Stage2ExperimentError(
                "Prometheus measurement attestation differs from final attestation"
            )
        if (
            _parse_model(
                files[manifest.cancellation_result_files[index].path],
                CancellationResult,
                label="cancellation result",
            )
            != repetition.runtime_control.cancellation_result
        ):
            raise Stage2ExperimentError("cancellation result differs from final attestation")
        for request in repetition.measured_requests:
            reconstructed_request = reconstruct_request_from_raw_evidence(
                {
                    field: files[prefix + getattr(request.raw_evidence, field).path]
                    for field in REQUEST_EVIDENCE_FIELDS
                },
                attestation.evidence_scope,
            )
            if reconstructed_request != (
                request.repetition_index,
                request.case_id,
                request.external_request_id,
                request.request_evidence,
                request.request_identity,
                request.lifecycle,
                request.wire_capture,
            ):
                raise Stage2ExperimentError(
                    f"measured request {request.external_request_id} does not reconstruct from raw"
                )
        cancellation_capture = cancellation_wire_capture_from_raw(
            files[prefix + repetition.cancellation_wire_file.path]
        )
        if cancellation_capture != repetition.cancellation_wire:
            raise Stage2ExperimentError(
                f"repetition {repetition.repetition_index} cancellation wire differs"
            )
        _require_exact_raw(
            files[prefix + repetition.cancellation_wire_file.path],
            cancellation_wire_raw_evidence_bytes(repetition, attestation.evidence_scope),
            label=f"repetition {repetition.repetition_index} cancellation HTTP/SSE wire",
        )
        for reference in repetition.cuda_execution.raw_evidence_files:
            _require_exact_raw(
                files[prefix + reference.path],
                cuda_raw_evidence_bytes(
                    repetition.cuda_execution,
                    attestation.evidence_scope,
                    reference.path,
                ),
                label=f"repetition {repetition.repetition_index} CUDA execution",
            )
    parsed_comparisons = tuple(
        _parse_model(
            files[reference.path],
            Stage2CrossRestartComparison,
            label="semantic comparison",
        )
        for reference in manifest.semantic_comparison_files
    )
    if parsed_comparisons != attestation.comparisons:
        raise Stage2ExperimentError("semantic comparisons differ from final attestation")
    return attestation


def reconstruct_experiment_attestation(root: Path) -> Stage2ReconstructedExperiment:
    if _path_has_symlink_ancestor(root):
        raise Stage2ExperimentError("aggregate directory cannot have symlink ancestors")
    manifest_path = root / AGGREGATE_MANIFEST_PATH
    if not manifest_path.is_file() or manifest_path.is_symlink():
        raise Stage2ExperimentError("aggregate manifest is missing or unsafe")
    manifest_bytes = manifest_path.read_bytes()
    manifest = _parse_model(
        manifest_bytes,
        Stage2AggregateExperimentManifest,
        label="aggregate manifest",
    )
    if manifest.state is AggregateRootState.INCOMPLETE:
        raise Stage2ExperimentError("aggregate root is incomplete")
    files = _validate_aggregate_inventory(root, manifest)
    manifest_mtime = manifest_path.stat().st_mtime_ns
    if any((root / entry.path).stat().st_mtime_ns >= manifest_mtime for entry in manifest.files):
        raise Stage2ExperimentError("aggregate manifest was not written strictly last")
    attestation = _validate_terminal_graph(root, manifest, files)
    return Stage2ReconstructedExperiment(
        attestation=attestation,
        aggregate_manifest=manifest,
        aggregate_manifest_sha256=hashlib.sha256(manifest_bytes).hexdigest(),
    )


def semantic_nonreproduction_result(
    comparison: Stage2CrossRestartComparison,
) -> AggregateComparisonState:
    """Expose the mandatory terminal state for a retained semantic mismatch."""

    return comparison.comparison.state
