"""Truthful HTTPX-layer transport provenance for the Stage 2A protocol.

Stage 2A constructs only deterministic CPU fixture records.  The capture labels in
this module describe HTTPX request/response objects, raw header pairs, and raw body
chunks.  They do not claim Ethernet, IP, TCP, TLS, kernel, proxy, or server-parser
observation.
"""

from __future__ import annotations

import base64
import binascii
import hashlib
import re
from typing import Annotated, Final, Literal, Self

from pydantic import Field, model_validator

from llm_inference_systems.canonical import sha256_identity
from llm_inference_systems.contracts import (
    Identifier,
    NonNegativeInt,
    Sha256,
    StrictModel,
)
from llm_inference_systems.stage2_contracts import Stage2EvidenceScope
from llm_inference_systems.stage2_runtime import Stage2LaunchSpec

HTTPX_EXCHANGE_CAPTURE_LAYER: Final = (
    "HTTPX_REQUEST_RESPONSE_OBJECTS_HEADERS_AND_AITER_RAW_BODY_CHUNKS"
)
HTTPX_REQUEST_HEADER_CAPTURE_SOURCE: Final = "HTTPX_REQUEST_OBJECT_AND_HEADERS_RAW"
HTTPX_RESPONSE_HEADER_CAPTURE_SOURCE: Final = "HTTPX_RESPONSE_OBJECT_AND_HEADERS_RAW"
HTTPX_REQUESTED_VERSION_SOURCE: Final = "HTTPX_CLIENT_CONFIGURATION_HTTP2_FALSE"
HTTPX_OBSERVED_VERSION_SOURCE: Final = "HTTPX_RESPONSE"

_SINGLETON_HEADER_NAMES: Final = frozenset(
    {
        "authorization",
        "connection",
        "content-length",
        "content-type",
        "cookie",
        "date",
        "host",
        "proxy-authorization",
        "server",
        "set-cookie",
        "transfer-encoding",
        "x-request-id",
    }
)
_SECRET_HEADER_NAME_RE: Final = re.compile(
    r"(?:^|[-_])(?:authorization|authentication|cookie|api[-_]?key|"
    r"access[-_]?key|auth|token|secret|credential|password)(?:$|[-_])",
    re.IGNORECASE,
)
_TOKEN_RE: Final = re.compile(r"[!#$%&'*+.^_`|~0-9A-Za-z-]+")
_MEDIA_TYPE_RE: Final = re.compile(r"[!#$%&'*+.^_`|~0-9A-Za-z-]+/[!#$%&'*+.^_`|~0-9A-Za-z-]+")
_FIXTURE_COMPLETION_REQUEST_HEADERS: Final = (
    "host",
    "accept",
    "accept-encoding",
    "connection",
    "user-agent",
    "x-request-id",
    "content-type",
    "content-length",
)
_FIXTURE_COMPLETION_RESPONSE_HEADERS: Final = (
    "date",
    "server",
    "x-request-id",
    "content-type",
    "transfer-encoding",
)
_FIXTURE_METRICS_REQUEST_HEADERS: Final = (
    "host",
    "accept-encoding",
    "connection",
    "user-agent",
    "accept",
)
_FIXTURE_METRICS_RESPONSE_HEADERS: Final = (
    "date",
    "server",
    "content-length",
    "content-type",
)


def _decode_canonical_base64(value: str, *, label: str) -> bytes:
    try:
        decoded = base64.b64decode(value, validate=True)
    except (binascii.Error, ValueError) as error:
        raise ValueError(f"{label} Base64 is invalid") from error
    if base64.b64encode(decoded).decode("ascii") != value:
        raise ValueError(f"{label} Base64 is not canonical")
    return decoded


def parse_content_type(value: str) -> tuple[str, tuple[tuple[str, str], ...]]:
    """Return a deterministic media type and retained normalized parameters."""

    parts: list[str] = []
    start = 0
    quoted = False
    escaped = False
    for index, character in enumerate(value):
        codepoint = ord(character)
        if (codepoint < 0x20 and character != "\t") or codepoint == 0x7F:
            raise ValueError("HTTP Content-Type contains a prohibited control character")
        if escaped:
            escaped = False
        elif quoted and character == "\\":
            escaped = True
        elif character == '"':
            quoted = not quoted
        elif character == ";" and not quoted:
            parts.append(value[start:index].strip(" \t"))
            start = index + 1
    if quoted or escaped:
        raise ValueError("HTTP Content-Type quoted parameter is malformed")
    parts.append(value[start:].strip(" \t"))
    media_type = parts[0].casefold() if parts else ""
    if not _MEDIA_TYPE_RE.fullmatch(media_type):
        raise ValueError("HTTP Content-Type media type is malformed")
    parameters: list[tuple[str, str]] = []
    seen: set[str] = set()
    for raw in parts[1:]:
        if not raw or "=" not in raw:
            raise ValueError("HTTP Content-Type parameter is malformed")
        name, parameter_value = (item.strip(" \t") for item in raw.split("=", 1))
        normalized_name = name.casefold()
        valid_parameter_value = bool(_TOKEN_RE.fullmatch(parameter_value))
        if parameter_value.startswith('"') and parameter_value.endswith('"'):
            inner = parameter_value[1:-1]
            position = 0
            valid_parameter_value = True
            while position < len(inner):
                character = inner[position]
                codepoint = ord(character)
                if character == "\\":
                    position += 1
                    if position == len(inner):
                        valid_parameter_value = False
                        break
                    codepoint = ord(inner[position])
                    if (codepoint < 0x20 and inner[position] != "\t") or codepoint == 0x7F:
                        valid_parameter_value = False
                        break
                elif (
                    character == '"'
                    or (codepoint < 0x20 and character != "\t")
                    or codepoint == 0x7F
                ):
                    valid_parameter_value = False
                    break
                position += 1
        if (
            not _TOKEN_RE.fullmatch(normalized_name)
            or not valid_parameter_value
            or normalized_name in seen
        ):
            raise ValueError("HTTP Content-Type parameter is malformed or ambiguous")
        seen.add(normalized_name)
        parameters.append((normalized_name, parameter_value))
    return media_type, tuple(parameters)


class FixtureWireCaptureProvenance(StrictModel):
    capture_kind: Literal["FIXTURE_CONSTRUCTOR"]
    evidence_scope: Literal[Stage2EvidenceScope.TEST_FIXTURE_ONLY]
    classification: Literal["SYNTHETIC_PROTOCOL_SHAPE_ONLY"]
    fixture_marker: Literal["TEST_FIXTURE_ONLY"]
    fixture_identity_sha256: Sha256
    identity_sha256: Sha256

    @model_validator(mode="after")
    def validate_identity(self) -> Self:
        if self.identity_sha256 != sha256_identity(
            self, omit_fields=frozenset({"identity_sha256"})
        ):
            raise ValueError("fixture transport provenance identity does not reconstruct")
        return self


class CollectorWireCaptureProvenance(StrictModel):
    capture_kind: Literal["COLLECTOR_CAPTURE"]
    evidence_scope: Literal[Stage2EvidenceScope.FUTURE_REAL_RUNTIME]
    classification: Literal["FUTURE_REAL_RUNTIME"]
    collector_identity_sha256: Sha256
    server_process_identity: Identifier
    model_snapshot_identity_sha256: Sha256
    environment_identity_sha256: Sha256
    identity_sha256: Sha256

    @model_validator(mode="after")
    def validate_identity(self) -> Self:
        if self.identity_sha256 != sha256_identity(
            self, omit_fields=frozenset({"identity_sha256"})
        ):
            raise ValueError("collector transport provenance identity does not reconstruct")
        return self


Stage2WireCaptureProvenance = Annotated[
    FixtureWireCaptureProvenance | CollectorWireCaptureProvenance,
    Field(discriminator="capture_kind"),
]


class Stage2LosslessHeaderField(StrictModel):
    ordinal: NonNegativeInt
    name_base64: str
    value_base64: str
    name_byte_count: NonNegativeInt
    value_byte_count: NonNegativeInt
    name_sha256: Sha256
    value_sha256: Sha256
    normalized_name: str
    normalized_value: str
    identity_sha256: Sha256

    @model_validator(mode="after")
    def validate_field(self) -> Self:
        name_bytes = _decode_canonical_base64(self.name_base64, label="header name")
        value_bytes = _decode_canonical_base64(self.value_base64, label="header value")
        if (
            len(name_bytes) != self.name_byte_count
            or len(value_bytes) != self.value_byte_count
            or hashlib.sha256(name_bytes).hexdigest() != self.name_sha256
            or hashlib.sha256(value_bytes).hexdigest() != self.value_sha256
        ):
            raise ValueError("header byte count or SHA-256 differs")
        try:
            name = name_bytes.decode("ascii")
            value = value_bytes.decode("ascii")
        except UnicodeDecodeError as error:
            raise ValueError("HTTP header field encoding is malformed") from error
        if not _TOKEN_RE.fullmatch(name):
            raise ValueError("HTTP header name is malformed")
        if any(byte < 0x20 or byte == 0x7F for byte in value_bytes):
            raise ValueError("HTTP header value contains a prohibited control byte")
        if self.normalized_name != name.casefold() or self.normalized_value != value.strip(" \t"):
            raise ValueError("normalized header view differs from retained bytes")
        if _SECRET_HEADER_NAME_RE.search(self.normalized_name):
            raise ValueError("secret-bearing HTTP header is prohibited from durable capture")
        if self.identity_sha256 != sha256_identity(
            self, omit_fields=frozenset({"identity_sha256"})
        ):
            raise ValueError("lossless header-field identity does not reconstruct")
        return self


class Stage2OrderedHeadersCapture(StrictModel):
    direction: Literal["TRANSMITTED_REQUEST", "RECEIVED_RESPONSE"]
    capture_source: Literal[
        "HTTPX_REQUEST_OBJECT_AND_HEADERS_RAW",
        "HTTPX_RESPONSE_OBJECT_AND_HEADERS_RAW",
    ]
    capture_complete_at_declared_layer: Literal[True]
    observation_offset_ns: NonNegativeInt
    observed_field_count: NonNegativeInt
    fields: tuple[Stage2LosslessHeaderField, ...] = Field(min_length=1)
    normalized_view: tuple[tuple[str, str], ...]
    identity_sha256: Sha256

    @model_validator(mode="after")
    def validate_headers(self) -> Self:
        expected_source = (
            HTTPX_REQUEST_HEADER_CAPTURE_SOURCE
            if self.direction == "TRANSMITTED_REQUEST"
            else HTTPX_RESPONSE_HEADER_CAPTURE_SOURCE
        )
        if self.capture_source != expected_source:
            raise ValueError("HTTP header capture source differs from its direction")
        if self.observed_field_count != len(self.fields):
            raise ValueError("HTTP header count differs from retained ordered fields")
        if tuple(item.ordinal for item in self.fields) != tuple(range(len(self.fields))):
            raise ValueError("HTTP header ordinals are missing, duplicated, or reordered")
        expected_view = tuple((item.normalized_name, item.normalized_value) for item in self.fields)
        if self.normalized_view != expected_view:
            raise ValueError("deterministic normalized header view does not reconstruct")
        by_name: dict[str, list[str]] = {}
        for name, value in expected_view:
            by_name.setdefault(name, []).append(value)
        duplicated_singletons = sorted(
            name for name in _SINGLETON_HEADER_NAMES if len(by_name.get(name, ())) > 1
        )
        if duplicated_singletons:
            raise ValueError("singleton HTTP header is duplicated or ambiguous")
        if "content-length" in by_name and "transfer-encoding" in by_name:
            raise ValueError("HTTP message has ambiguous framing headers")
        if self.identity_sha256 != sha256_identity(
            self, omit_fields=frozenset({"identity_sha256"})
        ):
            raise ValueError("ordered HTTP-header capture identity does not reconstruct")
        return self

    def values(self, name: str) -> tuple[str, ...]:
        return tuple(value for field, value in self.normalized_view if field == name.casefold())

    def effective(self, name: str) -> str:
        values = self.values(name)
        if len(values) != 1:
            raise ValueError(f"effective {name} header is missing or ambiguous")
        return values[0]


class Stage2HTTPExchangeCapture(StrictModel):
    """Complete exchange identity at the explicitly declared HTTPX layer."""

    schema_version: Literal["0.3.0"]
    measurement_protocol_version: Literal["0.3.0"]
    exchange_purpose: Literal[
        "MEASURED_COMPLETION",
        "CANCELLATION",
        "PROMETHEUS_BASELINE",
        "PROMETHEUS_FINAL",
    ]
    repetition_index: Literal[1, 2, 3]
    evidence_unit_id: Identifier
    external_request_id: Identifier | None
    capture_layer: Literal["HTTPX_REQUEST_RESPONSE_OBJECTS_HEADERS_AND_AITER_RAW_BODY_CHUNKS"]
    provenance: Stage2WireCaptureProvenance
    method: Literal["GET", "POST"]
    scheme: Literal["http"]
    host: Literal["127.0.0.1"]
    port: Literal[8000]
    request_target: Literal["/v1/completions", "/metrics"]
    requested_http_version: Literal["HTTP/1.1"]
    requested_http_version_source: Literal["HTTPX_CLIENT_CONFIGURATION_HTTP2_FALSE"]
    observed_response_http_version: Literal["HTTP/1.1"]
    observed_response_http_version_source: Literal["HTTPX_RESPONSE"]
    response_status: Literal[200]
    request_headers: Stage2OrderedHeadersCapture
    response_headers: Stage2OrderedHeadersCapture
    request_header_count: NonNegativeInt
    response_header_count: NonNegativeInt
    request_header_capture_complete_at_layer: Literal[True]
    response_header_capture_complete_at_layer: Literal[True]
    full_response_content_type: str
    normalized_response_media_type: Literal[
        "text/event-stream",
        "text/plain",
        "application/openmetrics-text",
    ]
    response_content_type_parameters: tuple[tuple[str, str], ...]
    request_body_transmission_observation_offset_ns: NonNegativeInt
    response_header_observation_offset_ns: NonNegativeInt
    request_body_byte_count: NonNegativeInt
    request_body_sha256: Sha256
    response_body_byte_count: NonNegativeInt
    response_body_sha256: Sha256
    response_body_inventory_sha256: Sha256
    response_body_completion_observation_offset_ns: NonNegativeInt
    transport_terminal_observation_offset_ns: NonNegativeInt
    transport_terminal_classification: Literal[
        "CLEAN_EOF",
        "CLEAN_RESPONSE_CLOSE",
        "INTENTIONAL_CLIENT_CLOSE_AFTER_FIRST_GENERATION_TOKEN",
    ]
    response_body_capture_complete_through_terminal_at_layer: Literal[True]
    server_process_identity: Identifier
    launch_spec_identity_sha256: Sha256
    identity_sha256: Sha256

    @model_validator(mode="after")
    def validate_exchange(self) -> Self:
        if self.capture_layer != HTTPX_EXCHANGE_CAPTURE_LAYER:
            raise ValueError("HTTP exchange capture layer differs")
        if self.request_headers.direction != "TRANSMITTED_REQUEST":
            raise ValueError("HTTP exchange request headers use the wrong direction")
        if self.response_headers.direction != "RECEIVED_RESPONSE":
            raise ValueError("HTTP exchange response headers use the wrong direction")
        if (
            self.request_header_count != self.request_headers.observed_field_count
            or self.response_header_count != self.response_headers.observed_field_count
            or self.request_header_capture_complete_at_layer
            != self.request_headers.capture_complete_at_declared_layer
            or self.response_header_capture_complete_at_layer
            != self.response_headers.capture_complete_at_declared_layer
        ):
            raise ValueError("HTTP exchange header count or completeness claim differs")
        if (
            self.request_body_transmission_observation_offset_ns
            < self.request_headers.observation_offset_ns
            or self.response_header_observation_offset_ns
            != self.response_headers.observation_offset_ns
            or self.response_header_observation_offset_ns
            <= self.request_body_transmission_observation_offset_ns
            or self.response_body_completion_observation_offset_ns
            < self.response_header_observation_offset_ns
            or self.transport_terminal_observation_offset_ns
            < self.response_body_completion_observation_offset_ns
        ):
            raise ValueError("HTTP exchange observation order is invalid")
        try:
            full_content_type = self.response_headers.effective("content-type")
            media_type, parameters = parse_content_type(full_content_type)
        except ValueError as error:
            raise ValueError("response Content-Type is missing, malformed, or ambiguous") from error
        if (
            self.full_response_content_type != full_content_type
            or self.normalized_response_media_type != media_type
            or self.response_content_type_parameters != parameters
        ):
            raise ValueError("response Content-Type views do not reconstruct")
        if self.request_headers.effective("host") != f"{self.host}:{self.port}":
            raise ValueError("transmitted Host header differs from the exchange endpoint")
        request_lengths = self.request_headers.values("content-length")
        response_lengths = self.response_headers.values("content-length")
        if request_lengths and request_lengths != (str(self.request_body_byte_count),):
            raise ValueError("transmitted Content-Length differs from exact request bytes")
        if response_lengths and response_lengths != (str(self.response_body_byte_count),):
            raise ValueError("received Content-Length differs from exact response bytes")
        transfer_encodings = self.response_headers.values("transfer-encoding")
        if transfer_encodings and tuple(value.casefold() for value in transfer_encodings) != (
            "chunked",
        ):
            raise ValueError("received Transfer-Encoding is unsupported or ambiguous")
        completion = self.exchange_purpose in {"MEASURED_COMPLETION", "CANCELLATION"}
        if completion:
            if (
                self.method != "POST"
                or self.request_target != "/v1/completions"
                or self.external_request_id is None
                or self.request_headers.effective("x-request-id") != self.external_request_id
                or self.response_headers.effective("x-request-id") != self.external_request_id
                or parse_content_type(self.request_headers.effective("content-type"))[0]
                != "application/json"
                or self.normalized_response_media_type != "text/event-stream"
                or self.request_body_byte_count == 0
                or request_lengths != (str(self.request_body_byte_count),)
            ):
                raise ValueError("completion HTTP exchange identity or media contract differs")
            expected_terminal = (
                "CLEAN_EOF"
                if self.exchange_purpose == "MEASURED_COMPLETION"
                else "INTENTIONAL_CLIENT_CLOSE_AFTER_FIRST_GENERATION_TOKEN"
            )
            if self.transport_terminal_classification != expected_terminal:
                raise ValueError("completion HTTP exchange terminal classification differs")
        else:
            if (
                self.method != "GET"
                or self.request_target != "/metrics"
                or self.external_request_id is not None
                or self.request_body_byte_count != 0
                or self.request_body_sha256 != hashlib.sha256(b"").hexdigest()
                or self.normalized_response_media_type
                not in {"text/plain", "application/openmetrics-text"}
                or self.transport_terminal_classification != "CLEAN_RESPONSE_CLOSE"
            ):
                raise ValueError("Prometheus scrape HTTP exchange identity differs")
        if (
            isinstance(self.provenance, CollectorWireCaptureProvenance)
            and self.provenance.server_process_identity != self.server_process_identity
        ):
            raise ValueError("collector provenance differs from the HTTP server process")
        expected_request_names = (
            _FIXTURE_COMPLETION_REQUEST_HEADERS if completion else _FIXTURE_METRICS_REQUEST_HEADERS
        )
        captured_request_names = tuple(name for name, _ in self.request_headers.normalized_view)
        for name in expected_request_names:
            if captured_request_names.count(name) != 1:
                raise ValueError(
                    "complete HTTPX request-header capture omits or duplicates an "
                    "ordinary client field"
                )
        expected_response_names = (
            _FIXTURE_COMPLETION_RESPONSE_HEADERS
            if completion
            else _FIXTURE_METRICS_RESPONSE_HEADERS
        )
        captured_response_names = tuple(name for name, _ in self.response_headers.normalized_view)
        for name in expected_response_names:
            if captured_response_names.count(name) != 1:
                raise ValueError(
                    "complete HTTPX response-header capture omits or duplicates an "
                    "ordinary server field"
                )
        if isinstance(self.provenance, FixtureWireCaptureProvenance) and (
            captured_request_names != expected_request_names
            or captured_response_names != expected_response_names
        ):
            raise ValueError(
                "fixture HTTPX header capture omits, adds, or reorders an observed field"
            )
        if self.identity_sha256 != sha256_identity(
            self, omit_fields=frozenset({"identity_sha256"})
        ):
            raise ValueError("HTTP exchange identity does not reconstruct")
        return self

    def require_launch_and_process(
        self,
        launch_spec: Stage2LaunchSpec,
        *,
        server_process_identity: str,
    ) -> None:
        if (
            self.host != launch_spec.host
            or self.port != launch_spec.port
            or self.launch_spec_identity_sha256 != sha256_identity(launch_spec)
            or self.server_process_identity != server_process_identity
        ):
            raise ValueError("HTTP exchange differs from accepted launch or server process")
