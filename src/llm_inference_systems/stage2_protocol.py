"""CPU-only Stage 2 request construction, identity correlation, and SSE reconciliation."""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Callable
from dataclasses import dataclass
from itertools import pairwise

from pydantic import ValidationError

from llm_inference_systems.canonical import sha256_identity
from llm_inference_systems.sse import IncrementalSSEParser, SSEProtocolError
from llm_inference_systems.stage2_contracts import (
    MetricAvailability,
    RawLogRecord,
    RequestIdentityChain,
    Stage2CancellationRequest,
    Stage2CompletionRequest,
    Stage2PerRequestMetrics,
    Stage2RequestEnvelope,
    Stage2RequestEvidence,
    Stage2StreamOptions,
    Stage2TimingRecord,
    Stage2TokenEvent,
    Stage2Usage,
)


class Stage2ProtocolError(ValueError):
    """Raised when future-runtime evidence violates the Stage 2 protocol."""


def _json_without_duplicate_keys(data: str) -> object:
    def pairs_hook(pairs: list[tuple[str, object]]) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, value in pairs:
            if key in result:
                raise Stage2ProtocolError("completion response JSON contains a duplicate field")
            result[key] = value
        return result

    try:
        return json.loads(data, object_pairs_hook=pairs_hook)
    except json.JSONDecodeError as error:
        raise Stage2ProtocolError("malformed SSE JSON") from error


@dataclass(frozen=True, slots=True)
class RetainedBodyChunk:
    observation_offset_ns: int
    data: bytes
    sha256: str


@dataclass(frozen=True, slots=True)
class RetainedSSEEvent:
    ordinal: int
    observation_offset_ns: int
    kind: str
    data: str | None
    comments: tuple[str, ...]


def build_completion_request(
    external_base_id: str,
    prompt_token_ids: tuple[int, ...],
) -> Stage2RequestEnvelope:
    body = Stage2CompletionRequest(
        model="qwen2.5-0.5b-instruct-stage2",
        prompt=prompt_token_ids,
        request_id=external_base_id,
        stream=True,
        stream_options=Stage2StreamOptions(include_usage=True),
        return_token_ids=True,
        stream_interval=1,
        add_special_tokens=False,
        temperature=0,
        top_p=1,
        seed=0,
        n=1,
        max_tokens=32,
        min_tokens=32,
        ignore_eos=True,
        echo=False,
    )
    return Stage2RequestEnvelope(x_request_id=external_base_id, body=body)


def build_cancellation_request(
    external_base_id: str,
    prompt_token_ids: tuple[int, ...],
) -> Stage2CancellationRequest:
    return Stage2CancellationRequest(
        model="qwen2.5-0.5b-instruct-stage2",
        prompt=prompt_token_ids,
        request_id=external_base_id,
        stream=True,
        stream_options=Stage2StreamOptions(include_usage=True),
        return_token_ids=True,
        stream_interval=1,
        add_special_tokens=False,
        temperature=0,
        top_p=1,
        seed=0,
        n=1,
        max_tokens=512,
        min_tokens=512,
        ignore_eos=True,
        echo=False,
    )


def validate_effective_request(
    requested: Stage2CompletionRequest,
    effective: Stage2CompletionRequest,
) -> None:
    if requested.model_dump(mode="json") != effective.model_dump(mode="json"):
        raise Stage2ProtocolError("effective request differs from the frozen request")


def correlate_request_logs(
    external_base_id: str,
    records: tuple[RawLogRecord, ...],
    *,
    cancellation: bool,
) -> RequestIdentityChain:
    lines = tuple(record.raw_record for record in records)
    serving_item = f"cmpl-{external_base_id}-0"
    received_pattern = re.compile(rf"\bReceived request {re.escape(serving_item)}:")
    internal_pattern = re.compile(rf"\bAdded request ({re.escape(serving_item)}-[0-9a-f]{{8}})\.")
    external_abort_pattern = re.compile(rf"\bRequest {re.escape(serving_item)} aborted\.")
    received = [record for record in records if received_pattern.search(record.raw_record)]
    internal_matches = [
        (record, match)
        for record in records
        if (match := internal_pattern.search(record.raw_record))
    ]
    external_aborts = [
        record for record in records if external_abort_pattern.search(record.raw_record)
    ]
    if len(received) != 1:
        raise Stage2ProtocolError("request logger correlation is missing or ambiguous")
    if len(internal_matches) != 1:
        raise Stage2ProtocolError("internal request-add correlation is missing or ambiguous")
    request_add_record, internal_match = internal_matches[0]
    internal_id = internal_match.group(1)
    internal_abort_pattern = re.compile(rf"\bAborted request\(s\) {re.escape(internal_id)}\.")
    internal_aborts = [
        record for record in records if internal_abort_pattern.search(record.raw_record)
    ]

    generic_ids = {
        match.group(0)
        for line in lines
        for match in re.finditer(r"cmpl-[A-Za-z0-9._-]+-0(?:-[0-9a-f]{8})?", line)
    }
    if not generic_ids.issubset({serving_item, internal_id}):
        raise Stage2ProtocolError("cross-request log correlation detected")
    if cancellation:
        if len(external_aborts) != 1 or len(internal_aborts) != 1:
            raise Stage2ProtocolError("cancellation requires both external and internal abort logs")
    elif external_aborts or internal_aborts:
        raise Stage2ProtocolError("unexpected abort logs observed for a completed request")
    return RequestIdentityChain(
        external_base_id=external_base_id,
        response_body_id=f"cmpl-{external_base_id}",
        serving_item_id=serving_item,
        internal_engine_id=internal_id,
        request_received_log=received[0],
        request_add_log=request_add_record,
        external_abort_log=external_aborts[0] if external_aborts else None,
        internal_abort_log=internal_aborts[0] if internal_aborts else None,
    )


def retain_raw_log_records(
    lines: tuple[str, ...],
    *,
    source_stream_id: str,
    first_observation_offset_ns: int = 0,
    observation_offsets_ns: tuple[int, ...] | None = None,
) -> tuple[RawLogRecord, ...]:
    """Retain exact fixture log bytes with deterministic offsets and hashes."""

    if observation_offsets_ns is not None and (
        len(observation_offsets_ns) != len(lines)
        or observation_offsets_ns != tuple(sorted(observation_offsets_ns))
    ):
        raise Stage2ProtocolError("raw log observation offsets must match and be monotonic")
    records: list[RawLogRecord] = []
    byte_offset = 0
    for ordinal, line in enumerate(lines):
        encoded = line.encode("utf-8")
        records.append(
            RawLogRecord(
                source_stream_id=source_stream_id,
                record_ordinal=ordinal,
                byte_start=byte_offset,
                byte_end=byte_offset + len(encoded),
                observation_offset_ns=(
                    observation_offsets_ns[ordinal]
                    if observation_offsets_ns is not None
                    else first_observation_offset_ns + ordinal
                ),
                raw_record=line,
                raw_record_sha256=hashlib.sha256(encoded).hexdigest(),
            )
        )
        byte_offset += len(encoded) + 1
    return tuple(records)


def _object(value: object, *, field: str) -> dict[str, object]:
    if not isinstance(value, dict) or any(not isinstance(key, str) for key in value):
        raise Stage2ProtocolError(f"{field} must be an object")
    return value


def _array(value: object, *, field: str) -> list[object]:
    if not isinstance(value, list):
        raise Stage2ProtocolError(f"{field} must be an array")
    return value


def _token_ids(value: object, *, field: str) -> tuple[int, ...]:
    values = _array(value, field=field)
    if any(isinstance(item, bool) or not isinstance(item, int) or item < 0 for item in values):
        raise Stage2ProtocolError(f"{field} must contain nonnegative integer token IDs")
    return tuple(item for item in values if isinstance(item, int))


def _metrics(value: object) -> Stage2PerRequestMetrics:
    raw = _object(value, field="metrics")
    if any(
        item is not None
        and (
            isinstance(item, bool)
            or not isinstance(item, (int, float))
            or not math.isfinite(float(item))
            or float(item) < 0
        )
        for item in raw.values()
    ):
        raise Stage2ProtocolError("per-request metrics must be finite nonnegative numbers or null")
    try:
        return Stage2PerRequestMetrics.model_validate(raw)
    except ValidationError as error:
        raise Stage2ProtocolError(
            "per-request metrics require the exact five-field shape"
        ) from error


class Stage2StreamValidator:
    """Incrementally validate the four-terminal future completions stream."""

    def __init__(
        self,
        *,
        external_base_id: str,
        sent_prompt_token_ids: tuple[int, ...],
        dispatch_offset_ns: int,
        fixture_identity_sha256: str | None = None,
        frame_clock: Callable[[], int] | None = None,
    ) -> None:
        request = build_completion_request(external_base_id, sent_prompt_token_ids)
        self._external_base_id = external_base_id
        self._sent_prompt_token_ids = request.body.prompt
        self._dispatch_offset_ns = dispatch_offset_ns
        self._fixture_identity_sha256 = fixture_identity_sha256
        self._frame_clock = frame_clock
        self._parser = IncrementalSSEParser()
        self._response_headers_offset_ns: int | None = None
        self._first_body_offset_ns: int | None = None
        self._generation_terminal_offset_ns: int | None = None
        self._usage_terminal_offset_ns: int | None = None
        self._protocol_terminal_offset_ns: int | None = None
        self._transport_terminal_offset_ns: int | None = None
        self._token_events: list[Stage2TokenEvent] = []
        self._returned_prompt_ids: tuple[int, ...] | None = None
        self._output_ids: list[int] = []
        self._output_text_parts: list[str] = []
        self._usage: Stage2Usage | None = None
        self._server_metrics: Stage2PerRequestMetrics | None = None
        self._terminal_carried_tokens: bool | None = None
        self._last_observation_offset_ns = dispatch_offset_ns
        self._raw_body_chunks: list[RetainedBodyChunk] = []
        self._parsed_sse_events: list[RetainedSSEEvent] = []

    @property
    def retained_raw_body_chunks(self) -> tuple[RetainedBodyChunk, ...]:
        return tuple(self._raw_body_chunks)

    @property
    def parsed_sse_events(self) -> tuple[RetainedSSEEvent, ...]:
        return tuple(self._parsed_sse_events)

    def accept_response_headers(self, x_request_id: str, observation_offset_ns: int) -> None:
        if self._response_headers_offset_ns is not None:
            raise Stage2ProtocolError("duplicate response headers")
        if x_request_id != self._external_base_id:
            raise Stage2ProtocolError("response X-Request-Id differs")
        self._advance(observation_offset_ns)
        self._response_headers_offset_ns = observation_offset_ns

    def _advance(self, observation_offset_ns: int) -> None:
        if observation_offset_ns < self._last_observation_offset_ns:
            raise Stage2ProtocolError("stream observations are not monotonic")
        self._last_observation_offset_ns = observation_offset_ns

    def feed(self, chunk: bytes, observation_offset_ns: int) -> None:
        if chunk:
            self._raw_body_chunks.append(
                RetainedBodyChunk(
                    observation_offset_ns=observation_offset_ns,
                    data=chunk,
                    sha256=hashlib.sha256(chunk).hexdigest(),
                )
            )
        if self._transport_terminal_offset_ns is not None:
            raise Stage2ProtocolError("data observed after transport terminal")
        if self._response_headers_offset_ns is None:
            raise Stage2ProtocolError("response body observed before response headers")
        self._advance(observation_offset_ns)
        if chunk and self._first_body_offset_ns is None:
            self._first_body_offset_ns = observation_offset_ns
        try:
            frames = self._parser.feed(chunk)
        except SSEProtocolError as error:
            raise Stage2ProtocolError(str(error)) from error
        frame_offset_ns = observation_offset_ns
        for index, frame in enumerate(frames):
            if index and self._frame_clock is not None:
                frame_offset_ns = self._frame_clock()
                self._advance(frame_offset_ns)
            self._parsed_sse_events.append(
                RetainedSSEEvent(
                    ordinal=len(self._parsed_sse_events),
                    observation_offset_ns=frame_offset_ns,
                    kind=frame.kind,
                    data=frame.data,
                    comments=frame.comments,
                )
            )
            if self._protocol_terminal_offset_ns is not None:
                raise Stage2ProtocolError("post-protocol-terminal SSE data observed")
            if frame.kind == "comment":
                continue
            if frame.kind == "done":
                if self._usage_terminal_offset_ns is None:
                    raise Stage2ProtocolError("[DONE] observed before usage terminal")
                if frame_offset_ns <= self._usage_terminal_offset_ns:
                    raise Stage2ProtocolError("protocol terminal must follow usage terminal")
                self._protocol_terminal_offset_ns = frame_offset_ns
                continue
            if frame.data is None:
                raise Stage2ProtocolError("SSE data frame is empty")
            decoded = _json_without_duplicate_keys(frame.data)
            self._accept_response(_object(decoded, field="completion response"), frame_offset_ns)

    def _accept_response(self, value: dict[str, object], observation_offset_ns: int) -> None:
        if value.get("id") != f"cmpl-{self._external_base_id}":
            raise Stage2ProtocolError("completion response body ID differs")
        choices = _array(value.get("choices"), field="choices")
        if not choices:
            self._accept_usage(value, observation_offset_ns)
            return
        if len(choices) != 1:
            raise Stage2ProtocolError("each generation event must contain exactly one choice")
        if self._usage_terminal_offset_ns is not None:
            raise Stage2ProtocolError("generation data observed after usage terminal")
        choice = _object(choices[0], field="choice")
        if choice.get("index") != 0:
            raise Stage2ProtocolError("completion choice index differs")
        text = choice.get("text")
        if not isinstance(text, str):
            raise Stage2ProtocolError("completion choice text must be a string")
        raw_ids = choice.get("token_ids", [])
        output_ids = _token_ids(raw_ids, field="choice.token_ids")
        finish_reason = choice.get("finish_reason")
        if finish_reason not in (None, "length"):
            raise Stage2ProtocolError("finish reason must be null or length")
        if self._generation_terminal_offset_ns is not None:
            raise Stage2ProtocolError("generation data observed after generation terminal")
        if "prompt_token_ids" in choice and choice["prompt_token_ids"] is not None:
            if self._returned_prompt_ids is not None:
                raise Stage2ProtocolError("duplicate returned prompt token IDs")
            self._returned_prompt_ids = _token_ids(
                choice["prompt_token_ids"], field="choice.prompt_token_ids"
            )
        self._output_ids.extend(output_ids)
        if len(self._output_ids) > 32:
            raise Stage2ProtocolError("more than 32 output token IDs observed")
        self._output_text_parts.append(text)
        self._token_events.append(
            Stage2TokenEvent(
                observation_offset_ns=observation_offset_ns,
                output_token_ids=output_ids,
                text=text,
                finish_reason=finish_reason,
                prompt_token_ids=(
                    self._returned_prompt_ids
                    if len(self._token_events) == 0 and self._returned_prompt_ids is not None
                    else None
                ),
            )
        )
        if finish_reason == "length":
            if len(self._output_ids) != 32:
                raise Stage2ProtocolError("generation terminal requires exactly 32 output IDs")
            self._generation_terminal_offset_ns = observation_offset_ns
            self._terminal_carried_tokens = bool(output_ids)

    def _accept_usage(self, value: dict[str, object], observation_offset_ns: int) -> None:
        if self._generation_terminal_offset_ns is None:
            raise Stage2ProtocolError("usage terminal observed before generation terminal")
        if self._usage_terminal_offset_ns is not None:
            raise Stage2ProtocolError("duplicate usage terminal")
        if observation_offset_ns <= self._generation_terminal_offset_ns:
            raise Stage2ProtocolError("usage terminal must follow generation terminal")
        usage = _object(value.get("usage"), field="usage")
        try:
            self._usage = Stage2Usage.model_validate(usage)
        except ValueError as error:
            raise Stage2ProtocolError("usage reconciliation failed") from error
        if "metrics" not in value:
            raise Stage2ProtocolError("usage terminal requires per-request metrics")
        self._server_metrics = _metrics(value["metrics"])
        self._usage_terminal_offset_ns = observation_offset_ns

    def close_transport(
        self,
        observation_offset_ns: int,
        *,
        identity_chain: RequestIdentityChain,
    ) -> Stage2RequestEvidence:
        if self._transport_terminal_offset_ns is not None:
            raise Stage2ProtocolError("duplicate transport terminal")
        self._advance(observation_offset_ns)
        try:
            self._parser.finalize()
        except SSEProtocolError as error:
            raise Stage2ProtocolError(str(error)) from error
        if self._protocol_terminal_offset_ns is None:
            raise Stage2ProtocolError("transport closed before protocol terminal")
        if observation_offset_ns <= self._protocol_terminal_offset_ns:
            raise Stage2ProtocolError("transport terminal must follow protocol terminal")
        self._transport_terminal_offset_ns = observation_offset_ns
        expected_response_id = f"cmpl-{self._external_base_id}"
        expected_serving_item_id = f"{expected_response_id}-0"
        if (
            identity_chain.external_base_id != self._external_base_id
            or identity_chain.response_body_id != expected_response_id
            or identity_chain.serving_item_id != expected_serving_item_id
            or identity_chain.external_abort_log is not None
            or identity_chain.internal_abort_log is not None
        ):
            raise Stage2ProtocolError("validated request-log identity chain differs")
        return self._evidence(identity_chain)

    def _evidence(self, identity_chain: RequestIdentityChain) -> Stage2RequestEvidence:
        required = (
            self._response_headers_offset_ns,
            self._first_body_offset_ns,
            self._generation_terminal_offset_ns,
            self._usage_terminal_offset_ns,
            self._protocol_terminal_offset_ns,
            self._transport_terminal_offset_ns,
        )
        if any(value is None for value in required):
            raise Stage2ProtocolError("stream is missing a required terminal or timing boundary")
        if self._returned_prompt_ids != self._sent_prompt_token_ids:
            raise Stage2ProtocolError("returned prompt token IDs differ from sent IDs")
        if (
            self._usage is None
            or self._server_metrics is None
            or self._terminal_carried_tokens is None
        ):
            raise Stage2ProtocolError("stream is missing usage or generation terminal evidence")
        assert self._response_headers_offset_ns is not None
        assert self._first_body_offset_ns is not None
        assert self._usage_terminal_offset_ns is not None
        assert self._protocol_terminal_offset_ns is not None
        assert self._transport_terminal_offset_ns is not None
        token_events = tuple(event for event in self._token_events if event.output_token_ids)
        if not token_events:
            raise Stage2ProtocolError("stream contains no output-token event")
        first_token_offset = token_events[0].observation_offset_ns
        grouped = any(len(event.output_token_ids) > 1 for event in token_events)
        assert self._generation_terminal_offset_ns is not None
        if grouped:
            tpot = MetricAvailability(value_ns=None, unavailable_reason="GROUPED_TOKEN_EVENT")
            itl = MetricAvailability(value_ns=None, unavailable_reason="GROUPED_TOKEN_EVENT")
        else:
            tpot = MetricAvailability(
                value_ns=(self._generation_terminal_offset_ns - first_token_offset) / 31,
                unavailable_reason=None,
            )
            gaps = tuple(
                float(right.observation_offset_ns - left.observation_offset_ns)
                for left, right in pairwise(token_events)
            )
            itl = MetricAvailability(
                value_ns=sum(gaps) / len(gaps),
                unavailable_reason=None,
            )
        event_gaps = tuple(
            right.observation_offset_ns - left.observation_offset_ns
            for left, right in pairwise(token_events)
        )
        output_text = "".join(self._output_text_parts)
        response_id = f"cmpl-{self._external_base_id}"
        serving_item_id = f"{response_id}-0"
        return Stage2RequestEvidence(
            fixture_identity_sha256=self._fixture_identity_sha256,
            request_identity_chain_sha256=sha256_identity(identity_chain),
            external_request_id=self._external_base_id,
            response_request_id=response_id,
            serving_item_request_id=serving_item_id,
            internal_engine_request_id=identity_chain.internal_engine_id,
            sent_prompt_token_ids=self._sent_prompt_token_ids,
            returned_prompt_token_ids=self._returned_prompt_ids,
            token_events=tuple(self._token_events),
            final_output_token_ids=tuple(self._output_ids),
            finish_reason="length",
            terminal_event_carried_token_ids=self._terminal_carried_tokens,
            usage=self._usage,
            local_prompt_token_count=None,
            local_output_token_count=None,
            server_per_request_metrics=self._server_metrics,
            disagreements=(),
            output_text=output_text,
            output_text_sha256=hashlib.sha256(output_text.encode("utf-8")).hexdigest(),
            timing=Stage2TimingRecord(
                dispatch_offset_ns=self._dispatch_offset_ns,
                response_headers_offset_ns=self._response_headers_offset_ns,
                first_response_body_bytes_offset_ns=self._first_body_offset_ns,
                first_output_token_offset_ns=first_token_offset,
                generation_terminal_offset_ns=self._generation_terminal_offset_ns,
                usage_terminal_offset_ns=self._usage_terminal_offset_ns,
                protocol_terminal_offset_ns=self._protocol_terminal_offset_ns,
                transport_terminal_offset_ns=self._transport_terminal_offset_ns,
            ),
            client_generation_tpot=tpot,
            token_observation_itl=itl,
            stream_output_gap_ns=event_gaps,
        )
