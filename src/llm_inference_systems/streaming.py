"""HTTPX streaming execution and raw evidence capture for the loopback fixture."""

from __future__ import annotations

import asyncio
import base64
import hashlib
import json
import time

import httpx

from llm_inference_systems.fixture_tokens import (
    FixtureTokenError,
    parse_input_tokens,
    parse_output_tokens,
)
from llm_inference_systems.sse import IncrementalSSEParser, SSEFrame, SSEProtocolError
from llm_inference_systems.stage1_contracts import (
    EvidenceBoundary,
    FailureOrigin,
    FixtureCaseDefinition,
    ServerEventKind,
    ServerEventRecord,
    Stage1FailureKind,
    Stage1FailureRecord,
    Stage1Phase,
    Stage1RequestRecord,
    Stage1RunConfiguration,
    Stage1TerminalClass,
    Stage1TimingRecord,
    StreamEvidenceKind,
    StreamEvidenceRecord,
)


class EvidenceCollector:
    """Assign synchronized global sequence numbers to client and server evidence."""

    def __init__(self, run_origin_ns: int) -> None:
        self.run_origin_ns = run_origin_ns
        self.stream_events: list[StreamEvidenceRecord] = []
        self.server_events: list[ServerEventRecord] = []
        self._stream_lock = asyncio.Lock()
        self._server_lock = asyncio.Lock()

    def offset_ns(self) -> int:
        return time.monotonic_ns() - self.run_origin_ns

    async def record_stream_event(
        self,
        *,
        request_id: str,
        case_id: str,
        phase: Stage1Phase,
        kind: StreamEvidenceKind,
        observation_offset_ns: int | None = None,
        raw_chunk_sequence: int | None = None,
        raw_bytes: bytes | None = None,
        sse_event_sequence: int | None = None,
        sse_data: str | None = None,
        token_delta_count: int | None = None,
        terminal_class: Stage1TerminalClass | None = None,
        failure_kind: Stage1FailureKind | None = None,
    ) -> StreamEvidenceRecord:
        async with self._stream_lock:
            offset = self.offset_ns() if observation_offset_ns is None else observation_offset_ns
            encoded = base64.b64encode(raw_bytes).decode("ascii") if raw_bytes is not None else None
            digest = hashlib.sha256(raw_bytes).hexdigest() if raw_bytes is not None else None
            event = StreamEvidenceRecord(
                boundary=EvidenceBoundary(),
                sequence=len(self.stream_events),
                request_id=request_id,
                case_id=case_id,
                phase=phase,
                kind=kind,
                observation_offset_ns=offset,
                raw_chunk_sequence=raw_chunk_sequence,
                raw_bytes_base64=encoded,
                raw_byte_count=len(raw_bytes) if raw_bytes is not None else None,
                raw_bytes_sha256=digest,
                sse_event_sequence=sse_event_sequence,
                sse_data=sse_data,
                token_delta_count=token_delta_count,
                terminal_class=terminal_class,
                failure_kind=failure_kind,
            )
            self.stream_events.append(event)
            return event

    async def record_server_event(
        self,
        *,
        request_id: str,
        case_id: str,
        kind: ServerEventKind,
        action_index: int | None = None,
        token_delta_count: int | None = None,
        http_status: int | None = None,
    ) -> None:
        async with self._server_lock:
            self.server_events.append(
                ServerEventRecord(
                    boundary=EvidenceBoundary(),
                    sequence=len(self.server_events),
                    request_id=request_id,
                    case_id=case_id,
                    kind=kind,
                    observation_offset_ns=self.offset_ns(),
                    action_index=action_index,
                    token_delta_count=token_delta_count,
                    http_status=http_status,
                )
            )


def _payload_text(frame: SSEFrame) -> str:
    if frame.kind != "data" or frame.data is None:
        raise SSEProtocolError("expected a data-bearing fixture event")
    try:
        value = json.loads(frame.data)
    except json.JSONDecodeError as error:
        raise SSEProtocolError("fixture SSE data is malformed JSON") from error
    if not isinstance(value, dict) or set(value) != {"choices"}:
        raise SSEProtocolError("fixture SSE payload must contain only choices")
    choices = value["choices"]
    if not isinstance(choices, list) or len(choices) != 1:
        raise SSEProtocolError("fixture SSE choices must contain exactly one item")
    choice = choices[0]
    if not isinstance(choice, dict) or set(choice) != {"text"}:
        raise SSEProtocolError("fixture SSE choice must contain only text")
    text = choice["text"]
    if not isinstance(text, str):
        raise SSEProtocolError("fixture SSE choice text must be a string")
    return text


async def execute_streaming_request(
    client: httpx.AsyncClient,
    case: FixtureCaseDefinition,
    configuration: Stage1RunConfiguration,
    *,
    phase: Stage1Phase,
    request_id: str,
    collector: EvidenceCollector,
) -> Stage1RequestRecord:
    """Execute one actual loopback request and retain partial evidence on every outcome."""

    input_count = len(parse_input_tokens(case.input_text))
    request = client.build_request(
        "POST",
        "/v1/completions",
        headers={
            "X-LIS-Request-ID": request_id,
            "X-LIS-Fixture-Case-ID": case.case_id,
        },
        json={
            "model": configuration.model,
            "prompt": case.input_text,
            "max_tokens": case.maximum_output_tokens,
            "temperature": configuration.temperature,
            "stream": configuration.stream,
        },
    )
    dispatch_offset = collector.offset_ns()
    response_headers_offset: int | None = None
    first_body_offset: int | None = None
    first_token_offset: int | None = None
    last_token_offset: int | None = None
    http_status: int | None = None
    token_deltas: list[int] = []
    token_event_offsets: list[int] = []
    raw_chunk_sequence = 0
    sse_event_sequence = 0
    terminal_class = Stage1TerminalClass.FAILED
    failure_kind: Stage1FailureKind | None = None
    failure_origin: FailureOrigin | None = None
    error_code: str | None = None
    response: httpx.Response | None = None
    parser = IncrementalSSEParser()
    try:
        response = await client.send(request, stream=True)
        response_headers_offset = collector.offset_ns()
        http_status = response.status_code
        if not 200 <= response.status_code < 300:
            async for raw_chunk in response.aiter_raw():
                if not raw_chunk:
                    continue
                observed = collector.offset_ns()
                if first_body_offset is None:
                    first_body_offset = observed
                await collector.record_stream_event(
                    request_id=request_id,
                    case_id=case.case_id,
                    phase=phase,
                    kind=StreamEvidenceKind.RAW_BODY_CHUNK,
                    observation_offset_ns=observed,
                    raw_chunk_sequence=raw_chunk_sequence,
                    raw_bytes=raw_chunk,
                )
                raw_chunk_sequence += 1
            terminal_class = Stage1TerminalClass.FAILED
            failure_kind = Stage1FailureKind.HTTP_STATUS
            failure_origin = FailureOrigin.HTTP_STATUS
            error_code = f"http-status-{response.status_code}"
        else:
            async for raw_chunk in response.aiter_raw():
                if not raw_chunk:
                    continue
                observed = collector.offset_ns()
                if first_body_offset is None:
                    first_body_offset = observed
                await collector.record_stream_event(
                    request_id=request_id,
                    case_id=case.case_id,
                    phase=phase,
                    kind=StreamEvidenceKind.RAW_BODY_CHUNK,
                    observation_offset_ns=observed,
                    raw_chunk_sequence=raw_chunk_sequence,
                    raw_bytes=raw_chunk,
                )
                raw_chunk_sequence += 1
                for frame in parser.feed(raw_chunk):
                    parsed_offset = collector.offset_ns()
                    if frame.kind == "comment":
                        await collector.record_stream_event(
                            request_id=request_id,
                            case_id=case.case_id,
                            phase=phase,
                            kind=StreamEvidenceKind.SSE_COMMENT,
                            observation_offset_ns=parsed_offset,
                            sse_event_sequence=sse_event_sequence,
                            sse_data="\n".join(frame.comments) if frame.comments else None,
                        )
                    elif frame.kind == "done":
                        await collector.record_stream_event(
                            request_id=request_id,
                            case_id=case.case_id,
                            phase=phase,
                            kind=StreamEvidenceKind.SSE_DONE,
                            observation_offset_ns=parsed_offset,
                            sse_event_sequence=sse_event_sequence,
                            sse_data="[DONE]",
                        )
                    else:
                        text = _payload_text(frame)
                        tokens = parse_output_tokens(text)
                        token_count = len(tokens)
                        token_deltas.append(token_count)
                        token_event_offsets.append(parsed_offset)
                        if first_token_offset is None:
                            first_token_offset = parsed_offset
                        last_token_offset = parsed_offset
                        await collector.record_stream_event(
                            request_id=request_id,
                            case_id=case.case_id,
                            phase=phase,
                            kind=StreamEvidenceKind.SSE_TOKEN_EVENT,
                            observation_offset_ns=parsed_offset,
                            sse_event_sequence=sse_event_sequence,
                            sse_data=frame.data,
                            token_delta_count=token_count,
                        )
                    sse_event_sequence += 1
            parser.finalize()
            if sum(token_deltas) != case.expected_output_token_count:
                raise FixtureTokenError("fixture output count differs from the declared case")
            terminal_class = Stage1TerminalClass.SUCCESS
    except httpx.TimeoutException:
        terminal_class = Stage1TerminalClass.TIMEOUT
        failure_kind = Stage1FailureKind.TIMEOUT
        failure_origin = FailureOrigin.HTTP_CLIENT
        error_code = "httpx-timeout"
    except (SSEProtocolError, json.JSONDecodeError):
        terminal_class = Stage1TerminalClass.FAILED
        failure_kind = Stage1FailureKind.PROTOCOL_MALFORMED_STREAM
        failure_origin = FailureOrigin.SSE_PARSER
        error_code = "malformed-sse-stream"
    except FixtureTokenError:
        terminal_class = Stage1TerminalClass.FAILED
        failure_kind = Stage1FailureKind.TOKEN_ACCOUNTING
        failure_origin = FailureOrigin.TOKEN_ACCOUNTING
        error_code = "fixture-token-accounting"
    except httpx.TransportError:
        terminal_class = Stage1TerminalClass.FAILED
        failure_kind = Stage1FailureKind.TRANSPORT
        failure_origin = FailureOrigin.HTTP_CLIENT
        error_code = "httpx-transport"
    except asyncio.CancelledError:
        terminal_class = Stage1TerminalClass.CANCELLED
        failure_kind = Stage1FailureKind.CANCELLED
        failure_origin = FailureOrigin.LOAD_GENERATOR
        error_code = "request-cancelled"
    except Exception:
        terminal_class = Stage1TerminalClass.FAILED
        failure_kind = Stage1FailureKind.UNEXPECTED
        failure_origin = FailureOrigin.HARNESS
        error_code = "unexpected-request-failure"
    finally:
        if response is not None:
            await response.aclose()

    terminal_offset = collector.offset_ns()
    failure = None
    if terminal_class is not Stage1TerminalClass.SUCCESS:
        assert failure_kind is not None
        assert failure_origin is not None
        assert error_code is not None
        failure = Stage1FailureRecord(
            kind=failure_kind,
            origin=failure_origin,
            error_code=error_code,
            occurred_offset_ns=terminal_offset,
            timeout_policy=(
                configuration.timeout_policy if failure_kind is Stage1FailureKind.TIMEOUT else None
            ),
        )
    e2e_ns = terminal_offset - dispatch_offset
    slo_satisfied = (
        terminal_class is Stage1TerminalClass.SUCCESS
        and e2e_ns <= configuration.slo.successful_end_to_end_threshold_ns
    )
    await collector.record_stream_event(
        request_id=request_id,
        case_id=case.case_id,
        phase=phase,
        kind=StreamEvidenceKind.REQUEST_TERMINAL,
        observation_offset_ns=terminal_offset,
        terminal_class=terminal_class,
        failure_kind=failure_kind,
    )
    return Stage1RequestRecord(
        boundary=EvidenceBoundary(),
        request_id=request_id,
        case_id=case.case_id,
        phase=phase,
        terminal_class=terminal_class,
        http_status=http_status,
        timing=Stage1TimingRecord(
            dispatch_offset_ns=dispatch_offset,
            response_headers_offset_ns=response_headers_offset,
            first_response_body_bytes_offset_ns=first_body_offset,
            first_output_token_offset_ns=first_token_offset,
            last_output_token_offset_ns=last_token_offset,
            terminal_offset_ns=terminal_offset,
        ),
        input_token_count=input_count,
        output_token_count=sum(token_deltas),
        token_count_source="FIXTURE_EXACT",
        token_event_delta_counts=tuple(token_deltas),
        token_event_observation_offsets_ns=tuple(token_event_offsets),
        per_token_observation_complete=all(delta == 1 for delta in token_deltas),
        slo_satisfied=slo_satisfied,
        failure=failure,
    )
