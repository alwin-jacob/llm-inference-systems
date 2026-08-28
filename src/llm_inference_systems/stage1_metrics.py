"""Pure Stage 1 summary, concurrency, and semantic-fingerprint derivation."""

from __future__ import annotations

import base64
import hashlib
from dataclasses import dataclass
from itertools import pairwise

from llm_inference_systems.canonical import sha256_identity
from llm_inference_systems.metrics import NANOSECONDS_PER_SECOND, percentile_type7
from llm_inference_systems.stage1_contracts import (
    TIMING_DISCLAIMER,
    EvidenceBoundary,
    RateValue,
    Stage1MetricDistribution,
    Stage1MetricName,
    Stage1RequestRecord,
    Stage1RunConfiguration,
    Stage1RunSummary,
    Stage1TerminalClass,
    StreamEvidenceKind,
    StreamEvidenceRecord,
)


@dataclass(frozen=True, slots=True)
class Stage1RequestMetrics:
    end_to_end_ns: float | None
    ttft_ns: float | None
    tpot_ns: float | None
    itl_ns: tuple[float, ...]
    observed_token_span_per_interval_ns: float | None


def derive_stage1_request_metrics(request: Stage1RequestRecord) -> Stage1RequestMetrics:
    """Apply the v0.2.0 terminal-boundary TPOT definition without changing Stage 0."""

    if request.terminal_class is not Stage1TerminalClass.SUCCESS:
        return Stage1RequestMetrics(None, None, None, (), None)
    timing = request.timing
    end_to_end = float(timing.terminal_offset_ns - timing.dispatch_offset_ns)
    ttft = (
        float(timing.first_output_token_offset_ns - timing.dispatch_offset_ns)
        if timing.first_output_token_offset_ns is not None
        else None
    )
    if request.output_token_count < 2 or timing.first_output_token_offset_ns is None:
        tpot = None
    else:
        tpot = (timing.terminal_offset_ns - timing.first_output_token_offset_ns) / (
            request.output_token_count - 1
        )
    if (
        request.output_token_count >= 2
        and request.per_token_observation_complete
        and len(request.token_event_observation_offsets_ns) == request.output_token_count
    ):
        itl = tuple(
            float(current - previous)
            for previous, current in pairwise(request.token_event_observation_offsets_ns)
        )
    else:
        itl = ()
    if (
        request.output_token_count >= 2
        and timing.first_output_token_offset_ns is not None
        and timing.last_output_token_offset_ns is not None
    ):
        observed_span = (
            timing.last_output_token_offset_ns - timing.first_output_token_offset_ns
        ) / (request.output_token_count - 1)
    else:
        observed_span = None
    return Stage1RequestMetrics(end_to_end, ttft, tpot, itl, observed_span)


def derive_observed_max_client_concurrency(
    events: tuple[StreamEvidenceRecord, ...],
) -> int:
    """Reconstruct active client count solely from ordered lifecycle evidence."""

    active = 0
    maximum = 0
    started: set[str] = set()
    ended: set[str] = set()
    for expected_sequence, event in enumerate(events):
        if event.sequence != expected_sequence:
            raise ValueError("stream evidence sequence must be contiguous")
        if event.kind is StreamEvidenceKind.CLIENT_REQUEST_STARTED:
            if event.request_id in started:
                raise ValueError("request lifecycle start is duplicated")
            started.add(event.request_id)
            active += 1
            maximum = max(maximum, active)
        elif event.kind is StreamEvidenceKind.CLIENT_REQUEST_ENDED:
            if event.request_id not in started or event.request_id in ended:
                raise ValueError("request lifecycle end has no unique matching start")
            ended.add(event.request_id)
            active -= 1
            if active < 0:
                raise ValueError("active client count became negative")
    if active != 0 or started != ended:
        raise ValueError("request lifecycle evidence is incomplete")
    return maximum


def _distribution(
    metric: Stage1MetricName,
    values: tuple[float, ...],
    *,
    denominator: str,
    inclusion_semantics: str,
) -> Stage1MetricDistribution:
    if not values:
        return Stage1MetricDistribution(
            metric=metric,
            unit="nanoseconds",
            percentile_algorithm="HYNDMAN_FAN_TYPE_7",
            sample_count=0,
            p50=None,
            p95=None,
            p99=None,
            denominator=denominator,
            inclusion_semantics=inclusion_semantics,
            unavailable_reason="NO_SAMPLES",
        )
    return Stage1MetricDistribution(
        metric=metric,
        unit="nanoseconds",
        percentile_algorithm="HYNDMAN_FAN_TYPE_7",
        sample_count=len(values),
        p50=round(percentile_type7(values, 0.50) or 0.0, 6),
        p95=round(percentile_type7(values, 0.95) or 0.0, 6),
        p99=round(percentile_type7(values, 0.99) or 0.0, 6),
        denominator=denominator,
        inclusion_semantics=inclusion_semantics,
        unavailable_reason=None,
    )


def derive_stage1_summary(
    configuration: Stage1RunConfiguration,
    requests: tuple[Stage1RequestRecord, ...],
    stream_events: tuple[StreamEvidenceRecord, ...],
) -> Stage1RunSummary:
    warmup = tuple(request for request in requests if request.phase == "WARMUP")
    measured = tuple(request for request in requests if request.phase == "MEASURED")
    if not measured:
        raise ValueError("Stage 1 fixture summary requires measured requests")
    successful = tuple(
        request for request in measured if request.terminal_class is Stage1TerminalClass.SUCCESS
    )
    failed_non_timeout = tuple(
        request for request in measured if request.terminal_class is Stage1TerminalClass.FAILED
    )
    timed_out = tuple(
        request for request in measured if request.terminal_class is Stage1TerminalClass.TIMEOUT
    )
    cancelled = tuple(
        request for request in measured if request.terminal_class is Stage1TerminalClass.CANCELLED
    )
    for request in measured:
        expected_slo = (
            request.terminal_class is Stage1TerminalClass.SUCCESS
            and request.timing.terminal_offset_ns - request.timing.dispatch_offset_ns
            <= configuration.slo.successful_end_to_end_threshold_ns
        )
        if request.slo_satisfied != expected_slo:
            raise ValueError("stored request SLO result differs from pure reconstruction")
    measurement_start = min(request.timing.dispatch_offset_ns for request in measured)
    measurement_end = max(request.timing.terminal_offset_ns for request in measured)
    measurement_window_ns = measurement_end - measurement_start
    if measurement_window_ns <= 0:
        raise ValueError("measurement window must be positive")
    seconds = measurement_window_ns / NANOSECONDS_PER_SECOND
    metrics = tuple(derive_stage1_request_metrics(request) for request in successful)
    e2e_values = tuple(value.end_to_end_ns for value in metrics if value.end_to_end_ns is not None)
    ttft_values = tuple(value.ttft_ns for value in metrics if value.ttft_ns is not None)
    tpot_values = tuple(value.tpot_ns for value in metrics if value.tpot_ns is not None)
    itl_values = tuple(gap for value in metrics for gap in value.itl_ns)
    span_values = tuple(
        value.observed_token_span_per_interval_ns
        for value in metrics
        if value.observed_token_span_per_interval_ns is not None
    )
    attempted = len(measured)
    observed_maximum = derive_observed_max_client_concurrency(stream_events)
    successful_output_tokens = sum(request.output_token_count for request in successful)
    successful_total_tokens = sum(
        request.input_token_count + request.output_token_count for request in successful
    )
    good_count = sum(request.slo_satisfied for request in measured)
    return Stage1RunSummary(
        boundary=EvidenceBoundary(),
        timing_disclaimer=TIMING_DISCLAIMER,
        measurement_window_ns=measurement_window_ns,
        attempted_measured_requests=attempted,
        terminal_measured_requests=attempted,
        successful_measured_requests=len(successful),
        failed_non_timeout_measured_requests=len(failed_non_timeout),
        timed_out_measured_requests=len(timed_out),
        cancelled_measured_requests=len(cancelled),
        warmup_record_count=len(warmup),
        warmup_excluded_count=len(warmup),
        requested_client_concurrency=configuration.load_shape.requested_client_concurrency,
        observed_max_client_concurrency=observed_maximum,
        configured_server_maximum_batch_size=None,
        observed_server_batch_size=None,
        offered_request_rate=round(attempted / seconds, 9),
        terminal_request_rate=round(attempted / seconds, 9),
        successful_request_throughput=round(len(successful) / seconds, 9),
        output_token_throughput=round(successful_output_tokens / seconds, 9),
        total_token_throughput=round(successful_total_tokens / seconds, 9),
        slo_satisfying_request_count=good_count,
        goodput=round(good_count / seconds, 9),
        failure_rate=RateValue(
            numerator=len(failed_non_timeout),
            denominator=attempted,
            value=len(failed_non_timeout) / attempted if attempted else None,
        ),
        timeout_rate=RateValue(
            numerator=len(timed_out),
            denominator=attempted,
            value=len(timed_out) / attempted if attempted else None,
        ),
        end_to_end_success=_distribution(
            "END_TO_END_SUCCESS_NS",
            e2e_values,
            denominator="successful measured requests",
            inclusion_semantics=(
                "Successful measured requests only; warmup and all failures excluded."
            ),
        ),
        ttft=_distribution(
            "TTFT_NS",
            ttft_values,
            denominator="successful measured requests with first output token",
            inclusion_semantics="Dispatch to first parsed fixture output-token event.",
        ),
        tpot=_distribution(
            "TPOT_NS",
            tpot_values,
            denominator="successful measured requests with at least two exact output tokens",
            inclusion_semantics=(
                "Terminal success minus first output token, divided by exact output tokens "
                "minus one."
            ),
        ),
        itl=_distribution(
            "ITL_NS",
            itl_values,
            denominator=(
                "adjacent distinct per-token observations from successful measured requests"
            ),
            inclusion_semantics="Unavailable when one SSE event contains multiple fixture tokens.",
        ),
        observed_token_span_per_interval=_distribution(
            "OBSERVED_TOKEN_SPAN_PER_INTERVAL_NS",
            span_values,
            denominator="successful measured requests with at least two exact output tokens",
            inclusion_semantics=(
                "Last observed token event minus first observed token event, divided by token "
                "intervals; "
                "this metric is not TPOT."
            ),
        ),
    )


def semantic_fingerprint(
    *,
    workload_sha256: str,
    configuration_sha256: str,
    fixture_sha256: str,
    requests: tuple[Stage1RequestRecord, ...],
    stream_events: tuple[StreamEvidenceRecord, ...],
    summary: Stage1RunSummary,
) -> str:
    """Exclude timestamps, ports, run IDs, and raw-chunk boundaries from semantic identity."""

    body_hashes: dict[str, str] = {}
    protocol_shapes: dict[str, list[dict[str, object]]] = {}
    for request in requests:
        chunks = sorted(
            (
                event
                for event in stream_events
                if event.request_id == request.request_id
                and event.kind is StreamEvidenceKind.RAW_BODY_CHUNK
            ),
            key=lambda event: event.raw_chunk_sequence or 0,
        )
        body = b"".join(
            base64.b64decode(event.raw_bytes_base64 or "", validate=True) for event in chunks
        )
        body_hashes[request.request_id] = hashlib.sha256(body).hexdigest()
        protocol_shapes[request.request_id] = [
            {
                "kind": event.kind.value,
                "token_delta_count": event.token_delta_count,
            }
            for event in stream_events
            if event.request_id == request.request_id
            and event.kind
            in {
                StreamEvidenceKind.SSE_COMMENT,
                StreamEvidenceKind.SSE_TOKEN_EVENT,
                StreamEvidenceKind.SSE_DONE,
            }
        ]
    value = {
        "workload_sha256": workload_sha256,
        "configuration_sha256": configuration_sha256,
        "fixture_sha256": fixture_sha256,
        "requests": [
            {
                "request_id": request.request_id,
                "case_id": request.case_id,
                "phase": request.phase,
                "terminal_class": request.terminal_class.value,
                "failure_kind": request.failure.kind.value if request.failure else None,
                "input_token_count": request.input_token_count,
                "output_token_count": request.output_token_count,
                "token_count_source": request.token_count_source,
                "token_event_delta_counts": request.token_event_delta_counts,
                "per_token_observation_complete": request.per_token_observation_complete,
                "slo_satisfied": request.slo_satisfied,
                "body_sha256": body_hashes[request.request_id],
                "protocol_shape": protocol_shapes[request.request_id],
            }
            for request in requests
        ],
        "failure_rate": summary.failure_rate.model_dump(mode="json"),
        "timeout_rate": summary.timeout_rate.model_dump(mode="json"),
        "observed_max_client_concurrency": summary.observed_max_client_concurrency,
        "server_batch_observed": summary.observed_server_batch_size is not None,
    }
    return sha256_identity(value)
