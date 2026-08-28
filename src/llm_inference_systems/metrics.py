"""Pure Stage 0 metric derivation from request records."""

from __future__ import annotations

import math
from dataclasses import dataclass
from itertools import pairwise
from typing import Final

from llm_inference_systems.canonical import slo_policy_identity
from llm_inference_systems.contracts import (
    IdentitySource,
    MetricDistribution,
    MetricName,
    MetricSource,
    MetricUnavailableReason,
    RequestOutcome,
    RequestPhase,
    RequestRecord,
    RunConfiguration,
    RunSummary,
)

NANOSECONDS_PER_SECOND: Final = 1_000_000_000
PERCENTILE_ALGORITHM: Final = "HYNDMAN_FAN_TYPE_7"


@dataclass(frozen=True, slots=True)
class RequestMetrics:
    ttft_ns: float | None
    end_to_end_ns: float | None
    tpot_ns: float | None
    itl_ns: tuple[float, ...]
    ttft_unavailable_reason: MetricUnavailableReason | None
    tpot_unavailable_reason: MetricUnavailableReason | None
    itl_unavailable_reason: MetricUnavailableReason | None


def percentile_type7(values: tuple[float, ...], probability: float) -> float | None:
    """Return a Hyndman-Fan Type 7 percentile without mutating the sample."""

    if not 0.0 <= probability <= 1.0:
        raise ValueError("percentile probability must be between zero and one")
    if any(not math.isfinite(value) for value in values):
        raise ValueError("percentile samples must be finite")
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    h = (len(ordered) - 1) * probability
    lower = math.floor(h)
    upper = math.ceil(h)
    return ordered[lower] + (h - lower) * (ordered[upper] - ordered[lower])


def distribution(
    metric: MetricName,
    values: tuple[float, ...],
    *,
    denominator: str,
) -> MetricDistribution:
    if not values:
        return MetricDistribution(
            metric=metric,
            source=MetricSource.HARNESS_DERIVED,
            denominator=denominator,
            unit="nanoseconds",
            percentile_algorithm=PERCENTILE_ALGORITHM,
            sample_count=0,
            p50=None,
            p95=None,
            p99=None,
            unavailable_reason=MetricUnavailableReason.NO_SAMPLES,
        )
    return MetricDistribution(
        metric=metric,
        source=MetricSource.HARNESS_DERIVED,
        denominator=denominator,
        unit="nanoseconds",
        percentile_algorithm=PERCENTILE_ALGORITHM,
        sample_count=len(values),
        p50=percentile_type7(values, 0.50),
        p95=percentile_type7(values, 0.95),
        p99=percentile_type7(values, 0.99),
        unavailable_reason=None,
    )


def _token_observation_offsets(
    request: RequestRecord,
) -> tuple[tuple[int, ...], MetricUnavailableReason | None]:
    output_count = request.output_tokens.value
    if output_count is None:
        return (), MetricUnavailableReason.MISSING_TOKEN_COUNT
    observations: list[int] = []
    for event in request.stream_events:
        if event.per_token_observation_offsets_ns is not None:
            observations.extend(event.per_token_observation_offsets_ns)
        elif event.output_tokens_in_chunk == 1:
            observations.append(event.event_offset_ns)
        else:
            return (), MetricUnavailableReason.MULTI_TOKEN_CHUNK_WITHOUT_TOKEN_TIMESTAMPS
    if not observations:
        return (), MetricUnavailableReason.MISSING_TOKEN_TIMESTAMPS
    if len(observations) != output_count:
        return (), MetricUnavailableReason.TOKEN_TIMESTAMP_COUNT_MISMATCH
    if observations != sorted(observations) or len(observations) != len(set(observations)):
        return (), MetricUnavailableReason.MISSING_TOKEN_TIMESTAMPS
    return tuple(observations), None


def derive_request_metrics(request: RequestRecord) -> RequestMetrics:
    if request.outcome is not RequestOutcome.SUCCESS:
        return RequestMetrics(
            ttft_ns=None,
            end_to_end_ns=None,
            tpot_ns=None,
            itl_ns=(),
            ttft_unavailable_reason=MetricUnavailableReason.REQUEST_FAILED,
            tpot_unavailable_reason=MetricUnavailableReason.REQUEST_FAILED,
            itl_unavailable_reason=MetricUnavailableReason.REQUEST_FAILED,
        )

    timing = request.timing
    if timing.first_output_token_offset_ns is None:
        ttft = None
        ttft_reason = MetricUnavailableReason.FIRST_OUTPUT_TOKEN_NOT_OBSERVED
    else:
        ttft = float(timing.first_output_token_offset_ns - timing.dispatch_offset_ns)
        ttft_reason = None

    end_to_end = float(timing.terminal_offset_ns - timing.dispatch_offset_ns)
    output_count = request.output_tokens.value
    if output_count is None:
        tpot = None
        tpot_reason = MetricUnavailableReason.MISSING_TOKEN_COUNT
    elif output_count < 2:
        tpot = None
        tpot_reason = MetricUnavailableReason.INSUFFICIENT_OUTPUT_TOKENS
    elif timing.first_output_token_offset_ns is None or timing.last_output_token_offset_ns is None:
        tpot = None
        tpot_reason = MetricUnavailableReason.MISSING_TOKEN_TIMESTAMPS
    else:
        tpot = (timing.last_output_token_offset_ns - timing.first_output_token_offset_ns) / (
            output_count - 1
        )
        tpot_reason = None

    observations, observation_reason = _token_observation_offsets(request)
    if observation_reason is not None:
        itl: tuple[float, ...] = ()
        itl_reason = observation_reason
    elif len(observations) < 2:
        itl = ()
        itl_reason = MetricUnavailableReason.INSUFFICIENT_OUTPUT_TOKENS
    else:
        itl = tuple(float(current - previous) for previous, current in pairwise(observations))
        itl_reason = None

    return RequestMetrics(
        ttft_ns=ttft,
        end_to_end_ns=end_to_end,
        tpot_ns=tpot,
        itl_ns=itl,
        ttft_unavailable_reason=ttft_reason,
        tpot_unavailable_reason=tpot_reason,
        itl_unavailable_reason=itl_reason,
    )


def _satisfies_slo(request: RequestRecord, configuration: RunConfiguration) -> bool:
    if request.outcome is not RequestOutcome.SUCCESS:
        return False
    metrics = derive_request_metrics(request)
    slo = configuration.slo
    checks = (
        (slo.ttft_threshold_ns, metrics.ttft_ns),
        (slo.tpot_threshold_ns, metrics.tpot_ns),
        (slo.end_to_end_threshold_ns, metrics.end_to_end_ns),
        (slo.itl_threshold_ns, max(metrics.itl_ns) if metrics.itl_ns else None),
    )
    return all(
        threshold is None or (value is not None and value <= threshold)
        for threshold, value in checks
    )


def derive_summary(
    configuration: RunConfiguration,
    requests: tuple[RequestRecord, ...],
    *,
    measurement_window_ns: int,
    observed_maximum_active_client_requests: int,
    observed_server_batch_size: int | None = None,
) -> RunSummary:
    if measurement_window_ns <= 0:
        raise ValueError("measurement window must be positive")
    if observed_maximum_active_client_requests < 0:
        raise ValueError("observed client concurrency must be nonnegative")

    warmup = tuple(request for request in requests if request.phase is RequestPhase.WARMUP)
    measured = tuple(request for request in requests if request.phase is RequestPhase.MEASURED)
    successful = tuple(request for request in measured if request.outcome is RequestOutcome.SUCCESS)
    failed = tuple(request for request in measured if request.outcome is not RequestOutcome.SUCCESS)
    timeout_count = sum(request.outcome is RequestOutcome.TIMEOUT for request in measured)
    cancelled_count = sum(request.outcome is RequestOutcome.CANCELLED for request in measured)
    protocol_outcomes = {RequestOutcome.PROTOCOL_ERROR, RequestOutcome.MALFORMED_STREAM}
    protocol_error_count = sum(request.outcome in protocol_outcomes for request in measured)

    request_metrics = tuple(derive_request_metrics(request) for request in successful)
    ttft_values = tuple(metric.ttft_ns for metric in request_metrics if metric.ttft_ns is not None)
    e2e_values = tuple(
        metric.end_to_end_ns for metric in request_metrics if metric.end_to_end_ns is not None
    )
    tpot_values = tuple(metric.tpot_ns for metric in request_metrics if metric.tpot_ns is not None)
    itl_values = tuple(value for metric in request_metrics for value in metric.itl_ns)

    seconds = measurement_window_ns / NANOSECONDS_PER_SECOND
    output_values = [request.output_tokens.value for request in successful]
    input_values = [request.input_tokens.value for request in successful]
    output_throughput = (
        sum(value for value in output_values if value is not None) / seconds
        if all(value is not None for value in output_values)
        else None
    )
    total_throughput = (
        (
            sum(value for value in output_values if value is not None)
            + sum(value for value in input_values if value is not None)
        )
        / seconds
        if all(value is not None for value in (*input_values, *output_values))
        else None
    )
    good_count = sum(_satisfies_slo(request, configuration) for request in measured)
    attempted = len(measured)

    return RunSummary(
        measurement_window_ns=measurement_window_ns,
        attempted_count=attempted,
        terminal_count=attempted,
        successful_count=len(successful),
        failed_count=len(failed),
        timeout_count=timeout_count,
        cancelled_count=cancelled_count,
        protocol_error_count=protocol_error_count,
        warmup_record_count=len(warmup),
        warmup_excluded_count=len(warmup),
        requested_client_concurrency=configuration.load_shape.requested_client_concurrency,
        observed_maximum_active_client_requests=observed_maximum_active_client_requests,
        configured_server_maximum_batch_size=configuration.configured_server_maximum_batch_size,
        configured_server_batch_source=configuration.configured_server_batch_source,
        observed_server_batch_size=observed_server_batch_size,
        observed_server_batch_source=(
            IdentitySource.DIRECTLY_OBSERVED if observed_server_batch_size is not None else None
        ),
        offered_request_rate=attempted / seconds,
        terminal_request_rate=attempted / seconds,
        successful_request_throughput=len(successful) / seconds,
        output_token_throughput=output_throughput,
        total_token_throughput=total_throughput,
        goodput=good_count / seconds,
        failure_rate=(len(failed) / attempted if attempted else 0.0),
        timeout_rate=(timeout_count / attempted if attempted else 0.0),
        goodput_slo_policy_sha256=slo_policy_identity(configuration.slo),
        ttft=distribution(
            MetricName.TTFT_NS,
            ttft_values,
            denominator="successful measured requests with first-output-token observation",
        ),
        end_to_end_success=distribution(
            MetricName.END_TO_END_SUCCESS_NS,
            e2e_values,
            denominator="successful measured requests",
        ),
        tpot=distribution(
            MetricName.TPOT_NS,
            tpot_values,
            denominator="successful measured requests with at least two counted output tokens",
        ),
        itl=distribution(
            MetricName.ITL_NS,
            itl_values,
            denominator="adjacent per-token observations from successful measured requests",
        ),
    )
