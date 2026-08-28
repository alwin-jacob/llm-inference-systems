"""Adversarial tests for deterministic metric semantics."""

from __future__ import annotations

import math

import pytest

from llm_inference_systems.contracts import (
    MetricUnavailableReason,
    RequestOutcome,
    RequestPhase,
    RequestRecord,
    SLODefinition,
    StreamEventRecord,
    TimingRecord,
)
from llm_inference_systems.metrics import (
    derive_request_metrics,
    derive_summary,
    percentile_type7,
)
from tests.factories import (
    failed_request,
    known_count,
    load_configuration,
    success_request,
    unknown_count,
)


def test_percentile_empty_is_unavailable() -> None:
    assert percentile_type7((), 0.5) is None


def test_percentile_singleton_is_itself() -> None:
    assert percentile_type7((7.0,), 0.99) == 7.0


def test_percentile_type7_even_sample_interpolates() -> None:
    values = (4.0, 1.0, 3.0, 2.0)
    assert percentile_type7(values, 0.50) == 2.5
    assert percentile_type7(values, 0.95) == pytest.approx(3.85)
    assert percentile_type7(values, 0.99) == pytest.approx(3.97)


@pytest.mark.parametrize("value", [math.nan, math.inf, -math.inf])
def test_percentile_rejects_nonfinite_samples(value: float) -> None:
    with pytest.raises(ValueError, match="finite"):
        percentile_type7((1.0, value), 0.5)


@pytest.mark.parametrize("probability", [-0.01, 1.01])
def test_percentile_rejects_invalid_probability(probability: float) -> None:
    with pytest.raises(ValueError, match="between"):
        percentile_type7((1.0,), probability)


def test_ttft_uses_first_output_token_not_earlier_first_byte() -> None:
    request = success_request(
        "ttft",
        dispatch_ns=100,
        first_byte_ns=110,
        token_offsets_ns=(150, 170),
        terminal_ns=180,
    )
    metrics = derive_request_metrics(request)
    assert metrics.ttft_ns == 50.0
    assert metrics.ttft_ns != 10.0


@pytest.mark.parametrize(
    ("token_offsets", "expected", "reason"),
    [
        ((), None, MetricUnavailableReason.INSUFFICIENT_OUTPUT_TOKENS),
        ((20,), None, MetricUnavailableReason.INSUFFICIENT_OUTPUT_TOKENS),
        ((20, 30, 50), 15.0, None),
    ],
)
def test_tpot_uses_output_tokens_minus_one(
    token_offsets: tuple[int, ...],
    expected: float | None,
    reason: MetricUnavailableReason | None,
) -> None:
    request = success_request(
        "tpot",
        token_offsets_ns=token_offsets,
        first_byte_ns=5,
    )
    metrics = derive_request_metrics(request)
    assert metrics.tpot_ns == expected
    assert metrics.tpot_unavailable_reason is reason


def test_itl_uses_true_per_token_sequence() -> None:
    metrics = derive_request_metrics(success_request("itl", token_offsets_ns=(20, 30, 55)))
    assert metrics.itl_ns == (10.0, 25.0)
    assert metrics.itl_unavailable_reason is None


def test_itl_unavailable_for_multi_token_chunk_without_individual_timestamps() -> None:
    request = RequestRecord(
        request_id="chunked",
        case_id="case-alpha",
        phase=RequestPhase.MEASURED,
        outcome=RequestOutcome.SUCCESS,
        timing=TimingRecord(
            dispatch_offset_ns=0,
            first_response_byte_offset_ns=5,
            first_output_token_offset_ns=20,
            last_output_token_offset_ns=20,
            terminal_offset_ns=25,
        ),
        stream_events=(
            StreamEventRecord(
                chunk_index=0,
                event_offset_ns=20,
                output_tokens_in_chunk=2,
                per_token_observation_offsets_ns=None,
            ),
        ),
        input_tokens=known_count(2),
        output_tokens=known_count(2),
        failure=None,
    )
    metrics = derive_request_metrics(request)
    assert metrics.itl_ns == ()
    assert (
        metrics.itl_unavailable_reason
        is MetricUnavailableReason.MULTI_TOKEN_CHUNK_WITHOUT_TOKEN_TIMESTAMPS
    )


def test_failed_request_is_terminal_but_not_success_throughput() -> None:
    requests = (
        success_request("success", token_offsets_ns=(20, 30, 40)),
        failed_request("failure", RequestOutcome.TRANSPORT_ERROR),
    )
    summary = derive_summary(
        load_configuration(),
        requests,
        measurement_window_ns=1_000_000_000,
        observed_maximum_active_client_requests=1,
    )
    assert summary.offered_request_rate == 2.0
    assert summary.terminal_request_rate == 2.0
    assert summary.successful_request_throughput == 1.0
    assert summary.failed_count == 1
    assert summary.output_token_throughput == 3.0


def test_total_token_throughput_unavailable_when_success_input_unknown() -> None:
    request = success_request("unknown-input", input_tokens=unknown_count())
    summary = derive_summary(
        load_configuration(),
        (request,),
        measurement_window_ns=1_000_000_000,
        observed_maximum_active_client_requests=1,
    )
    assert summary.output_token_throughput == 3.0
    assert summary.total_token_throughput is None


@pytest.mark.parametrize(
    "outcome",
    [RequestOutcome.TRANSPORT_ERROR, RequestOutcome.TIMEOUT],
)
def test_failure_or_timeout_never_satisfies_goodput(outcome: RequestOutcome) -> None:
    summary = derive_summary(
        load_configuration(),
        (failed_request("bad", outcome),),
        measurement_window_ns=1_000_000_000,
        observed_maximum_active_client_requests=1,
    )
    assert summary.goodput == 0.0


def test_unavailable_applicable_slo_metric_is_unsatisfied() -> None:
    configuration = load_configuration().model_copy(
        update={
            "slo": SLODefinition(
                policy_name="itl-required",
                itl_threshold_ns=100,
            )
        }
    )
    request = success_request("one-token", token_offsets_ns=(20,))
    summary = derive_summary(
        configuration,
        (request,),
        measurement_window_ns=1_000_000_000,
        observed_maximum_active_client_requests=1,
    )
    assert derive_request_metrics(request).itl_ns == ()
    assert summary.goodput == 0.0


def test_all_applicable_slos_must_pass() -> None:
    configuration = load_configuration().model_copy(
        update={
            "slo": SLODefinition(
                policy_name="one-failing-threshold",
                ttft_threshold_ns=25,
                tpot_threshold_ns=5,
            )
        }
    )
    request = success_request("mixed-slo", token_offsets_ns=(20, 30, 40))
    summary = derive_summary(
        configuration,
        (request,),
        measurement_window_ns=1_000_000_000,
        observed_maximum_active_client_requests=1,
    )
    assert derive_request_metrics(request).ttft_ns == 20.0
    assert derive_request_metrics(request).tpot_ns == 10.0
    assert summary.goodput == 0.0


def test_warmup_retained_but_excluded_from_summary() -> None:
    warmup = success_request("warmup", phase=RequestPhase.WARMUP)
    measured = success_request("measured")
    summary = derive_summary(
        load_configuration(),
        (warmup, measured),
        measurement_window_ns=1_000_000_000,
        observed_maximum_active_client_requests=1,
    )
    assert summary.warmup_record_count == 1
    assert summary.warmup_excluded_count == 1
    assert summary.attempted_count == 1
    assert summary.successful_count == 1


def test_client_concurrency_and_server_batch_are_separate() -> None:
    summary = derive_summary(
        load_configuration(),
        (success_request("measured"),),
        measurement_window_ns=1_000_000_000,
        observed_maximum_active_client_requests=2,
        observed_server_batch_size=7,
    )
    assert summary.requested_client_concurrency == 2
    assert summary.observed_maximum_active_client_requests == 2
    assert summary.observed_server_batch_size == 7


def test_observed_client_activity_cannot_exceed_requested_concurrency() -> None:
    with pytest.raises(ValueError, match="requested concurrency"):
        derive_summary(
            load_configuration(),
            (success_request("measured"),),
            measurement_window_ns=1_000_000_000,
            observed_maximum_active_client_requests=3,
        )


def test_measurement_window_must_be_positive() -> None:
    with pytest.raises(ValueError, match="positive"):
        derive_summary(
            load_configuration(),
            (success_request("measured"),),
            measurement_window_ns=0,
            observed_maximum_active_client_requests=1,
        )
