"""Stage 1 timing, population, failure-rate, and concurrency semantics."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from llm_inference_systems.artifact_io import ValidatedBundle
from llm_inference_systems.stage1_contracts import (
    RateValue,
    Stage1FailureKind,
    Stage1TerminalClass,
    StreamEvidenceKind,
)
from llm_inference_systems.stage1_metrics import (
    derive_observed_max_client_concurrency,
    derive_stage1_request_metrics,
)


def _bundle(
    stage1_bundle_pair: tuple[object, ValidatedBundle, object, ValidatedBundle],
) -> ValidatedBundle:
    return stage1_bundle_pair[1]


def test_dispatch_headers_body_token_terminal_ordering(
    stage1_bundle_pair: tuple[object, ValidatedBundle, object, ValidatedBundle],
) -> None:
    bundle = _bundle(stage1_bundle_pair)
    for request in bundle.requests:
        timing = request.timing
        assert timing.dispatch_offset_ns <= timing.terminal_offset_ns
        if timing.response_headers_offset_ns is not None:
            assert timing.dispatch_offset_ns <= timing.response_headers_offset_ns
        if timing.first_response_body_bytes_offset_ns is not None:
            assert timing.response_headers_offset_ns is not None
            assert timing.response_headers_offset_ns <= timing.first_response_body_bytes_offset_ns


def test_ttft_uses_first_token_not_first_body_bytes(
    stage1_bundle_pair: tuple[object, ValidatedBundle, object, ValidatedBundle],
) -> None:
    request = next(
        item
        for item in _bundle(stage1_bundle_pair).requests
        if item.case_id == "success-first-body-before-token"
    )
    metrics = derive_stage1_request_metrics(request)
    assert request.timing.first_response_body_bytes_offset_ns is not None
    assert request.timing.first_output_token_offset_ns is not None
    assert request.timing.first_response_body_bytes_offset_ns < (
        request.timing.first_output_token_offset_ns
    )
    assert metrics.ttft_ns == (
        request.timing.first_output_token_offset_ns - request.timing.dispatch_offset_ns
    )


def test_canonical_tpot_is_distinct_from_observed_token_span(
    stage1_bundle_pair: tuple[object, ValidatedBundle, object, ValidatedBundle],
) -> None:
    request = next(
        item for item in _bundle(stage1_bundle_pair).requests if item.case_id == "success-three-a"
    )
    metrics = derive_stage1_request_metrics(request)
    assert metrics.tpot_ns is not None
    assert metrics.observed_token_span_per_interval_ns is not None
    assert metrics.tpot_ns > metrics.observed_token_span_per_interval_ns


def test_one_token_tpot_is_unavailable(
    stage1_bundle_pair: tuple[object, ValidatedBundle, object, ValidatedBundle],
) -> None:
    request = next(
        item
        for item in _bundle(stage1_bundle_pair).requests
        if item.case_id == "success-single-output-token"
    )
    assert derive_stage1_request_metrics(request).tpot_ns is None


def test_multi_token_event_cannot_fabricate_itl(
    stage1_bundle_pair: tuple[object, ValidatedBundle, object, ValidatedBundle],
) -> None:
    request = next(
        item
        for item in _bundle(stage1_bundle_pair).requests
        if item.case_id == "success-multi-token-event"
    )
    assert request.output_token_count == 3
    assert request.token_event_delta_counts == (1, 2)
    assert not request.per_token_observation_complete
    assert derive_stage1_request_metrics(request).itl_ns == ()


def test_single_token_events_produce_adjacent_itl(
    stage1_bundle_pair: tuple[object, ValidatedBundle, object, ValidatedBundle],
) -> None:
    request = next(
        item for item in _bundle(stage1_bundle_pair).requests if item.case_id == "success-three-b"
    )
    metrics = derive_stage1_request_metrics(request)
    assert len(metrics.itl_ns) == 2
    assert all(value > 0 for value in metrics.itl_ns)


def test_failure_and_timeout_rates_use_attempted_measured_denominator(
    stage1_bundle_pair: tuple[object, ValidatedBundle, object, ValidatedBundle],
) -> None:
    summary = _bundle(stage1_bundle_pair).summary
    assert summary.attempted_measured_requests == 8
    assert summary.failure_rate == RateValue(numerator=2, denominator=8, value=0.25)
    assert summary.timeout_rate == RateValue(numerator=1, denominator=8, value=0.125)


def test_zero_attempt_rate_is_explicitly_unavailable() -> None:
    assert RateValue(numerator=0, denominator=0, value=None).value is None
    with pytest.raises(ValidationError, match="zero-denominator"):
        RateValue(numerator=0, denominator=0, value=0.0)


def test_failed_partial_tokens_do_not_enter_successful_token_throughput(
    stage1_bundle_pair: tuple[object, ValidatedBundle, object, ValidatedBundle],
) -> None:
    bundle = _bundle(stage1_bundle_pair)
    malformed = next(
        request
        for request in bundle.requests
        if request.case_id == "malformed-after-partial-output"
    )
    assert malformed.output_token_count == 1
    successful_tokens = sum(
        request.output_token_count
        for request in bundle.requests
        if request.phase == "MEASURED" and request.terminal_class is Stage1TerminalClass.SUCCESS
    )
    seconds = bundle.summary.measurement_window_ns / 1_000_000_000
    assert successful_tokens == 12
    assert bundle.summary.output_token_throughput == round(successful_tokens / seconds, 9)


def test_warmup_is_retained_and_excluded(
    stage1_bundle_pair: tuple[object, ValidatedBundle, object, ValidatedBundle],
) -> None:
    bundle = _bundle(stage1_bundle_pair)
    assert bundle.requests[0].phase == "WARMUP"
    assert bundle.summary.warmup_record_count == 1
    assert bundle.summary.warmup_excluded_count == 1
    assert bundle.summary.attempted_measured_requests == 8


def test_failure_taxonomy_retains_partial_protocol_http_and_timeout(
    stage1_bundle_pair: tuple[object, ValidatedBundle, object, ValidatedBundle],
) -> None:
    requests = {request.case_id: request for request in _bundle(stage1_bundle_pair).requests}
    assert requests["malformed-after-partial-output"].failure is not None
    assert (
        requests["malformed-after-partial-output"].failure.kind
        is Stage1FailureKind.PROTOCOL_MALFORMED_STREAM
    )
    assert requests["http-error"].failure is not None
    assert requests["http-error"].failure.kind is Stage1FailureKind.HTTP_STATUS
    assert requests["timeout-after-partial-body"].failure is not None
    assert requests["timeout-after-partial-body"].failure.kind is Stage1FailureKind.TIMEOUT


def test_timeout_partial_body_is_retained(
    stage1_bundle_pair: tuple[object, ValidatedBundle, object, ValidatedBundle],
) -> None:
    bundle = _bundle(stage1_bundle_pair)
    timeout = next(
        request for request in bundle.requests if request.case_id == "timeout-after-partial-body"
    )
    assert any(
        event.request_id == timeout.request_id and event.kind is StreamEvidenceKind.RAW_BODY_CHUNK
        for event in bundle.stream_events
    )


def test_concurrency_is_independently_reconstructed_and_batch_is_unobserved(
    stage1_bundle_pair: tuple[object, ValidatedBundle, object, ValidatedBundle],
) -> None:
    bundle = _bundle(stage1_bundle_pair)
    reconstructed = derive_observed_max_client_concurrency(bundle.stream_events)
    assert reconstructed == bundle.summary.observed_max_client_concurrency == 2
    assert bundle.summary.requested_client_concurrency == 2
    assert bundle.summary.configured_server_maximum_batch_size is None
    assert bundle.summary.observed_server_batch_size is None


def test_fixture_distributions_are_qualified_and_small_sampled(
    stage1_bundle_pair: tuple[object, ValidatedBundle, object, ValidatedBundle],
) -> None:
    summary = _bundle(stage1_bundle_pair).summary
    assert summary.end_to_end_success.sample_count == 5
    assert summary.ttft.sample_count == 5
    assert summary.tpot.sample_count == 4
    assert summary.itl.sample_count == 5
    assert summary.end_to_end_success.metric == "END_TO_END_SUCCESS_NS"
    assert summary.timing_disclaimer.startswith("These measurements are loopback fixture")
