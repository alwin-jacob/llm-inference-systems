from __future__ import annotations

from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from llm_inference_systems.stage2_contracts import (
    REQUIRED_PREIMPORT_ENVIRONMENT,
    RUNTIME_PHASE_ORDER,
    BundleState,
    OfflineProcessRecord,
    ProcessClass,
    ProviderShape,
    ResourceBudgetInputs,
    ResourceEstimate,
    RuntimeImplementationRecord,
    RuntimePhaseRecord,
)
from llm_inference_systems.stage2_control import (
    AggregateComparisonState,
    CancellationClassification,
    CancellationProbe,
    DrainSample,
    FinishedReasonDelta,
    RestartSemanticRecord,
    Stage2ControlError,
    calculate_resource_budget,
    compare_three_restarts,
    describe_tiny_n_metric,
    evaluate_cancellation,
    validate_aggregate_commit,
    validate_offline_process_separation,
    validate_resource_gate,
    validate_runtime_phases,
)
from llm_inference_systems.stage2_prometheus import (
    PrometheusProtocolError,
    PrometheusSnapshot,
    derive_counter_delta,
    parse_prometheus_snapshot,
    require_fresh_snapshot,
    require_quiescent,
    select_exact_series,
    validate_measured_window_deltas,
)


def _exposition(
    *,
    prompt: int = 0,
    generation: int = 0,
    length: int = 0,
    running: int = 0,
    engine: str = "0",
) -> str:
    labels = f'engine="{engine}",model_name="qwen2.5-0.5b-instruct-stage2"'
    success = (
        f'engine="{engine}",finished_reason="length",model_name="qwen2.5-0.5b-instruct-stage2"'
    )
    return "\n".join(
        (
            f"vllm:num_requests_running{{{labels}}} {running}.0",
            f"vllm:num_requests_waiting{{{labels}}} 0.0",
            f"vllm:kv_cache_usage_perc{{{labels}}} 0.25",
            f"vllm:prompt_tokens_total{{{labels}}} {prompt}.0",
            f"vllm:generation_tokens_total{{{labels}}} {generation}.0",
            f"vllm:request_success_total{{{success}}} {length}.0",
            f"vllm:num_preemptions_total{{{labels}}} 0.0",
            f"vllm:prefix_cache_queries_total{{{labels}}} 0.0",
            f"vllm:prefix_cache_hits_total{{{labels}}} 0.0",
            "",
        )
    )


def _snapshot(raw: str, offset: int, process: str = "process-a") -> PrometheusSnapshot:
    return parse_prometheus_snapshot(
        raw,
        process_start_id=process,
        scrape_wall_clock_utc=datetime(2026, 8, 28, tzinfo=UTC),
        scrape_monotonic_offset_ns=offset,
    )


def test_exact_prometheus_series_and_full_inventory_are_retained() -> None:
    snapshot = _snapshot(_exposition(prompt=5), 100)
    sample = select_exact_series(snapshot, "vllm:prompt_tokens_total")
    assert sample.value == 5
    assert snapshot.raw_exposition == _exposition(prompt=5)
    assert snapshot.label_inventory["vllm:prompt_tokens_total"] == (("engine", "model_name"),)
    require_quiescent(snapshot)


@pytest.mark.parametrize(
    "line",
    [
        "vllm:prompt_tokens_total 1.0 123\n",
        "vllm:prompt_tokens_total NaN\n",
        "vllm:prompt_tokens_total +Inf\n",
        "vllm:prompt_tokens_total -1\n",
        'vllm:prompt_tokens_total{engine="0",engine="1"} 1\n',
        'vllm:prompt_tokens_total{engine="unterminated} 1\n',
    ],
)
def test_malformed_nonfinite_negative_or_duplicate_labels_are_rejected(line: str) -> None:
    with pytest.raises(PrometheusProtocolError):
        _snapshot(line, 1)


def test_absent_duplicate_ambiguous_or_wrong_label_series_are_rejected() -> None:
    snapshot = _snapshot(_exposition(engine="1"), 1)
    with pytest.raises(PrometheusProtocolError, match="absent, duplicate, or ambiguous"):
        select_exact_series(snapshot, "vllm:prompt_tokens_total")
    duplicated = _snapshot(_exposition() + _exposition(), 1)
    with pytest.raises(PrometheusProtocolError, match="absent, duplicate, or ambiguous"):
        select_exact_series(duplicated, "vllm:prompt_tokens_total")


def test_snapshot_freshness_provenance_is_enforced() -> None:
    snapshot = _snapshot(_exposition(), 100)
    require_fresh_snapshot(snapshot, reference_monotonic_offset_ns=150, maximum_age_ns=50)
    with pytest.raises(PrometheusProtocolError, match="stale"):
        require_fresh_snapshot(snapshot, reference_monotonic_offset_ns=151, maximum_age_ns=50)


def test_same_process_nondecreasing_counter_delta() -> None:
    before = _snapshot(_exposition(prompt=10), 100)
    after = _snapshot(_exposition(prompt=20), 200)
    delta = derive_counter_delta(before, after, "vllm:prompt_tokens_total")
    assert delta.delta == 10
    with pytest.raises(PrometheusProtocolError, match="across restarts"):
        derive_counter_delta(
            before, _snapshot(_exposition(prompt=20), 200, "process-b"), "vllm:prompt_tokens_total"
        )
    with pytest.raises(PrometheusProtocolError, match="reset"):
        derive_counter_delta(
            after, _snapshot(_exposition(prompt=5), 300), "vllm:prompt_tokens_total"
        )
    with pytest.raises(PrometheusProtocolError, match="only to Stage 2 counters"):
        derive_counter_delta(before, after, "vllm:num_requests_running")


def test_changed_label_inventory_is_rejected_before_subtraction() -> None:
    before = _snapshot(_exposition(prompt=1), 1)
    changed = _exposition(prompt=2) + (
        'vllm:prompt_tokens_total{engine="0",extra="changed",'
        'model_name="qwen2.5-0.5b-instruct-stage2"} 2.0\n'
    )
    after = _snapshot(changed, 2)
    with pytest.raises(PrometheusProtocolError, match="label inventory changed"):
        derive_counter_delta(before, after, "vllm:prompt_tokens_total")


def test_expected_16_by_64_to_32_counter_deltas() -> None:
    before = _snapshot(_exposition(), 1)
    after = _snapshot(_exposition(prompt=1024, generation=512, length=16), 2)
    deltas = (
        derive_counter_delta(before, after, "vllm:prompt_tokens_total"),
        derive_counter_delta(before, after, "vllm:generation_tokens_total"),
        derive_counter_delta(
            before,
            after,
            "vllm:request_success_total",
            finished_reason="length",
        ),
        derive_counter_delta(before, after, "vllm:num_preemptions_total"),
        derive_counter_delta(before, after, "vllm:prefix_cache_queries_total"),
        derive_counter_delta(before, after, "vllm:prefix_cache_hits_total"),
    )
    validate_measured_window_deltas(deltas)


def _phases() -> tuple[RuntimePhaseRecord, ...]:
    return tuple(
        RuntimePhaseRecord(
            phase=phase,
            started_offset_ns=index * 10,
            ended_offset_ns=index * 10 + 5,
            passed=True,
        )
        for index, phase in enumerate(RUNTIME_PHASE_ORDER)
    )


def test_runtime_phases_are_strictly_ordered_and_jit_monitored() -> None:
    validate_runtime_phases(_phases())
    with pytest.raises(Stage2ControlError, match="missing, duplicated, or reordered"):
        validate_runtime_phases(tuple(reversed(_phases())))
    records = list(_phases())
    records[10] = records[10].model_copy(update={"post_warmup_jit_observed": True})
    with pytest.raises(Stage2ControlError, match="post-warmup monitored JIT"):
        validate_runtime_phases(tuple(records))


def test_resolved_model_implementation_requires_independent_provenance() -> None:
    record = RuntimeImplementationRecord(
        runtime_package_name="future-runtime-package",
        resolved_model_implementation="resolved-architecture-fixture",
        resolved_implementation_source="RUNTIME_REPORTED",
    )
    assert record.resolved_model_implementation == "resolved-architecture-fixture"
    with pytest.raises(ValidationError, match="retained together"):
        RuntimeImplementationRecord(
            runtime_package_name="future-runtime-package",
            resolved_model_implementation=None,
            resolved_implementation_source="RUNTIME_REPORTED",
        )


def _offline_records() -> tuple[OfflineProcessRecord, ...]:
    records: list[OfflineProcessRecord] = []
    for index, process_class in enumerate(ProcessClass):
        offline = process_class is not ProcessClass.ONLINE_SNAPSHOT_DOWNLOAD
        records.append(
            OfflineProcessRecord(
                process_class=process_class,
                process_nonce=f"process-{index}",
                environment_set_before_import=True,
                offline_environment=dict(REQUIRED_PREIMPORT_ENVIRONMENT) if offline else {},
                token_variables_unset_without_reading=True,
                verified_local_snapshot_relative_path=(
                    "snapshots/qwen-fixture" if offline else None
                ),
                imported_runtime_or_tokenizer=offline,
            )
        )
    return tuple(records)


def test_offline_processes_are_fresh_and_environment_is_preimport() -> None:
    validate_offline_process_separation(_offline_records())
    records = list(_offline_records())
    records[1] = records[1].model_copy(update={"process_nonce": records[0].process_nonce})
    with pytest.raises(Stage2ControlError, match="reuse"):
        validate_offline_process_separation(tuple(records))
    with pytest.raises(ValidationError, match="before import"):
        OfflineProcessRecord.model_validate(
            {
                **_offline_records()[1].model_dump(mode="python"),
                "environment_set_before_import": False,
            }
        )


def _budget_inputs(total: int) -> ResourceBudgetInputs:
    parts = (total // 4, total // 4, total // 4, total - 3 * (total // 4))
    values = [
        ResourceEstimate(bytes=value, source=f"source-{index}") for index, value in enumerate(parts)
    ]
    return ResourceBudgetInputs(
        runtime_and_cuda_package_download=values[0],
        expected_installed_environment=values[1],
        model_tokenizer_snapshot=values[2],
        temporary_extraction_and_cache=values[3],
    )


def test_dynamic_resource_budget_boundary_rounding_and_fixed_floor() -> None:
    floor = calculate_resource_budget(_budget_inputs(1))
    assert floor.required_free_before_setup == 14_000_000_000
    total = 10_000_000_001
    result = calculate_resource_budget(_budget_inputs(total))
    assert result.required_setup_bytes == total
    assert result.required_free_before_setup == (total * 5 + 3) // 4 + 2_000_000_000
    with pytest.raises(Stage2ControlError, match="all resource estimates"):
        calculate_resource_budget(None)
    with pytest.raises(OverflowError):
        calculate_resource_budget(_budget_inputs(2**63 - 1))


def test_fixed_and_dynamic_resource_gate() -> None:
    shape = ProviderShape(
        operating_system="Linux",
        architecture="x86_64",
        logical_cpu_count=4,
        memory_total_bytes=28_000_000_000,
        filesystem_total_bytes=19_000_000_000,
        initial_free_bytes=15_000_000_000,
        post_setup_free_bytes=5_000_000_000,
        physical_gpu_models=("NVIDIA T4", "NVIDIA T4"),
        runtime_visible_gpu_count=1,
    )
    validate_resource_gate(shape, calculate_resource_budget(_budget_inputs(1_000_000_000)))
    with pytest.raises(Stage2ControlError, match="dynamic resource budget"):
        validate_resource_gate(shape, calculate_resource_budget(_budget_inputs(12_000_000_000)))
    with pytest.raises(ValidationError):
        ProviderShape.model_validate({**shape.model_dump(), "logical_cpu_count": 3})


def _cancellation(abort_delta: int = 0) -> CancellationProbe:
    samples = tuple(
        DrainSample(
            observation_offset_ns=index * 100_000_000,
            running_requests=0,
            waiting_requests=0,
            generation_tokens_total=10,
        )
        for index in range(40)
    )
    return CancellationProbe(
        client_close_offset_ns=0,
        first_generation_token_observed=True,
        external_abort_log_observed=True,
        internal_abort_log_observed=True,
        identity_chain_valid=True,
        later_terminal_reason=None,
        samples=samples,
        finished_reason_deltas=tuple(
            FinishedReasonDelta(
                finished_reason=reason,
                delta=abort_delta if reason == "abort" else 0,
            )
            for reason in ("abort", "length", "stop", "error", "repetition")
        ),
        residual_process_or_request_state=False,
    )


@pytest.mark.parametrize("abort_delta", [0, 1])
def test_cancellation_accepts_abort_counter_zero_or_one(abort_delta: int) -> None:
    assert (
        evaluate_cancellation(_cancellation(abort_delta))
        is CancellationClassification.SERVER_ABORT_ACKNOWLEDGED_AND_DRAINED
    )


def test_cancellation_rejects_nonabort_finish_or_residual_work() -> None:
    deltas = list(_cancellation().finished_reason_deltas)
    deltas[1] = deltas[1].model_copy(update={"delta": 1})
    probe = _cancellation().model_copy(update={"finished_reason_deltas": tuple(deltas)})
    assert evaluate_cancellation(probe) is CancellationClassification.LATER_COMPLETION
    samples = list(_cancellation().samples)
    samples[5] = samples[5].model_copy(update={"running_requests": 1})
    residual = _cancellation().model_copy(update={"samples": tuple(samples[:15])})
    assert evaluate_cancellation(residual) is CancellationClassification.RESIDUAL_WORK_TIMEOUT


def test_cancellation_rejected_classifications_and_continuous_cadence() -> None:
    assert (
        evaluate_cancellation(_cancellation().model_copy(update={"identity_chain_valid": False}))
        is CancellationClassification.ID_CORRELATION_FAILURE
    )
    assert (
        evaluate_cancellation(
            _cancellation().model_copy(update={"external_abort_log_observed": False})
        )
        is CancellationClassification.UNKNOWN_ACKNOWLEDGEMENT
    )
    assert (
        evaluate_cancellation(
            _cancellation().model_copy(update={"later_terminal_reason": "unrecognized"})
        )
        is CancellationClassification.TERMINAL_UNKNOWN
    )
    samples = list(_cancellation().samples)
    samples.pop(20)
    cadence_gap = _cancellation().model_copy(update={"samples": tuple(samples)})
    assert evaluate_cancellation(cadence_gap) is CancellationClassification.RESIDUAL_WORK_TIMEOUT
    samples = list(_cancellation().samples)
    samples[30] = samples[30].model_copy(update={"generation_tokens_total": 11})
    late_generation = _cancellation().model_copy(update={"samples": tuple(samples)})
    assert (
        evaluate_cancellation(late_generation) is CancellationClassification.RESIDUAL_WORK_TIMEOUT
    )


def _restart(index: int) -> RestartSemanticRecord:
    return RestartSemanticRecord(
        repetition_index=index,
        case_id="case-001",
        sent_prompt_token_ids=tuple(range(64)),
        returned_prompt_token_ids=tuple(range(64)),
        output_token_ids=tuple(range(32)),
        finish_reason="length",
        prompt_tokens=64,
        completion_tokens=32,
        total_tokens=96,
        output_text_sha256="0" * 64,
    )


def test_three_restart_exact_match_and_semantic_mismatches() -> None:
    records = (_restart(1), _restart(2), _restart(3))
    assert compare_three_restarts(records).state is AggregateComparisonState.COMMITTED
    changed = _restart(3).model_copy(update={"output_token_ids": (*range(31), 999)})
    result = compare_three_restarts((_restart(1), _restart(2), changed))
    assert result.state is AggregateComparisonState.INVALID_SEMANTIC_NONREPRODUCTION
    assert result.pooled_performance_interpretation_allowed is False


@pytest.mark.parametrize(
    "update",
    [
        {"sent_prompt_token_ids": tuple(range(1, 65))},
        {"returned_prompt_token_ids": tuple(range(1, 65))},
        {"finish_reason": "stop"},
        {"prompt_tokens": 63},
        {"completion_tokens": 31},
        {"total_tokens": 95},
        {"output_text_sha256": "1" * 64},
    ],
)
def test_three_restart_every_semantic_field_mismatch_invalidates(
    update: dict[str, object],
) -> None:
    changed = _restart(3).model_copy(update=update)
    result = compare_three_restarts((_restart(1), _restart(2), changed))
    assert result.state is AggregateComparisonState.INVALID_SEMANTIC_NONREPRODUCTION


def test_aggregate_commits_only_three_committed_repetitions_and_passing_cases() -> None:
    passing = compare_three_restarts((_restart(1), _restart(2), _restart(3)))
    assert (
        validate_aggregate_commit(
            (BundleState.COMMITTED,) * 3,
            (passing,),
            expected_case_ids=("case-001",),
        )
        is BundleState.COMMITTED
    )
    with pytest.raises(Stage2ControlError, match="three committed"):
        validate_aggregate_commit(
            (BundleState.COMMITTED, BundleState.INVALID, BundleState.COMMITTED),
            (passing,),
            expected_case_ids=("case-001",),
        )
    failing = compare_three_restarts(
        (_restart(1), _restart(2), _restart(3).model_copy(update={"total_tokens": 95}))
    )
    with pytest.raises(Stage2ControlError, match="semantic comparison"):
        validate_aggregate_commit(
            (BundleState.COMMITTED,) * 3,
            (failing,),
            expected_case_ids=("case-001",),
        )
    with pytest.raises(Stage2ControlError, match="every expected case"):
        validate_aggregate_commit(
            (BundleState.COMMITTED,) * 3,
            (passing,),
            expected_case_ids=("case-001", "case-002"),
        )


@pytest.mark.parametrize("kind", ["missing", "duplicate", "replacement"])
def test_three_restart_missing_duplicate_or_replacement_is_rejected(kind: str) -> None:
    records: tuple[RestartSemanticRecord, ...]
    if kind == "missing":
        records = (_restart(1), _restart(2))
    elif kind == "duplicate":
        records = (_restart(1), _restart(2), _restart(2))
    else:
        records = (
            _restart(1),
            _restart(2),
            _restart(3).model_copy(update={"replacement_run": True}),
        )
    with pytest.raises(Stage2ControlError):
        compare_three_restarts(records)


def test_tiny_n_reporting_prohibits_p99_and_goodput() -> None:
    metric = describe_tiny_n_metric("ttft_ns", (1.0, 2.0, 3.0), restart_group=1)
    assert metric.sample_count == 3
    assert metric.p99 is None
    assert metric.goodput_or_capacity_interpretation_allowed is False
    with pytest.raises(Stage2ControlError, match="p99"):
        describe_tiny_n_metric(
            "ttft_ns",
            (1.0, 2.0, 3.0),
            restart_group=1,
            requested_percentiles=(50, 95, 99),
        )
