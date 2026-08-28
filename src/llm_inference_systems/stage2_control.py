"""Deterministic Stage 2 experiment control, cancellation, and comparison logic."""

from __future__ import annotations

import hashlib
from enum import StrEnum
from itertools import pairwise
from typing import Annotated, Literal

from pydantic import Field

from llm_inference_systems.canonical import canonical_json_bytes
from llm_inference_systems.contracts import NonNegativeFloat, NonNegativeInt, Sha256, StrictModel
from llm_inference_systems.metrics import percentile_type7
from llm_inference_systems.stage2_contracts import (
    RUNTIME_PHASE_ORDER,
    BundleState,
    OfflineProcessRecord,
    ProcessClass,
    ProcessOperation,
    ProviderShape,
    ResourceBudgetInputs,
    ResourceBudgetResult,
    RuntimePhaseRecord,
    Stage2BundleManifest,
)

MAX_SIGNED_64 = 2**63 - 1


class Stage2ControlError(ValueError):
    """Raised when an experiment-control invariant is violated."""


def validate_runtime_phases(records: tuple[RuntimePhaseRecord, ...]) -> None:
    if tuple(record.phase for record in records) != RUNTIME_PHASE_ORDER:
        raise Stage2ControlError("runtime phases are missing, duplicated, or reordered")
    if any(not record.passed for record in records):
        raise Stage2ControlError("a runtime phase did not pass")
    for left, right in pairwise(records):
        if right.started_offset_ns < left.ended_offset_ns:
            raise Stage2ControlError("runtime phases overlap or regress")
    if any(record.post_warmup_jit_observed for record in records):
        raise Stage2ControlError("post-warmup monitored JIT invalidates the repetition")


def validate_offline_process_separation(records: tuple[OfflineProcessRecord, ...]) -> None:
    expected = tuple(ProcessClass)
    if tuple(record.process_class for record in records) != expected:
        raise Stage2ControlError("online, tokenizer, and three runtime processes must be separate")
    nonces = tuple(record.process_nonce for record in records)
    if len(nonces) != len(set(nonces)):
        raise Stage2ControlError("online or offline process reuse is prohibited")
    downloader = records[0]
    if (
        downloader.completed_operation is not ProcessOperation.SNAPSHOT_DOWNLOADED_AND_MANIFESTED
        or downloader.imported_runtime_or_tokenizer
    ):
        raise Stage2ControlError("online downloader process cannot become an offline runtime")
    offline = records[1:]
    if not all(
        record.token_variables_unset_without_reading
        and record.environment_set_before_import
        and record.imported_runtime_or_tokenizer
        and bool(record.verified_local_snapshot_relative_path)
        for record in offline
    ):
        raise Stage2ControlError("offline processes must unset token variables without reading")


def calculate_resource_budget(inputs: ResourceBudgetInputs | None) -> ResourceBudgetResult:
    if inputs is None:
        raise Stage2ControlError("all resource estimates and sources are required")
    estimates = (
        inputs.runtime_and_cuda_package_download.bytes,
        inputs.expected_installed_environment.bytes,
        inputs.model_tokenizer_snapshot.bytes,
        inputs.temporary_extraction_and_cache.bytes,
    )
    total = sum(estimates)
    if total > MAX_SIGNED_64 or total * 5 > MAX_SIGNED_64 * 4:
        raise OverflowError("resource-budget calculation exceeds signed 64-bit bounds")
    with_margin = (total * 5 + 3) // 4 + 2_000_000_000
    if with_margin > MAX_SIGNED_64:
        raise OverflowError("resource-budget margin exceeds signed 64-bit bounds")
    required_free = max(14_000_000_000, with_margin)
    return ResourceBudgetResult(
        required_setup_bytes=total,
        required_free_before_setup=required_free,
    )


def validate_resource_gate(shape: ProviderShape, budget: ResourceBudgetResult) -> None:
    if shape.initial_free_bytes < budget.required_free_before_setup:
        raise Stage2ControlError("initial free space fails the dynamic resource budget")


class CancellationClassification(StrEnum):
    SERVER_ABORT_ACKNOWLEDGED_AND_DRAINED = "SERVER_ABORT_ACKNOWLEDGED_AND_DRAINED"
    UNKNOWN_ACKNOWLEDGEMENT = "UNKNOWN_ACKNOWLEDGEMENT"
    LATER_COMPLETION = "LATER_COMPLETION"
    RESIDUAL_WORK_TIMEOUT = "RESIDUAL_WORK_TIMEOUT"
    TERMINAL_UNKNOWN = "TERMINAL_UNKNOWN"
    ID_CORRELATION_FAILURE = "ID_CORRELATION_FAILURE"


class DrainSample(StrictModel):
    observation_offset_ns: NonNegativeInt
    running_requests: NonNegativeInt
    waiting_requests: NonNegativeInt
    generation_tokens_total: NonNegativeFloat


class FinishedReasonDelta(StrictModel):
    finished_reason: str
    delta: NonNegativeFloat


class CancellationProbe(StrictModel):
    client_close_offset_ns: NonNegativeInt
    first_generation_token_observed: bool
    external_abort_log_observed: bool
    internal_abort_log_observed: bool
    identity_chain_valid: bool
    later_terminal_reason: str | None
    samples: tuple[DrainSample, ...]
    finished_reason_deltas: tuple[FinishedReasonDelta, ...]
    residual_process_or_request_state: bool


def _success_counter_valid(deltas: tuple[FinishedReasonDelta, ...]) -> bool:
    values: dict[str, float] = {}
    for item in deltas:
        if item.finished_reason in values:
            return False
        values[item.finished_reason] = item.delta
    if set(values) != {"abort", "length", "stop", "error", "repetition"}:
        return False
    abort_delta = values.pop("abort", 0.0)
    return abort_delta in (0.0, 1.0) and all(delta == 0 for delta in values.values())


def evaluate_cancellation(probe: CancellationProbe) -> CancellationClassification:
    if not probe.identity_chain_valid:
        return CancellationClassification.ID_CORRELATION_FAILURE
    if not probe.first_generation_token_observed:
        return CancellationClassification.TERMINAL_UNKNOWN
    if not (probe.external_abort_log_observed and probe.internal_abort_log_observed):
        return CancellationClassification.UNKNOWN_ACKNOWLEDGEMENT
    if probe.later_terminal_reason in {"length", "stop", "error", "repetition"}:
        return CancellationClassification.LATER_COMPLETION
    if probe.later_terminal_reason is not None:
        return CancellationClassification.TERMINAL_UNKNOWN
    if not _success_counter_valid(probe.finished_reason_deltas):
        return CancellationClassification.LATER_COMPLETION
    samples = probe.samples
    if not samples or any(
        right.observation_offset_ns <= left.observation_offset_ns
        for left, right in pairwise(samples)
    ):
        return CancellationClassification.RESIDUAL_WORK_TIMEOUT
    if samples[0].observation_offset_ns < probe.client_close_offset_ns:
        return CancellationClassification.RESIDUAL_WORK_TIMEOUT

    quiescent_end_index: int | None = None
    for start in range(max(0, len(samples) - 9)):
        window = samples[start : start + 10]
        if len(window) != 10:
            continue
        zero = all(item.running_requests == 0 and item.waiting_requests == 0 for item in window)
        cadence = all(
            right.observation_offset_ns - left.observation_offset_ns == 100_000_000
            for left, right in pairwise(window)
        )
        if zero and cadence:
            quiescent_end_index = start + 9
            break
    if quiescent_end_index is None:
        return CancellationClassification.RESIDUAL_WORK_TIMEOUT
    quiescent_end = samples[quiescent_end_index]
    stable_end_ns = quiescent_end.observation_offset_ns + 1_000_000_000
    stable = tuple(
        item
        for item in samples[quiescent_end_index:]
        if item.observation_offset_ns <= stable_end_ns
    )
    if not stable or stable[-1].observation_offset_ns != stable_end_ns:
        return CancellationClassification.RESIDUAL_WORK_TIMEOUT
    if any(
        right.observation_offset_ns - left.observation_offset_ns != 100_000_000
        for left, right in pairwise(stable)
    ):
        return CancellationClassification.RESIDUAL_WORK_TIMEOUT
    if any(
        item.generation_tokens_total != quiescent_end.generation_tokens_total for item in stable
    ):
        return CancellationClassification.RESIDUAL_WORK_TIMEOUT
    cooldown_end_ns = stable_end_ns + 2_000_000_000
    cooldown = tuple(
        item
        for item in samples[quiescent_end_index:]
        if item.observation_offset_ns <= cooldown_end_ns
    )
    if not cooldown or cooldown[-1].observation_offset_ns != cooldown_end_ns:
        return CancellationClassification.RESIDUAL_WORK_TIMEOUT
    if any(
        right.observation_offset_ns - left.observation_offset_ns != 100_000_000
        for left, right in pairwise(cooldown)
    ):
        return CancellationClassification.RESIDUAL_WORK_TIMEOUT
    if cooldown_end_ns - probe.client_close_offset_ns > 10_000_000_000:
        return CancellationClassification.RESIDUAL_WORK_TIMEOUT
    if any(
        item.running_requests != 0
        or item.waiting_requests != 0
        or item.generation_tokens_total != quiescent_end.generation_tokens_total
        for item in cooldown
    ):
        return CancellationClassification.RESIDUAL_WORK_TIMEOUT
    later_retained = tuple(item for item in samples if item.observation_offset_ns > cooldown_end_ns)
    if any(
        item.running_requests != 0
        or item.waiting_requests != 0
        or item.generation_tokens_total != quiescent_end.generation_tokens_total
        for item in later_retained
    ):
        return CancellationClassification.RESIDUAL_WORK_TIMEOUT
    if probe.residual_process_or_request_state:
        return CancellationClassification.RESIDUAL_WORK_TIMEOUT
    return CancellationClassification.SERVER_ABORT_ACKNOWLEDGED_AND_DRAINED


class RestartSemanticRecord(StrictModel):
    repetition_index: Annotated[int, Field(ge=1, le=3)]
    bundle_manifest_sha256: Sha256
    case_id: str
    sent_prompt_token_ids: tuple[int, ...] = Field(min_length=64, max_length=64)
    returned_prompt_token_ids: tuple[int, ...] = Field(min_length=64, max_length=64)
    output_token_ids: tuple[int, ...] = Field(min_length=32, max_length=32)
    finish_reason: Literal["length"]
    prompt_tokens: Literal[64]
    completion_tokens: Literal[32]
    total_tokens: Literal[96]
    output_text_sha256: Sha256
    replacement_run: bool = False


class AggregateComparisonState(StrEnum):
    COMMITTED = "COMMITTED"
    INVALID_SEMANTIC_NONREPRODUCTION = "INVALID_SEMANTIC_NONREPRODUCTION"


class RestartComparison(StrictModel):
    case_id: str
    state: AggregateComparisonState
    pooled_performance_interpretation_allowed: Literal[False]
    lis_obs_003_advancement_allowed: Literal[False]
    mismatches: tuple[str, ...]


def compare_three_restarts(records: tuple[RestartSemanticRecord, ...]) -> RestartComparison:
    indexes = tuple(record.repetition_index for record in records)
    if len(records) != 3 or set(indexes) != {1, 2, 3} or len(set(indexes)) != 3:
        raise Stage2ControlError("exactly three non-replaceable restart records are required")
    if any(record.replacement_run for record in records):
        raise Stage2ControlError("a fourth or replacement restart cannot be selected")
    ordered = tuple(sorted(records, key=lambda record: record.repetition_index))
    baseline = ordered[0]
    fields = (
        "case_id",
        "sent_prompt_token_ids",
        "returned_prompt_token_ids",
        "output_token_ids",
        "finish_reason",
        "prompt_tokens",
        "completion_tokens",
        "total_tokens",
        "output_text_sha256",
    )
    mismatches = tuple(
        f"restart-{candidate.repetition_index}:{field}"
        for candidate in ordered[1:]
        for field in fields
        if getattr(candidate, field) != getattr(baseline, field)
    )
    return RestartComparison(
        case_id=baseline.case_id,
        state=(
            AggregateComparisonState.INVALID_SEMANTIC_NONREPRODUCTION
            if mismatches
            else AggregateComparisonState.COMMITTED
        ),
        pooled_performance_interpretation_allowed=False,
        lis_obs_003_advancement_allowed=False,
        mismatches=mismatches,
    )


def bundle_manifest_sha256(manifest: Stage2BundleManifest) -> str:
    """Return the exact identity of the canonical manifest file bytes."""

    return hashlib.sha256(canonical_json_bytes(manifest) + b"\n").hexdigest()


def validate_aggregate_commit(
    repetition_manifests: tuple[Stage2BundleManifest, ...],
    case_records: tuple[tuple[RestartSemanticRecord, ...], ...],
    case_comparisons: tuple[RestartComparison, ...],
    *,
    expected_case_ids: tuple[str, ...],
) -> BundleState:
    if tuple(manifest.repetition_index for manifest in repetition_manifests) != (1, 2, 3):
        raise Stage2ControlError("aggregate commit requires exactly three committed repetitions")
    manifest_identities = {
        manifest.repetition_index: bundle_manifest_sha256(manifest)
        for manifest in repetition_manifests
    }
    comparison_case_ids = tuple(comparison.case_id for comparison in case_comparisons)
    record_case_ids = tuple(records[0].case_id for records in case_records if records)
    if (
        not expected_case_ids
        or len(expected_case_ids) != len(set(expected_case_ids))
        or comparison_case_ids != expected_case_ids
        or record_case_ids != expected_case_ids
        or len(case_records) != len(expected_case_ids)
    ):
        raise Stage2ControlError("aggregate commit requires every expected case comparison")
    for records in case_records:
        if tuple(record.repetition_index for record in records) != (1, 2, 3):
            raise Stage2ControlError("case records require all three repetition identities")
        if any(
            record.bundle_manifest_sha256 != manifest_identities[record.repetition_index]
            for record in records
        ):
            raise Stage2ControlError("case record is not bound to its repetition bundle")
    reconstructed = tuple(compare_three_restarts(records) for records in case_records)
    if reconstructed != case_comparisons:
        raise Stage2ControlError("semantic comparison does not reconstruct from bundle records")
    if any(
        comparison.state is not AggregateComparisonState.COMMITTED or comparison.mismatches
        for comparison in case_comparisons
    ):
        raise Stage2ControlError("aggregate commit requires passing semantic comparison")
    return BundleState.COMMITTED


class DescriptiveMetric(StrictModel):
    metric: str
    sample_count: NonNegativeInt
    restart_group: Annotated[int, Field(ge=1, le=3)]
    p50: NonNegativeFloat
    p95: NonNegativeFloat
    p99: None = None
    goodput_or_capacity_interpretation_allowed: Literal[False] = False


def describe_tiny_n_metric(
    metric: str,
    values: tuple[float, ...],
    *,
    restart_group: int,
    requested_percentiles: tuple[int, ...] = (50, 95),
) -> DescriptiveMetric:
    if 99 in requested_percentiles:
        raise Stage2ControlError("Stage 2 p99 calculation and display are prohibited")
    if set(requested_percentiles) != {50, 95} or not values:
        raise Stage2ControlError("Stage 2 reporting permits only nonempty p50/p95 descriptions")
    p50 = percentile_type7(values, 0.50)
    p95 = percentile_type7(values, 0.95)
    if p50 is None or p95 is None:
        raise Stage2ControlError("descriptive metric unexpectedly has no samples")
    return DescriptiveMetric(
        metric=metric,
        sample_count=len(values),
        restart_group=restart_group,
        p50=p50,
        p95=p95,
        p99=None,
        goodput_or_capacity_interpretation_allowed=False,
    )
