"""Deterministic Stage 2 experiment control, cancellation, and comparison logic."""

from __future__ import annotations

import hashlib
from enum import StrEnum
from itertools import pairwise
from typing import Annotated, Final, Literal, Self

from pydantic import Field, model_validator

from llm_inference_systems.canonical import canonical_json_bytes, sha256_identity
from llm_inference_systems.contracts import (
    Identifier,
    NonNegativeFloat,
    NonNegativeInt,
    Sha256,
    StrictModel,
)
from llm_inference_systems.metrics import percentile_type7
from llm_inference_systems.stage2_contracts import (
    RUNTIME_PHASE_ORDER,
    BundleState,
    ProviderShape,
    RawLogCapture,
    RequestIdentityChain,
    ResourceBudgetInputs,
    ResourceBudgetResult,
    RuntimePhaseRecord,
    Stage2BundleManifest,
)
from llm_inference_systems.stage2_prometheus import (
    CounterDelta,
    PrometheusProtocolError,
    PrometheusSnapshot,
    derive_counter_delta,
    require_quiescent,
    select_exact_series,
)
from llm_inference_systems.stage2_runtime import (
    GpuMemorySample,
    Stage2ProcessRecord,
    validate_gpu_memory_stability,
    validate_process_sequence,
)

MAX_SIGNED_64 = 2**63 - 1

PHASE_EVIDENCE_KINDS: Final = {
    phase: phase.value.casefold() + "-evidence-v0.3.0" for phase in RUNTIME_PHASE_ORDER
}


class Stage2ControlError(ValueError):
    """Raised when an experiment-control invariant is violated."""


def validate_runtime_phases(records: tuple[RuntimePhaseRecord, ...]) -> None:
    if tuple(record.phase for record in records) != RUNTIME_PHASE_ORDER:
        raise Stage2ControlError("runtime phases are missing, duplicated, or reordered")
    if any(not record.passed for record in records):
        raise Stage2ControlError("a runtime phase did not pass")
    if any(
        record.evidence_kind != PHASE_EVIDENCE_KINDS[record.phase]
        or record.evidence_references != (f"raw/phases/{record.phase.value.casefold()}.json",)
        for record in records
    ):
        raise Stage2ControlError("runtime phase lacks its exact phase-specific evidence")
    for left, right in pairwise(records):
        if right.started_offset_ns < left.ended_offset_ns:
            raise Stage2ControlError("runtime phases overlap or regress")
    if any(record.post_warmup_jit_observed for record in records):
        raise Stage2ControlError("post-warmup monitored JIT invalidates the repetition")


def validate_offline_process_separation(records: tuple[Stage2ProcessRecord, ...]) -> None:
    try:
        validate_process_sequence(records)
    except ValueError as error:
        raise Stage2ControlError(str(error)) from error


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


class FirstGenerationDeliveryEvidence(StrictModel):
    external_request_id: Identifier
    response_body_id: Identifier
    generation_event_ordinal: NonNegativeInt
    body_chunk_ordinal: NonNegativeInt
    observation_offset_ns: NonNegativeInt
    output_token_ids: tuple[NonNegativeInt, ...] = Field(min_length=1)


class ResidualStateEvidence(StrictModel):
    observation_offset_ns: NonNegativeInt
    raw_process_inventory: str
    raw_process_inventory_sha256: Sha256
    active_request_ids: tuple[str, ...]
    project_process_ids: tuple[NonNegativeInt, ...]

    def hashes_reconstruct(self) -> bool:
        return (
            self.raw_process_inventory_sha256
            == hashlib.sha256(self.raw_process_inventory.encode("utf-8")).hexdigest()
        )


class CancellationScrapeObservationEvidence(StrictModel):
    phase: Literal["PRE_DISPATCH", "DRAIN", "STABLE_GENERATION", "COOLDOWN", "LATER"]
    phase_ordinal: NonNegativeInt
    scheduled_offset_ns: NonNegativeInt
    request_dispatch_offset_ns: NonNegativeInt
    response_completion_offset_ns: NonNegativeInt
    snapshot_identity_sha256: Sha256

    @model_validator(mode="after")
    def validate_observation(self) -> Self:
        if not (
            self.scheduled_offset_ns
            <= self.request_dispatch_offset_ns
            <= self.response_completion_offset_ns
        ):
            raise ValueError("cancellation scrape schedule/dispatch/completion order differs")
        return self


class CancellationProbe(StrictModel):
    repetition_index: Annotated[int, Field(ge=1, le=3)]
    server_process_identity: Identifier
    identity_chain: RequestIdentityChain
    raw_log_capture: RawLogCapture
    raw_log_capture_sha256: Sha256
    raw_log_start_byte_offset: NonNegativeInt
    dispatch_offset_ns: NonNegativeInt
    first_generation_delivery: FirstGenerationDeliveryEvidence
    client_close_offset_ns: NonNegativeInt
    pre_dispatch_snapshots: tuple[PrometheusSnapshot, ...] = Field(min_length=10, max_length=10)
    drain_snapshots: tuple[PrometheusSnapshot, ...] = Field(min_length=10, max_length=10)
    stable_generation_snapshots: tuple[PrometheusSnapshot, ...] = Field(min_length=2)
    cooldown_snapshots: tuple[PrometheusSnapshot, ...] = Field(min_length=2)
    later_retained_snapshots: tuple[PrometheusSnapshot, ...]
    scrape_observations: tuple[CancellationScrapeObservationEvidence, ...] = Field(min_length=52)
    residual_state: ResidualStateEvidence


class CancellationResult(StrictModel):
    classification: CancellationClassification
    accepted: bool
    evidence_identity_sha256: Sha256
    prompt_token_delta: CounterDelta | None
    generation_token_delta: CounterDelta | None
    finished_reason_deltas: tuple[CounterDelta, ...]
    auxiliary_counter_deltas: tuple[CounterDelta, ...]
    observed_abort_delta: NonNegativeFloat | None

    @model_validator(mode="after")
    def validate_result(self) -> Self:
        success = (
            self.classification is CancellationClassification.SERVER_ABORT_ACKNOWLEDGED_AND_DRAINED
        )
        if self.accepted != success:
            raise ValueError("cancellation acceptance must match its classification")
        if success and (
            self.prompt_token_delta is None
            or self.generation_token_delta is None
            or len(self.finished_reason_deltas) != 5
            or len(self.auxiliary_counter_deltas) != 3
            or self.observed_abort_delta is None
        ):
            raise ValueError("accepted cancellation result lacks derived counter evidence")
        if self.observed_abort_delta is not None and self.observed_abort_delta not in (0.0, 1.0):
            raise ValueError("observed abort delta must remain exactly zero or one")
        if success:
            assert self.prompt_token_delta is not None
            assert self.generation_token_delta is not None
            reason_values = {
                dict(delta.labels).get("finished_reason"): delta.delta
                for delta in self.finished_reason_deltas
            }
            if (
                self.prompt_token_delta.metric != "vllm:prompt_tokens_total"
                or self.prompt_token_delta.delta != 64.0
                or self.generation_token_delta.metric != "vllm:generation_tokens_total"
                or self.generation_token_delta.delta < 1.0
                or reason_values.get("abort") != self.observed_abort_delta
                or any(
                    reason_values.get(reason) != 0.0
                    for reason in ("length", "stop", "error", "repetition")
                )
                or tuple(delta.metric for delta in self.auxiliary_counter_deltas)
                != (
                    "vllm:num_preemptions_total",
                    "vllm:prefix_cache_queries_total",
                    "vllm:prefix_cache_hits_total",
                )
                or any(delta.delta != 0.0 for delta in self.auxiliary_counter_deltas)
            ):
                raise ValueError("accepted cancellation counter evidence differs from the contract")
        if not success and any(
            value is not None
            for value in (
                self.prompt_token_delta,
                self.generation_token_delta,
                self.observed_abort_delta,
            )
        ):
            raise ValueError("invalid cancellation cannot retain accepted derived deltas")
        if not success and (self.finished_reason_deltas or self.auxiliary_counter_deltas):
            raise ValueError("invalid cancellation cannot retain derived counter sequences")
        return self


def _invalid_cancellation(
    probe: CancellationProbe,
    classification: CancellationClassification,
) -> CancellationResult:
    return CancellationResult(
        classification=classification,
        accepted=False,
        evidence_identity_sha256=sha256_identity(probe),
        prompt_token_delta=None,
        generation_token_delta=None,
        finished_reason_deltas=(),
        auxiliary_counter_deltas=(),
        observed_abort_delta=None,
    )


def _snapshot_offsets(snapshots: tuple[PrometheusSnapshot, ...]) -> tuple[int, ...]:
    return tuple(snapshot.scrape_monotonic_offset_ns for snapshot in snapshots)


def _scheduled_offsets(
    observations: tuple[CancellationScrapeObservationEvidence, ...],
) -> tuple[int, ...]:
    return tuple(observation.scheduled_offset_ns for observation in observations)


def _spaced_at_least(snapshots: tuple[PrometheusSnapshot, ...], spacing_ns: int) -> bool:
    return all(right - left >= spacing_ns for left, right in pairwise(_snapshot_offsets(snapshots)))


def _same_process(snapshots: tuple[PrometheusSnapshot, ...]) -> bool:
    return len({snapshot.process_start_id for snapshot in snapshots}) == 1


def _generation_value(snapshot: PrometheusSnapshot) -> float:
    return select_exact_series(snapshot, "vllm:generation_tokens_total").value


_CANCELLATION_COUNTER_SELECTORS: Final = (
    ("vllm:prompt_tokens_total", None),
    ("vllm:generation_tokens_total", None),
    ("vllm:num_preemptions_total", None),
    ("vllm:prefix_cache_queries_total", None),
    ("vllm:prefix_cache_hits_total", None),
    ("vllm:request_success_total", "abort"),
    ("vllm:request_success_total", "length"),
    ("vllm:request_success_total", "stop"),
    ("vllm:request_success_total", "error"),
    ("vllm:request_success_total", "repetition"),
)


def _validate_counter_progression(snapshots: tuple[PrometheusSnapshot, ...]) -> bool:
    try:
        for left, right in pairwise(snapshots):
            for metric, reason in _CANCELLATION_COUNTER_SELECTORS:
                derive_counter_delta(left, right, metric, finished_reason=reason)
    except PrometheusProtocolError:
        return False
    return True


def _all_selected_counters_stable(snapshots: tuple[PrometheusSnapshot, ...]) -> bool:
    try:
        return all(
            derive_counter_delta(left, right, metric, finished_reason=reason).delta == 0
            for left, right in pairwise(snapshots)
            for metric, reason in _CANCELLATION_COUNTER_SELECTORS
        )
    except PrometheusProtocolError:
        return False


def evaluate_cancellation(probe: CancellationProbe) -> CancellationResult:
    chain = probe.identity_chain
    if (
        chain.response_body_id != f"cmpl-{chain.external_base_id}"
        or chain.serving_item_id != f"cmpl-{chain.external_base_id}-0"
    ):
        return _invalid_cancellation(probe, CancellationClassification.ID_CORRELATION_FAILURE)
    if chain.external_abort_log is None or chain.internal_abort_log is None:
        return _invalid_cancellation(probe, CancellationClassification.UNKNOWN_ACKNOWLEDGEMENT)
    from llm_inference_systems.stage2_protocol import (
        Stage2ProtocolError,
        correlate_request_logs,
    )

    try:
        reconstructed_chain = correlate_request_logs(
            chain.external_base_id,
            probe.raw_log_capture.records,
            cancellation=True,
        )
    except Stage2ProtocolError:
        return _invalid_cancellation(probe, CancellationClassification.ID_CORRELATION_FAILURE)
    if (
        reconstructed_chain != chain
        or probe.raw_log_capture_sha256 != probe.raw_log_capture.raw_bytes_sha256
        or probe.raw_log_capture.source_stream_id != f"{probe.server_process_identity}-raw-log"
        or chain.request_received_log.byte_start < probe.raw_log_start_byte_offset
        or probe.first_generation_delivery.external_request_id != chain.external_base_id
        or probe.first_generation_delivery.response_body_id != chain.response_body_id
    ):
        return _invalid_cancellation(probe, CancellationClassification.ID_CORRELATION_FAILURE)
    if not (
        probe.dispatch_offset_ns
        < probe.first_generation_delivery.observation_offset_ns
        <= probe.client_close_offset_ns
    ):
        return _invalid_cancellation(probe, CancellationClassification.TERMINAL_UNKNOWN)
    if not (
        probe.dispatch_offset_ns
        <= chain.request_received_log.observation_offset_ns
        <= chain.request_add_log.observation_offset_ns
        <= probe.first_generation_delivery.observation_offset_ns
        <= probe.client_close_offset_ns
        <= chain.internal_abort_log.observation_offset_ns
        <= chain.external_abort_log.observation_offset_ns
    ):
        return _invalid_cancellation(probe, CancellationClassification.TERMINAL_UNKNOWN)

    pre = probe.pre_dispatch_snapshots
    drain = probe.drain_snapshots
    stable = probe.stable_generation_snapshots
    cooldown = probe.cooldown_snapshots
    all_snapshots = (*pre, *drain, *stable, *cooldown, *probe.later_retained_snapshots)
    if len(pre) != 10 or len(drain) != 10:
        return _invalid_cancellation(probe, CancellationClassification.RESIDUAL_WORK_TIMEOUT)
    if not _same_process(all_snapshots) or any(
        snapshot.process_start_id != probe.server_process_identity for snapshot in all_snapshots
    ):
        return _invalid_cancellation(probe, CancellationClassification.RESIDUAL_WORK_TIMEOUT)
    phase_snapshots = (
        ("PRE_DISPATCH", pre),
        ("DRAIN", drain),
        ("STABLE_GENERATION", stable),
        ("COOLDOWN", cooldown),
        ("LATER", probe.later_retained_snapshots),
    )
    expected_observation_keys = tuple(
        (phase, ordinal, sha256_identity(snapshot))
        for phase, snapshots in phase_snapshots
        for ordinal, snapshot in enumerate(snapshots)
    )
    actual_observation_keys = tuple(
        (observation.phase, observation.phase_ordinal, observation.snapshot_identity_sha256)
        for observation in probe.scrape_observations
    )
    if (
        actual_observation_keys != expected_observation_keys
        or any(
            observation.response_completion_offset_ns != snapshot.scrape_monotonic_offset_ns
            for observation, snapshot in zip(probe.scrape_observations, all_snapshots, strict=True)
        )
        or any(
            not (
                observation.scheduled_offset_ns
                <= observation.request_dispatch_offset_ns
                <= observation.response_completion_offset_ns
            )
            for observation in probe.scrape_observations
        )
        or _snapshot_offsets(all_snapshots) != tuple(sorted(_snapshot_offsets(all_snapshots)))
    ):
        return _invalid_cancellation(probe, CancellationClassification.RESIDUAL_WORK_TIMEOUT)
    counts = tuple(len(snapshots) for _, snapshots in phase_snapshots)
    boundaries = tuple(sum(counts[:index]) for index in range(len(counts) + 1))
    pre_observations = probe.scrape_observations[boundaries[0] : boundaries[1]]
    drain_observations = probe.scrape_observations[boundaries[1] : boundaries[2]]
    stable_observations = probe.scrape_observations[boundaries[2] : boundaries[3]]
    cooldown_observations = probe.scrape_observations[boundaries[3] : boundaries[4]]
    if (
        not all(
            right - left >= 100_000_000
            for left, right in pairwise(_scheduled_offsets(pre_observations))
        )
        or not _spaced_at_least(pre, 100_000_000)
        or pre[-1].scrape_monotonic_offset_ns >= probe.dispatch_offset_ns
    ):
        return _invalid_cancellation(probe, CancellationClassification.RESIDUAL_WORK_TIMEOUT)
    try:
        for snapshot in (*pre, *drain, *stable, *cooldown, *probe.later_retained_snapshots):
            require_quiescent(snapshot)
    except PrometheusProtocolError:
        return _invalid_cancellation(probe, CancellationClassification.RESIDUAL_WORK_TIMEOUT)
    if (
        drain[0].scrape_monotonic_offset_ns < probe.client_close_offset_ns
        or chain.external_abort_log.observation_offset_ns > drain[0].scrape_monotonic_offset_ns
        or not all(
            right - left >= 100_000_000
            for left, right in pairwise(_scheduled_offsets(drain_observations))
        )
        or not _spaced_at_least(drain, 100_000_000)
        or stable[0].scrape_monotonic_offset_ns != drain[-1].scrape_monotonic_offset_ns
        or stable[0] != drain[-1]
        or (
            stable_observations[0].scheduled_offset_ns,
            stable_observations[0].request_dispatch_offset_ns,
            stable_observations[0].response_completion_offset_ns,
        )
        != (
            drain_observations[-1].scheduled_offset_ns,
            drain_observations[-1].request_dispatch_offset_ns,
            drain_observations[-1].response_completion_offset_ns,
        )
        or stable[-1].scrape_monotonic_offset_ns - stable[0].scrape_monotonic_offset_ns
        < 1_000_000_000
        or not all(
            right - left == 100_000_000
            for left, right in pairwise(_scheduled_offsets(stable_observations))
        )
        or cooldown[0].scrape_monotonic_offset_ns != stable[-1].scrape_monotonic_offset_ns
        or (
            cooldown_observations[0].scheduled_offset_ns,
            cooldown_observations[0].request_dispatch_offset_ns,
            cooldown_observations[0].response_completion_offset_ns,
        )
        != (
            stable_observations[-1].scheduled_offset_ns,
            stable_observations[-1].request_dispatch_offset_ns,
            stable_observations[-1].response_completion_offset_ns,
        )
        or cooldown[-1].scrape_monotonic_offset_ns - cooldown[0].scrape_monotonic_offset_ns
        < 2_000_000_000
        or not all(
            right - left == 100_000_000
            for left, right in pairwise(_scheduled_offsets(cooldown_observations))
        )
        or cooldown[-1].scrape_monotonic_offset_ns - probe.client_close_offset_ns > 10_000_000_000
        or chain.external_abort_log.observation_offset_ns > cooldown[-1].scrape_monotonic_offset_ns
        or any(
            snapshot.scrape_monotonic_offset_ns <= cooldown[-1].scrape_monotonic_offset_ns
            for snapshot in probe.later_retained_snapshots
        )
        or tuple(snapshot.scrape_monotonic_offset_ns for snapshot in probe.later_retained_snapshots)
        != tuple(
            sorted(
                snapshot.scrape_monotonic_offset_ns for snapshot in probe.later_retained_snapshots
            )
        )
    ):
        return _invalid_cancellation(probe, CancellationClassification.RESIDUAL_WORK_TIMEOUT)
    chronological = (*pre, *drain, *stable[1:], *cooldown[1:], *probe.later_retained_snapshots)
    stable_and_later = (*stable, *cooldown[1:], *probe.later_retained_snapshots)
    if not _validate_counter_progression(chronological) or not _all_selected_counters_stable(pre):
        return _invalid_cancellation(probe, CancellationClassification.RESIDUAL_WORK_TIMEOUT)
    last_retained_snapshot = (
        probe.later_retained_snapshots[-1] if probe.later_retained_snapshots else cooldown[-1]
    )
    try:
        retained_reason_deltas = {
            reason: derive_counter_delta(
                pre[-1],
                last_retained_snapshot,
                "vllm:request_success_total",
                finished_reason=reason,
            ).delta
            for reason in ("abort", "length", "stop", "error", "repetition")
        }
    except PrometheusProtocolError:
        return _invalid_cancellation(probe, CancellationClassification.RESIDUAL_WORK_TIMEOUT)
    if retained_reason_deltas["abort"] not in (0.0, 1.0) or any(
        retained_reason_deltas[reason] != 0.0
        for reason in ("length", "stop", "error", "repetition")
    ):
        return _invalid_cancellation(probe, CancellationClassification.LATER_COMPLETION)
    if not _all_selected_counters_stable(stable_and_later):
        return _invalid_cancellation(probe, CancellationClassification.RESIDUAL_WORK_TIMEOUT)
    try:
        stable_value = _generation_value(stable[0])
        if any(
            _generation_value(snapshot) != stable_value
            for snapshot in (*stable, *cooldown, *probe.later_retained_snapshots)
        ):
            return _invalid_cancellation(probe, CancellationClassification.RESIDUAL_WORK_TIMEOUT)
        baseline = pre[-1]
        final = cooldown[-1]
        prompt_delta = derive_counter_delta(baseline, final, "vllm:prompt_tokens_total")
        generation_delta = derive_counter_delta(baseline, final, "vllm:generation_tokens_total")
        reasons = ("abort", "length", "stop", "error", "repetition")
        reason_deltas = tuple(
            derive_counter_delta(
                baseline,
                final,
                "vllm:request_success_total",
                finished_reason=reason,
            )
            for reason in reasons
        )
        auxiliary_deltas = tuple(
            derive_counter_delta(baseline, final, metric)
            for metric in (
                "vllm:num_preemptions_total",
                "vllm:prefix_cache_queries_total",
                "vllm:prefix_cache_hits_total",
            )
        )
    except PrometheusProtocolError:
        return _invalid_cancellation(probe, CancellationClassification.RESIDUAL_WORK_TIMEOUT)
    observed = {dict(delta.labels)["finished_reason"]: delta.delta for delta in reason_deltas}
    if observed["abort"] not in (0.0, 1.0) or any(
        observed[reason] != 0.0 for reason in ("length", "stop", "error", "repetition")
    ):
        return _invalid_cancellation(probe, CancellationClassification.LATER_COMPLETION)
    if (
        prompt_delta.delta != 64.0
        or generation_delta.delta < 1.0
        or any(delta.delta != 0.0 for delta in auxiliary_deltas)
    ):
        return _invalid_cancellation(probe, CancellationClassification.RESIDUAL_WORK_TIMEOUT)
    last_retained_offset = (
        probe.later_retained_snapshots[-1].scrape_monotonic_offset_ns
        if probe.later_retained_snapshots
        else cooldown[-1].scrape_monotonic_offset_ns
    )
    if (
        not probe.residual_state.hashes_reconstruct()
        or probe.residual_state.observation_offset_ns < last_retained_offset
        or probe.residual_state.active_request_ids
        or probe.residual_state.project_process_ids
    ):
        return _invalid_cancellation(probe, CancellationClassification.RESIDUAL_WORK_TIMEOUT)
    return CancellationResult(
        classification=CancellationClassification.SERVER_ABORT_ACKNOWLEDGED_AND_DRAINED,
        accepted=True,
        evidence_identity_sha256=sha256_identity(probe),
        prompt_token_delta=prompt_delta,
        generation_token_delta=generation_delta,
        finished_reason_deltas=reason_deltas,
        auxiliary_counter_deltas=auxiliary_deltas,
        observed_abort_delta=observed["abort"],
    )


class ProcessExitEvidence(StrictModel):
    process_identity: Identifier
    exit_code: Literal[0]
    observed_offset_ns: NonNegativeInt
    raw_evidence_sha256: Sha256


class Stage2RuntimeControlEvidence(StrictModel):
    repetition_index: Annotated[int, Field(ge=1, le=3)]
    phases: tuple[RuntimePhaseRecord, ...]
    process_records: tuple[Stage2ProcessRecord, ...]
    gpu_memory_samples: tuple[GpuMemorySample, ...] = Field(min_length=5, max_length=5)
    gpu_memory_tolerance_bytes: NonNegativeInt
    stabilization_request_count: Literal[3]
    stabilization_request_ids: tuple[Identifier, ...] = Field(min_length=3, max_length=3)
    workload_shape_warmup_count: Literal[4]
    workload_shape_warmup_request_ids: tuple[Identifier, ...] = Field(min_length=4, max_length=4)
    cancellation_probe: CancellationProbe
    cancellation_result: CancellationResult
    steady_state_snapshots: tuple[PrometheusSnapshot, ...] = Field(min_length=10, max_length=10)
    prefix_cache_query_delta: CounterDelta
    prefix_cache_hit_delta: CounterDelta
    post_warmup_jit_event_hashes: tuple[Sha256, ...]
    quiet_interval_start_offset_ns: NonNegativeInt
    quiet_interval_end_offset_ns: NonNegativeInt
    measured_request_count: Literal[16]
    measured_request_ids: tuple[Identifier, ...] = Field(min_length=16, max_length=16)
    requested_client_concurrency: Literal[2]
    measured_client_slot_assignments: tuple[Literal[0, 1], ...] = Field(
        min_length=16, max_length=16
    )
    final_drain_completed_offset_ns: NonNegativeInt
    final_metric_scrape: PrometheusSnapshot
    shutdown_processes: tuple[ProcessExitEvidence, ...] = Field(min_length=1)
    residual_process_ids: tuple[NonNegativeInt, ...]
    residual_active_request_ids: tuple[str, ...]
    residual_verification_offset_ns: NonNegativeInt
    residual_verification_evidence_sha256: Sha256

    @model_validator(mode="after")
    def validate_runtime_control(self) -> Self:
        validate_runtime_phases(self.phases)
        validate_offline_process_separation(self.process_records)
        tolerance = validate_gpu_memory_stability(self.gpu_memory_samples)
        if self.gpu_memory_tolerance_bytes != tolerance:
            raise ValueError("GPU-memory tolerance does not reconstruct from the first sample")
        identity_groups = (
            self.stabilization_request_ids,
            self.workload_shape_warmup_request_ids,
            self.measured_request_ids,
        )
        if any(len(values) != len(set(values)) for values in identity_groups):
            raise ValueError("excluded and measured request identities must be unique")
        all_request_ids = (
            *self.stabilization_request_ids,
            *self.workload_shape_warmup_request_ids,
            self.cancellation_probe.identity_chain.external_base_id,
            *self.measured_request_ids,
        )
        if len(all_request_ids) != len(set(all_request_ids)):
            raise ValueError("excluded, cancellation, and measured request IDs must be disjoint")
        if set(self.measured_client_slot_assignments) != {0, 1}:
            raise ValueError("measured request identities must exercise both requested clients")
        if evaluate_cancellation(self.cancellation_probe) != self.cancellation_result:
            raise ValueError("cancellation result does not reconstruct from raw probe evidence")
        expected_server = self.process_records[self.repetition_index + 1].process_identity
        if (
            self.cancellation_probe.repetition_index != self.repetition_index
            or self.cancellation_probe.server_process_identity != expected_server
        ):
            raise ValueError("cancellation probe differs from its repetition or server process")
        if not self.cancellation_result.accepted:
            raise ValueError("runtime control requires an accepted isolated cancellation probe")
        steady = self.steady_state_snapshots
        if not _same_process(steady) or not _spaced_at_least(steady, 100_000_000):
            raise ValueError("steady-state samples lack exact process identity or spacing")
        try:
            for snapshot in steady:
                require_quiescent(snapshot)
            expected_queries = derive_counter_delta(
                steady[0], steady[-1], "vllm:prefix_cache_queries_total"
            )
            expected_hits = derive_counter_delta(
                steady[0], steady[-1], "vllm:prefix_cache_hits_total"
            )
        except PrometheusProtocolError as error:
            raise ValueError("steady-state exact-series evidence is invalid") from error
        if (
            self.prefix_cache_query_delta != expected_queries
            or self.prefix_cache_hit_delta != expected_hits
            or expected_queries.delta != 0
            or expected_hits.delta != 0
        ):
            raise ValueError("prefix-cache query and hit deltas must reconstruct as zero")
        if self.post_warmup_jit_event_hashes:
            raise ValueError("post-warmup monitored JIT evidence invalidates runtime control")
        memory_phase = self.phases[RUNTIME_PHASE_ORDER.index("ALLOCATOR_KV_STABILIZATION")]
        if not (
            memory_phase.started_offset_ns <= self.gpu_memory_samples[0].observation_offset_ns
            and self.gpu_memory_samples[-1].observation_offset_ns <= memory_phase.ended_offset_ns
        ):
            raise ValueError("GPU-memory samples fall outside allocator/KV stabilization")
        steady_phase = self.phases[RUNTIME_PHASE_ORDER.index("STEADY_STATE_GATE")]
        if (
            steady_phase.started_offset_ns > steady[0].scrape_monotonic_offset_ns
            or self.quiet_interval_start_offset_ns < steady[-1].scrape_monotonic_offset_ns
            or self.quiet_interval_end_offset_ns - self.quiet_interval_start_offset_ns
            < 2_000_000_000
            or self.quiet_interval_end_offset_ns > steady_phase.ended_offset_ns
        ):
            raise ValueError("steady-state samples or quiet interval fall outside their phase")
        measured_phase = self.phases[RUNTIME_PHASE_ORDER.index("MEASURED_WINDOW")]
        cancellation_phase = self.phases[RUNTIME_PHASE_ORDER.index("CANCELLATION_PROBE_DRAIN")]
        if cancellation_phase.ended_offset_ns > measured_phase.started_offset_ns:
            raise ValueError("cancellation gate must complete before measurement")
        if self.quiet_interval_end_offset_ns > measured_phase.started_offset_ns:
            raise ValueError("steady-state quiet interval must complete before measurement")
        if not (
            cancellation_phase.started_offset_ns
            <= self.cancellation_probe.pre_dispatch_snapshots[0].scrape_monotonic_offset_ns
            and self.cancellation_probe.residual_state.observation_offset_ns
            <= cancellation_phase.ended_offset_ns
        ):
            raise ValueError("cancellation evidence falls outside the cancellation phase")
        final_drain_phase = self.phases[RUNTIME_PHASE_ORDER.index("FINAL_METRICS_DRAIN")]
        if not (
            final_drain_phase.started_offset_ns
            <= self.final_drain_completed_offset_ns
            < self.final_metric_scrape.scrape_monotonic_offset_ns
            <= final_drain_phase.ended_offset_ns
        ):
            raise ValueError(
                "final drain completion or metric scrape is outside the final-drain phase"
            )
        try:
            require_quiescent(self.final_metric_scrape)
        except PrometheusProtocolError as error:
            raise ValueError("final metric scrape is not drained and quiescent") from error
        expected_server = self.process_records[self.repetition_index + 1].process_identity
        if (
            self.final_metric_scrape.process_start_id != steady[0].process_start_id
            or self.final_metric_scrape.process_start_id != expected_server
            or self.cancellation_probe.pre_dispatch_snapshots[0].process_start_id != expected_server
        ):
            raise ValueError("runtime metric evidence is not bound to its repetition server")
        shutdown_phase = self.phases[RUNTIME_PHASE_ORDER.index("SHUTDOWN")]
        if any(
            not (
                shutdown_phase.started_offset_ns
                <= item.observed_offset_ns
                <= shutdown_phase.ended_offset_ns
            )
            for item in self.shutdown_processes
        ):
            raise ValueError("server or worker shutdown evidence is outside the shutdown phase")
        residual_phase = self.phases[RUNTIME_PHASE_ORDER.index("NO_RESIDUAL_PROCESS_VERIFICATION")]
        if not (
            residual_phase.started_offset_ns
            <= self.residual_verification_offset_ns
            <= residual_phase.ended_offset_ns
        ):
            raise ValueError("residual-process evidence is outside its verification phase")
        if self.residual_process_ids or self.residual_active_request_ids:
            raise ValueError("runtime control retains a residual process or active request")
        if len({item.process_identity for item in self.shutdown_processes}) != len(
            self.shutdown_processes
        ) or (
            len(self.shutdown_processes) != 2
            or expected_server not in {item.process_identity for item in self.shutdown_processes}
        ):
            raise ValueError("shutdown evidence must identify one repetition server and worker")
        expected_phase_identities = {
            "OFFLINE_SNAPSHOT_VERIFICATION": sha256_identity(self.process_records[:2]),
            "RUNTIME_PROCESS_START": sha256_identity(self.process_records[2:]),
            "JIT_COMPILATION_STATE": sha256_identity(self.post_warmup_jit_event_hashes),
            "ALLOCATOR_KV_STABILIZATION": sha256_identity(
                {
                    "gpu_memory_samples": self.gpu_memory_samples,
                    "gpu_memory_tolerance_bytes": self.gpu_memory_tolerance_bytes,
                }
            ),
            "EXCLUDED_STABILIZATION_REQUESTS": sha256_identity(self.stabilization_request_ids),
            "EXCLUDED_SHAPE_WARMUPS": sha256_identity(self.workload_shape_warmup_request_ids),
            "CANCELLATION_PROBE_DRAIN": sha256_identity(
                {"probe": self.cancellation_probe, "result": self.cancellation_result}
            ),
            "STEADY_STATE_GATE": sha256_identity(
                {
                    "prefix_cache_hit_delta": self.prefix_cache_hit_delta,
                    "prefix_cache_query_delta": self.prefix_cache_query_delta,
                    "quiet_interval_end_offset_ns": self.quiet_interval_end_offset_ns,
                    "quiet_interval_start_offset_ns": self.quiet_interval_start_offset_ns,
                    "steady_state_snapshots": self.steady_state_snapshots,
                }
            ),
            "MEASURED_WINDOW": sha256_identity(
                {
                    "measured_client_slot_assignments": self.measured_client_slot_assignments,
                    "measured_request_ids": self.measured_request_ids,
                    "requested_client_concurrency": self.requested_client_concurrency,
                }
            ),
            "FINAL_METRICS_DRAIN": sha256_identity(
                {
                    "final_drain_completed_offset_ns": self.final_drain_completed_offset_ns,
                    "final_metric_scrape": self.final_metric_scrape,
                }
            ),
            "SHUTDOWN": sha256_identity(self.shutdown_processes),
            "NO_RESIDUAL_PROCESS_VERIFICATION": sha256_identity(
                {
                    "residual_active_request_ids": self.residual_active_request_ids,
                    "residual_process_ids": self.residual_process_ids,
                    "residual_verification_evidence_sha256": (
                        self.residual_verification_evidence_sha256
                    ),
                    "residual_verification_offset_ns": self.residual_verification_offset_ns,
                }
            ),
        }
        if any(
            self.phases[RUNTIME_PHASE_ORDER.index(phase)].evidence_identity_sha256 != identity
            for phase, identity in expected_phase_identities.items()
        ):
            raise ValueError("phase-specific evidence identity does not reconstruct")
        return self


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
