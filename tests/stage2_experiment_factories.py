from __future__ import annotations

import hashlib
import os
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal, cast

from llm_inference_systems.canonical import canonical_json_bytes, sha256_identity
from llm_inference_systems.stage2_attestation import (
    CudaBackedExecutionAttestation,
    LinuxEnvironmentManifest,
    NvidiaT4ResourceAttestation,
    PublicSafetyAttestation,
    RequestIdentityAttestation,
    ServerRestartIdentity,
)
from llm_inference_systems.stage2_contracts import (
    BundleFileEntry,
    BundleState,
    Stage2BundleManifest,
    Stage2EvidenceScope,
    Stage2RequestEvidence,
)
from llm_inference_systems.stage2_control import bundle_manifest_sha256, compare_three_restarts
from llm_inference_systems.stage2_experiment import (
    STAGE2_EXPERIMENT_CASE_IDS,
    AggregateRootState,
    ManifestBoundFile,
    Stage2AggregateExperimentManifest,
    Stage2CrossRestartComparison,
    Stage2ExperimentAttestation,
    Stage2ExperimentClassification,
    Stage2ExperimentSummary,
    Stage2ExperimentWorkload,
    Stage2MeasuredRequestAttestation,
    Stage2RepetitionAttestation,
    Stage2RepetitionCudaAttestation,
    Stage2RequestLifecycle,
    Stage2RequestRawEvidence,
    Stage2RestartSemanticAttestation,
    Stage2WorkloadCase,
    build_cancellation_client_stream_raw_evidence_bytes,
    build_cuda_raw_evidence_bytes,
    build_request_raw_evidence_payloads,
    derive_aggregate_validation_result,
    derive_metric_availability,
    derive_metric_availability_summary,
    environment_raw_evidence_bytes,
    execution_lock_raw_evidence_bytes,
    nvidia_raw_evidence_bytes,
    public_safety_raw_evidence_bytes,
    reconstruct_experiment_repetition,
    scoped_raw_evidence_bytes,
    snapshot_read_only_raw_evidence_bytes,
    write_aggregate_manifest_last,
)
from tests.stage2_factories import (
    FIXTURE_IDENTITY,
    PROMPT,
    make_log_chain,
    make_real_runtime_attestation,
    make_request_evidence,
    make_runtime_control,
)


def _json_bytes(value: object) -> bytes:
    return canonical_json_bytes(value) + b"\n"


def _reference(path: str, data: bytes) -> ManifestBoundFile:
    return ManifestBoundFile(
        path=path,
        sha256=hashlib.sha256(data).hexdigest(),
        size=len(data),
    )


def _phase_payloads(control: object) -> dict[str, bytes]:
    from llm_inference_systems.stage2_control import Stage2RuntimeControlEvidence

    typed = cast(Stage2RuntimeControlEvidence, control)
    expected: dict[str, object] = {
        "OFFLINE_SNAPSHOT_VERIFICATION": typed.process_records[:2],
        "RUNTIME_PROCESS_START": typed.process_records[2:],
        "JIT_COMPILATION_STATE": typed.post_warmup_jit_event_hashes,
        "ALLOCATOR_KV_STABILIZATION": {
            "gpu_memory_samples": typed.gpu_memory_samples,
            "gpu_memory_tolerance_bytes": typed.gpu_memory_tolerance_bytes,
        },
        "EXCLUDED_STABILIZATION_REQUESTS": typed.stabilization_request_ids,
        "EXCLUDED_SHAPE_WARMUPS": typed.workload_shape_warmup_request_ids,
        "CANCELLATION_PROBE_DRAIN": {
            "probe": typed.cancellation_probe,
            "result": typed.cancellation_result,
        },
        "STEADY_STATE_GATE": {
            "prefix_cache_hit_delta": typed.prefix_cache_hit_delta,
            "prefix_cache_query_delta": typed.prefix_cache_query_delta,
            "quiet_interval_end_offset_ns": typed.quiet_interval_end_offset_ns,
            "quiet_interval_start_offset_ns": typed.quiet_interval_start_offset_ns,
            "steady_state_snapshots": typed.steady_state_snapshots,
        },
        "MEASURED_WINDOW": {
            "measured_client_slot_assignments": typed.measured_client_slot_assignments,
            "measured_request_ids": typed.measured_request_ids,
            "requested_client_concurrency": typed.requested_client_concurrency,
        },
        "FINAL_METRICS_DRAIN": typed.final_metric_scrape,
        "SHUTDOWN": typed.shutdown_processes,
        "NO_RESIDUAL_PROCESS_VERIFICATION": {
            "residual_active_request_ids": typed.residual_active_request_ids,
            "residual_process_ids": typed.residual_process_ids,
            "residual_verification_evidence_sha256": (typed.residual_verification_evidence_sha256),
            "residual_verification_offset_ns": typed.residual_verification_offset_ns,
        },
    }
    payloads: dict[str, bytes] = {}
    for phase in typed.phases:
        data = (
            canonical_json_bytes(expected[phase.phase.value])
            if phase.phase.value in expected
            else phase.phase.value.encode()
        )
        assert hashlib.sha256(data).hexdigest() == phase.evidence_identity_sha256
        payloads[phase.evidence_references[0]] = data
    return payloads


def _raw_request_payloads(
    *,
    repetition_index: Literal[1, 2, 3],
    case_id: str,
    external_id: str,
    request_evidence: Stage2RequestEvidence,
    request_identity: RequestIdentityAttestation,
    lifecycle: Stage2RequestLifecycle,
) -> tuple[Stage2RequestRawEvidence, dict[str, bytes]]:
    values: dict[str, ManifestBoundFile] = {}
    derived = build_request_raw_evidence_payloads(
        repetition_index=repetition_index,
        case_id=case_id,
        external_request_id=external_id,
        request_evidence=request_evidence,
        request_identity=request_identity,
        lifecycle=lifecycle,
        evidence_scope=Stage2EvidenceScope.TEST_FIXTURE_ONLY,
    )
    payloads: dict[str, bytes] = {}
    for field, data in derived.items():
        path = f"raw/requests/{external_id}/{field}.json"
        payloads[path] = data
        values[field] = _reference(path, data)
    return Stage2RequestRawEvidence.model_validate(values), payloads


def _make_repetition(
    repetition_index: int,
    *,
    environment_resource_identity_sha256: str,
    launch_spec_identity_sha256: str,
    semantic_mismatch_case_id: str | None = None,
    swap_request_raw_references: bool = False,
) -> tuple[Stage2RepetitionAttestation, dict[str, bytes]]:
    control = make_runtime_control(repetition_index=repetition_index, future_shape=False)
    measured_phase = next(
        phase for phase in control.phases if phase.phase.value == "MEASURED_WINDOW"
    )
    payloads = _phase_payloads(control)
    typed_repetition_index = cast(Literal[1, 2, 3], repetition_index)
    request_components: list[
        tuple[
            str,
            str,
            Stage2RequestEvidence,
            RequestIdentityAttestation,
            Stage2RequestLifecycle,
            Stage2RequestRawEvidence,
        ]
    ] = []
    for index, (case_id, external_id) in enumerate(
        zip(STAGE2_EXPERIMENT_CASE_IDS, control.measured_request_ids, strict=True)
    ):
        pair_index = index // 2
        slot_index = index % 2
        base_offset = measured_phase.started_offset_ns + 100_000_000 + pair_index * 1_000
        base_offset += slot_index * 10
        evidence = make_request_evidence(
            fixture_identity_sha256=FIXTURE_IDENTITY,
            external_id=external_id,
            base_offset_ns=base_offset,
            output_token_ids=(
                (*range(31), 999)
                if repetition_index == 3 and case_id == semantic_mismatch_case_id
                else tuple(range(32))
            ),
        )
        chain = make_log_chain(
            external_id,
            first_observation_offset_ns=base_offset + 1,
            fixture_marked=True,
        )
        request_identity = RequestIdentityAttestation(
            identity_chain=chain,
            identity_sha256=sha256_identity(chain),
        )
        lifecycle = Stage2RequestLifecycle(
            dispatch_offset_ns=evidence.timing.dispatch_offset_ns,
            terminal_offset_ns=evidence.timing.transport_terminal_offset_ns,
            measurement_phase_start_ns=measured_phase.started_offset_ns,
            measurement_phase_end_ns=measured_phase.ended_offset_ns,
            measurement_phase_identity_sha256=measured_phase.evidence_identity_sha256,
        )
        raw, raw_payloads = _raw_request_payloads(
            repetition_index=typed_repetition_index,
            case_id=case_id,
            external_id=external_id,
            request_evidence=evidence,
            request_identity=request_identity,
            lifecycle=lifecycle,
        )
        payloads.update(raw_payloads)
        request_components.append(
            (case_id, external_id, evidence, request_identity, lifecycle, raw)
        )
    if swap_request_raw_references and repetition_index == 1:
        first = request_components[0]
        second = request_components[1]
        request_components[0] = (*first[:-1], second[-1])
        request_components[1] = (*second[:-1], first[-1])
    cancellation_result_data = _json_bytes(control.cancellation_result)
    cancellation_result = _reference("cancellation-result.json", cancellation_result_data)
    payloads[cancellation_result.path] = cancellation_result_data
    cancellation_stream_data = build_cancellation_client_stream_raw_evidence_bytes(
        repetition_index=typed_repetition_index,
        external_request_id=control.cancellation_probe.identity_chain.external_base_id,
        evidence_scope=Stage2EvidenceScope.TEST_FIXTURE_ONLY,
    )
    cancellation_stream = _reference(
        "raw/cancellation/client-stream.json", cancellation_stream_data
    )
    payloads[cancellation_stream.path] = cancellation_stream_data
    server_process = f"server-process-{repetition_index}"
    runtime_control_sha = sha256_identity(control)
    execution_shape: dict[str, object] = {
        "torch_version": "2.13.0+cu129",
        "vllm_version": "0.28.0",
        "cuda_available": True,
        "runtime_visible_gpu_count": 1,
        "cuda_device_index": 0,
        "server_process_identity": server_process,
    }
    cuda_raw_data = build_cuda_raw_evidence_bytes(
        repetition_index=typed_repetition_index,
        server_process_identity=server_process,
        runtime_control_sha256=runtime_control_sha,
        environment_resource_identity_sha256=environment_resource_identity_sha256,
        execution=execution_shape,
        evidence_scope=Stage2EvidenceScope.TEST_FIXTURE_ONLY,
        evidence_path="raw/cuda/runtime-evidence.json",
    )
    cuda_raw = _reference("raw/cuda/runtime-evidence.json", cuda_raw_data)
    payloads[cuda_raw.path] = cuda_raw_data
    derived_payloads = reconstruct_experiment_repetition(
        {path: data for path, data in payloads.items() if path.startswith("raw/")}
    )
    payloads.update(derived_payloads)
    manifest = Stage2BundleManifest(
        schema_version="0.3.0",
        measurement_protocol_version="0.3.0",
        state=BundleState.COMMITTED,
        repetition_index=repetition_index,
        source_commit="a" * 40,
        created_at_utc=datetime(2026, 8, 28, tzinfo=UTC),
        files=tuple(
            BundleFileEntry(
                path=path,
                sha256=hashlib.sha256(data).hexdigest(),
                size=len(data),
            )
            for path, data in sorted(payloads.items())
        ),
        reconstruction_sha256=sha256_identity(
            {
                path: hashlib.sha256(data).hexdigest()
                for path, data in sorted(derived_payloads.items())
            }
        ),
    )
    manifest_sha = bundle_manifest_sha256(manifest)
    requests: list[Stage2MeasuredRequestAttestation] = []
    for (
        case_id,
        external_id,
        evidence_value,
        request_identity,
        lifecycle,
        raw,
    ) in request_components:
        evidence = evidence_value
        values: dict[str, object] = {
            "schema_version": "0.3.0",
            "repetition_index": repetition_index,
            "case_id": case_id,
            "external_request_id": external_id,
            "request_evidence": evidence,
            "request_identity": request_identity,
            "lifecycle": lifecycle,
            "raw_evidence": raw,
            "metric_availability": derive_metric_availability(
                evidence.server_per_request_metrics, evidence
            ),
            "repetition_manifest_sha256": manifest_sha,
        }
        values["attestation_sha256"] = sha256_identity(values)
        requests.append(Stage2MeasuredRequestAttestation.model_validate(values))
    execution_values: dict[str, object] = {
        **execution_shape,
        "raw_execution_evidence_sha256": cuda_raw.sha256,
    }
    execution_values["identity_sha256"] = sha256_identity(execution_values)
    execution = CudaBackedExecutionAttestation.model_validate(execution_values)
    cuda_values: dict[str, object] = {
        "schema_version": "0.3.0",
        "repetition_index": repetition_index,
        "server_process_identity": server_process,
        "runtime_control_sha256": runtime_control_sha,
        "repetition_manifest_sha256": manifest_sha,
        "environment_resource_identity_sha256": environment_resource_identity_sha256,
        "execution": execution,
        "raw_evidence_files": (cuda_raw,),
    }
    cuda_values["identity_sha256"] = sha256_identity(cuda_values)
    cuda = Stage2RepetitionCudaAttestation.model_validate(cuda_values)
    restart = ServerRestartIdentity(
        repetition_index=cast(Literal[1, 2, 3], repetition_index),
        server_process_identity=server_process,
        worker_process_identities=(f"worker-process-{repetition_index}",),
        launch_spec_identity_sha256=launch_spec_identity_sha256,
        output_bundle_identity_sha256=manifest_sha,
    )
    request_tuple = tuple(requests)
    values = {
        "schema_version": "0.3.0",
        "repetition_index": repetition_index,
        "runtime_control": control,
        "runtime_control_sha256": runtime_control_sha,
        "server_restart": restart,
        "repetition_manifest": manifest,
        "repetition_manifest_sha256": manifest_sha,
        "cancellation_result_file": cancellation_result,
        "cancellation_client_stream_file": cancellation_stream,
        "cuda_execution": cuda,
        "measured_requests": request_tuple,
        "requested_client_concurrency": 2,
        "observed_max_active_concurrency": 2,
        "positive_duration_overlap_observed": True,
        "metric_availability_summary": derive_metric_availability_summary(request_tuple),
    }
    values["identity_sha256"] = sha256_identity(values)
    return Stage2RepetitionAttestation.model_validate(values), payloads


def make_experiment_attestation(
    *,
    semantic_mismatch_case_id: str | None = None,
    swap_request_raw_references: bool = False,
) -> tuple[
    Stage2ExperimentAttestation,
    tuple[dict[str, bytes], ...],
]:
    prior_shape = make_real_runtime_attestation()
    evidence_scope = Stage2EvidenceScope.TEST_FIXTURE_ONLY
    environment_raw = environment_raw_evidence_bytes(prior_shape.linux_environment, evidence_scope)
    linux_values = prior_shape.linux_environment.model_dump(mode="python")
    linux_values["environment_evidence_sha256"] = hashlib.sha256(environment_raw).hexdigest()
    linux_values["identity_sha256"] = sha256_identity(
        linux_values, omit_fields=frozenset({"identity_sha256"})
    )
    linux = LinuxEnvironmentManifest.model_validate(linux_values)
    nvidia_raw = nvidia_raw_evidence_bytes(prior_shape.nvidia_resources, evidence_scope)
    nvidia_values = prior_shape.nvidia_resources.model_dump(mode="python")
    nvidia_values["isolation_evidence_sha256"] = hashlib.sha256(nvidia_raw).hexdigest()
    nvidia_values["identity_sha256"] = sha256_identity(
        nvidia_values, omit_fields=frozenset({"identity_sha256"})
    )
    nvidia = NvidiaT4ResourceAttestation.model_validate(nvidia_values)
    resolver_raw, installed_raw = execution_lock_raw_evidence_bytes(
        prior_shape.execution_lock, evidence_scope
    )
    execution_lock_values = prior_shape.execution_lock.model_dump(mode="python")
    execution_lock_values["resolver_lock_sha256"] = hashlib.sha256(resolver_raw).hexdigest()
    execution_lock_values["installed_distribution_inventory_sha256"] = hashlib.sha256(
        installed_raw
    ).hexdigest()
    execution_lock_values["identity_sha256"] = sha256_identity(
        execution_lock_values, omit_fields=frozenset({"identity_sha256"})
    )
    execution_lock = type(prior_shape.execution_lock).model_validate(execution_lock_values)
    snapshot_raw = snapshot_read_only_raw_evidence_bytes(
        prior_shape.snapshot_manifest, evidence_scope
    )
    snapshot_values = prior_shape.snapshot_manifest.model_dump(mode="python")
    snapshot_values["read_only_transition"]["verification_evidence_sha256"] = hashlib.sha256(
        snapshot_raw
    ).hexdigest()
    snapshot = type(prior_shape.snapshot_manifest).model_validate(snapshot_values)
    environment_resource_identity = sha256_identity({"linux": linux, "nvidia": nvidia})
    built = tuple(
        _make_repetition(
            index,
            environment_resource_identity_sha256=environment_resource_identity,
            launch_spec_identity_sha256=sha256_identity(prior_shape.launch_spec),
            semantic_mismatch_case_id=semantic_mismatch_case_id,
            swap_request_raw_references=swap_request_raw_references,
        )
        for index in (1, 2, 3)
    )
    repetitions = cast(
        tuple[
            Stage2RepetitionAttestation,
            Stage2RepetitionAttestation,
            Stage2RepetitionAttestation,
        ],
        tuple(item[0] for item in built),
    )
    workload_cases = tuple(
        Stage2WorkloadCase(
            case_id=case_id,
            sent_prompt_token_ids=PROMPT,
            sent_prompt_token_ids_sha256=sha256_identity(PROMPT),
        )
        for case_id in STAGE2_EXPERIMENT_CASE_IDS
    )
    workload_values: dict[str, object] = {
        "schema_version": "0.3.0",
        "workload_name": "stage2-fixed-16-case-v1",
        "cases": workload_cases,
    }
    workload_values["identity_sha256"] = sha256_identity(workload_values)
    workload = Stage2ExperimentWorkload.model_validate(workload_values)
    comparisons: list[Stage2CrossRestartComparison] = []
    for case_index, case_id in enumerate(STAGE2_EXPERIMENT_CASE_IDS):
        semantic_records: list[Stage2RestartSemanticAttestation] = []
        for repetition in repetitions:
            request = repetition.measured_requests[case_index]
            evidence = request.request_evidence
            semantic_records.append(
                Stage2RestartSemanticAttestation(
                    repetition_index=repetition.repetition_index,
                    case_id=case_id,
                    measured_request_attestation_sha256=request.attestation_sha256,
                    sent_prompt_token_ids=evidence.sent_prompt_token_ids,
                    returned_prompt_token_ids=evidence.returned_prompt_token_ids,
                    output_token_ids=evidence.final_output_token_ids,
                    finish_reason=evidence.finish_reason,
                    prompt_tokens=evidence.usage.prompt_tokens,
                    completion_tokens=evidence.usage.completion_tokens,
                    total_tokens=evidence.usage.total_tokens,
                    output_text_sha256=evidence.output_text_sha256,
                    repetition_manifest_sha256=repetition.repetition_manifest_sha256,
                )
            )
        records_tuple = cast(
            tuple[
                Stage2RestartSemanticAttestation,
                Stage2RestartSemanticAttestation,
                Stage2RestartSemanticAttestation,
            ],
            tuple(semantic_records),
        )
        comparison_values: dict[str, object] = {
            "schema_version": "0.3.0",
            "case_id": case_id,
            "semantic_records": records_tuple,
            "comparison": compare_three_restarts(
                tuple(record.as_restart_record() for record in records_tuple)
            ),
        }
        comparison_values["identity_sha256"] = sha256_identity(comparison_values)
        comparisons.append(Stage2CrossRestartComparison.model_validate(comparison_values))
    comparison_tuple = tuple(comparisons)
    all_requests = tuple(
        request for repetition in repetitions for request in repetition.measured_requests
    )
    availability = derive_metric_availability_summary(all_requests)
    aggregate_result = derive_aggregate_validation_result(repetitions, comparison_tuple)
    summary_values: dict[str, object] = {
        "repetition_count": 3,
        "measured_request_count": 48,
        "cancellation_probe_count": 3,
        "cuda_attestation_count": 3,
        "semantic_comparison_count": 16,
        "requested_client_concurrency": 2,
        "observed_max_active_concurrency_per_repetition": (2, 2, 2),
        "fixture_or_protocol_shape_only": True,
        "runtime_claim_advancement_allowed": False,
        "performance_claim_advancement_allowed": False,
    }
    summary_values["identity_sha256"] = sha256_identity(summary_values)
    summary = Stage2ExperimentSummary.model_validate(summary_values)
    scan_inventory_sha = sha256_identity(
        tuple(repetition.repetition_manifest.files for repetition in repetitions)
    )
    public_safety_raw = scoped_raw_evidence_bytes(
        evidence_kind="public_safety_scan",
        evidence_scope=evidence_scope,
        content={
            "finding_count": 0,
            "passed": True,
            "scan_inventory_sha256": scan_inventory_sha,
        },
    )
    raw_scan_sha = hashlib.sha256(public_safety_raw).hexdigest()
    public_safety = PublicSafetyAttestation(
        passed=True,
        finding_count=0,
        scan_inventory_sha256=scan_inventory_sha,
        raw_scan_evidence_sha256=raw_scan_sha,
        scan_result_sha256=sha256_identity(
            {
                "finding_count": 0,
                "passed": True,
                "raw_scan_evidence_sha256": raw_scan_sha,
                "scan_inventory_sha256": scan_inventory_sha,
            }
        ),
    )
    values: dict[str, object] = {
        "schema_version": "0.3.0",
        "measurement_protocol_version": "0.3.0",
        "experiment_id": "stage2-synthetic-experiment-v1",
        "evidence_scope": Stage2EvidenceScope.TEST_FIXTURE_ONLY,
        "classification": Stage2ExperimentClassification.SYNTHETIC_PROTOCOL_SHAPE_ONLY,
        "workload": workload,
        "launch_spec": prior_shape.launch_spec,
        "snapshot_manifest": snapshot,
        "execution_lock": execution_lock,
        "linux_environment": linux,
        "nvidia_resources": nvidia,
        "public_safety": public_safety,
        "repetitions": repetitions,
        "comparisons": comparison_tuple,
        "experiment_metric_availability": availability,
        "aggregate_validation_result": aggregate_result,
        "summary": summary,
    }
    values["identity_sha256"] = sha256_identity(values)
    return Stage2ExperimentAttestation.model_validate(values), tuple(item[1] for item in built)


def make_aggregate_manifest(
    attestation: Stage2ExperimentAttestation,
    repetition_payloads: tuple[dict[str, bytes], ...],
) -> tuple[Stage2AggregateExperimentManifest, dict[str, bytes]]:
    payloads: dict[str, bytes] = {}
    resolver_raw, installed_raw = execution_lock_raw_evidence_bytes(
        attestation.execution_lock, attestation.evidence_scope
    )
    reviewed_lock_raw = (
        Path(__file__).resolve().parents[1] / "execution-lock/stage2-execution-lock.json"
    ).read_bytes()
    for repetition, raw_payloads in zip(attestation.repetitions, repetition_payloads, strict=True):
        prefix = f"repetition-{repetition.repetition_index:02d}"
        payloads.update({f"{prefix}/{path}": data for path, data in raw_payloads.items()})
        payloads[f"{prefix}/evidence-manifest.json"] = _json_bytes(repetition.repetition_manifest)
        payloads[f"{prefix}/cancellation-result.json"] = _json_bytes(
            repetition.runtime_control.cancellation_result
        )
        payloads[f"attestations/cuda-repetition-{repetition.repetition_index:02d}.json"] = (
            _json_bytes(repetition.cuda_execution)
        )
    payloads.update(
        {
            "shared/resource-environment-manifest.json": _json_bytes(attestation.linux_environment),
            "shared/raw/resource-environment-evidence.json": environment_raw_evidence_bytes(
                attestation.linux_environment, attestation.evidence_scope
            ),
            "shared/nvidia-isolation-evidence.json": _json_bytes(attestation.nvidia_resources),
            "shared/raw/nvidia-isolation-evidence.json": nvidia_raw_evidence_bytes(
                attestation.nvidia_resources, attestation.evidence_scope
            ),
            "shared/execution-lock-snapshot.json": _json_bytes(attestation.execution_lock),
            "shared/raw/runtime-resolver-lock.json": resolver_raw,
            "shared/raw/installed-distribution-inventory.json": installed_raw,
            "shared/raw/reviewed-stage2-execution-lock.json": reviewed_lock_raw,
            "shared/model-tokenizer-snapshot-manifest.json": _json_bytes(
                attestation.snapshot_manifest
            ),
            "shared/raw/snapshot-read-only-verification.json": (
                snapshot_read_only_raw_evidence_bytes(
                    attestation.snapshot_manifest, attestation.evidence_scope
                )
            ),
            "shared/launch-specification.json": _json_bytes(attestation.launch_spec),
            "shared/public-safety-result.json": _json_bytes(attestation.public_safety),
            "shared/raw/public-safety-scan.json": public_safety_raw_evidence_bytes(
                attestation.public_safety, attestation.evidence_scope
            ),
            "shared/workload-definition.json": _json_bytes(attestation.workload),
            "derived/metric-availability-summary.json": _json_bytes(
                attestation.experiment_metric_availability
            ),
            "derived/experiment-summary.json": _json_bytes(attestation.summary),
            "derived/aggregate-validation-result.json": _json_bytes(
                attestation.aggregate_validation_result
            ),
            "derived/final-attestation.json": _json_bytes(attestation),
        }
    )
    for comparison in attestation.comparisons:
        payloads[f"comparisons/{comparison.case_id}.json"] = _json_bytes(comparison)
    entries = tuple(
        BundleFileEntry(
            path=path,
            sha256=hashlib.sha256(data).hexdigest(),
            size=len(data),
        )
        for path, data in sorted(payloads.items())
    )
    by_path = {entry.path: entry for entry in entries}

    def reference(path: str) -> ManifestBoundFile:
        entry = by_path[path]
        return ManifestBoundFile(path=path, sha256=entry.sha256, size=entry.size)

    values = {
        "schema_version": "0.3.0",
        "measurement_protocol_version": "0.3.0",
        "experiment_id": attestation.experiment_id,
        "state": (
            AggregateRootState.COMMITTED
            if attestation.aggregate_validation_result.state is BundleState.COMMITTED
            else AggregateRootState.INVALID
        ),
        "failure_reason": attestation.aggregate_validation_result.failure_reason,
        "evidence_scope": attestation.evidence_scope,
        "created_at_utc": datetime(2026, 8, 28, tzinfo=UTC),
        "files": entries,
        "resource_environment_manifest": reference("shared/resource-environment-manifest.json"),
        "resource_environment_raw_evidence": reference(
            "shared/raw/resource-environment-evidence.json"
        ),
        "nvidia_isolation_evidence": reference("shared/nvidia-isolation-evidence.json"),
        "nvidia_isolation_raw_evidence": reference("shared/raw/nvidia-isolation-evidence.json"),
        "execution_lock_snapshot": reference("shared/execution-lock-snapshot.json"),
        "runtime_resolver_lock_evidence": reference("shared/raw/runtime-resolver-lock.json"),
        "runtime_installed_distribution_inventory": reference(
            "shared/raw/installed-distribution-inventory.json"
        ),
        "reviewed_execution_lock": reference("shared/raw/reviewed-stage2-execution-lock.json"),
        "model_tokenizer_snapshot_manifest": reference(
            "shared/model-tokenizer-snapshot-manifest.json"
        ),
        "snapshot_read_only_verification_evidence": reference(
            "shared/raw/snapshot-read-only-verification.json"
        ),
        "launch_specification": reference("shared/launch-specification.json"),
        "public_safety_result": reference("shared/public-safety-result.json"),
        "public_safety_raw_scan_evidence": reference("shared/raw/public-safety-scan.json"),
        "shared_workload_definition": reference("shared/workload-definition.json"),
        "repetition_manifest_files": tuple(
            reference(f"repetition-{index:02d}/evidence-manifest.json") for index in (1, 2, 3)
        ),
        "cuda_execution_attestation_files": tuple(
            reference(f"attestations/cuda-repetition-{index:02d}.json") for index in (1, 2, 3)
        ),
        "cancellation_result_files": tuple(
            reference(f"repetition-{index:02d}/cancellation-result.json") for index in (1, 2, 3)
        ),
        "semantic_comparison_files": tuple(
            reference(f"comparisons/{case_id}.json") for case_id in STAGE2_EXPERIMENT_CASE_IDS
        ),
        "metric_availability_summary": reference("derived/metric-availability-summary.json"),
        "experiment_summary": reference("derived/experiment-summary.json"),
        "aggregate_validation_result": reference("derived/aggregate-validation-result.json"),
        "final_attestation": reference("derived/final-attestation.json"),
    }
    return Stage2AggregateExperimentManifest.model_validate(values), payloads


def write_synthetic_experiment_directory(
    root: Path,
    *,
    semantic_mismatch_case_id: str | None = None,
    swap_request_raw_references: bool = False,
) -> Stage2AggregateExperimentManifest:
    attestation, repetition_payloads = make_experiment_attestation(
        semantic_mismatch_case_id=semantic_mismatch_case_id,
        swap_request_raw_references=swap_request_raw_references,
    )
    manifest, payloads = make_aggregate_manifest(attestation, repetition_payloads)
    root.mkdir()
    timestamp = 1_700_000_000_000_000_000
    repetition_manifest_paths = {
        f"repetition-{index:02d}/evidence-manifest.json" for index in (1, 2, 3)
    }
    for path, data in sorted(payloads.items()):
        if path in repetition_manifest_paths:
            continue
        destination = root / path
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(data)
        os.utime(destination, ns=(timestamp, timestamp))
    for path in sorted(repetition_manifest_paths):
        destination = root / path
        destination.write_bytes(payloads[path])
        os.utime(destination, ns=(timestamp + 1_000, timestamp + 1_000))
    write_aggregate_manifest_last(root, manifest)
    return manifest
