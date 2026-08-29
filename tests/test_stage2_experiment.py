from __future__ import annotations

import hashlib
import json
import os
from functools import cache
from pathlib import Path

import pytest
from pydantic import ValidationError

from llm_inference_systems.canonical import canonical_json_bytes, sha256_identity
from llm_inference_systems.stage2_contracts import BundleFileEntry, Stage2EvidenceScope
from llm_inference_systems.stage2_control import AggregateComparisonState
from llm_inference_systems.stage2_experiment import (
    ManifestBoundFile,
    Stage2AggregateExperimentManifest,
    Stage2CrossRestartComparison,
    Stage2ExperimentAttestation,
    Stage2ExperimentClassification,
    Stage2ExperimentError,
    Stage2MeasuredRequestAttestation,
    Stage2MetricAvailability,
    Stage2RepetitionAttestation,
    Stage2RepetitionCudaAttestation,
    Stage2RequestRawEvidence,
    _contains_fixture_value,
    derive_aggregate_validation_result,
    reconstruct_experiment_attestation,
    reconstruct_request_from_raw_evidence,
    request_raw_evidence_payloads,
    semantic_nonreproduction_result,
    write_aggregate_manifest_last,
)
from tests.stage2_experiment_factories import (
    make_aggregate_manifest,
    make_experiment_attestation,
    write_synthetic_experiment_directory,
)


@cache
def _attestation() -> Stage2ExperimentAttestation:
    return make_experiment_attestation()[0]


def test_complete_16_by_3_shape_is_accepted_only_as_synthetic_fixture() -> None:
    attestation = _attestation()
    assert len(attestation.repetitions) == 3
    assert sum(len(item.measured_requests) for item in attestation.repetitions) == 48
    assert len(attestation.comparisons) == 16
    assert attestation.summary.fixture_or_protocol_shape_only is True
    assert attestation.summary.runtime_claim_advancement_allowed is False
    assert attestation.summary.performance_claim_advancement_allowed is False


@pytest.mark.parametrize("count", [1, 15, 17])
def test_incomplete_or_extra_measured_request_cardinality_is_rejected(count: int) -> None:
    repetition = _attestation().repetitions[0]
    values = repetition.model_dump(mode="python")
    source = list(repetition.measured_requests)
    if count == 17:
        source.append(source[-1])
    values["measured_requests"] = tuple(source[:count])
    with pytest.raises(ValidationError):
        Stage2RepetitionAttestation.model_validate(values)


def test_parsed_request_absent_from_control_and_set_mismatch_are_rejected() -> None:
    repetition = _attestation().repetitions[0]
    values = repetition.model_dump(mode="python")
    values["runtime_control"]["measured_request_ids"] = (
        "undeclared-request",
        *repetition.runtime_control.measured_request_ids[1:],
    )
    values["runtime_control_sha256"] = sha256_identity(values["runtime_control"])
    with pytest.raises(ValidationError):
        Stage2RepetitionAttestation.model_validate(values)


def test_request_id_overlap_with_excluded_group_is_rejected() -> None:
    repetition = _attestation().repetitions[0]
    values = repetition.model_dump(mode="python")
    values["runtime_control"]["stabilization_request_ids"] = (
        repetition.measured_requests[0].external_request_id,
        *repetition.runtime_control.stabilization_request_ids[1:],
    )
    with pytest.raises(ValidationError):
        Stage2RepetitionAttestation.model_validate(values)


def test_external_request_id_reused_across_repetitions_is_rejected() -> None:
    attestation = _attestation()
    second = attestation.repetitions[1]
    duplicate = second.measured_requests[0].model_copy(
        update={
            "external_request_id": (
                attestation.repetitions[0].measured_requests[0].external_request_id
            )
        }
    )
    second_copy = second.model_copy(
        update={"measured_requests": (duplicate, *second.measured_requests[1:])}
    )
    values = attestation.model_dump(mode="python")
    values["repetitions"] = (
        attestation.repetitions[0],
        second_copy,
        attestation.repetitions[2],
    )
    with pytest.raises(ValidationError):
        Stage2ExperimentAttestation.model_validate(values)


def test_slot_labels_without_lifecycle_overlap_are_rejected() -> None:
    repetition = _attestation().repetitions[0]
    requests = []
    phase = repetition.measured_requests[0].lifecycle
    for index, request in enumerate(repetition.measured_requests):
        dispatch = phase.measurement_phase_start_ns + 100 + index * 100
        lifecycle = request.lifecycle.model_copy(
            update={"dispatch_offset_ns": dispatch, "terminal_offset_ns": dispatch + 50}
        )
        requests.append(request.model_copy(update={"lifecycle": lifecycle}))
    values = repetition.model_dump(mode="python")
    values["measured_requests"] = tuple(requests)
    with pytest.raises(ValidationError):
        Stage2RepetitionAttestation.model_validate(values)


def test_observed_concurrency_above_two_is_rejected() -> None:
    repetition = _attestation().repetitions[0]
    first = repetition.measured_requests[0].lifecycle
    requests = []
    for index, request in enumerate(repetition.measured_requests):
        if index < 3:
            lifecycle = request.lifecycle.model_copy(
                update={
                    "dispatch_offset_ns": first.dispatch_offset_ns + index,
                    "terminal_offset_ns": first.terminal_offset_ns,
                }
            )
            request = request.model_copy(update={"lifecycle": lifecycle})
        requests.append(request)
    values = repetition.model_dump(mode="python")
    values["measured_requests"] = tuple(requests)
    with pytest.raises(ValidationError):
        Stage2RepetitionAttestation.model_validate(values)


def test_lifecycle_outside_measured_phase_is_rejected() -> None:
    repetition = _attestation().repetitions[0]
    request = repetition.measured_requests[0]
    lifecycle = request.lifecycle.model_copy(
        update={"measurement_phase_start_ns": request.lifecycle.dispatch_offset_ns + 1}
    )
    changed = request.model_copy(update={"lifecycle": lifecycle})
    values = repetition.model_dump(mode="python")
    values["measured_requests"] = (changed, *repetition.measured_requests[1:])
    with pytest.raises(ValidationError):
        Stage2RepetitionAttestation.model_validate(values)


def test_missing_request_raw_file_field_is_rejected() -> None:
    raw = _attestation().repetitions[0].measured_requests[0].raw_evidence.model_dump(mode="python")
    raw.pop("request_headers")
    with pytest.raises(ValidationError):
        Stage2RequestRawEvidence.model_validate(raw)


def test_request_hash_or_cancellation_stream_not_in_manifest_is_rejected() -> None:
    repetition = _attestation().repetitions[0]
    request = repetition.measured_requests[0]
    raw_values = request.raw_evidence.model_dump(mode="python")
    raw_values["request_body"]["sha256"] = "0" * 64
    request_values = request.model_dump(mode="python")
    request_values["raw_evidence"] = Stage2RequestRawEvidence.model_validate(raw_values)
    request_values["attestation_sha256"] = sha256_identity(
        request_values, omit_fields=frozenset({"attestation_sha256"})
    )
    changed_request = Stage2MeasuredRequestAttestation.model_validate(request_values)
    values = repetition.model_dump(mode="python")
    values["measured_requests"] = (changed_request, *repetition.measured_requests[1:])
    with pytest.raises(ValidationError, match="absent from the repetition manifest"):
        Stage2RepetitionAttestation.model_validate(values)

    values = repetition.model_dump(mode="python")
    values["cancellation_client_stream_file"]["sha256"] = "0" * 64
    with pytest.raises(ValidationError, match="absent from the repetition manifest"):
        Stage2RepetitionAttestation.model_validate(values)


def test_missing_or_caller_overridden_metric_availability_is_rejected() -> None:
    request = _attestation().repetitions[0].measured_requests[0]
    values = request.model_dump(mode="python")
    values.pop("metric_availability")
    with pytest.raises(ValidationError):
        Stage2MeasuredRequestAttestation.model_validate(values)
    availability = request.metric_availability.model_dump(mode="python")
    availability["server_ttft_advancement_allowed"] = False
    with pytest.raises(ValidationError, match="derived from availability"):
        Stage2MetricAvailability.model_validate(availability)


def test_all_null_server_metrics_cannot_leave_advancement_enabled() -> None:
    values = {
        "server_ttft_available": False,
        "server_generation_time_available": False,
        "server_queue_time_available": False,
        "server_mean_itl_available": False,
        "server_tokens_per_second_available": False,
        "client_generation_tpot_available": True,
        "token_observation_itl_available": True,
        "server_ttft_advancement_allowed": True,
        "server_generation_time_advancement_allowed": True,
        "server_queue_time_advancement_allowed": True,
        "server_mean_itl_advancement_allowed": True,
        "server_tokens_per_second_advancement_allowed": True,
        "client_generation_tpot_advancement_allowed": True,
        "token_observation_itl_advancement_allowed": True,
    }
    with pytest.raises(ValidationError, match="derived from availability"):
        Stage2MetricAvailability.model_validate(values)


def test_missing_comparison_or_wrong_repetition_count_is_rejected() -> None:
    attestation = _attestation()
    values = attestation.model_dump(mode="python")
    values["comparisons"] = attestation.comparisons[:-1]
    with pytest.raises(ValidationError):
        Stage2ExperimentAttestation.model_validate(values)
    comparison = attestation.comparisons[0].model_dump(mode="python")
    comparison["semantic_records"] = comparison["semantic_records"][:-1]
    with pytest.raises(ValidationError):
        Stage2CrossRestartComparison.model_validate(comparison)


def test_output_token_mismatch_forces_semantic_nonreproduction_and_blocks_commit() -> None:
    attestation = _attestation()
    comparison = attestation.comparisons[0]
    records = list(comparison.semantic_records)
    records[-1] = records[-1].model_copy(update={"output_token_ids": (*range(31), 999)})
    values: dict[str, object] = {
        "schema_version": "0.3.0",
        "case_id": comparison.case_id,
        "semantic_records": tuple(records),
        "comparison": comparison.comparison.model_copy(
            update={
                "state": AggregateComparisonState.INVALID_SEMANTIC_NONREPRODUCTION,
                "mismatches": ("restart-3:output_token_ids",),
            }
        ),
    }
    values["identity_sha256"] = sha256_identity(values)
    invalid = Stage2CrossRestartComparison.model_validate(values)
    assert (
        semantic_nonreproduction_result(invalid)
        is AggregateComparisonState.INVALID_SEMANTIC_NONREPRODUCTION
    )
    result = derive_aggregate_validation_result(
        attestation.repetitions,
        (invalid, *attestation.comparisons[1:]),
    )
    assert result.state is AggregateComparisonState.INVALID_SEMANTIC_NONREPRODUCTION
    assert result.failure_reason == "INVALID_SEMANTIC_NONREPRODUCTION"
    assert result.invalid_case_ids == (invalid.case_id,)
    final_values = attestation.model_dump(mode="python")
    final_values["comparisons"] = (invalid, *attestation.comparisons[1:])
    with pytest.raises(ValidationError):
        Stage2ExperimentAttestation.model_validate(final_values)


def test_semantic_nonreproduction_is_retained_in_a_durable_invalid_root(
    tmp_path: Path,
) -> None:
    root = tmp_path / "invalid-aggregate"
    manifest = write_synthetic_experiment_directory(
        root,
        semantic_mismatch_case_id="stage2-case-v1-01",
    )
    reconstructed = reconstruct_experiment_attestation(root)
    assert manifest.state.value == "INVALID"
    assert manifest.failure_reason == "INVALID_SEMANTIC_NONREPRODUCTION"
    assert (
        reconstructed.attestation.aggregate_validation_result.state
        is AggregateComparisonState.INVALID_SEMANTIC_NONREPRODUCTION
    )
    assert reconstructed.attestation.aggregate_validation_result.invalid_case_ids == (
        "stage2-case-v1-01",
    )


def test_request_identity_suffix_must_match_the_retained_log_chain() -> None:
    request = _attestation().repetitions[0].measured_requests[0]
    values = request.model_dump(mode="python")
    values["request_evidence"]["internal_engine_request_id"] = (
        f"cmpl-{request.external_request_id}-0-cafebabe"
    )
    values["attestation_sha256"] = sha256_identity(
        values, omit_fields=frozenset({"attestation_sha256"})
    )
    with pytest.raises(ValidationError, match="identity chain does not reconcile"):
        Stage2MeasuredRequestAttestation.model_validate(values)


@pytest.mark.parametrize(
    ("field", "key", "contradiction"),
    [
        ("request_headers", "x_request_id", "contradictory-request"),
        ("terminal_boundary", "finish_reason", "stop"),
        ("token_usage_reconciliation", "sent_prompt_token_ids", list(reversed(range(64)))),
    ],
)
def test_request_raw_reconstruction_rejects_contradictory_retained_fields(
    field: str,
    key: str,
    contradiction: object,
) -> None:
    request = _attestation().repetitions[0].measured_requests[0]
    raw = request_raw_evidence_payloads(request, Stage2EvidenceScope.TEST_FIXTURE_ONLY)
    changed = json.loads(raw[field])
    changed["content"][key] = contradiction
    raw[field] = canonical_json_bytes(changed) + b"\n"
    with pytest.raises(Stage2ExperimentError, match="contradict"):
        reconstruct_request_from_raw_evidence(raw, Stage2EvidenceScope.TEST_FIXTURE_ONLY)


def test_repetition_server_process_must_match_runtime_control() -> None:
    repetition = _attestation().repetitions[0]
    values = repetition.model_dump(mode="python")
    wrong_process = "wrong-runtime-server"
    values["server_restart"]["server_process_identity"] = wrong_process
    execution = values["cuda_execution"]["execution"]
    execution["server_process_identity"] = wrong_process
    execution["identity_sha256"] = sha256_identity(
        execution, omit_fields=frozenset({"identity_sha256"})
    )
    values["cuda_execution"]["server_process_identity"] = wrong_process
    values["cuda_execution"]["execution"] = execution
    values["cuda_execution"]["identity_sha256"] = sha256_identity(
        values["cuda_execution"], omit_fields=frozenset({"identity_sha256"})
    )
    values["identity_sha256"] = sha256_identity(values, omit_fields=frozenset({"identity_sha256"}))
    with pytest.raises(ValidationError, match="runtime-control server process"):
        Stage2RepetitionAttestation.model_validate(values)


def test_aggregate_validation_boolean_bypass_is_structurally_rejected() -> None:
    values = _attestation().model_dump(mode="python")
    values["aggregate_validation_passed"] = True
    with pytest.raises(ValidationError):
        Stage2ExperimentAttestation.model_validate(values)


def test_missing_or_wrong_cuda_attestation_is_rejected() -> None:
    attestation = _attestation()
    values = attestation.model_dump(mode="python")
    values["repetitions"] = values["repetitions"][:2]
    with pytest.raises(ValidationError):
        Stage2ExperimentAttestation.model_validate(values)
    cuda = attestation.repetitions[0].cuda_execution.model_dump(mode="python")
    cuda["server_process_identity"] = "wrong-process"
    with pytest.raises(ValidationError, match="repetition server process"):
        Stage2RepetitionCudaAttestation.model_validate(cuda)


@pytest.mark.parametrize(
    "field",
    [
        "resource_environment_manifest",
        "nvidia_isolation_evidence",
        "execution_lock_snapshot",
        "runtime_resolver_lock_evidence",
        "runtime_installed_distribution_inventory",
        "reviewed_execution_lock",
        "model_tokenizer_snapshot_manifest",
        "snapshot_read_only_verification_evidence",
        "launch_specification",
        "public_safety_result",
        "repetition_manifest_files",
        "semantic_comparison_files",
        "experiment_summary",
        "final_attestation",
    ],
)
def test_aggregate_root_omission_is_rejected(field: str) -> None:
    attestation, repetition_payloads = make_experiment_attestation()
    manifest, _ = make_aggregate_manifest(attestation, repetition_payloads)
    values = manifest.model_dump(mode="python")
    if isinstance(values[field], tuple):
        values[field] = values[field][:-1]
    else:
        values.pop(field)
    with pytest.raises(ValidationError):
        Stage2AggregateExperimentManifest.model_validate(values)


def test_fixture_execution_cannot_request_live_runtime_classification() -> None:
    values = _attestation().model_dump(mode="python")
    values["evidence_scope"] = Stage2EvidenceScope.FUTURE_REAL_RUNTIME
    values["classification"] = Stage2ExperimentClassification.FUTURE_REAL_RUNTIME
    values["summary"]["fixture_or_protocol_shape_only"] = False
    values["summary"]["identity_sha256"] = sha256_identity(
        values["summary"], omit_fields=frozenset({"identity_sha256"})
    )
    with pytest.raises(ValidationError, match="fixture evidence"):
        Stage2ExperimentAttestation.model_validate(values)


def test_swapped_request_raw_references_are_rejected_during_publication(
    tmp_path: Path,
) -> None:
    with pytest.raises(Stage2ExperimentError, match="does not reconstruct from raw"):
        write_synthetic_experiment_directory(
            tmp_path / "swapped-request-raw",
            swap_request_raw_references=True,
        )


def _rebind_manifest_to_payloads(
    manifest: Stage2AggregateExperimentManifest,
    payloads: dict[str, bytes],
) -> Stage2AggregateExperimentManifest:
    entries = tuple(
        BundleFileEntry(
            path=path,
            sha256=hashlib.sha256(data).hexdigest(),
            size=len(data),
        )
        for path, data in sorted(payloads.items())
    )
    entries_by_path = {entry.path: entry for entry in entries}

    def reference(path: str) -> ManifestBoundFile:
        entry = entries_by_path[path]
        return ManifestBoundFile(path=path, sha256=entry.sha256, size=entry.size)

    values = manifest.model_dump(mode="python")
    values["files"] = entries
    for field_name, value in tuple(values.items()):
        if isinstance(value, dict) and "path" in value:
            values[field_name] = reference(value["path"])
        elif isinstance(value, tuple) and value and isinstance(value[0], dict):
            values[field_name] = tuple(reference(item["path"]) for item in value)
    return Stage2AggregateExperimentManifest.model_validate(values)


def _write_aggregate_payloads(root: Path, payloads: dict[str, bytes]) -> None:
    root.mkdir()
    timestamp = 1_700_000_000_000_000_000
    repetition_manifests = {f"repetition-{index:02d}/evidence-manifest.json" for index in (1, 2, 3)}
    for relative, data in sorted(payloads.items()):
        if relative in repetition_manifests:
            continue
        destination = root / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(data)
        os.utime(destination, ns=(timestamp, timestamp))
    for relative in sorted(repetition_manifests):
        destination = root / relative
        destination.write_bytes(payloads[relative])
        os.utime(destination, ns=(timestamp + 1_000, timestamp + 1_000))


def test_manifest_writer_rejects_semantically_invalid_committed_payload(
    tmp_path: Path,
) -> None:
    attestation, repetition_payloads = make_experiment_attestation()
    manifest, payloads = make_aggregate_manifest(attestation, repetition_payloads)
    payloads[manifest.final_attestation.path] = canonical_json_bytes({}) + b"\n"
    rebound = _rebind_manifest_to_payloads(manifest, payloads)
    root = tmp_path / "invalid-publication"
    _write_aggregate_payloads(root, payloads)
    with pytest.raises(Stage2ExperimentError, match="final experiment attestation is invalid"):
        write_aggregate_manifest_last(root, rebound)
    assert not (root / "aggregate-experiment-manifest.json").exists()


def test_manifest_writer_uses_complete_public_safety_patterns(tmp_path: Path) -> None:
    attestation, repetition_payloads = make_experiment_attestation()
    manifest, payloads = make_aggregate_manifest(attestation, repetition_payloads)
    credential_name = "HUGGINGFACEHUB_" + "API_TOKEN"
    payloads["extra-sensitive.txt"] = f"{credential_name}=fixture-secret-value\n".encode()
    rebound = _rebind_manifest_to_payloads(manifest, payloads)
    root = tmp_path / "sensitive-publication"
    _write_aggregate_payloads(root, payloads)
    with pytest.raises(Stage2ExperimentError, match="prohibited private material"):
        write_aggregate_manifest_last(root, rebound)


def test_known_synthetic_future_marker_cannot_receive_live_scope(tmp_path: Path) -> None:
    attestation, repetition_payloads = make_experiment_attestation()
    manifest, payloads = make_aggregate_manifest(attestation, repetition_payloads)
    payloads["extra-synthetic.txt"] = b"SYNTHETIC_FUTURE_SHAPE_ONLY\n"
    values = _rebind_manifest_to_payloads(manifest, payloads).model_dump(mode="python")
    values["evidence_scope"] = Stage2EvidenceScope.FUTURE_REAL_RUNTIME
    future_manifest = Stage2AggregateExperimentManifest.model_validate(values)
    root = tmp_path / "synthetic-live-publication"
    _write_aggregate_payloads(root, payloads)
    with pytest.raises(Stage2ExperimentError, match="fixture-marked raw evidence"):
        write_aggregate_manifest_last(root, future_manifest)


def test_live_marker_check_ignores_field_names_and_rejects_marker_values() -> None:
    clean_live_shape = {
        "evidence_scope": "FUTURE_REAL_RUNTIME",
        "fixture_identity_sha256": None,
        "fixture_or_protocol_shape_only": False,
        "source_stream_id": "runtime-server-log",
    }
    assert _contains_fixture_value(clean_live_shape) is False
    assert _contains_fixture_value({**clean_live_shape, "source_stream_id": "synthetic-shape-log"})


def test_complete_aggregate_reconstructs_only_from_retained_files(tmp_path: Path) -> None:
    root = tmp_path / "aggregate"
    manifest = write_synthetic_experiment_directory(root)
    reconstructed = reconstruct_experiment_attestation(root)
    assert reconstructed.aggregate_manifest == manifest
    assert reconstructed.attestation.summary.measured_request_count == 48
    assert reconstructed.attestation.summary.fixture_or_protocol_shape_only is True


def test_reconstruction_rejects_missing_request_file_symlink_and_tamper(tmp_path: Path) -> None:
    root = tmp_path / "missing"
    write_synthetic_experiment_directory(root)
    request_path = next((root / "repetition-01/raw/requests").rglob("request_body.json"))
    request_path.unlink()
    with pytest.raises(Stage2ExperimentError, match="inventory"):
        reconstruct_experiment_attestation(root)

    root = tmp_path / "tamper"
    write_synthetic_experiment_directory(root)
    request_path = next((root / "repetition-01/raw/requests").rglob("request_body.json"))
    request_path.write_text("tampered")
    with pytest.raises(Stage2ExperimentError, match="size or SHA-256"):
        reconstruct_experiment_attestation(root)

    root = tmp_path / "symlink"
    write_synthetic_experiment_directory(root)
    request_path = next((root / "repetition-01/raw/requests").rglob("request_body.json"))
    target = root / "target.json"
    target.write_text("safe")
    request_path.unlink()
    request_path.symlink_to(target)
    with pytest.raises(Stage2ExperimentError, match="symlinks"):
        reconstruct_experiment_attestation(root)


def test_reconstruction_rejects_aggregate_manifest_not_strictly_last(tmp_path: Path) -> None:
    root = tmp_path / "aggregate"
    write_synthetic_experiment_directory(root)
    manifest_path = root / "aggregate-experiment-manifest.json"
    later_file = root / "derived/experiment-summary.json"
    later = manifest_path.stat().st_mtime_ns + 1_000_000
    os.utime(later_file, ns=(later, later))
    with pytest.raises(Stage2ExperimentError, match="strictly last"):
        reconstruct_experiment_attestation(root)


def test_environment_hardware_or_safety_raw_hash_without_root_file_is_rejected(
    tmp_path: Path,
) -> None:
    root = tmp_path / "aggregate"
    write_synthetic_experiment_directory(root)
    manifest_path = root / "aggregate-experiment-manifest.json"
    values = Stage2AggregateExperimentManifest.model_validate_json(
        manifest_path.read_bytes()
    ).model_dump(mode="python")
    values["resource_environment_raw_evidence"] = ManifestBoundFile(
        path="shared/raw/resource-environment-evidence.json",
        sha256="0" * 64,
        size=values["resource_environment_raw_evidence"]["size"],
    )
    with pytest.raises(ValidationError, match="not bound"):
        Stage2AggregateExperimentManifest.model_validate(values)
