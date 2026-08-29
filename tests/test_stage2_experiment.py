from __future__ import annotations

import base64
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
    build_request_raw_evidence_payloads,
    derive_aggregate_validation_result,
    reconstruct_experiment_attestation,
    reconstruct_experiment_repetition,
    reconstruct_request_from_raw_evidence,
    replay_stage2_wire_capture,
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


def test_request_hash_or_cancellation_wire_not_in_manifest_is_rejected() -> None:
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
    values["cancellation_wire_file"]["sha256"] = "0" * 64
    with pytest.raises(ValidationError, match="absent from the repetition manifest"):
        Stage2RepetitionAttestation.model_validate(values)


def test_repetition_structurally_requires_one_prometheus_measurement_attestation() -> None:
    values = _attestation().repetitions[0].model_dump(mode="python")
    values.pop("prometheus_measurement")
    with pytest.raises(ValidationError):
        Stage2RepetitionAttestation.model_validate(values)


@pytest.mark.parametrize(
    "field",
    ["baseline_raw_exposition_file", "final_raw_exposition_file"],
)
def test_prometheus_raw_exposition_references_must_bind_to_repetition_manifest(
    field: str,
) -> None:
    repetition = _attestation().repetitions[0]
    measurement_values = repetition.prometheus_measurement.model_dump(mode="python")
    measurement_values[field]["sha256"] = "0" * 64
    measurement_values["evidence_sha256"] = sha256_identity(
        measurement_values, omit_fields=frozenset({"evidence_sha256"})
    )
    with pytest.raises(ValidationError, match="raw scrape file identity"):
        type(repetition.prometheus_measurement).model_validate(measurement_values)


def test_prometheus_parsed_snapshots_cannot_exist_without_raw_exposition_references() -> None:
    measurement = _attestation().repetitions[0].prometheus_measurement
    values = measurement.model_dump(mode="python")
    values.pop("baseline_raw_exposition_file")
    with pytest.raises(ValidationError):
        type(measurement).model_validate(values)


@pytest.mark.parametrize(
    ("mutation", "value"),
    [
        ("server_process_identity", "other-process"),
        ("served_model_label", "other-model"),
        ("engine_label", "1"),
        ("baseline_at_dispatch", None),
        ("final_before_drain", None),
        ("wrong_delta", None),
    ],
)
def test_prometheus_measurement_rejects_process_phase_label_and_delta_drift(
    mutation: str,
    value: object,
) -> None:
    measurement = _attestation().repetitions[0].prometheus_measurement
    values = measurement.model_dump(mode="python")
    if mutation in {"server_process_identity", "served_model_label", "engine_label"}:
        values[mutation] = value
    elif mutation == "baseline_at_dispatch":
        values["first_measured_request_dispatch_offset_ns"] = values["baseline_snapshot"][
            "scrape_monotonic_offset_ns"
        ]
    elif mutation == "final_before_drain":
        values["final_drain_boundary_offset_ns"] = (
            values["final_snapshot"]["scrape_monotonic_offset_ns"] + 1
        )
    else:
        values["counter_deltas"][0]["after"] += 1
        values["counter_deltas"][0]["delta"] += 1
    values["evidence_sha256"] = sha256_identity(values, omit_fields=frozenset({"evidence_sha256"}))
    with pytest.raises(ValidationError):
        type(measurement).model_validate(values)


def test_prometheus_measurement_rejects_stale_scrape_gate_distance() -> None:
    measurement = _attestation().repetitions[0].prometheus_measurement
    values = measurement.model_dump(mode="python")
    values["final_drain_boundary_offset_ns"] = (
        values["final_snapshot"]["scrape_monotonic_offset_ns"] - 1_000_000_001
    )
    values["evidence_sha256"] = sha256_identity(values, omit_fields=frozenset({"evidence_sha256"}))
    with pytest.raises(ValidationError, match="stale"):
        type(measurement).model_validate(values)


def test_prometheus_measurement_rejects_stale_baseline_scrape() -> None:
    measurement = _attestation().repetitions[0].prometheus_measurement
    values = measurement.model_dump(mode="python")
    stale_offset = values["first_measured_request_dispatch_offset_ns"] - 1_000_000_001
    values["measured_phase_start_offset_ns"] = stale_offset
    original_offset = values["baseline_snapshot"]["scrape_monotonic_offset_ns"]
    transport_shift = stale_offset - original_offset
    values["baseline_snapshot"]["scrape_monotonic_offset_ns"] = stale_offset
    baseline = type(measurement.baseline_snapshot).model_validate(values["baseline_snapshot"])
    values["baseline_parsed_snapshot_file"]["sha256"] = hashlib.sha256(
        canonical_json_bytes(baseline) + b"\n"
    ).hexdigest()
    capture = values["baseline_capture"]
    capture["scrape_monotonic_offset_ns"] = stale_offset
    exchange = capture["http_exchange"]
    exchange["request_headers"]["observation_offset_ns"] += transport_shift
    exchange["response_headers"]["observation_offset_ns"] += transport_shift
    exchange["request_body_transmission_observation_offset_ns"] += transport_shift
    exchange["response_header_observation_offset_ns"] += transport_shift
    exchange["response_body_completion_observation_offset_ns"] += transport_shift
    exchange["transport_terminal_observation_offset_ns"] += transport_shift
    exchange["request_headers"]["identity_sha256"] = sha256_identity(
        exchange["request_headers"], omit_fields=frozenset({"identity_sha256"})
    )
    exchange["response_headers"]["identity_sha256"] = sha256_identity(
        exchange["response_headers"], omit_fields=frozenset({"identity_sha256"})
    )
    exchange["identity_sha256"] = sha256_identity(
        exchange, omit_fields=frozenset({"identity_sha256"})
    )
    capture["identity_sha256"] = sha256_identity(
        capture, omit_fields=frozenset({"identity_sha256"})
    )
    baseline_capture = type(measurement.baseline_capture).model_validate(capture)
    raw_capture_bytes = canonical_json_bytes(baseline_capture) + b"\n"
    values["baseline_raw_exposition_file"]["size"] = len(raw_capture_bytes)
    values["baseline_raw_exposition_file"]["sha256"] = hashlib.sha256(raw_capture_bytes).hexdigest()
    values["evidence_sha256"] = sha256_identity(values, omit_fields=frozenset({"evidence_sha256"}))
    with pytest.raises(ValidationError, match="stale"):
        type(measurement).model_validate(values)


def test_runtime_control_requires_drain_completion_before_final_scrape() -> None:
    control = _attestation().repetitions[0].runtime_control
    values = control.model_dump(mode="python")
    values["final_drain_completed_offset_ns"] = values["final_metric_scrape"][
        "scrape_monotonic_offset_ns"
    ]
    with pytest.raises(ValidationError, match="final drain completion"):
        type(control).model_validate(values)


@pytest.mark.parametrize("mutation", ["repetition", "scope"])
def test_repetition_reconstruction_binds_raw_prometheus_capture_metadata(
    mutation: str,
) -> None:
    _, repetition_payloads = make_experiment_attestation()
    raw = {path: data for path, data in repetition_payloads[0].items() if path.startswith("raw/")}
    path = "raw/prometheus/measured-window-baseline.json"
    capture = json.loads(raw[path])
    if mutation == "repetition":
        capture["repetition_index"] = 2
    else:
        capture["evidence_scope"] = "FUTURE_REAL_RUNTIME"
        capture["capture_source"] = "FUTURE_RUNTIME_PROMETHEUS_COLLECTOR"
    capture["identity_sha256"] = sha256_identity(
        capture, omit_fields=frozenset({"identity_sha256"})
    )
    raw[path] = canonical_json_bytes(capture) + b"\n"
    with pytest.raises(Stage2ExperimentError, match=r"Prometheus|repetition|scope"):
        reconstruct_experiment_repetition(raw)


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
    with pytest.raises(Stage2ExperimentError, match=r"wire|replay|invalid"):
        reconstruct_request_from_raw_evidence(raw, Stage2EvidenceScope.TEST_FIXTURE_ONLY)


@pytest.mark.parametrize(
    ("field", "replacement"),
    [
        ("temperature", 0.5),
        ("max_tokens", 31),
        ("include_usage", False),
        ("return_token_ids", False),
        ("stream_interval", 2),
        ("add_special_tokens", True),
        ("seed", 1),
        ("n", 2),
        ("min_tokens", 31),
        ("ignore_eos", False),
        ("echo", True),
        ("model", "drifted-model"),
        ("prompt", list(range(63))),
    ],
)
def test_exact_request_body_bytes_reject_every_fixed_field_drift(
    field: str,
    replacement: object,
) -> None:
    request = _attestation().repetitions[0].measured_requests[0]
    payloads = request_raw_evidence_payloads(request, Stage2EvidenceScope.TEST_FIXTURE_ONLY)
    body_record = json.loads(payloads["request_body"])
    exact = json.loads(base64.b64decode(body_record["content"]["exact_bytes_base64"]))
    if field == "include_usage":
        exact["stream_options"]["include_usage"] = replacement
    else:
        exact[field] = replacement
    changed_bytes = canonical_json_bytes(exact)
    body_record["content"]["exact_bytes_base64"] = base64.b64encode(changed_bytes).decode("ascii")
    body_record["content"]["byte_count"] = len(changed_bytes)
    body_record["content"]["sha256"] = hashlib.sha256(changed_bytes).hexdigest()
    payloads["request_body"] = canonical_json_bytes(body_record) + b"\n"
    with pytest.raises(Stage2ExperimentError):
        reconstruct_request_from_raw_evidence(payloads, Stage2EvidenceScope.TEST_FIXTURE_ONLY)


def test_exact_request_body_rejects_missing_unknown_and_duplicate_fields() -> None:
    request = _attestation().repetitions[0].measured_requests[0]
    for mutation in ("missing-include-usage", "unknown-field", "duplicate-field"):
        payloads = request_raw_evidence_payloads(request, Stage2EvidenceScope.TEST_FIXTURE_ONLY)
        body_record = json.loads(payloads["request_body"])
        exact = json.loads(base64.b64decode(body_record["content"]["exact_bytes_base64"]))
        if mutation == "missing-include-usage":
            exact["stream_options"].pop("include_usage")
        elif mutation == "unknown-field":
            exact["unexpected"] = True
        changed_bytes = canonical_json_bytes(exact)
        if mutation == "duplicate-field":
            changed_bytes = changed_bytes.replace(
                b'"temperature":0', b'"temperature":0,"temperature":0'
            )
        body_record["content"]["exact_bytes_base64"] = base64.b64encode(changed_bytes).decode(
            "ascii"
        )
        body_record["content"]["byte_count"] = len(changed_bytes)
        body_record["content"]["sha256"] = hashlib.sha256(changed_bytes).hexdigest()
        payloads["request_body"] = canonical_json_bytes(body_record) + b"\n"
        with pytest.raises(Stage2ExperimentError):
            reconstruct_request_from_raw_evidence(payloads, Stage2EvidenceScope.TEST_FIXTURE_ONLY)


@pytest.mark.parametrize(
    "mutation",
    [
        "missing-request-body",
        "request-id-mismatch",
        "missing-content-type",
        "ambiguous-content-type",
        "missing-response-id",
        "duplicate-response-id",
        "response-id-mismatch",
        "missing-chunks",
        "reordered-chunks",
        "duplicated-chunk",
        "truncated-chunk",
        "appended-chunk",
        "altered-chunk",
        "chunk-byte-count",
        "chunk-sha256",
        "parsed-events-mismatch",
        "typed-evidence-mismatch",
        "missing-transport-close",
        "transport-close-before-done",
    ],
)
def test_wire_replay_rejects_raw_and_derived_bypass_attempts(mutation: str) -> None:
    request = _attestation().repetitions[0].measured_requests[0]
    payloads = request_raw_evidence_payloads(request, Stage2EvidenceScope.TEST_FIXTURE_ONLY)
    if mutation == "missing-request-body":
        payloads.pop("request_body")
    elif mutation in {"request-id-mismatch", "missing-content-type", "ambiguous-content-type"}:
        record = json.loads(payloads["request_headers"])
        fields = record["content"]["fields"]
        if mutation == "request-id-mismatch":
            fields[0]["normalized_value"] = "different-request"
        elif mutation == "missing-content-type":
            fields.pop()
        else:
            fields.append(fields[-1])
        payloads["request_headers"] = canonical_json_bytes(record) + b"\n"
    elif mutation in {"missing-response-id", "duplicate-response-id", "response-id-mismatch"}:
        record = json.loads(payloads["response_headers"])
        fields = record["content"]["fields"]
        if mutation == "missing-response-id":
            fields.clear()
        elif mutation == "duplicate-response-id":
            fields.append(fields[0])
        else:
            fields[0]["normalized_value"] = "different-request"
        payloads["response_headers"] = canonical_json_bytes(record) + b"\n"
    elif mutation in {
        "missing-chunks",
        "reordered-chunks",
        "duplicated-chunk",
        "truncated-chunk",
        "appended-chunk",
        "altered-chunk",
        "chunk-byte-count",
        "chunk-sha256",
        "missing-transport-close",
        "transport-close-before-done",
    }:
        record = json.loads(payloads["raw_response_body"])
        chunks = record["content"]["response_body_chunks"]
        if mutation == "missing-chunks":
            chunks.clear()
        elif mutation == "reordered-chunks":
            chunks[0], chunks[1] = chunks[1], chunks[0]
        elif mutation == "duplicated-chunk":
            chunks.insert(1, chunks[0])
        elif mutation == "truncated-chunk":
            chunks.pop()
        elif mutation == "appended-chunk":
            chunks.append(chunks[-1])
        elif mutation == "altered-chunk":
            chunks[0]["exact_bytes_base64"] = "WA=="
        elif mutation == "chunk-byte-count":
            chunks[0]["decoded_byte_count"] += 1
        elif mutation == "chunk-sha256":
            chunks[0]["sha256"] = "0" * 64
        elif mutation == "missing-transport-close":
            record["content"].pop("transport_close")
        else:
            record["content"]["transport_close"]["close_observation_offset_ns"] = chunks[-1][
                "observation_offset_ns"
            ]
        payloads["raw_response_body"] = canonical_json_bytes(record) + b"\n"
    elif mutation == "parsed-events-mismatch":
        record = json.loads(payloads["parsed_sse_events"])
        record["content"]["events"][0]["data"] = "{}"
        payloads["parsed_sse_events"] = canonical_json_bytes(record) + b"\n"
    else:
        record = json.loads(payloads["token_usage_reconciliation"])
        record["content"]["typed_request_evidence"]["final_output_token_ids"][-1] = 999
        payloads["token_usage_reconciliation"] = canonical_json_bytes(record) + b"\n"
    with pytest.raises(Stage2ExperimentError):
        reconstruct_request_from_raw_evidence(payloads, Stage2EvidenceScope.TEST_FIXTURE_ONLY)


def test_fixture_wire_helper_cannot_emit_future_runtime_scope() -> None:
    request = _attestation().repetitions[0].measured_requests[0]
    with pytest.raises(Stage2ExperimentError, match="prohibited"):
        request_raw_evidence_payloads(request, Stage2EvidenceScope.FUTURE_REAL_RUNTIME)


@pytest.mark.parametrize(
    "header_name",
    [b"Authorization", b"X-Goog-Api-Key", b"Private-Token", b"X-Amz-Security-Token"],
)
def test_wire_headers_reject_secret_bearing_fields_before_durable_capture(
    header_name: bytes,
) -> None:
    headers = _attestation().repetitions[0].measured_requests[0].wire_capture.request_headers
    field_values: dict[str, object] = {
        "ordinal": len(headers.fields),
        "name_base64": base64.b64encode(header_name).decode("ascii"),
        "value_base64": base64.b64encode(b"super-secret-value").decode("ascii"),
        "name_byte_count": len(header_name),
        "value_byte_count": len(b"super-secret-value"),
        "name_sha256": hashlib.sha256(header_name).hexdigest(),
        "value_sha256": hashlib.sha256(b"super-secret-value").hexdigest(),
        "normalized_name": header_name.decode("ascii").casefold(),
        "normalized_value": "super-secret-value",
    }
    field_values["identity_sha256"] = sha256_identity(field_values)
    with pytest.raises(ValidationError, match="secret-bearing"):
        type(headers.fields[0]).model_validate(field_values)


@pytest.mark.parametrize("control_byte", [b"\x00", b"\x01", b"\x1f", b"\x7f"])
def test_lossless_header_fields_reject_prohibited_control_bytes(control_byte: bytes) -> None:
    field = (
        _attestation().repetitions[0].measured_requests[0].wire_capture.request_headers.fields[0]
    )
    values = field.model_dump(mode="python")
    malformed = b"safe" + control_byte + b"value"
    values.update(
        {
            "value_base64": base64.b64encode(malformed).decode("ascii"),
            "value_byte_count": len(malformed),
            "value_sha256": hashlib.sha256(malformed).hexdigest(),
            "normalized_value": malformed.decode("ascii"),
        }
    )
    values["identity_sha256"] = sha256_identity(values, omit_fields=frozenset({"identity_sha256"}))
    with pytest.raises(ValidationError, match="control byte"):
        type(field).model_validate(values)


def test_coalesced_terminal_frames_replay_from_retained_frame_observations() -> None:
    request = _attestation().repetitions[0].measured_requests[0]
    capture = request.wire_capture
    terminal_chunks = capture.response_body_chunks[-3:]
    data = b"".join(chunk.exact_bytes() for chunk in terminal_chunks)
    chunk_values = terminal_chunks[0].model_dump(mode="python")
    chunk_values.update(
        {
            "completed_sse_frame_observation_offsets_ns": tuple(
                offset
                for chunk in terminal_chunks
                for offset in chunk.completed_sse_frame_observation_offsets_ns
            ),
            "exact_bytes_base64": base64.b64encode(data).decode("ascii"),
            "decoded_byte_count": len(data),
            "sha256": hashlib.sha256(data).hexdigest(),
        }
    )
    chunk_values["identity_sha256"] = sha256_identity(
        chunk_values, omit_fields=frozenset({"identity_sha256"})
    )
    combined = type(terminal_chunks[0]).model_validate(chunk_values)
    chunks = (*capture.response_body_chunks[:-3], combined)
    close_values = capture.transport_close.model_dump(mode="python")
    close_values["raw_response_body_inventory_sha256"] = sha256_identity(chunks)
    close_values["identity_sha256"] = sha256_identity(
        close_values, omit_fields=frozenset({"identity_sha256"})
    )
    close = type(capture.transport_close).model_validate(close_values)
    capture_values = capture.model_dump(mode="python")
    capture_values["response_body_chunks"] = chunks
    capture_values["transport_close"] = close
    capture_values["http_exchange"]["response_body_inventory_sha256"] = sha256_identity(chunks)
    capture_values["http_exchange"]["identity_sha256"] = sha256_identity(
        capture_values["http_exchange"],
        omit_fields=frozenset({"identity_sha256"}),
    )
    capture_values["identity_sha256"] = sha256_identity(
        capture_values, omit_fields=frozenset({"identity_sha256"})
    )
    coalesced = type(capture).model_validate(capture_values)
    replayed, events = replay_stage2_wire_capture(
        coalesced, request.request_identity.identity_chain
    )
    assert replayed == request.request_evidence
    assert tuple(event.observation_offset_ns for event in events[-3:]) == tuple(
        offset
        for chunk in terminal_chunks
        for offset in chunk.completed_sse_frame_observation_offsets_ns
    )
    payloads = build_request_raw_evidence_payloads(
        wire_capture=coalesced,
        request_identity=request.request_identity,
        lifecycle=request.lifecycle,
    )
    assert (
        reconstruct_request_from_raw_evidence(payloads, Stage2EvidenceScope.TEST_FIXTURE_ONLY)[-1]
        == coalesced
    )


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
        "prometheus_measurement_attestation_files",
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
    with pytest.raises(ValidationError, match=r"wire provenance|fixture evidence"):
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


def test_aggregate_writer_and_reconstructor_reject_symlinked_ancestors(tmp_path: Path) -> None:
    real_parent = tmp_path / "real-parent"
    real_parent.mkdir()
    linked_parent = tmp_path / "linked-parent"
    linked_parent.symlink_to(real_parent, target_is_directory=True)

    attestation, repetition_payloads = make_experiment_attestation()
    manifest, payloads = make_aggregate_manifest(attestation, repetition_payloads)
    linked_root = linked_parent / "writer-aggregate"
    _write_aggregate_payloads(linked_root, payloads)
    with pytest.raises(Stage2ExperimentError, match="unsafe"):
        write_aggregate_manifest_last(linked_root, manifest)

    real_root = real_parent / "reconstruction-aggregate"
    write_synthetic_experiment_directory(real_root)
    with pytest.raises(Stage2ExperimentError, match="symlink ancestors"):
        reconstruct_experiment_attestation(linked_parent / real_root.name)


@pytest.mark.parametrize("boundary", ["baseline", "final"])
def test_aggregate_rejects_omitted_prometheus_raw_exposition(
    tmp_path: Path,
    boundary: str,
) -> None:
    root = tmp_path / f"missing-prometheus-{boundary}"
    write_synthetic_experiment_directory(root)
    path = root / "repetition-01/raw/prometheus" / f"measured-window-{boundary}.json"
    path.unlink()
    with pytest.raises(Stage2ExperimentError, match="inventory"):
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
