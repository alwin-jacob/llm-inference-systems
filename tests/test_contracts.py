"""Adversarial tests for strict Stage 0 contracts."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest
from pydantic import ValidationError

from llm_inference_systems.contracts import (
    EvidenceScope,
    FailureRecord,
    IdentitySource,
    LoadShape,
    LoadShapeKind,
    RequestOutcome,
    RequestPhase,
    RequestRecord,
    RunArtifact,
    RunConfiguration,
    SamplingConfiguration,
    SchedulingPolicy,
    StreamEventRecord,
    TimingRecord,
    TokenCount,
    TokenCountQuality,
    TokenCountSource,
    WorkloadCase,
    WorkloadDefinition,
    WorkloadOrdering,
)
from tests.factories import (
    artifact,
    known_count,
    load_configuration,
    load_workload,
    standard_requests,
    success_request,
    unknown_count,
)


def test_contracts_forbid_extra_fields() -> None:
    data = load_workload().model_dump(mode="python")
    data["unexpected"] = True
    with pytest.raises(ValidationError, match="extra_forbidden"):
        WorkloadDefinition.model_validate(data)


def test_workload_requires_explicit_version() -> None:
    data = load_workload().model_dump(mode="python")
    data["schema_version"] = "0.2.0"
    with pytest.raises(ValidationError, match="literal_error"):
        WorkloadDefinition.model_validate(data)


def test_run_configuration_requires_measurement_contract_version() -> None:
    data = load_configuration().model_dump(mode="python")
    data["measurement_contract_version"] = "0.2.0"
    with pytest.raises(ValidationError, match="literal_error"):
        RunConfiguration.model_validate(data)


@pytest.mark.parametrize("temperature", [-0.1, float("nan"), float("inf")])
def test_sampling_rejects_negative_or_nonfinite_temperature(temperature: float) -> None:
    with pytest.raises(ValidationError):
        SamplingConfiguration(
            seed=0,
            temperature=temperature,
            top_p=1.0,
            maximum_output_tokens=1,
        )


def test_workload_case_ids_must_be_unique() -> None:
    case = WorkloadCase(case_id="duplicate", prompt="synthetic prompt")
    with pytest.raises(ValidationError, match="unique"):
        WorkloadDefinition(
            schema_version="0.1.0",
            name="duplicate-workload",
            description="synthetic fixture",
            ordering_policy=WorkloadOrdering.DECLARED,
            prompt_transformation="literal",
            cases=(case, case),
        )


def test_sorted_workload_ids_must_be_sorted() -> None:
    with pytest.raises(ValidationError, match="sorted"):
        WorkloadDefinition(
            schema_version="0.1.0",
            name="unsorted-workload",
            description="synthetic fixture",
            ordering_policy=WorkloadOrdering.SORTED_CASE_ID,
            prompt_transformation="literal",
            cases=(
                WorkloadCase(case_id="z", prompt="z"),
                WorkloadCase(case_id="a", prompt="a"),
            ),
        )


def test_artifact_request_ids_must_be_unique() -> None:
    value = artifact()
    requests = list(value.requests)
    requests[1] = requests[1].model_copy(update={"request_id": requests[0].request_id})
    data = value.model_dump(mode="python")
    data["requests"] = tuple(requests)
    with pytest.raises(ValidationError, match="unique"):
        RunArtifact.model_validate(data)


@pytest.mark.parametrize(
    "created_at",
    [
        datetime.fromisoformat("2026-08-27T12:00:00"),
        datetime(2026, 8, 27, 12, 0, tzinfo=timezone(timedelta(hours=1))),
    ],
)
def test_artifact_timestamp_must_be_utc(created_at: datetime) -> None:
    value = artifact()
    data = value.model_dump(mode="python")
    data["created_at"] = created_at
    with pytest.raises(ValidationError, match=r"timezone|UTC"):
        RunArtifact.model_validate(data)


def test_stage0_artifact_rejects_real_runtime_scope() -> None:
    value = artifact()
    data = value.model_dump(mode="python")
    data["evidence_scope"] = EvidenceScope.REAL_RUNTIME
    with pytest.raises(ValidationError, match="literal_error"):
        RunArtifact.model_validate(data)


def test_fixture_artifact_requires_synthetic_identity_sources() -> None:
    value = artifact()
    runtime = value.runtime_identity.model_copy(
        update={"identity_source": IdentitySource.RUNTIME_REPORTED}
    )
    data = value.model_dump(mode="python")
    data["runtime_identity"] = runtime
    with pytest.raises(ValidationError, match="synthetic-fixture"):
        RunArtifact.model_validate(data)


def test_fixture_artifact_rejects_gpu_assertion() -> None:
    value = artifact()
    hardware = value.hardware_identity.model_copy(update={"gpu_model": "synthetic-gpu"})
    data = value.model_dump(mode="python")
    data["hardware_identity"] = hardware
    with pytest.raises(ValidationError, match="GPU"):
        RunArtifact.model_validate(data)


def test_client_concurrency_is_not_server_batch_size() -> None:
    configuration = load_configuration()
    assert configuration.load_shape.requested_client_concurrency == 2
    assert configuration.configured_server_maximum_batch_size is None
    assert "batch" not in LoadShape.model_fields
    assert "requested_client_concurrency" in LoadShape.model_fields


def test_server_batch_value_and_source_must_appear_together() -> None:
    data = load_configuration().model_dump(mode="python")
    data["configured_server_maximum_batch_size"] = 4
    with pytest.raises(ValidationError, match="present together"):
        RunConfiguration.model_validate(data)


def test_configured_server_batch_cannot_be_direct_observation() -> None:
    data = load_configuration().model_dump(mode="python")
    data["configured_server_maximum_batch_size"] = 4
    data["configured_server_batch_source"] = IdentitySource.DIRECTLY_OBSERVED
    with pytest.raises(ValidationError, match="cannot be directly observed"):
        RunConfiguration.model_validate(data)


def test_tokenizer_derived_count_requires_exact_tokenizer_identity() -> None:
    with pytest.raises(ValidationError, match="tokenizer ID and revision"):
        TokenCount(
            value=3,
            source=TokenCountSource.TOKENIZER_DERIVED,
            quality=TokenCountQuality.DERIVED,
        )


def test_server_reported_count_requires_provider_field() -> None:
    with pytest.raises(ValidationError, match="provider field"):
        TokenCount(
            value=3,
            source=TokenCountSource.SERVER_REPORTED,
            quality=TokenCountQuality.DERIVED,
        )


def test_unknown_token_count_has_explicit_unavailable_provenance() -> None:
    count = unknown_count()
    assert count.value is None
    assert count.source is TokenCountSource.UNKNOWN
    assert count.quality is TokenCountQuality.UNAVAILABLE


def test_available_token_count_cannot_claim_unknown_source() -> None:
    with pytest.raises(ValidationError, match="UNKNOWN source"):
        TokenCount(
            value=3,
            source=TokenCountSource.UNKNOWN,
            quality=TokenCountQuality.EXACT,
        )


def test_fixture_exact_count_requires_exact_quality() -> None:
    with pytest.raises(ValidationError, match="EXACT quality"):
        TokenCount(
            value=3,
            source=TokenCountSource.FIXTURE_EXACT,
            quality=TokenCountQuality.DERIVED,
        )


def test_failed_request_requires_retained_matching_failure() -> None:
    with pytest.raises(ValidationError, match="matching failure"):
        RequestRecord(
            request_id="missing-failure",
            case_id="case-alpha",
            phase=RequestPhase.MEASURED,
            outcome=RequestOutcome.TIMEOUT,
            timing=TimingRecord(dispatch_offset_ns=0, terminal_offset_ns=10),
            stream_events=(),
            input_tokens=known_count(1),
            output_tokens=unknown_count(),
            failure=None,
        )


def test_success_cannot_hide_failure_record() -> None:
    failure = FailureRecord(
        kind=RequestOutcome.PROTOCOL_ERROR,
        occurred_offset_ns=10,
        error_code="synthetic-error",
    )
    with pytest.raises(ValidationError, match="successful requests"):
        RequestRecord(
            request_id="bad-success",
            case_id="case-alpha",
            phase=RequestPhase.MEASURED,
            outcome=RequestOutcome.SUCCESS,
            timing=TimingRecord(dispatch_offset_ns=0, terminal_offset_ns=10),
            stream_events=(),
            input_tokens=known_count(1),
            output_tokens=known_count(0),
            failure=failure,
        )


def test_stream_token_total_must_match_available_count() -> None:
    with pytest.raises(ValidationError, match="stream token total"):
        RequestRecord(
            request_id="bad-count",
            case_id="case-alpha",
            phase=RequestPhase.MEASURED,
            outcome=RequestOutcome.SUCCESS,
            timing=TimingRecord(
                dispatch_offset_ns=0,
                first_output_token_offset_ns=10,
                last_output_token_offset_ns=10,
                terminal_offset_ns=20,
            ),
            stream_events=(
                StreamEventRecord(
                    chunk_index=0,
                    event_offset_ns=10,
                    output_tokens_in_chunk=1,
                    per_token_observation_offsets_ns=(10,),
                ),
            ),
            input_tokens=known_count(1),
            output_tokens=known_count(2),
            failure=None,
        )


def test_per_token_observation_must_be_inside_request_interval() -> None:
    with pytest.raises(ValidationError, match="within the request interval"):
        RequestRecord(
            request_id="bad-offset",
            case_id="case-alpha",
            phase=RequestPhase.MEASURED,
            outcome=RequestOutcome.SUCCESS,
            timing=TimingRecord(
                dispatch_offset_ns=10,
                first_output_token_offset_ns=15,
                last_output_token_offset_ns=15,
                terminal_offset_ns=20,
            ),
            stream_events=(
                StreamEventRecord(
                    chunk_index=0,
                    event_offset_ns=15,
                    output_tokens_in_chunk=1,
                    per_token_observation_offsets_ns=(5,),
                ),
            ),
            input_tokens=known_count(1),
            output_tokens=known_count(1),
            failure=None,
        )


def test_first_response_byte_cannot_follow_first_token() -> None:
    with pytest.raises(ValidationError, match="first response byte"):
        TimingRecord(
            dispatch_offset_ns=0,
            first_response_byte_offset_ns=11,
            first_output_token_offset_ns=10,
            last_output_token_offset_ns=10,
            terminal_offset_ns=20,
        )


def test_artifact_summary_counts_must_match_retained_failures() -> None:
    value = artifact()
    bad_summary = value.summary.model_copy(update={"failed_count": 3, "successful_count": 1})
    data = value.model_dump(mode="python")
    data["summary"] = bad_summary
    with pytest.raises(ValidationError, match="outcome counts"):
        RunArtifact.model_validate(data)


def test_artifact_counts_must_match_configured_warmup_and_measurement() -> None:
    configuration = load_configuration().model_copy(update={"warmup_request_count": 0})
    with pytest.raises(ValidationError, match="configured warmup"):
        artifact(configuration)


def test_request_case_must_belong_to_workload() -> None:
    requests = list(standard_requests())
    requests[0] = requests[0].model_copy(update={"case_id": "not-declared"})
    with pytest.raises(ValidationError, match="declared workload"):
        artifact(requests=tuple(requests))


@pytest.mark.parametrize("path", ["/absolute/workload.json", "../outside.json"])
def test_workload_path_must_remain_repository_relative(path: str) -> None:
    data = load_configuration().model_dump(mode="python")
    data["workload_path"] = path
    with pytest.raises(ValidationError, match="repository-relative"):
        RunConfiguration.model_validate(data)


def test_load_shape_requires_positive_client_concurrency() -> None:
    with pytest.raises(ValidationError):
        LoadShape(
            kind=LoadShapeKind.CLOSED_LOOP,
            requested_client_concurrency=0,
            scheduling_policy=SchedulingPolicy.NEXT_AVAILABLE_CLIENT,
        )


def test_success_factory_retains_separate_byte_and_token_timing() -> None:
    request = success_request("separate", first_byte_ns=5, token_offsets_ns=(10, 20))
    assert request.timing.first_response_byte_offset_ns == 5
    assert request.timing.first_output_token_offset_ns == 10
