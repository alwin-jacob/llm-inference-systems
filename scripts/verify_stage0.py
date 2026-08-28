"""Deterministically exercise Stage 0 contracts using in-memory fixture evidence."""

from __future__ import annotations

import hashlib
from datetime import UTC, datetime
from pathlib import Path

from llm_inference_systems.canonical import (
    canonical_json,
    configuration_identity,
    sha256_identity,
    verify_artifact_content_hash,
    verify_report_content_hash,
    with_artifact_content_hash,
    workload_identity,
)
from llm_inference_systems.comparison import check_compatibility, create_comparison_report
from llm_inference_systems.contracts import (
    ARTIFACT_SCHEMA_VERSION,
    COMPARISON_CONTRACT_VERSION,
    ESSENTIAL_COMPARISON_FIELDS,
    MEASUREMENT_CONTRACT_VERSION,
    ComparisonKind,
    ComparisonPolicy,
    EnvironmentIdentity,
    EvidenceScope,
    FailureComparisonPolicy,
    FailureRecord,
    HardwareIdentity,
    IdentitySource,
    ModelIdentity,
    RequestOutcome,
    RequestPhase,
    RequestRecord,
    RunArtifact,
    RunConfiguration,
    RuntimeIdentity,
    StreamEventRecord,
    TimingRecord,
    TokenCount,
    TokenCountQuality,
    TokenCountSource,
    WorkloadDefinition,
)
from llm_inference_systems.metrics import derive_request_metrics, derive_summary

FIXED_TIME = datetime(2026, 8, 27, 12, 0, tzinfo=UTC)
MEASUREMENT_WINDOW_NS = 2_000_000_000


def _known_count(value: int) -> TokenCount:
    return TokenCount(
        value=value,
        source=TokenCountSource.FIXTURE_EXACT,
        quality=TokenCountQuality.EXACT,
    )


def _unknown_count() -> TokenCount:
    return TokenCount(
        value=None,
        source=TokenCountSource.UNKNOWN,
        quality=TokenCountQuality.UNAVAILABLE,
    )


def _success(
    request_id: str,
    case_id: str,
    phase: RequestPhase,
    start_ns: int,
    token_offsets_ns: tuple[int, int, int],
    terminal_ns: int,
) -> RequestRecord:
    events = tuple(
        StreamEventRecord(
            chunk_index=index,
            event_offset_ns=offset,
            output_tokens_in_chunk=1,
            per_token_observation_offsets_ns=(offset,),
        )
        for index, offset in enumerate(token_offsets_ns)
    )
    return RequestRecord(
        request_id=request_id,
        case_id=case_id,
        phase=phase,
        outcome=RequestOutcome.SUCCESS,
        timing=TimingRecord(
            dispatch_offset_ns=start_ns,
            first_response_byte_offset_ns=start_ns + 4_000_000,
            first_output_token_offset_ns=token_offsets_ns[0],
            last_output_token_offset_ns=token_offsets_ns[-1],
            terminal_offset_ns=terminal_ns,
        ),
        stream_events=events,
        input_tokens=_known_count(2),
        output_tokens=_known_count(3),
        failure=None,
    )


def _failure(
    request_id: str,
    case_id: str,
    outcome: RequestOutcome,
    start_ns: int,
    terminal_ns: int,
) -> RequestRecord:
    return RequestRecord(
        request_id=request_id,
        case_id=case_id,
        phase=RequestPhase.MEASURED,
        outcome=outcome,
        timing=TimingRecord(
            dispatch_offset_ns=start_ns,
            first_response_byte_offset_ns=None,
            first_output_token_offset_ns=None,
            last_output_token_offset_ns=None,
            terminal_offset_ns=terminal_ns,
        ),
        stream_events=(),
        input_tokens=_known_count(2),
        output_tokens=_unknown_count(),
        failure=FailureRecord(
            kind=outcome,
            occurred_offset_ns=terminal_ns,
            error_code=f"synthetic-{outcome.value.casefold()}",
        ),
    )


def _requests(*, candidate: bool) -> tuple[RequestRecord, ...]:
    adjustment = -2_000_000 if candidate else 0
    return (
        _success(
            "warmup-001",
            "case-alpha",
            RequestPhase.WARMUP,
            0,
            (10_000_000, 20_000_000, 30_000_000),
            35_000_000,
        ),
        _success(
            "measured-001",
            "case-alpha",
            RequestPhase.MEASURED,
            100_000_000,
            (
                120_000_000 + adjustment,
                130_000_000 + adjustment,
                140_000_000 + adjustment,
            ),
            145_000_000 + adjustment,
        ),
        _success(
            "measured-002",
            "case-beta",
            RequestPhase.MEASURED,
            200_000_000,
            (
                225_000_000 + adjustment,
                237_000_000 + adjustment,
                249_000_000 + adjustment,
            ),
            255_000_000 + adjustment,
        ),
        _failure(
            "measured-003",
            "case-alpha",
            RequestOutcome.TIMEOUT,
            300_000_000,
            500_000_000,
        ),
        _failure(
            "measured-004",
            "case-beta",
            RequestOutcome.PROTOCOL_ERROR,
            600_000_000,
            610_000_000,
        ),
    )


def _artifact(
    root: Path,
    workload: WorkloadDefinition,
    configuration: RunConfiguration,
    *,
    candidate: bool,
) -> RunArtifact:
    requests = _requests(candidate=candidate)
    summary = derive_summary(
        configuration,
        requests,
        measurement_window_ns=MEASUREMENT_WINDOW_NS,
        observed_maximum_active_client_requests=2,
    )
    lock_digest = hashlib.sha256((root / "uv.lock").read_bytes()).hexdigest()
    artifact = RunArtifact(
        schema_version=ARTIFACT_SCHEMA_VERSION,
        measurement_contract_version=MEASUREMENT_CONTRACT_VERSION,
        evidence_scope=EvidenceScope.TEST_FIXTURE_ONLY,
        created_at=FIXED_TIME,
        artifact_content_sha256=None,
        workload_identity=workload_identity(workload),
        configuration_identity=configuration_identity(configuration),
        configuration=configuration,
        runtime_identity=RuntimeIdentity(
            runtime_name="synthetic-fixture-runtime",
            exact_version="fixture-v1",
            source_revision="fixture-revision-v1",
            effective_configuration_sha256=sha256_identity({"fixture_mode": "stage0"}),
            identity_source=IdentitySource.SYNTHETIC_FIXTURE,
        ),
        model_identity=configuration.model_identity,
        tokenizer_identity=configuration.tokenizer_identity,
        hardware_identity=HardwareIdentity(
            platform="synthetic-fixture-platform",
            architecture="synthetic",
            processor="synthetic-fixture-processor",
            memory_bytes=1,
            gpu_model=None,
            identity_source=IdentitySource.SYNTHETIC_FIXTURE,
        ),
        environment_identity=EnvironmentIdentity(
            os_name="synthetic-fixture-os",
            architecture="synthetic",
            python_version="fixture-only",
            package_lock_sha256=lock_digest,
            identity_source=IdentitySource.SYNTHETIC_FIXTURE,
        ),
        requests=requests,
        summary=summary,
    )
    return with_artifact_content_hash(artifact)


def _policy(baseline: RunArtifact, candidate: RunArtifact) -> ComparisonPolicy:
    if baseline.artifact_content_sha256 is None or candidate.artifact_content_sha256 is None:
        raise AssertionError("fixture artifacts must be finalized")
    return ComparisonPolicy(
        schema_version=ARTIFACT_SCHEMA_VERSION,
        comparison_contract_version=COMPARISON_CONTRACT_VERSION,
        baseline_artifact_sha256=baseline.artifact_content_sha256,
        candidate_artifact_sha256=candidate.artifact_content_sha256,
        comparison_kind=ComparisonKind.SAME_RUNTIME_AND_HARDWARE,
        fields_required_identical=(
            *ESSENTIAL_COMPARISON_FIELDS,
            "runtime_identity",
            "hardware_identity",
        ),
        fields_allowed_to_differ=(),
        slo_policy_sha256=baseline.summary.goodput_slo_policy_sha256,
        minimum_successful_requests=2,
        minimum_metric_samples=2,
        failure_policy=FailureComparisonPolicy.REDUCE_GOODPUT,
    )


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    workload = WorkloadDefinition.model_validate_json(
        (root / "examples/workloads/deterministic-smoke-v1.json").read_bytes()
    )
    configuration = RunConfiguration.model_validate_json(
        (root / "examples/configs/stage0-contract-v1.json").read_bytes()
    )
    if workload_identity(workload) != configuration.workload_identity:
        raise AssertionError("checked-in workload identity is incorrect")

    baseline = _artifact(root, workload, configuration, candidate=False)
    candidate = _artifact(root, workload, configuration, candidate=True)
    if not verify_artifact_content_hash(baseline) or not verify_artifact_content_hash(candidate):
        raise AssertionError("artifact self-hash verification failed")
    if baseline.summary.warmup_record_count != 1 or baseline.summary.attempted_count != 4:
        raise AssertionError("warmup exclusion proof failed")
    if baseline.summary.failed_count != 2 or baseline.summary.timeout_count != 1:
        raise AssertionError("failure retention proof failed")
    if baseline.summary.goodput != 1.0:
        raise AssertionError("failure-aware goodput proof failed")
    success_metrics = derive_request_metrics(baseline.requests[1])
    if not success_metrics.itl_ns or success_metrics.tpot_ns is None:
        raise AssertionError("per-token metric proof failed")
    if baseline.summary.output_token_throughput != 3.0:
        raise AssertionError("output-token throughput proof failed")
    if baseline.summary.total_token_throughput != 5.0:
        raise AssertionError("total-token throughput proof failed")
    if baseline.summary.offered_request_rate != 2.0:
        raise AssertionError("request throughput proof failed")

    policy = _policy(baseline, candidate)
    report = create_comparison_report(
        baseline,
        candidate,
        policy,
        created_at=FIXED_TIME,
    )
    if not report.compatibility.compatible or not verify_report_content_hash(report):
        raise AssertionError("compatible comparison proof failed")

    different_model = ModelIdentity(
        model_id="different-synthetic-fixture-model",
        exact_revision="fixture-v1",
        prompt_template_sha256=configuration.model_identity.prompt_template_sha256,
        identity_source=IdentitySource.SYNTHETIC_FIXTURE,
    )
    incompatible_configuration = configuration.model_copy(
        update={"model_identity": different_model}
    )
    incompatible_candidate = _artifact(
        root,
        workload,
        incompatible_configuration,
        candidate=True,
    )
    incompatible_policy = _policy(baseline, incompatible_candidate)
    if check_compatibility(baseline, incompatible_candidate, incompatible_policy).compatible:
        raise AssertionError("incompatible comparison was not rejected")

    result = {
        "compatible_comparison": True,
        "evidence_scope": EvidenceScope.TEST_FIXTURE_ONLY.value,
        "gpu_execution": False,
        "performance_claim_allowed": False,
        "real_runtime_execution": False,
        "verified": {
            "failure_retention": True,
            "goodput": True,
            "incompatible_comparison_rejected": True,
            "itl": True,
            "request_throughput": True,
            "tpot": True,
            "total_token_throughput": True,
            "ttft": True,
            "warmup_exclusion": True,
        },
        "workload_sha256": workload_identity(workload).content_sha256,
    }
    print(canonical_json(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
