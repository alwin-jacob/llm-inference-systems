"""Synthetic object factories used only by Stage 0 tests."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

from llm_inference_systems.canonical import (
    configuration_identity,
    sha256_identity,
    with_artifact_content_hash,
)
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
from llm_inference_systems.metrics import derive_summary

ROOT = Path(__file__).resolve().parents[1]
FIXED_TIME = datetime(2026, 8, 27, 12, 0, tzinfo=UTC)


def load_workload() -> WorkloadDefinition:
    return WorkloadDefinition.model_validate_json(
        (ROOT / "examples/workloads/deterministic-smoke-v1.json").read_bytes()
    )


def load_configuration() -> RunConfiguration:
    return RunConfiguration.model_validate_json(
        (ROOT / "examples/configs/stage0-contract-v1.json").read_bytes()
    )


def known_count(value: int) -> TokenCount:
    return TokenCount(
        value=value,
        source=TokenCountSource.FIXTURE_EXACT,
        quality=TokenCountQuality.EXACT,
    )


def unknown_count() -> TokenCount:
    return TokenCount(
        value=None,
        source=TokenCountSource.UNKNOWN,
        quality=TokenCountQuality.UNAVAILABLE,
    )


def success_request(
    request_id: str,
    *,
    case_id: str = "case-alpha",
    phase: RequestPhase = RequestPhase.MEASURED,
    dispatch_ns: int = 0,
    token_offsets_ns: tuple[int, ...] = (20, 30, 40),
    first_byte_ns: int | None = 10,
    terminal_ns: int | None = None,
    input_tokens: TokenCount | None = None,
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
    terminal = terminal_ns
    if terminal is None:
        terminal = (token_offsets_ns[-1] + 5) if token_offsets_ns else dispatch_ns + 5
    first_token = token_offsets_ns[0] if token_offsets_ns else None
    last_token = token_offsets_ns[-1] if token_offsets_ns else None
    return RequestRecord(
        request_id=request_id,
        case_id=case_id,
        phase=phase,
        outcome=RequestOutcome.SUCCESS,
        timing=TimingRecord(
            dispatch_offset_ns=dispatch_ns,
            first_response_byte_offset_ns=first_byte_ns,
            first_output_token_offset_ns=first_token,
            last_output_token_offset_ns=last_token,
            terminal_offset_ns=terminal,
        ),
        stream_events=events,
        input_tokens=input_tokens or known_count(2),
        output_tokens=known_count(len(token_offsets_ns)),
        failure=None,
    )


def failed_request(
    request_id: str,
    outcome: RequestOutcome,
    *,
    case_id: str = "case-alpha",
    phase: RequestPhase = RequestPhase.MEASURED,
    dispatch_ns: int = 0,
    terminal_ns: int = 50,
) -> RequestRecord:
    return RequestRecord(
        request_id=request_id,
        case_id=case_id,
        phase=phase,
        outcome=outcome,
        timing=TimingRecord(
            dispatch_offset_ns=dispatch_ns,
            terminal_offset_ns=terminal_ns,
        ),
        stream_events=(),
        input_tokens=known_count(2),
        output_tokens=unknown_count(),
        failure=FailureRecord(
            kind=outcome,
            occurred_offset_ns=terminal_ns,
            error_code=f"synthetic-{outcome.value.casefold()}",
        ),
    )


def standard_requests() -> tuple[RequestRecord, ...]:
    return (
        success_request(
            "warmup-001",
            phase=RequestPhase.WARMUP,
            token_offsets_ns=(10_000_000, 20_000_000, 30_000_000),
            first_byte_ns=4_000_000,
            terminal_ns=35_000_000,
        ),
        success_request(
            "measured-001",
            dispatch_ns=100_000_000,
            token_offsets_ns=(120_000_000, 130_000_000, 140_000_000),
            first_byte_ns=104_000_000,
            terminal_ns=145_000_000,
        ),
        success_request(
            "measured-002",
            case_id="case-beta",
            dispatch_ns=200_000_000,
            token_offsets_ns=(225_000_000, 237_000_000, 249_000_000),
            first_byte_ns=204_000_000,
            terminal_ns=255_000_000,
        ),
        failed_request(
            "measured-003",
            RequestOutcome.TIMEOUT,
            dispatch_ns=300_000_000,
            terminal_ns=500_000_000,
        ),
        failed_request(
            "measured-004",
            RequestOutcome.PROTOCOL_ERROR,
            case_id="case-beta",
            dispatch_ns=600_000_000,
            terminal_ns=610_000_000,
        ),
    )


def artifact(
    configuration: RunConfiguration | None = None,
    *,
    requests: tuple[RequestRecord, ...] | None = None,
    runtime_name: str = "synthetic-fixture-runtime",
    processor: str = "synthetic-fixture-processor",
) -> RunArtifact:
    config = configuration or load_configuration()
    retained = requests or standard_requests()
    summary = derive_summary(
        config,
        retained,
        measurement_window_ns=2_000_000_000,
        observed_maximum_active_client_requests=2,
    )
    value = RunArtifact(
        schema_version=ARTIFACT_SCHEMA_VERSION,
        measurement_contract_version=MEASUREMENT_CONTRACT_VERSION,
        evidence_scope=EvidenceScope.TEST_FIXTURE_ONLY,
        created_at=FIXED_TIME,
        artifact_content_sha256=None,
        workload_identity=config.workload_identity,
        configuration_identity=configuration_identity(config),
        configuration=config,
        runtime_identity=RuntimeIdentity(
            runtime_name=runtime_name,
            exact_version="fixture-v1",
            source_revision="fixture-revision-v1",
            effective_configuration_sha256=sha256_identity({"fixture": "stage0"}),
            identity_source=IdentitySource.SYNTHETIC_FIXTURE,
        ),
        model_identity=config.model_identity,
        tokenizer_identity=config.tokenizer_identity,
        hardware_identity=HardwareIdentity(
            platform="synthetic-fixture-platform",
            architecture="synthetic",
            processor=processor,
            memory_bytes=1,
            gpu_model=None,
            identity_source=IdentitySource.SYNTHETIC_FIXTURE,
        ),
        environment_identity=EnvironmentIdentity(
            os_name="synthetic-fixture-os",
            architecture="synthetic",
            python_version="fixture-only",
            package_lock_sha256="0" * 64,
            identity_source=IdentitySource.SYNTHETIC_FIXTURE,
        ),
        requests=retained,
        summary=summary,
    )
    return with_artifact_content_hash(value)


def comparison_policy(
    baseline: RunArtifact,
    candidate: RunArtifact,
    *,
    kind: ComparisonKind = ComparisonKind.SAME_RUNTIME_AND_HARDWARE,
    minimum_successful_requests: int = 2,
    minimum_metric_samples: int = 2,
    failure_policy: FailureComparisonPolicy = FailureComparisonPolicy.REDUCE_GOODPUT,
) -> ComparisonPolicy:
    assert baseline.artifact_content_sha256 is not None
    assert candidate.artifact_content_sha256 is not None
    required = list(ESSENTIAL_COMPARISON_FIELDS)
    allowed: list[str] = []
    if kind is ComparisonKind.CROSS_RUNTIME:
        allowed.append("runtime_identity")
    else:
        required.append("runtime_identity")
    if kind is ComparisonKind.CROSS_HARDWARE:
        allowed.append("hardware_identity")
    else:
        required.append("hardware_identity")
    return ComparisonPolicy(
        schema_version=ARTIFACT_SCHEMA_VERSION,
        comparison_contract_version=COMPARISON_CONTRACT_VERSION,
        baseline_artifact_sha256=baseline.artifact_content_sha256,
        candidate_artifact_sha256=candidate.artifact_content_sha256,
        comparison_kind=kind,
        fields_required_identical=tuple(required),
        fields_allowed_to_differ=tuple(allowed),
        slo_policy_sha256=baseline.summary.goodput_slo_policy_sha256,
        minimum_successful_requests=minimum_successful_requests,
        minimum_metric_samples=minimum_metric_samples,
        failure_policy=failure_policy,
    )
