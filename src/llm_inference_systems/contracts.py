"""Strict, versioned Stage 0 data contracts."""

from __future__ import annotations

from datetime import timedelta
from enum import StrEnum
from pathlib import PurePosixPath
from typing import Annotated, Final, Literal, Self

from pydantic import (
    AwareDatetime,
    BaseModel,
    ConfigDict,
    Field,
    StringConstraints,
    model_validator,
)

ARTIFACT_SCHEMA_VERSION: Final[Literal["0.1.0"]] = "0.1.0"
MEASUREMENT_CONTRACT_VERSION: Final[Literal["0.1.0"]] = "0.1.0"
COMPARISON_CONTRACT_VERSION: Final[Literal["0.1.0"]] = "0.1.0"

NonNegativeInt = Annotated[int, Field(ge=0)]
PositiveInt = Annotated[int, Field(gt=0)]
NonNegativeFloat = Annotated[float, Field(ge=0, allow_inf_nan=False)]
FiniteFloat = Annotated[float, Field(allow_inf_nan=False)]
Probability = Annotated[float, Field(ge=0, le=1, allow_inf_nan=False)]
Identifier = Annotated[str, StringConstraints(min_length=1, max_length=160)]
Sha256 = Annotated[str, StringConstraints(pattern=r"^[0-9a-f]{64}$")]


class StrictModel(BaseModel):
    """Base for durable contracts: immutable, strict, and closed to unknown fields."""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)


class EvidenceScope(StrEnum):
    """The strongest evidence an artifact is allowed to represent."""

    TEST_FIXTURE_ONLY = "TEST_FIXTURE_ONLY"
    REAL_RUNTIME = "REAL_RUNTIME"


class IdentitySource(StrEnum):
    SYNTHETIC_FIXTURE = "SYNTHETIC_FIXTURE"
    CONFIGURED = "CONFIGURED"
    RUNTIME_REPORTED = "RUNTIME_REPORTED"
    DIRECTLY_OBSERVED = "DIRECTLY_OBSERVED"


class WorkloadOrdering(StrEnum):
    DECLARED = "DECLARED"
    SORTED_CASE_ID = "SORTED_CASE_ID"


class RequestPhase(StrEnum):
    WARMUP = "WARMUP"
    MEASURED = "MEASURED"


class RequestOutcome(StrEnum):
    SUCCESS = "SUCCESS"
    TIMEOUT = "TIMEOUT"
    TRANSPORT_ERROR = "TRANSPORT_ERROR"
    HTTP_ERROR = "HTTP_ERROR"
    PROTOCOL_ERROR = "PROTOCOL_ERROR"
    MALFORMED_STREAM = "MALFORMED_STREAM"
    TOKEN_ACCOUNTING_ERROR = "TOKEN_ACCOUNTING_ERROR"
    CANCELLED = "CANCELLED"
    UNEXPECTED_ERROR = "UNEXPECTED_ERROR"


class LoadShapeKind(StrEnum):
    CLOSED_LOOP = "CLOSED_LOOP"


class SchedulingPolicy(StrEnum):
    NEXT_AVAILABLE_CLIENT = "NEXT_AVAILABLE_CLIENT"


class WarmupPolicyKind(StrEnum):
    FIXED_COUNT = "FIXED_COUNT"


class TokenCountSource(StrEnum):
    FIXTURE_EXACT = "FIXTURE_EXACT"
    TOKENIZER_DERIVED = "TOKENIZER_DERIVED"
    SERVER_REPORTED = "SERVER_REPORTED"
    UNKNOWN = "UNKNOWN"


class TokenCountQuality(StrEnum):
    EXACT = "EXACT"
    DERIVED = "DERIVED"
    UNAVAILABLE = "UNAVAILABLE"


class MetricSource(StrEnum):
    HARNESS_DERIVED = "HARNESS_DERIVED"
    RUNTIME_REPORTED = "RUNTIME_REPORTED"


class MetricName(StrEnum):
    TTFT_NS = "TTFT_NS"
    END_TO_END_SUCCESS_NS = "END_TO_END_SUCCESS_NS"
    TPOT_NS = "TPOT_NS"
    ITL_NS = "ITL_NS"


class MetricUnavailableReason(StrEnum):
    NO_SAMPLES = "NO_SAMPLES"
    REQUEST_FAILED = "REQUEST_FAILED"
    FIRST_OUTPUT_TOKEN_NOT_OBSERVED = "FIRST_OUTPUT_TOKEN_NOT_OBSERVED"
    INSUFFICIENT_OUTPUT_TOKENS = "INSUFFICIENT_OUTPUT_TOKENS"
    MISSING_TOKEN_COUNT = "MISSING_TOKEN_COUNT"
    MISSING_TOKEN_TIMESTAMPS = "MISSING_TOKEN_TIMESTAMPS"
    MULTI_TOKEN_CHUNK_WITHOUT_TOKEN_TIMESTAMPS = "MULTI_TOKEN_CHUNK_WITHOUT_TOKEN_TIMESTAMPS"
    TOKEN_TIMESTAMP_COUNT_MISMATCH = "TOKEN_TIMESTAMP_COUNT_MISMATCH"


class ComparisonKind(StrEnum):
    SAME_RUNTIME_AND_HARDWARE = "SAME_RUNTIME_AND_HARDWARE"
    CROSS_RUNTIME = "CROSS_RUNTIME"
    CROSS_HARDWARE = "CROSS_HARDWARE"


class FailureComparisonPolicy(StrEnum):
    INVALIDATE = "INVALIDATE"
    REDUCE_GOODPUT = "REDUCE_GOODPUT"


class WorkloadCase(StrictModel):
    case_id: Identifier
    prompt: Annotated[str, StringConstraints(min_length=1, max_length=16_384)]
    expected_output_token_count: NonNegativeInt | None = None


class WorkloadDefinition(StrictModel):
    schema_version: Literal["0.1.0"]
    name: Identifier
    description: Annotated[str, StringConstraints(min_length=1, max_length=1_024)]
    ordering_policy: WorkloadOrdering
    prompt_transformation: Identifier
    cases: tuple[WorkloadCase, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_case_ids(self) -> Self:
        case_ids = [case.case_id for case in self.cases]
        if len(case_ids) != len(set(case_ids)):
            raise ValueError("workload case IDs must be unique")
        if self.ordering_policy is WorkloadOrdering.SORTED_CASE_ID and case_ids != sorted(case_ids):
            raise ValueError("case IDs must be sorted for SORTED_CASE_ID ordering")
        return self


class WorkloadIdentity(StrictModel):
    content_sha256: Sha256
    case_ids: tuple[Identifier, ...] = Field(min_length=1)
    ordering_policy: WorkloadOrdering


class ConfigurationIdentity(StrictModel):
    content_sha256: Sha256


class RuntimeIdentity(StrictModel):
    runtime_name: Identifier
    exact_version: Identifier
    source_revision: Identifier
    effective_configuration_sha256: Sha256
    identity_source: IdentitySource

    @model_validator(mode="after")
    def reject_ambiguous_names(self) -> Self:
        if self.runtime_name.casefold() in {"default", "untuned"}:
            raise ValueError("runtime identity cannot be named only default or untuned")
        return self


class ModelIdentity(StrictModel):
    model_id: Identifier
    exact_revision: Identifier
    prompt_template_sha256: Sha256
    identity_source: IdentitySource


class TokenizerIdentity(StrictModel):
    tokenizer_id: Identifier
    exact_revision: Identifier
    identity_source: IdentitySource


class HardwareIdentity(StrictModel):
    platform: Identifier
    architecture: Identifier
    processor: Identifier
    memory_bytes: PositiveInt
    gpu_model: Identifier | None = None
    identity_source: IdentitySource


class EnvironmentIdentity(StrictModel):
    os_name: Identifier
    architecture: Identifier
    python_version: Identifier
    package_lock_sha256: Sha256
    identity_source: IdentitySource


class LoadShape(StrictModel):
    kind: Literal[LoadShapeKind.CLOSED_LOOP]
    requested_client_concurrency: PositiveInt
    scheduling_policy: Literal[SchedulingPolicy.NEXT_AVAILABLE_CLIENT]


class TimeoutPolicy(StrictModel):
    connect_timeout_ns: PositiveInt
    first_output_token_timeout_ns: PositiveInt
    request_timeout_ns: PositiveInt

    @model_validator(mode="after")
    def validate_timeout_order(self) -> Self:
        if self.connect_timeout_ns > self.request_timeout_ns:
            raise ValueError("connect timeout cannot exceed request timeout")
        if self.first_output_token_timeout_ns > self.request_timeout_ns:
            raise ValueError("first-output-token timeout cannot exceed request timeout")
        return self


class SamplingConfiguration(StrictModel):
    seed: NonNegativeInt
    temperature: NonNegativeFloat
    top_p: Probability
    maximum_output_tokens: PositiveInt
    stop_sequences: tuple[Identifier, ...] = ()


class SLODefinition(StrictModel):
    policy_name: Identifier
    ttft_threshold_ns: PositiveInt | None = None
    tpot_threshold_ns: PositiveInt | None = None
    end_to_end_threshold_ns: PositiveInt | None = None
    itl_threshold_ns: PositiveInt | None = None

    @model_validator(mode="after")
    def require_threshold(self) -> Self:
        thresholds = (
            self.ttft_threshold_ns,
            self.tpot_threshold_ns,
            self.end_to_end_threshold_ns,
            self.itl_threshold_ns,
        )
        if not any(value is not None for value in thresholds):
            raise ValueError("an SLO policy must declare at least one threshold")
        return self


class RunConfiguration(StrictModel):
    schema_version: Literal["0.1.0"]
    measurement_contract_version: Literal["0.1.0"]
    workload_path: Identifier
    workload_identity: WorkloadIdentity
    model_identity: ModelIdentity
    tokenizer_identity: TokenizerIdentity
    prompt_transformation: Identifier
    load_shape: LoadShape
    timeout_policy: TimeoutPolicy
    sampling: SamplingConfiguration
    slo: SLODefinition
    warmup_policy: Literal[WarmupPolicyKind.FIXED_COUNT]
    warmup_request_count: NonNegativeInt
    measured_request_count: PositiveInt
    configured_server_maximum_batch_size: PositiveInt | None = None
    configured_server_batch_source: IdentitySource | None = None

    @model_validator(mode="after")
    def validate_configuration(self) -> Self:
        path = PurePosixPath(self.workload_path)
        if path.is_absolute() or ".." in path.parts:
            raise ValueError(
                "workload_path must be repository-relative and cannot traverse parents"
            )
        if (self.configured_server_maximum_batch_size is None) != (
            self.configured_server_batch_source is None
        ):
            raise ValueError("configured server batch size and source must be present together")
        if self.configured_server_batch_source is IdentitySource.DIRECTLY_OBSERVED:
            raise ValueError("a configured server batch limit cannot be directly observed")
        return self


class TokenCount(StrictModel):
    value: NonNegativeInt | None
    source: TokenCountSource
    quality: TokenCountQuality
    tokenizer_id: Identifier | None = None
    tokenizer_revision: Identifier | None = None
    provider_field_name: Identifier | None = None

    @model_validator(mode="after")
    def validate_provenance(self) -> Self:
        if self.value is None:
            if self.source is not TokenCountSource.UNKNOWN:
                raise ValueError("an unavailable count must use UNKNOWN source")
            if self.quality is not TokenCountQuality.UNAVAILABLE:
                raise ValueError("an unavailable count must use UNAVAILABLE quality")
        else:
            if self.source is TokenCountSource.UNKNOWN:
                raise ValueError("an available count cannot use UNKNOWN source")
            if self.quality is TokenCountQuality.UNAVAILABLE:
                raise ValueError("an available count cannot use UNAVAILABLE quality")
            if (
                self.source is TokenCountSource.FIXTURE_EXACT
                and self.quality is not TokenCountQuality.EXACT
            ):
                raise ValueError("fixture-exact counts require EXACT quality")
            if (
                self.source is TokenCountSource.TOKENIZER_DERIVED
                and self.quality is not TokenCountQuality.DERIVED
            ):
                raise ValueError("tokenizer-derived counts require DERIVED quality")
        if self.source is TokenCountSource.TOKENIZER_DERIVED:
            if self.tokenizer_id is None or self.tokenizer_revision is None:
                raise ValueError("tokenizer-derived counts require tokenizer ID and revision")
        elif self.tokenizer_id is not None or self.tokenizer_revision is not None:
            raise ValueError("tokenizer provenance is only valid for tokenizer-derived counts")
        if self.source is TokenCountSource.SERVER_REPORTED:
            if self.provider_field_name is None:
                raise ValueError("server-reported counts require a provider field name")
        elif self.provider_field_name is not None:
            raise ValueError("provider field is only valid for server-reported counts")
        return self


class TimingRecord(StrictModel):
    dispatch_offset_ns: NonNegativeInt
    first_response_byte_offset_ns: NonNegativeInt | None = None
    first_output_token_offset_ns: NonNegativeInt | None = None
    last_output_token_offset_ns: NonNegativeInt | None = None
    terminal_offset_ns: NonNegativeInt

    @model_validator(mode="after")
    def validate_offsets(self) -> Self:
        observations = (
            self.first_response_byte_offset_ns,
            self.first_output_token_offset_ns,
            self.last_output_token_offset_ns,
        )
        if self.terminal_offset_ns < self.dispatch_offset_ns:
            raise ValueError("terminal offset cannot precede dispatch")
        if any(
            value is not None
            and (value < self.dispatch_offset_ns or value > self.terminal_offset_ns)
            for value in observations
        ):
            raise ValueError("observations must fall between dispatch and terminal offsets")
        if (
            self.first_output_token_offset_ns is None
            and self.last_output_token_offset_ns is not None
        ):
            raise ValueError("last output token requires a first output token")
        if (
            self.first_output_token_offset_ns is not None
            and self.last_output_token_offset_ns is not None
            and self.last_output_token_offset_ns < self.first_output_token_offset_ns
        ):
            raise ValueError("last output token cannot precede first output token")
        if (
            self.first_response_byte_offset_ns is not None
            and self.first_output_token_offset_ns is not None
            and self.first_response_byte_offset_ns > self.first_output_token_offset_ns
        ):
            raise ValueError("first response byte cannot follow the first output token")
        return self


class StreamEventRecord(StrictModel):
    chunk_index: NonNegativeInt
    event_offset_ns: NonNegativeInt
    output_tokens_in_chunk: PositiveInt
    per_token_observation_offsets_ns: tuple[NonNegativeInt, ...] | None = None

    @model_validator(mode="after")
    def validate_token_observations(self) -> Self:
        offsets = self.per_token_observation_offsets_ns
        if offsets is not None:
            if len(offsets) != self.output_tokens_in_chunk:
                raise ValueError("per-token timestamps must match tokens in the chunk")
            if tuple(sorted(offsets)) != offsets or len(offsets) != len(set(offsets)):
                raise ValueError("per-token timestamps must be strictly ordered")
            if offsets[-1] > self.event_offset_ns:
                raise ValueError("a token observation cannot follow its enclosing stream event")
        return self


class FailureRecord(StrictModel):
    kind: RequestOutcome
    occurred_offset_ns: NonNegativeInt
    error_code: Identifier

    @model_validator(mode="after")
    def reject_success(self) -> Self:
        if self.kind is RequestOutcome.SUCCESS:
            raise ValueError("a failure record cannot have SUCCESS kind")
        return self


class RequestRecord(StrictModel):
    request_id: Identifier
    case_id: Identifier
    phase: RequestPhase
    outcome: RequestOutcome
    timing: TimingRecord
    stream_events: tuple[StreamEventRecord, ...] = ()
    input_tokens: TokenCount
    output_tokens: TokenCount
    failure: FailureRecord | None = None

    @model_validator(mode="after")
    def validate_semantics(self) -> Self:
        if self.outcome is RequestOutcome.SUCCESS and self.failure is not None:
            raise ValueError("successful requests cannot contain a failure record")
        if self.outcome is not RequestOutcome.SUCCESS and (
            self.failure is None or self.failure.kind is not self.outcome
        ):
            raise ValueError("failed requests require a matching failure record")
        if self.failure is not None and not (
            self.timing.dispatch_offset_ns
            <= self.failure.occurred_offset_ns
            <= self.timing.terminal_offset_ns
        ):
            raise ValueError("failure offset must be within the request interval")
        chunk_indexes = [event.chunk_index for event in self.stream_events]
        if chunk_indexes != list(range(len(self.stream_events))):
            raise ValueError("stream chunk indexes must be contiguous from zero")
        event_offsets = [event.event_offset_ns for event in self.stream_events]
        if event_offsets != sorted(event_offsets) or len(event_offsets) != len(set(event_offsets)):
            raise ValueError("stream event offsets must be strictly ordered")
        if any(
            offset < self.timing.dispatch_offset_ns or offset > self.timing.terminal_offset_ns
            for offset in event_offsets
        ):
            raise ValueError("stream events must be within the request interval")
        token_offsets = tuple(
            offset
            for event in self.stream_events
            for offset in (event.per_token_observation_offsets_ns or ())
        )
        if any(
            offset < self.timing.dispatch_offset_ns or offset > self.timing.terminal_offset_ns
            for offset in token_offsets
        ):
            raise ValueError("per-token observations must be within the request interval")
        observed_output_tokens = sum(event.output_tokens_in_chunk for event in self.stream_events)
        if (
            self.output_tokens.value is not None
            and observed_output_tokens != self.output_tokens.value
        ):
            raise ValueError("stream token total must match the available output-token count")
        if observed_output_tokens == 0:
            if self.timing.first_output_token_offset_ns is not None:
                raise ValueError("first output token requires a retained stream event")
            if self.timing.last_output_token_offset_ns is not None:
                raise ValueError("last output token requires a retained stream event")
        else:
            first_observation = (
                self.stream_events[0].per_token_observation_offsets_ns or (event_offsets[0],)
            )[0]
            last_observation = (
                self.stream_events[-1].per_token_observation_offsets_ns or (event_offsets[-1],)
            )[-1]
            if self.timing.first_output_token_offset_ns != first_observation:
                raise ValueError("first-output-token timing must match the first observation")
            if self.timing.last_output_token_offset_ns != last_observation:
                raise ValueError("last-output-token timing must match the last observation")
        return self


class MetricDistribution(StrictModel):
    metric: MetricName
    source: MetricSource
    denominator: Identifier
    unit: Literal["nanoseconds"]
    percentile_algorithm: Literal["HYNDMAN_FAN_TYPE_7"]
    sample_count: NonNegativeInt
    p50: NonNegativeFloat | None
    p95: NonNegativeFloat | None
    p99: NonNegativeFloat | None
    unavailable_reason: Literal[MetricUnavailableReason.NO_SAMPLES] | None = None

    @model_validator(mode="after")
    def validate_availability(self) -> Self:
        percentiles = (self.p50, self.p95, self.p99)
        if self.sample_count == 0:
            if any(value is not None for value in percentiles):
                raise ValueError("empty distributions cannot contain percentiles")
            if self.unavailable_reason is not MetricUnavailableReason.NO_SAMPLES:
                raise ValueError("empty distributions require NO_SAMPLES")
        elif any(value is None for value in percentiles) or self.unavailable_reason is not None:
            raise ValueError("nonempty distributions require all percentiles and no reason")
        return self


class RunSummary(StrictModel):
    measurement_window_ns: PositiveInt
    attempted_count: NonNegativeInt
    terminal_count: NonNegativeInt
    successful_count: NonNegativeInt
    failed_count: NonNegativeInt
    timeout_count: NonNegativeInt
    cancelled_count: NonNegativeInt
    protocol_error_count: NonNegativeInt
    warmup_record_count: NonNegativeInt
    warmup_excluded_count: NonNegativeInt
    requested_client_concurrency: PositiveInt
    observed_maximum_active_client_requests: NonNegativeInt
    configured_server_maximum_batch_size: PositiveInt | None
    configured_server_batch_source: IdentitySource | None
    observed_server_batch_size: PositiveInt | None
    observed_server_batch_source: IdentitySource | None
    offered_request_rate: NonNegativeFloat
    terminal_request_rate: NonNegativeFloat
    successful_request_throughput: NonNegativeFloat
    output_token_throughput: NonNegativeFloat | None
    total_token_throughput: NonNegativeFloat | None
    goodput: NonNegativeFloat
    failure_rate: Probability
    timeout_rate: Probability
    goodput_slo_policy_sha256: Sha256
    ttft: MetricDistribution
    end_to_end_success: MetricDistribution
    tpot: MetricDistribution
    itl: MetricDistribution

    @model_validator(mode="after")
    def validate_counts_and_sources(self) -> Self:
        if self.terminal_count > self.attempted_count:
            raise ValueError("terminal count cannot exceed attempted count")
        if self.successful_count + self.failed_count != self.terminal_count:
            raise ValueError("successful plus failed count must equal terminal count")
        if any(
            value > self.failed_count
            for value in (self.timeout_count, self.cancelled_count, self.protocol_error_count)
        ):
            raise ValueError("failure subtype count cannot exceed failed count")
        if self.warmup_record_count != self.warmup_excluded_count:
            raise ValueError("all retained warmup records must be excluded from summaries")
        if self.observed_maximum_active_client_requests > self.requested_client_concurrency:
            raise ValueError("observed client activity cannot exceed requested concurrency")
        if (self.configured_server_maximum_batch_size is None) != (
            self.configured_server_batch_source is None
        ):
            raise ValueError("configured server batch size and source must be present together")
        if self.configured_server_batch_source is IdentitySource.DIRECTLY_OBSERVED:
            raise ValueError("configured server batch size cannot be directly observed")
        if (self.observed_server_batch_size is None) != (self.observed_server_batch_source is None):
            raise ValueError("observed server batch size and source must be present together")
        if (
            self.observed_server_batch_source is not None
            and self.observed_server_batch_source is not IdentitySource.DIRECTLY_OBSERVED
        ):
            raise ValueError("observed server batch size must be directly observed")
        return self


class RunArtifact(StrictModel):
    schema_version: Literal["0.1.0"]
    measurement_contract_version: Literal["0.1.0"]
    evidence_scope: Literal[EvidenceScope.TEST_FIXTURE_ONLY]
    created_at: AwareDatetime
    artifact_content_sha256: Sha256 | None = None
    workload_identity: WorkloadIdentity
    configuration_identity: ConfigurationIdentity
    configuration: RunConfiguration
    runtime_identity: RuntimeIdentity
    model_identity: ModelIdentity
    tokenizer_identity: TokenizerIdentity
    hardware_identity: HardwareIdentity
    environment_identity: EnvironmentIdentity
    requests: tuple[RequestRecord, ...] = Field(min_length=1)
    summary: RunSummary

    @model_validator(mode="after")
    def validate_artifact(self) -> Self:
        offset = self.created_at.utcoffset()
        if offset is None or offset != timedelta(0):
            raise ValueError("created_at must be timezone-aware UTC")
        request_ids = [request.request_id for request in self.requests]
        if len(request_ids) != len(set(request_ids)):
            raise ValueError("request IDs must be unique")
        measured = sum(request.phase is RequestPhase.MEASURED for request in self.requests)
        warmup = len(self.requests) - measured
        if measured != self.summary.attempted_count:
            raise ValueError("summary attempted count must match measured request records")
        if warmup != self.summary.warmup_record_count:
            raise ValueError("summary warmup count must match retained warmup records")
        if measured != self.configuration.measured_request_count:
            raise ValueError("measured records must match the configured measured request count")
        if warmup != self.configuration.warmup_request_count:
            raise ValueError("warmup records must match the configured warmup request count")
        allowed_case_ids = set(self.workload_identity.case_ids)
        if any(request.case_id not in allowed_case_ids for request in self.requests):
            raise ValueError("request case IDs must belong to the declared workload identity")
        measured_requests = tuple(
            request for request in self.requests if request.phase is RequestPhase.MEASURED
        )
        successful = sum(request.outcome is RequestOutcome.SUCCESS for request in measured_requests)
        failed = measured - successful
        timeouts = sum(request.outcome is RequestOutcome.TIMEOUT for request in measured_requests)
        cancelled = sum(
            request.outcome is RequestOutcome.CANCELLED for request in measured_requests
        )
        protocol_errors = sum(
            request.outcome in {RequestOutcome.PROTOCOL_ERROR, RequestOutcome.MALFORMED_STREAM}
            for request in measured_requests
        )
        summary_counts = (
            self.summary.terminal_count,
            self.summary.successful_count,
            self.summary.failed_count,
            self.summary.timeout_count,
            self.summary.cancelled_count,
            self.summary.protocol_error_count,
        )
        retained_counts = (
            measured,
            successful,
            failed,
            timeouts,
            cancelled,
            protocol_errors,
        )
        if summary_counts != retained_counts:
            raise ValueError("summary outcome counts must match retained measured records")
        if self.summary.requested_client_concurrency != (
            self.configuration.load_shape.requested_client_concurrency
        ):
            raise ValueError("summary client concurrency must match the run configuration")
        if (
            self.summary.configured_server_maximum_batch_size
            != self.configuration.configured_server_maximum_batch_size
            or self.summary.configured_server_batch_source
            != self.configuration.configured_server_batch_source
        ):
            raise ValueError("summary server batch configuration must match the run configuration")
        if self.configuration.workload_identity != self.workload_identity:
            raise ValueError("configuration and artifact workload identities must match")
        if self.configuration.model_identity != self.model_identity:
            raise ValueError("configuration and artifact model identities must match")
        if self.configuration.tokenizer_identity != self.tokenizer_identity:
            raise ValueError("configuration and artifact tokenizer identities must match")
        fixture_sources = (
            self.runtime_identity.identity_source,
            self.hardware_identity.identity_source,
            self.environment_identity.identity_source,
            self.model_identity.identity_source,
            self.tokenizer_identity.identity_source,
        )
        if any(source is not IdentitySource.SYNTHETIC_FIXTURE for source in fixture_sources):
            raise ValueError("fixture artifacts require synthetic-fixture identities")
        if self.hardware_identity.gpu_model is not None:
            raise ValueError("fixture artifacts cannot assert GPU evidence")
        return self


ESSENTIAL_COMPARISON_FIELDS = (
    "workload_identity.content_sha256",
    "workload_identity.case_ids",
    "workload_identity.ordering_policy",
    "model_identity.model_id",
    "model_identity.exact_revision",
    "tokenizer_identity.tokenizer_id",
    "tokenizer_identity.exact_revision",
    "configuration.sampling",
    "configuration.sampling.maximum_output_tokens",
    "configuration.prompt_transformation",
    "measurement_contract_version",
    "configuration.warmup_policy",
    "configuration.warmup_request_count",
    "configuration.timeout_policy",
    "configuration.load_shape",
    "summary.goodput_slo_policy_sha256",
)


class ComparisonPolicy(StrictModel):
    schema_version: Literal["0.1.0"]
    comparison_contract_version: Literal["0.1.0"]
    baseline_artifact_sha256: Sha256
    candidate_artifact_sha256: Sha256
    comparison_kind: ComparisonKind
    fields_required_identical: tuple[Identifier, ...]
    fields_allowed_to_differ: tuple[Identifier, ...]
    slo_policy_sha256: Sha256
    minimum_successful_requests: PositiveInt
    minimum_metric_samples: PositiveInt
    failure_policy: FailureComparisonPolicy

    @model_validator(mode="after")
    def validate_policy(self) -> Self:
        required = set(self.fields_required_identical)
        allowed = set(self.fields_allowed_to_differ)
        if len(required) != len(self.fields_required_identical):
            raise ValueError("required-identical fields must be unique")
        if len(allowed) != len(self.fields_allowed_to_differ):
            raise ValueError("allowed-difference fields must be unique")
        if required & allowed:
            raise ValueError("a field cannot be both required identical and allowed to differ")
        missing = set(ESSENTIAL_COMPARISON_FIELDS) - required
        if missing:
            raise ValueError(f"required comparison fields missing: {sorted(missing)}")
        if self.comparison_kind is ComparisonKind.CROSS_RUNTIME:
            if "runtime_identity" not in allowed:
                raise ValueError("cross-runtime policies must allow runtime_identity to differ")
        elif "runtime_identity" not in required:
            raise ValueError("non-cross-runtime policies must require runtime_identity")
        if self.comparison_kind is ComparisonKind.CROSS_HARDWARE:
            if "hardware_identity" not in allowed:
                raise ValueError("cross-hardware policies must allow hardware_identity to differ")
        elif "hardware_identity" not in required:
            raise ValueError("non-cross-hardware policies must require hardware_identity")
        return self


class ComparisonMismatch(StrictModel):
    field: Identifier
    baseline_value: str
    candidate_value: str
    allowed: bool


class ComparisonCompatibility(StrictModel):
    compatible: bool
    mismatches: tuple[ComparisonMismatch, ...]
    baseline_sample_requirement_met: bool
    candidate_sample_requirement_met: bool

    @model_validator(mode="after")
    def validate_compatibility(self) -> Self:
        expected = (
            not any(not mismatch.allowed for mismatch in self.mismatches)
            and self.baseline_sample_requirement_met
            and self.candidate_sample_requirement_met
        )
        if self.compatible != expected:
            raise ValueError("compatible flag does not match mismatches and sample requirements")
        return self


class MetricDelta(StrictModel):
    metric: Identifier
    baseline_value: FiniteFloat
    candidate_value: FiniteFloat
    absolute_delta: FiniteFloat
    relative_delta: FiniteFloat | None


class ComparisonReport(StrictModel):
    schema_version: Literal["0.1.0"]
    comparison_contract_version: Literal["0.1.0"]
    evidence_scope: Literal[EvidenceScope.TEST_FIXTURE_ONLY]
    created_at: AwareDatetime
    report_content_sha256: Sha256 | None = None
    policy_sha256: Sha256
    baseline_artifact_sha256: Sha256
    candidate_artifact_sha256: Sha256
    compatibility: ComparisonCompatibility
    deltas: tuple[MetricDelta, ...]

    @model_validator(mode="after")
    def validate_report(self) -> Self:
        offset = self.created_at.utcoffset()
        if offset is None or offset != timedelta(0):
            raise ValueError("created_at must be timezone-aware UTC")
        if not self.compatibility.compatible and self.deltas:
            raise ValueError("incompatible comparisons cannot contain metric deltas")
        return self
