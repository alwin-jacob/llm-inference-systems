"""Strict Stage 1 contracts isolated from the historical v0.1.0 schemas."""

from __future__ import annotations

from datetime import timedelta
from enum import StrEnum
from typing import Annotated, Final, Literal, Self

from pydantic import AwareDatetime, Field, StringConstraints, model_validator

from llm_inference_systems.contracts import (
    Identifier,
    NonNegativeFloat,
    NonNegativeInt,
    PositiveInt,
    Sha256,
    StrictModel,
)

STAGE1_SCHEMA_VERSION: Final[Literal["0.2.0"]] = "0.2.0"
STAGE1_MEASUREMENT_CONTRACT_VERSION: Final[Literal["0.2.0"]] = "0.2.0"
STAGE1_COMPARISON_CONTRACT_VERSION: Final[Literal["0.2.0"]] = "0.2.0"
FIXTURE_PROTOCOL: Final[Literal["OPENAI_COMPLETIONS_SSE_FIXTURE_SUBSET"]] = (
    "OPENAI_COMPLETIONS_SSE_FIXTURE_SUBSET"
)
FIXTURE_PROTOCOL_VERSION: Final[Literal["1.0.0"]] = "1.0.0"
TIMING_DISCLAIMER: Final = (
    "These measurements are loopback fixture measurements used to verify the benchmark "
    "harness and measurement semantics. They are not LLM-serving, model, runtime, GPU, "
    "or production-performance measurements."
)

PositiveFloat = Annotated[float, Field(gt=0, allow_inf_nan=False)]
FixtureText = Annotated[str, StringConstraints(min_length=1, max_length=16_384)]
SourceCommit = Annotated[
    str,
    StringConstraints(
        pattern=r"^(?:[0-9a-f]{40}|ARCHIVE_NO_GIT|WORKTREE_DIRTY)$",
    ),
]
Stage1Phase = Literal["WARMUP", "MEASURED"]
Stage1MetricName = Literal[
    "END_TO_END_SUCCESS_NS",
    "TTFT_NS",
    "TPOT_NS",
    "ITL_NS",
    "OBSERVED_TOKEN_SPAN_PER_INTERVAL_NS",
]


class EvidenceBoundary(StrictModel):
    evidence_scope: Literal["TEST_FIXTURE_ONLY"] = "TEST_FIXTURE_ONLY"
    synthetic_fixture: Literal[True] = True
    real_runtime_execution: Literal[False] = False
    model_execution: Literal[False] = False
    tokenizer_execution: Literal[False] = False
    gpu_execution: Literal[False] = False
    cuda_execution: Literal[False] = False
    performance_claim_allowed: Literal[False] = False
    historical_authentication_effect: Literal["NONE"] = "NONE"


class Stage1TerminalClass(StrEnum):
    SUCCESS = "SUCCESS"
    FAILED = "FAILED"
    TIMEOUT = "TIMEOUT"
    CANCELLED = "CANCELLED"


class Stage1FailureKind(StrEnum):
    HTTP_STATUS = "HTTP_STATUS"
    PROTOCOL_MALFORMED_STREAM = "PROTOCOL_MALFORMED_STREAM"
    TIMEOUT = "TIMEOUT"
    TRANSPORT = "TRANSPORT"
    TOKEN_ACCOUNTING = "TOKEN_ACCOUNTING"
    CANCELLED = "CANCELLED"
    UNEXPECTED = "UNEXPECTED"


class FailureOrigin(StrEnum):
    HTTP_CLIENT = "HTTP_CLIENT"
    HTTP_STATUS = "HTTP_STATUS"
    SSE_PARSER = "SSE_PARSER"
    TOKEN_ACCOUNTING = "TOKEN_ACCOUNTING"
    LOAD_GENERATOR = "LOAD_GENERATOR"
    HARNESS = "HARNESS"


class FixtureScenario(StrEnum):
    SUCCESS_THREE_SINGLE_TOKEN_CHUNKS_A = "SUCCESS_THREE_SINGLE_TOKEN_CHUNKS_A"
    SUCCESS_THREE_SINGLE_TOKEN_CHUNKS_B = "SUCCESS_THREE_SINGLE_TOKEN_CHUNKS_B"
    SUCCESS_MULTI_TOKEN_EVENT = "SUCCESS_MULTI_TOKEN_EVENT"
    SUCCESS_FIRST_BODY_BEFORE_FIRST_TOKEN = "SUCCESS_FIRST_BODY_BEFORE_FIRST_TOKEN"
    SUCCESS_SINGLE_OUTPUT_TOKEN = "SUCCESS_SINGLE_OUTPUT_TOKEN"
    MALFORMED_AFTER_PARTIAL_OUTPUT = "MALFORMED_AFTER_PARTIAL_OUTPUT"
    HTTP_ERROR = "HTTP_ERROR"
    TIMEOUT_AFTER_PARTIAL_BODY = "TIMEOUT_AFTER_PARTIAL_BODY"
    WARMUP_SUCCESS = "WARMUP_SUCCESS"


class FixtureActionKind(StrEnum):
    SSE_COMMENT = "SSE_COMMENT"
    SSE_TOKEN_EVENT = "SSE_TOKEN_EVENT"
    SSE_MALFORMED_DATA = "SSE_MALFORMED_DATA"
    SSE_DONE = "SSE_DONE"
    HTTP_ERROR = "HTTP_ERROR"
    STALL = "STALL"


class FixtureAction(StrictModel):
    kind: FixtureActionKind
    delay_ms_before: NonNegativeInt = 0
    text: FixtureText | None = None
    http_status: Annotated[int, Field(ge=400, le=599)] | None = None
    stall_seconds: PositiveFloat | None = None

    @model_validator(mode="after")
    def validate_action_shape(self) -> Self:
        text_kinds = {
            FixtureActionKind.SSE_COMMENT,
            FixtureActionKind.SSE_TOKEN_EVENT,
            FixtureActionKind.SSE_MALFORMED_DATA,
            FixtureActionKind.HTTP_ERROR,
        }
        if (self.text is not None) != (self.kind in text_kinds):
            raise ValueError("fixture action text presence does not match action kind")
        if (self.http_status is not None) != (self.kind is FixtureActionKind.HTTP_ERROR):
            raise ValueError("HTTP status is valid only for HTTP_ERROR actions")
        if (self.stall_seconds is not None) != (self.kind is FixtureActionKind.STALL):
            raise ValueError("stall_seconds is valid only for STALL actions")
        return self


class FixtureCaseDefinition(StrictModel):
    case_id: Identifier
    scenario: FixtureScenario
    input_text: FixtureText
    maximum_output_tokens: PositiveInt
    expected_terminal_class: Stage1TerminalClass
    expected_output_token_count: NonNegativeInt
    actions: tuple[FixtureAction, ...] = Field(min_length=1, max_length=16)


class FixtureDefinition(StrictModel):
    schema_version: Literal["0.2.0"]
    fixture_id: Identifier
    protocol: Literal["OPENAI_COMPLETIONS_SSE_FIXTURE_SUBSET"]
    protocol_version: Literal["1.0.0"]
    model_sentinel: Literal["fixture-no-model"]
    cases: tuple[FixtureCaseDefinition, ...] = Field(min_length=9, max_length=9)

    @model_validator(mode="after")
    def validate_cases(self) -> Self:
        case_ids = tuple(case.case_id for case in self.cases)
        if len(set(case_ids)) != len(case_ids):
            raise ValueError("fixture case IDs must be unique")
        scenarios = tuple(case.scenario for case in self.cases)
        if len(set(scenarios)) != len(scenarios):
            raise ValueError("fixture scenarios must be unique")
        return self


class Stage1WorkloadCase(StrictModel):
    case_id: Identifier
    prompt: FixtureText
    expected_terminal_class: Stage1TerminalClass
    expected_output_token_count: NonNegativeInt


class Stage1WorkloadDefinition(StrictModel):
    schema_version: Literal["0.2.0"]
    name: Identifier
    description: Annotated[str, StringConstraints(min_length=1, max_length=1_024)]
    ordering_policy: Literal["DECLARED"]
    prompt_transformation: Literal["fixture-token-markers-v1"]
    warmup_case: Stage1WorkloadCase
    measured_cases: tuple[Stage1WorkloadCase, ...] = Field(min_length=8, max_length=8)

    @model_validator(mode="after")
    def validate_case_ids(self) -> Self:
        case_ids = (self.warmup_case.case_id, *(case.case_id for case in self.measured_cases))
        if len(set(case_ids)) != len(case_ids):
            raise ValueError("workload case IDs must be unique")
        return self


class Stage1LoadShape(StrictModel):
    kind: Literal["CLOSED_LOOP"]
    requested_client_concurrency: Literal[2]
    scheduling_policy: Literal["DECLARED_ORDER_NEXT_AVAILABLE_CLIENT"]


class Stage1TimeoutPolicy(StrictModel):
    connect_timeout_seconds: PositiveFloat
    write_timeout_seconds: PositiveFloat
    pool_timeout_seconds: PositiveFloat
    read_timeout_seconds: PositiveFloat


class Stage1SLODefinition(StrictModel):
    policy_name: Identifier
    successful_end_to_end_threshold_ns: PositiveInt


class Stage1RunConfiguration(StrictModel):
    schema_version: Literal["0.2.0"]
    measurement_contract_version: Literal["0.2.0"]
    workload_path: Literal["examples/workloads/streaming-fixture-v1.json"]
    workload_sha256: Sha256
    fixture_path: Literal["examples/fixtures/streaming-fixture-v1.json"]
    fixture_sha256: Sha256
    load_shape: Stage1LoadShape
    timeout_policy: Stage1TimeoutPolicy
    slo: Stage1SLODefinition
    warmup_request_count: Literal[1]
    measured_request_count: Literal[8]
    model: Literal["fixture-no-model"]
    temperature: Annotated[float, Field(ge=0, le=0, allow_inf_nan=False)]
    stream: Literal[True]
    token_count_source: Literal["FIXTURE_EXACT"]
    configured_server_maximum_batch_size: None = None
    configured_server_batch_source: None = None


class Stage1TimingRecord(StrictModel):
    dispatch_offset_ns: NonNegativeInt
    response_headers_offset_ns: NonNegativeInt | None = None
    first_response_body_bytes_offset_ns: NonNegativeInt | None = None
    first_output_token_offset_ns: NonNegativeInt | None = None
    last_output_token_offset_ns: NonNegativeInt | None = None
    terminal_offset_ns: NonNegativeInt

    @model_validator(mode="after")
    def validate_chronology(self) -> Self:
        observations = (
            self.response_headers_offset_ns,
            self.first_response_body_bytes_offset_ns,
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
            raise ValueError("observations must fall inside the request interval")
        present = [value for value in observations if value is not None]
        if present != sorted(present):
            raise ValueError("request observations must be chronological")
        if (self.first_output_token_offset_ns is None) != (
            self.last_output_token_offset_ns is None
        ):
            raise ValueError("first and last output-token boundaries must appear together")
        return self


class Stage1FailureRecord(StrictModel):
    kind: Stage1FailureKind
    origin: FailureOrigin
    error_code: Identifier
    occurred_offset_ns: NonNegativeInt
    timeout_policy: Stage1TimeoutPolicy | None = None

    @model_validator(mode="after")
    def validate_timeout_policy(self) -> Self:
        if (self.timeout_policy is not None) != (self.kind is Stage1FailureKind.TIMEOUT):
            raise ValueError("timeout policy must be retained only for TIMEOUT")
        return self


class Stage1RequestRecord(StrictModel):
    boundary: EvidenceBoundary
    request_id: Identifier
    case_id: Identifier
    phase: Stage1Phase
    terminal_class: Stage1TerminalClass
    http_status: Annotated[int, Field(ge=100, le=599)] | None
    timing: Stage1TimingRecord
    input_token_count: NonNegativeInt
    output_token_count: NonNegativeInt
    token_count_source: Literal["FIXTURE_EXACT"]
    token_event_delta_counts: tuple[PositiveInt, ...]
    token_event_observation_offsets_ns: tuple[NonNegativeInt, ...]
    per_token_observation_complete: bool
    slo_satisfied: bool
    failure: Stage1FailureRecord | None = None

    @model_validator(mode="after")
    def validate_terminal_semantics(self) -> Self:
        if self.terminal_class is Stage1TerminalClass.SUCCESS:
            if self.failure is not None:
                raise ValueError("successful requests cannot retain a failure")
            if self.http_status is None or not 200 <= self.http_status < 300:
                raise ValueError("successful requests require a successful HTTP status")
        else:
            if self.failure is None:
                raise ValueError("non-success requests require a failure record")
            if self.slo_satisfied:
                raise ValueError("non-success requests cannot satisfy the SLO")
            if self.terminal_class is Stage1TerminalClass.TIMEOUT and (
                self.failure.kind is not Stage1FailureKind.TIMEOUT
            ):
                raise ValueError("TIMEOUT terminal requires TIMEOUT failure kind")
            if self.terminal_class is Stage1TerminalClass.CANCELLED and (
                self.failure.kind is not Stage1FailureKind.CANCELLED
            ):
                raise ValueError("CANCELLED terminal requires CANCELLED failure kind")
            if self.terminal_class is Stage1TerminalClass.FAILED and self.failure.kind in {
                Stage1FailureKind.TIMEOUT,
                Stage1FailureKind.CANCELLED,
            }:
                raise ValueError("FAILED terminal cannot use timeout or cancellation failure kind")
        if len(self.token_event_delta_counts) != len(self.token_event_observation_offsets_ns):
            raise ValueError("token deltas and event observations must align")
        if sum(self.token_event_delta_counts) != self.output_token_count:
            raise ValueError("token-event deltas must equal the exact fixture output count")
        if self.output_token_count:
            if (
                self.timing.first_output_token_offset_ns
                != (self.token_event_observation_offsets_ns[0])
            ):
                raise ValueError("first output-token boundary must match the first token event")
            if (
                self.timing.last_output_token_offset_ns
                != (self.token_event_observation_offsets_ns[-1])
            ):
                raise ValueError("last output-token boundary must match the last token event")
        if self.per_token_observation_complete and any(
            delta != 1 for delta in self.token_event_delta_counts
        ):
            raise ValueError("complete per-token observation requires one token per event")
        if self.failure is not None and not (
            self.timing.dispatch_offset_ns
            <= self.failure.occurred_offset_ns
            <= self.timing.terminal_offset_ns
        ):
            raise ValueError("failure occurrence must fall inside the request interval")
        return self


class StreamEvidenceKind(StrEnum):
    CLIENT_REQUEST_STARTED = "CLIENT_REQUEST_STARTED"
    RAW_BODY_CHUNK = "RAW_BODY_CHUNK"
    SSE_COMMENT = "SSE_COMMENT"
    SSE_TOKEN_EVENT = "SSE_TOKEN_EVENT"
    SSE_DONE = "SSE_DONE"
    REQUEST_TERMINAL = "REQUEST_TERMINAL"
    CLIENT_REQUEST_ENDED = "CLIENT_REQUEST_ENDED"


class StreamEvidenceRecord(StrictModel):
    boundary: EvidenceBoundary
    sequence: NonNegativeInt
    request_id: Identifier
    case_id: Identifier
    phase: Stage1Phase
    kind: StreamEvidenceKind
    observation_offset_ns: NonNegativeInt
    raw_chunk_sequence: NonNegativeInt | None = None
    raw_bytes_base64: str | None = None
    raw_byte_count: NonNegativeInt | None = None
    raw_bytes_sha256: Sha256 | None = None
    sse_event_sequence: NonNegativeInt | None = None
    sse_data: str | None = None
    token_delta_count: NonNegativeInt | None = None
    terminal_class: Stage1TerminalClass | None = None
    failure_kind: Stage1FailureKind | None = None

    @model_validator(mode="after")
    def validate_kind_payload(self) -> Self:
        raw_fields = (
            self.raw_chunk_sequence,
            self.raw_bytes_base64,
            self.raw_byte_count,
            self.raw_bytes_sha256,
        )
        if self.kind is StreamEvidenceKind.RAW_BODY_CHUNK:
            if any(value is None for value in raw_fields):
                raise ValueError("raw body records require reversible bytes and integrity fields")
        elif any(value is not None for value in raw_fields):
            raise ValueError("raw body fields are valid only for RAW_BODY_CHUNK")
        event_kinds = {
            StreamEvidenceKind.SSE_COMMENT,
            StreamEvidenceKind.SSE_TOKEN_EVENT,
            StreamEvidenceKind.SSE_DONE,
        }
        if (self.sse_event_sequence is not None) != (self.kind in event_kinds):
            raise ValueError("SSE sequence presence does not match evidence kind")
        if self.kind is StreamEvidenceKind.SSE_TOKEN_EVENT:
            if self.sse_data is None or self.token_delta_count is None:
                raise ValueError("token events require data and a token delta count")
            if self.token_delta_count <= 0:
                raise ValueError("token-event delta must be positive")
        elif self.token_delta_count is not None:
            raise ValueError("token delta is valid only for SSE_TOKEN_EVENT")
        if self.kind is StreamEvidenceKind.REQUEST_TERMINAL:
            if self.terminal_class is None:
                raise ValueError("terminal evidence requires a terminal class")
            if (self.failure_kind is None) != (self.terminal_class is Stage1TerminalClass.SUCCESS):
                raise ValueError("terminal failure kind does not match terminal class")
            if self.terminal_class is Stage1TerminalClass.TIMEOUT and (
                self.failure_kind is not Stage1FailureKind.TIMEOUT
            ):
                raise ValueError("TIMEOUT terminal event requires TIMEOUT failure kind")
            if self.terminal_class is Stage1TerminalClass.CANCELLED and (
                self.failure_kind is not Stage1FailureKind.CANCELLED
            ):
                raise ValueError("CANCELLED terminal event requires CANCELLED failure kind")
            if self.terminal_class is Stage1TerminalClass.FAILED and self.failure_kind in {
                Stage1FailureKind.TIMEOUT,
                Stage1FailureKind.CANCELLED,
            }:
                raise ValueError("FAILED terminal event cannot use timeout or cancellation kind")
        elif self.terminal_class is not None or self.failure_kind is not None:
            raise ValueError("terminal fields are valid only for REQUEST_TERMINAL")
        return self


class ServerEventKind(StrEnum):
    REQUEST_ACCEPTED = "REQUEST_ACCEPTED"
    RESPONSE_HEADERS_SENT = "RESPONSE_HEADERS_SENT"
    SSE_COMMENT_SENT = "SSE_COMMENT_SENT"
    SSE_TOKEN_EVENT_SENT = "SSE_TOKEN_EVENT_SENT"
    SSE_MALFORMED_DATA_SENT = "SSE_MALFORMED_DATA_SENT"
    SSE_DONE_SENT = "SSE_DONE_SENT"
    HTTP_ERROR_SENT = "HTTP_ERROR_SENT"
    STALL_STARTED = "STALL_STARTED"
    CLIENT_DISCONNECTED = "CLIENT_DISCONNECTED"
    REQUEST_HANDLER_ENDED = "REQUEST_HANDLER_ENDED"


class ServerEventRecord(StrictModel):
    boundary: EvidenceBoundary
    sequence: NonNegativeInt
    request_id: Identifier
    case_id: Identifier
    kind: ServerEventKind
    observation_offset_ns: NonNegativeInt
    action_index: NonNegativeInt | None = None
    token_delta_count: NonNegativeInt | None = None
    http_status: Annotated[int, Field(ge=100, le=599)] | None = None


class RateValue(StrictModel):
    numerator: NonNegativeInt
    denominator: NonNegativeInt
    value: Annotated[float, Field(ge=0, le=1, allow_inf_nan=False)] | None

    @model_validator(mode="after")
    def validate_value(self) -> Self:
        if self.denominator == 0:
            if self.value is not None:
                raise ValueError("zero-denominator rates must be unavailable")
        elif self.value != self.numerator / self.denominator:
            raise ValueError("rate value must equal numerator divided by denominator")
        return self


class Stage1MetricDistribution(StrictModel):
    metric: Stage1MetricName
    unit: Literal["nanoseconds"]
    percentile_algorithm: Literal["HYNDMAN_FAN_TYPE_7"]
    sample_count: NonNegativeInt
    p50: NonNegativeFloat | None
    p95: NonNegativeFloat | None
    p99: NonNegativeFloat | None
    denominator: Identifier
    inclusion_semantics: Annotated[str, StringConstraints(min_length=1, max_length=512)]
    unavailable_reason: Literal["NO_SAMPLES"] | None = None

    @model_validator(mode="after")
    def validate_distribution(self) -> Self:
        percentiles = (self.p50, self.p95, self.p99)
        if self.sample_count == 0:
            if any(value is not None for value in percentiles):
                raise ValueError("empty distributions cannot contain percentiles")
            if self.unavailable_reason != "NO_SAMPLES":
                raise ValueError("empty distributions require NO_SAMPLES")
        elif any(value is None for value in percentiles) or self.unavailable_reason is not None:
            raise ValueError("nonempty distributions require percentiles and no reason")
        return self


class Stage1RunSummary(StrictModel):
    boundary: EvidenceBoundary
    timing_disclaimer: Literal[
        "These measurements are loopback fixture measurements used to verify the benchmark "
        "harness and measurement semantics. They are not LLM-serving, model, runtime, GPU, "
        "or production-performance measurements."
    ]
    measurement_window_ns: PositiveInt
    attempted_measured_requests: NonNegativeInt
    terminal_measured_requests: NonNegativeInt
    successful_measured_requests: NonNegativeInt
    failed_non_timeout_measured_requests: NonNegativeInt
    timed_out_measured_requests: NonNegativeInt
    cancelled_measured_requests: NonNegativeInt
    warmup_record_count: NonNegativeInt
    warmup_excluded_count: NonNegativeInt
    requested_client_concurrency: PositiveInt
    observed_max_client_concurrency: NonNegativeInt
    configured_server_maximum_batch_size: None
    observed_server_batch_size: None
    offered_request_rate: NonNegativeFloat
    terminal_request_rate: NonNegativeFloat
    successful_request_throughput: NonNegativeFloat
    output_token_throughput: NonNegativeFloat
    total_token_throughput: NonNegativeFloat
    slo_satisfying_request_count: NonNegativeInt
    goodput: NonNegativeFloat
    failure_rate: RateValue
    timeout_rate: RateValue
    end_to_end_success: Stage1MetricDistribution
    ttft: Stage1MetricDistribution
    tpot: Stage1MetricDistribution
    itl: Stage1MetricDistribution
    observed_token_span_per_interval: Stage1MetricDistribution

    @model_validator(mode="after")
    def validate_counts(self) -> Self:
        if self.terminal_measured_requests != self.attempted_measured_requests:
            raise ValueError("all attempted fixture requests must reach a terminal state")
        if (
            self.successful_measured_requests
            + self.failed_non_timeout_measured_requests
            + self.timed_out_measured_requests
            + self.cancelled_measured_requests
            != self.terminal_measured_requests
        ):
            raise ValueError("terminal populations must reconcile")
        if self.warmup_record_count != self.warmup_excluded_count:
            raise ValueError("every retained warmup must be excluded")
        if self.observed_max_client_concurrency > self.requested_client_concurrency:
            raise ValueError("observed concurrency cannot exceed its requested bound")
        return self


class Stage1ExecutionManifest(StrictModel):
    boundary: EvidenceBoundary
    schema_version: Literal["0.2.0"]
    measurement_contract_version: Literal["0.2.0"]
    fixture_protocol: Literal["OPENAI_COMPLETIONS_SSE_FIXTURE_SUBSET"]
    fixture_protocol_version: Literal["1.0.0"]
    run_id: Identifier
    source_commit: SourceCommit
    package_version: Literal["0.2.0"]
    python_version: Identifier
    package_lock_sha256: Sha256
    environment_identity: Identifier
    runtime_identity: Literal["deterministic-loopback-http-fixture"]
    model_identity: Literal["fixture-no-model"]
    tokenizer_identity: Literal["not-executed-fixture-exact-markers"]
    started_at_utc: AwareDatetime
    ended_at_utc: AwareDatetime
    run_duration_ns: PositiveInt
    loopback_host: Literal["127.0.0.1"]
    loopback_port: Annotated[int, Field(ge=1, le=65_535)]
    workload: Stage1WorkloadDefinition
    configuration: Stage1RunConfiguration
    fixture: FixtureDefinition
    workload_sha256: Sha256
    configuration_sha256: Sha256
    fixture_sha256: Sha256
    raw_file_sha256: dict[str, Sha256]
    summary_sha256: Sha256
    semantic_fingerprint: Sha256
    content_sha256: Sha256 | None = None

    @model_validator(mode="after")
    def validate_utc_and_inputs(self) -> Self:
        for timestamp in (self.started_at_utc, self.ended_at_utc):
            if timestamp.utcoffset() != timedelta(0):
                raise ValueError("audit timestamps must use UTC")
        if self.ended_at_utc < self.started_at_utc:
            raise ValueError("run end cannot precede run start")
        if set(self.raw_file_sha256) != {
            "requests.jsonl",
            "stream-events.jsonl",
            "server-events.jsonl",
        }:
            raise ValueError("manifest must retain exactly the three raw file digests")
        return self


class Stage1ComparisonPolicy(StrictModel):
    schema_version: Literal["0.2.0"]
    comparison_contract_version: Literal["0.2.0"]
    policy_name: Identifier
    require_matching_semantic_fingerprint: Literal[True]
    require_matching_input_identities: Literal[True]
    require_matching_terminal_taxonomy: Literal[True]
    allow_failure_count_increase: Literal[False]
    allow_timeout_count_increase: Literal[False]
    performance_gating_enabled: Literal[False]


class SemanticCheck(StrictModel):
    check: Identifier
    passed: bool


class Stage1ComparisonReport(StrictModel):
    boundary: EvidenceBoundary
    schema_version: Literal["0.2.0"]
    comparison_contract_version: Literal["0.2.0"]
    timing_disclaimer: Literal[
        "These measurements are loopback fixture measurements used to verify the benchmark "
        "harness and measurement semantics. They are not LLM-serving, model, runtime, GPU, "
        "or production-performance measurements."
    ]
    created_at_utc: AwareDatetime
    policy_sha256: Sha256
    baseline_content_sha256: Sha256
    candidate_content_sha256: Sha256
    baseline_semantic_fingerprint: Sha256
    candidate_semantic_fingerprint: Sha256
    semantic_fingerprints_match: bool
    compatible: bool
    policy_passed: bool
    performance_interpretation_allowed: Literal[False]
    checks: tuple[SemanticCheck, ...] = Field(min_length=1)
    content_sha256: Sha256 | None = None

    @model_validator(mode="after")
    def validate_report(self) -> Self:
        if self.created_at_utc.utcoffset() != timedelta(0):
            raise ValueError("comparison timestamp must use UTC")
        checks_pass = all(check.passed for check in self.checks)
        if self.policy_passed != (self.compatible and checks_pass):
            raise ValueError("policy result must reconcile with compatibility and checks")
        return self
