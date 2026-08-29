"""Strict Stage 2A contracts for a future real-runtime evidence protocol.

These models are exercised only with CPU fixtures in Stage 2A.  They do not import,
launch, or assert execution of vLLM, a tokenizer, a model, CUDA, or a GPU.
"""

from __future__ import annotations

import hashlib
from datetime import timedelta
from enum import StrEnum
from itertools import pairwise
from pathlib import PurePosixPath
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

STAGE2_SCHEMA_VERSION: Final[Literal["0.3.0"]] = "0.3.0"
STAGE2_MEASUREMENT_PROTOCOL_VERSION: Final[Literal["0.3.0"]] = "0.3.0"
STAGE2_MODEL_NAME: Final = "qwen2.5-0.5b-instruct-stage2"
STAGE2_PROMPT_TOKEN_COUNT: Final = 64
STAGE2_OUTPUT_TOKEN_COUNT: Final = 32
STAGE2_MEASURED_REQUEST_COUNT: Final = 16
STAGE2_RESTART_COUNT: Final = 3

ExternalRequestId = Annotated[
    str,
    StringConstraints(pattern=r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$"),
]
TokenId = Annotated[int, Field(ge=0)]

REQUIRED_LAUNCH_ARGUMENTS: Final = (
    "--host=127.0.0.1",
    "--tensor-parallel-size=1",
    "--enforce-eager",
    "--optimization-level=0",
    "--jit-monitor-mode=error",
    "--no-enable-prefix-caching",
    "--stream-interval=1",
    "--enable-request-id-headers",
    "--enable-log-requests",
    "--enable-per-request-metrics",
)


class Stage2EvidenceScope(StrEnum):
    TEST_FIXTURE_ONLY = "TEST_FIXTURE_ONLY"
    FUTURE_REAL_RUNTIME = "FUTURE_REAL_RUNTIME"


class LoopbackEndpoint(StrictModel):
    host: Literal["127.0.0.1"]
    port: Annotated[int, Field(ge=1, le=65_535)]
    completions_path: Literal["/v1/completions"] = "/v1/completions"
    metrics_path: Literal["/metrics"] = "/metrics"

    @property
    def completions_url(self) -> str:
        return f"http://127.0.0.1:{self.port}{self.completions_path}"

    @property
    def metrics_url(self) -> str:
        return f"http://127.0.0.1:{self.port}{self.metrics_path}"


class Stage2ReportingPolicy(StrictModel):
    p50_descriptive: Literal[True] = True
    p95_descriptive: Literal[True] = True
    p99_prohibited: Literal[True] = True
    exact_sample_counts_required: Literal[True] = True
    restart_grouping_required: Literal[True] = True
    goodput_or_capacity_advancement_prohibited: Literal[True] = True


class Stage2RunConfiguration(StrictModel):
    schema_version: Literal["0.3.0"]
    measurement_protocol_version: Literal["0.3.0"]
    evidence_scope: Literal[Stage2EvidenceScope.TEST_FIXTURE_ONLY]
    endpoint: LoopbackEndpoint
    model: Literal["qwen2.5-0.5b-instruct-stage2"]
    sent_prompt_token_count: Literal[64]
    output_token_count: Literal[32]
    measured_request_count: Literal[16]
    required_restart_count: Literal[3]
    launch_arguments: tuple[str, ...]
    reporting_policy: Stage2ReportingPolicy

    @model_validator(mode="after")
    def validate_launch_contract(self) -> Self:
        if self.launch_arguments != REQUIRED_LAUNCH_ARGUMENTS:
            raise ValueError("launch arguments differ from the frozen Stage 2 contract")
        return self


class Stage2StreamOptions(StrictModel):
    include_usage: Literal[True]


class Stage2CompletionRequest(StrictModel):
    model: Literal["qwen2.5-0.5b-instruct-stage2"]
    prompt: tuple[TokenId, ...] = Field(min_length=64, max_length=64)
    request_id: ExternalRequestId
    stream: Literal[True]
    stream_options: Stage2StreamOptions
    return_token_ids: Literal[True]
    stream_interval: Literal[1]
    add_special_tokens: Literal[False]
    temperature: Literal[0]
    top_p: Literal[1]
    seed: Literal[0]
    n: Literal[1]
    max_tokens: Literal[32]
    min_tokens: Literal[32]
    ignore_eos: Literal[True]
    echo: Literal[False]


class Stage2CancellationRequest(StrictModel):
    model: Literal["qwen2.5-0.5b-instruct-stage2"]
    prompt: tuple[TokenId, ...] = Field(min_length=64, max_length=64)
    request_id: ExternalRequestId
    stream: Literal[True]
    stream_options: Stage2StreamOptions
    return_token_ids: Literal[True]
    stream_interval: Literal[1]
    add_special_tokens: Literal[False]
    temperature: Literal[0]
    top_p: Literal[1]
    seed: Literal[0]
    n: Literal[1]
    max_tokens: Literal[512]
    min_tokens: Literal[512]
    ignore_eos: Literal[True]
    echo: Literal[False]


class Stage2RequestEnvelope(StrictModel):
    x_request_id: ExternalRequestId
    body: Stage2CompletionRequest

    @model_validator(mode="after")
    def validate_identity(self) -> Self:
        if self.x_request_id != self.body.request_id:
            raise ValueError("X-Request-Id and JSON request_id differ")
        return self


class RawLogRecord(StrictModel):
    source_stream_id: Identifier
    record_ordinal: NonNegativeInt
    byte_start: NonNegativeInt
    byte_end: PositiveInt
    observation_offset_ns: NonNegativeInt
    raw_record: str
    raw_record_sha256: Sha256

    @model_validator(mode="after")
    def validate_raw_record(self) -> Self:
        encoded = self.raw_record.encode("utf-8")
        if self.byte_end <= self.byte_start:
            raise ValueError("raw log record byte interval must be positive")
        if self.byte_end - self.byte_start != len(encoded):
            raise ValueError("raw log record byte offsets do not match retained bytes")
        if self.raw_record_sha256 != hashlib.sha256(encoded).hexdigest():
            raise ValueError("raw log record SHA-256 does not match retained bytes")
        return self


class RequestIdentityChain(StrictModel):
    external_base_id: ExternalRequestId
    response_body_id: str
    serving_item_id: str
    internal_engine_id: str
    request_received_log: RawLogRecord
    request_add_log: RawLogRecord
    external_abort_log: RawLogRecord | None
    internal_abort_log: RawLogRecord | None

    @model_validator(mode="after")
    def validate_chain(self) -> Self:
        expected_response = f"cmpl-{self.external_base_id}"
        expected_item = f"{expected_response}-0"
        if self.response_body_id != expected_response or self.serving_item_id != expected_item:
            raise ValueError("request identity chain does not preserve the external ID")
        prefix = f"{expected_item}-"
        if not self.internal_engine_id.startswith(prefix):
            raise ValueError("internal engine ID does not preserve the serving item ID")
        suffix = self.internal_engine_id.removeprefix(prefix)
        if len(suffix) != 8 or any(character not in "0123456789abcdef" for character in suffix):
            raise ValueError("internal engine ID requires an eight-character lowercase hex suffix")
        records = tuple(
            record
            for record in (
                self.request_received_log,
                self.request_add_log,
                self.external_abort_log,
                self.internal_abort_log,
            )
            if record is not None
        )
        if len({(record.source_stream_id, record.record_ordinal) for record in records}) != len(
            records
        ):
            raise ValueError("request identity log records are duplicated")
        if len({record.source_stream_id for record in records}) != 1:
            raise ValueError("request identity log records do not share one source stream")
        if (self.external_abort_log is None) != (self.internal_abort_log is None):
            raise ValueError("external and internal abort records must be retained together")
        if (
            tuple(record.record_ordinal for record in records)
            != tuple(sorted(record.record_ordinal for record in records))
            or tuple(record.observation_offset_ns for record in records)
            != tuple(sorted(record.observation_offset_ns for record in records))
            or tuple(record.byte_start for record in records)
            != tuple(sorted(record.byte_start for record in records))
            or any(right.byte_start < left.byte_end for left, right in pairwise(records))
        ):
            raise ValueError("request identity log records are not monotonic and nonoverlapping")
        required_fragments = [
            (self.request_received_log, f"Received request {expected_item}:"),
            (self.request_add_log, f"Added request {self.internal_engine_id}."),
        ]
        if self.external_abort_log is not None:
            required_fragments.append(
                (self.external_abort_log, f"Request {expected_item} aborted.")
            )
        if self.internal_abort_log is not None:
            required_fragments.append(
                (self.internal_abort_log, f"Aborted request(s) {self.internal_engine_id}.")
            )
        if any(fragment not in record.raw_record for record, fragment in required_fragments):
            raise ValueError("raw log record does not match the request identity chain")
        return self


class Stage2Usage(StrictModel):
    prompt_tokens: Literal[64]
    completion_tokens: Literal[32]
    total_tokens: Literal[96]


class Stage2PerRequestMetrics(StrictModel):
    time_to_first_token_ms: NonNegativeFloat | None
    generation_time_ms: NonNegativeFloat | None
    queue_time_ms: NonNegativeFloat | None
    mean_itl_ms: NonNegativeFloat | None
    tokens_per_second: NonNegativeFloat | None


class Stage2TokenEvent(StrictModel):
    observation_offset_ns: NonNegativeInt
    output_token_ids: tuple[TokenId, ...]
    text: str
    finish_reason: Literal["length"] | None
    prompt_token_ids: tuple[TokenId, ...] | None = None


class Stage2TimingRecord(StrictModel):
    dispatch_offset_ns: NonNegativeInt
    response_headers_offset_ns: NonNegativeInt
    first_response_body_bytes_offset_ns: NonNegativeInt
    first_output_token_offset_ns: NonNegativeInt
    generation_terminal_offset_ns: NonNegativeInt
    usage_terminal_offset_ns: NonNegativeInt
    protocol_terminal_offset_ns: NonNegativeInt
    transport_terminal_offset_ns: NonNegativeInt

    @model_validator(mode="after")
    def validate_chronology(self) -> Self:
        values = (
            self.dispatch_offset_ns,
            self.response_headers_offset_ns,
            self.first_response_body_bytes_offset_ns,
            self.first_output_token_offset_ns,
            self.generation_terminal_offset_ns,
            self.usage_terminal_offset_ns,
            self.protocol_terminal_offset_ns,
            self.transport_terminal_offset_ns,
        )
        if tuple(sorted(values)) != values:
            raise ValueError("Stage 2 timing boundaries are not monotonic")
        if not (
            self.generation_terminal_offset_ns
            < self.usage_terminal_offset_ns
            < self.protocol_terminal_offset_ns
            < self.transport_terminal_offset_ns
        ):
            raise ValueError("generation, usage, protocol, and transport must be strict")
        return self


class MetricAvailability(StrictModel):
    value_ns: NonNegativeFloat | None
    unavailable_reason: Literal["GROUPED_TOKEN_EVENT", "INSUFFICIENT_OUTPUT_TOKENS"] | None

    @model_validator(mode="after")
    def validate_availability(self) -> Self:
        if (self.value_ns is None) != (self.unavailable_reason is not None):
            raise ValueError("metric value and unavailable reason must be exclusive")
        return self


class Stage2RequestEvidence(StrictModel):
    fixture_identity_sha256: Sha256 | None
    request_identity_chain_sha256: Sha256
    external_request_id: ExternalRequestId
    response_request_id: str
    serving_item_request_id: str
    internal_engine_request_id: str
    sent_prompt_token_ids: tuple[TokenId, ...] = Field(min_length=64, max_length=64)
    returned_prompt_token_ids: tuple[TokenId, ...] = Field(min_length=64, max_length=64)
    token_events: tuple[Stage2TokenEvent, ...] = Field(min_length=1)
    final_output_token_ids: tuple[TokenId, ...] = Field(min_length=32, max_length=32)
    finish_reason: Literal["length"]
    terminal_event_carried_token_ids: bool
    usage: Stage2Usage
    local_prompt_token_count: NonNegativeInt | None
    local_output_token_count: NonNegativeInt | None
    server_per_request_metrics: Stage2PerRequestMetrics
    disagreements: tuple[str, ...]
    output_text: str
    output_text_sha256: Sha256
    timing: Stage2TimingRecord
    client_generation_tpot: MetricAvailability
    token_observation_itl: MetricAvailability
    stream_output_gap_ns: tuple[NonNegativeInt, ...]

    @model_validator(mode="after")
    def validate_reconciliation(self) -> Self:
        base = self.external_request_id
        if self.response_request_id != f"cmpl-{base}":
            raise ValueError("response body ID does not match the external request ID")
        if self.serving_item_request_id != f"cmpl-{base}-0":
            raise ValueError("serving item ID does not match the external request ID")
        if not self.internal_engine_request_id.startswith(f"cmpl-{base}-0-"):
            raise ValueError("internal engine ID does not preserve the request chain")
        suffix = self.internal_engine_request_id.removeprefix(f"cmpl-{base}-0-")
        if len(suffix) != 8 or any(character not in "0123456789abcdef" for character in suffix):
            raise ValueError("internal engine ID requires an eight-character lowercase hex suffix")
        if self.sent_prompt_token_ids != self.returned_prompt_token_ids:
            raise ValueError("returned prompt IDs differ from sent prompt IDs")
        prompt_events = tuple(
            event for event in self.token_events if event.prompt_token_ids is not None
        )
        if (
            len(prompt_events) != 1
            or prompt_events[0] is not self.token_events[0]
            or prompt_events[0].prompt_token_ids != self.returned_prompt_token_ids
        ):
            raise ValueError("returned prompt IDs require one matching first token event")
        accumulated = tuple(
            token_id for event in self.token_events for token_id in event.output_token_ids
        )
        if accumulated != self.final_output_token_ids:
            raise ValueError("final output IDs do not reconstruct from token events")
        offsets = tuple(event.observation_offset_ns for event in self.token_events)
        if offsets != tuple(sorted(offsets)):
            raise ValueError("token-event observations are not monotonic")
        terminals = tuple(event for event in self.token_events if event.finish_reason is not None)
        if len(terminals) != 1 or terminals[0] is not self.token_events[-1]:
            raise ValueError("token events require one final generation terminal")
        terminal = terminals[0]
        if self.terminal_event_carried_token_ids != bool(terminal.output_token_ids):
            raise ValueError("terminal token-bearing evidence differs from the terminal event")
        token_bearing = tuple(event for event in self.token_events if event.output_token_ids)
        if not token_bearing:
            raise ValueError("request evidence contains no output-token event")
        if self.timing.first_output_token_offset_ns != token_bearing[0].observation_offset_ns:
            raise ValueError("first output-token timing differs from token evidence")
        if self.timing.generation_terminal_offset_ns != terminal.observation_offset_ns:
            raise ValueError("generation-terminal timing differs from token evidence")
        expected_text = "".join(event.text for event in self.token_events)
        if (
            self.output_text != expected_text
            or self.output_text_sha256 != hashlib.sha256(expected_text.encode("utf-8")).hexdigest()
        ):
            raise ValueError("output text or text SHA-256 does not reconstruct")
        expected_gaps = tuple(
            right.observation_offset_ns - left.observation_offset_ns
            for left, right in pairwise(token_bearing)
        )
        if self.stream_output_gap_ns != expected_gaps:
            raise ValueError("stream output gaps do not reconstruct from token evidence")
        grouped = any(len(event.output_token_ids) > 1 for event in token_bearing)
        if grouped:
            if (
                self.client_generation_tpot.unavailable_reason != "GROUPED_TOKEN_EVENT"
                or self.token_observation_itl.unavailable_reason != "GROUPED_TOKEN_EVENT"
            ):
                raise ValueError("grouped token events require TPOT and ITL unavailability")
        else:
            expected_tpot = (
                self.timing.generation_terminal_offset_ns - self.timing.first_output_token_offset_ns
            ) / 31
            expected_itl = sum(expected_gaps) / len(expected_gaps)
            if self.client_generation_tpot.value_ns != expected_tpot:
                raise ValueError("client-generation TPOT does not reconstruct")
            if self.token_observation_itl.value_ns != expected_itl:
                raise ValueError("token-observation ITL does not reconstruct")
        expected_disagreements: list[str] = []
        if self.local_prompt_token_count is not None and self.local_prompt_token_count != 64:
            expected_disagreements.append("LOCAL_PROMPT_COUNT_DIFFERS_FROM_SERVER_USAGE")
        if self.local_output_token_count is not None and self.local_output_token_count != 32:
            expected_disagreements.append("LOCAL_OUTPUT_COUNT_DIFFERS_FROM_SERVER_USAGE")
        if self.disagreements != tuple(expected_disagreements):
            raise ValueError("retained token-count disagreements are incomplete or unexpected")
        return self


class RuntimePhase(StrEnum):
    LIVE_RESOURCE_AUDIT = "LIVE_RESOURCE_AUDIT"
    ENVIRONMENT_LOCK_VERIFICATION = "ENVIRONMENT_LOCK_VERIFICATION"
    OFFLINE_SNAPSHOT_VERIFICATION = "OFFLINE_SNAPSHOT_VERIFICATION"
    RUNTIME_PROCESS_START = "RUNTIME_PROCESS_START"
    MODEL_WEIGHT_LOAD = "MODEL_WEIGHT_LOAD"
    ENGINE_READINESS_HEALTH = "ENGINE_READINESS_HEALTH"
    JIT_COMPILATION_STATE = "JIT_COMPILATION_STATE"
    CUDA_GRAPH_STATE = "CUDA_GRAPH_STATE"
    ALLOCATOR_KV_STABILIZATION = "ALLOCATOR_KV_STABILIZATION"
    EXCLUDED_STABILIZATION_REQUESTS = "EXCLUDED_STABILIZATION_REQUESTS"
    EXCLUDED_SHAPE_WARMUPS = "EXCLUDED_SHAPE_WARMUPS"
    CANCELLATION_PROBE_DRAIN = "CANCELLATION_PROBE_DRAIN"
    STEADY_STATE_GATE = "STEADY_STATE_GATE"
    MEASURED_WINDOW = "MEASURED_WINDOW"
    FINAL_METRICS_DRAIN = "FINAL_METRICS_DRAIN"
    SHUTDOWN = "SHUTDOWN"
    NO_RESIDUAL_PROCESS_VERIFICATION = "NO_RESIDUAL_PROCESS_VERIFICATION"


RUNTIME_PHASE_ORDER: Final = tuple(RuntimePhase)


class RuntimePhaseRecord(StrictModel):
    phase: RuntimePhase
    started_offset_ns: NonNegativeInt
    ended_offset_ns: NonNegativeInt
    passed: Literal[True]
    post_warmup_jit_observed: bool = False
    evidence_kind: Identifier
    evidence_identity_sha256: Sha256
    evidence_references: tuple[Identifier, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_interval(self) -> Self:
        if self.ended_offset_ns <= self.started_offset_ns:
            raise ValueError("runtime phase end must be strictly after its start")
        return self


class RuntimeImplementationRecord(StrictModel):
    runtime_package_name: Identifier
    resolved_model_implementation: Identifier | None
    resolved_implementation_source: Literal["RUNTIME_REPORTED", "DIRECTLY_OBSERVED", "UNAVAILABLE"]

    @model_validator(mode="after")
    def validate_resolution(self) -> Self:
        unavailable = self.resolved_implementation_source == "UNAVAILABLE"
        if unavailable != (self.resolved_model_implementation is None):
            raise ValueError("resolved implementation and its source must be retained together")
        return self


class ResourceEstimate(StrictModel):
    bytes: NonNegativeInt
    source: Identifier


class ResourceBudgetInputs(StrictModel):
    runtime_and_cuda_package_download: ResourceEstimate
    expected_installed_environment: ResourceEstimate
    model_tokenizer_snapshot: ResourceEstimate
    temporary_extraction_and_cache: ResourceEstimate


class ResourceBudgetResult(StrictModel):
    required_setup_bytes: NonNegativeInt
    required_free_before_setup: PositiveInt


class ProviderShape(StrictModel):
    operating_system: Literal["Linux"]
    architecture: Literal["x86_64"]
    logical_cpu_count: Annotated[int, Field(ge=4)]
    memory_total_bytes: Annotated[int, Field(ge=28_000_000_000)]
    filesystem_total_bytes: Annotated[int, Field(ge=19_000_000_000)]
    initial_free_bytes: Annotated[int, Field(ge=14_000_000_000)]
    post_setup_free_bytes: Annotated[int, Field(ge=5_000_000_000)]
    physical_gpu_models: tuple[Literal["NVIDIA T4"], Literal["NVIDIA T4"]]
    runtime_visible_gpu_count: Literal[1]


class BundleState(StrEnum):
    INCOMPLETE = "INCOMPLETE"
    INVALID = "INVALID"
    COMMITTED = "COMMITTED"


class BundleFileEntry(StrictModel):
    path: str
    sha256: Sha256
    size: NonNegativeInt

    @model_validator(mode="after")
    def validate_path(self) -> Self:
        path = PurePosixPath(self.path)
        if self.path != path.as_posix() or path.is_absolute() or ".." in path.parts:
            raise ValueError("bundle inventory path is unsafe")
        if not path.parts:
            raise ValueError("bundle inventory path is empty")
        return self


class Stage2ManifestBoundFile(StrictModel):
    """Path/size/digest identity for evidence retained by a committed manifest."""

    path: str
    sha256: Sha256
    size: NonNegativeInt

    @model_validator(mode="after")
    def validate_path(self) -> Self:
        path = PurePosixPath(self.path)
        if (
            self.path != path.as_posix()
            or path.is_absolute()
            or not path.parts
            or any(part in {"", ".", ".."} for part in path.parts)
        ):
            raise ValueError("manifest-bound evidence path must be normalized and relative")
        return self


class Stage2BundleManifest(StrictModel):
    schema_version: Literal["0.3.0"]
    measurement_protocol_version: Literal["0.3.0"]
    state: Literal[BundleState.COMMITTED]
    repetition_index: Annotated[int, Field(ge=1, le=3)]
    source_commit: Annotated[str, StringConstraints(pattern=r"^[0-9a-f]{40}$")]
    created_at_utc: AwareDatetime
    files: tuple[BundleFileEntry, ...] = Field(min_length=1)
    reconstruction_sha256: Sha256

    @model_validator(mode="after")
    def validate_manifest(self) -> Self:
        if self.created_at_utc.utcoffset() != timedelta(0):
            raise ValueError("bundle timestamp must use UTC")
        paths = tuple(entry.path for entry in self.files)
        if paths != tuple(sorted(paths)) or len(paths) != len(set(paths)):
            raise ValueError("bundle file inventory must be sorted and unique")
        if "evidence-manifest.json" in paths:
            raise ValueError("bundle manifest cannot inventory itself")
        return self


class ExecutionLockStatus(StrEnum):
    COMPLETE = "COMPLETE"
    INCOMPLETE = "INCOMPLETE"
    BLOCKED_BINARY_RETRIEVAL_AUTHORIZATION_REQUIRED = (
        "BLOCKED_BINARY_RETRIEVAL_AUTHORIZATION_REQUIRED"
    )


class VllmExecutionLockArtifact(StrictModel):
    package: Literal["vllm"]
    version: Literal["0.28.0"]
    filename: Literal["vllm-0.28.0+cu129-cp38-abi3-manylinux_2_28_x86_64.whl"]
    source_url: Literal[
        "https://github.com/vllm-project/vllm/releases/download/v0.28.0/"
        "vllm-0.28.0%2Bcu129-cp38-abi3-manylinux_2_28_x86_64.whl"
    ]
    sha256: Literal["8ec943b66a0c6b4351d0778e99d7bacfca5788dd8eedd49425092bacb61c4397"]
    hash_source: Literal["CONTROLLER_AUTHORIZED_SPEC"]


class TorchExecutionLockArtifact(StrictModel):
    package: Literal["torch"]
    version: Literal["2.13.0+cu129"]
    filename: Literal["torch-2.13.0+cu129-cp313-cp313-manylinux_2_28_x86_64.whl"]
    source_url: Literal[
        "https://download-r2.pytorch.org/whl/cu129/"
        "torch-2.13.0%2Bcu129-cp313-cp313-manylinux_2_28_x86_64.whl"
    ]
    sha256: Literal["6e3bcf183e3096db45bf539dc21f820963074986ece7a56550714f12863c76af"]
    hash_source: Literal["OFFICIAL_INDEX_METADATA"]


class TorchaudioExecutionLockArtifact(StrictModel):
    package: Literal["torchaudio"]
    version: Literal["2.11.0+cu129"]
    filename: Literal["torchaudio-2.11.0+cu129-cp313-cp313-manylinux_2_28_x86_64.whl"]
    source_url: Literal[
        "https://download-r2.pytorch.org/whl/cu129/"
        "torchaudio-2.11.0%2Bcu129-cp313-cp313-manylinux_2_28_x86_64.whl"
    ]
    sha256: Literal["45103fac849ffee337976ff19eac81725b3396e2c18e3f48ed92ba7669cb32d7"]
    hash_source: Literal["OFFICIAL_INDEX_METADATA"]


class TorchvisionExecutionLockArtifact(StrictModel):
    package: Literal["torchvision"]
    version: Literal["0.28.0+cu129"]
    filename: Literal["torchvision-0.28.0+cu129-cp313-cp313-manylinux_2_28_x86_64.whl"]
    source_url: Literal[
        "https://download-r2.pytorch.org/whl/cu129/"
        "torchvision-0.28.0%2Bcu129-cp313-cp313-manylinux_2_28_x86_64.whl"
    ]
    sha256: None
    hash_source: Literal["UNAVAILABLE"]


class Stage2ExecutionLock(StrictModel):
    schema_version: Literal["0.3.0"]
    status: Literal[ExecutionLockStatus.BLOCKED_BINARY_RETRIEVAL_AUTHORIZATION_REQUIRED]
    python_version: Literal["3.13.15"]
    uv_version: Literal["0.12.5"]
    vllm_version: Literal["0.28.0"]
    vllm_git_revision: Literal["2cf0a6915ce544dc493a0990f2ea38d81601128a"]
    pytorch_index: Literal["https://download.pytorch.org/whl/cu129"]
    flashinfer_index: Literal["https://flashinfer.ai/whl/cu129"]
    qwen_model_repository: Literal["Qwen/Qwen2.5-0.5B-Instruct"]
    qwen_snapshot_revision: Literal["7ae557604adf67be50417f59c2c2f167def9a775"]
    qwen_snapshot_source_url: Literal[
        "https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct/tree/"
        "7ae557604adf67be50417f59c2c2f167def9a775"
    ]
    artifacts: tuple[
        VllmExecutionLockArtifact,
        TorchExecutionLockArtifact,
        TorchaudioExecutionLockArtifact,
        TorchvisionExecutionLockArtifact,
    ]
    preimport_distribution_version_command: tuple[str, ...]
    installed: Literal[False]
    executed: Literal[False]
    resolver_lock_claimed_complete: Literal[False]
    unresolved: tuple[
        Literal[
            "The selected torchvision wheel SHA-256 is not exposed by official index metadata; "
            "computing it would require downloading the 9,290,444-byte binary."
        ]
    ]

    @model_validator(mode="after")
    def validate_lock(self) -> Self:
        identities = tuple((item.package, item.version) for item in self.artifacts)
        required = {
            ("vllm", "0.28.0"),
            ("torch", "2.13.0+cu129"),
            ("torchaudio", "2.11.0+cu129"),
            ("torchvision", "0.28.0+cu129"),
        }
        if len(identities) != 4 or set(identities) != required:
            raise ValueError("execution lock requires exactly one of each exact package")
        expected_artifacts = {
            "vllm": (
                "vllm-0.28.0+cu129-cp38-abi3-manylinux_2_28_x86_64.whl",
                "https://github.com/vllm-project/vllm/releases/download/v0.28.0/"
                "vllm-0.28.0%2Bcu129-cp38-abi3-manylinux_2_28_x86_64.whl",
            ),
            "torch": (
                "torch-2.13.0+cu129-cp313-cp313-manylinux_2_28_x86_64.whl",
                "https://download-r2.pytorch.org/whl/cu129/"
                "torch-2.13.0%2Bcu129-cp313-cp313-manylinux_2_28_x86_64.whl",
            ),
            "torchaudio": (
                "torchaudio-2.11.0+cu129-cp313-cp313-manylinux_2_28_x86_64.whl",
                "https://download-r2.pytorch.org/whl/cu129/"
                "torchaudio-2.11.0%2Bcu129-cp313-cp313-manylinux_2_28_x86_64.whl",
            ),
            "torchvision": (
                "torchvision-0.28.0+cu129-cp313-cp313-manylinux_2_28_x86_64.whl",
                "https://download-r2.pytorch.org/whl/cu129/"
                "torchvision-0.28.0%2Bcu129-cp313-cp313-manylinux_2_28_x86_64.whl",
            ),
        }
        for artifact in self.artifacts:
            if (artifact.filename, artifact.source_url) != expected_artifacts[artifact.package]:
                raise ValueError("execution-lock filename or official source URL differs")
        expected_known_hashes = {
            "vllm": (
                "8ec943b66a0c6b4351d0778e99d7bacfca5788dd8eedd49425092bacb61c4397",
                "CONTROLLER_AUTHORIZED_SPEC",
            ),
            "torch": (
                "6e3bcf183e3096db45bf539dc21f820963074986ece7a56550714f12863c76af",
                "OFFICIAL_INDEX_METADATA",
            ),
            "torchaudio": (
                "45103fac849ffee337976ff19eac81725b3396e2c18e3f48ed92ba7669cb32d7",
                "OFFICIAL_INDEX_METADATA",
            ),
        }
        for artifact in self.artifacts:
            if (
                artifact.package in expected_known_hashes
                and (
                    artifact.sha256,
                    artifact.hash_source,
                )
                != expected_known_hashes[artifact.package]
            ):
                raise ValueError("execution-lock artifact hash or provenance differs")
        missing_hash = any(item.sha256 is None for item in self.artifacts)
        if not missing_hash:
            raise ValueError("a binary-retrieval block requires a missing hash and unresolved item")
        if self.preimport_distribution_version_command != (
            "python",
            "-c",
            "import importlib.metadata; print(importlib.metadata.version('vllm'))",
        ):
            raise ValueError("pre-import version verification command differs")
        return self
