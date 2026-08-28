"""Strict non-executable Stage 2 future-runtime identity contracts.

The models in this module describe evidence that a separately authorized Stage 2B
collector would have to retain.  Stage 2A validates only synthetic instances and
does not import or launch any runtime, model, tokenizer, CUDA, or GPU package.
"""

from __future__ import annotations

from enum import StrEnum
from itertools import pairwise
from pathlib import PurePosixPath
from typing import Final, Literal, Self

from pydantic import Field, model_validator

from llm_inference_systems.canonical import sha256_identity
from llm_inference_systems.contracts import (
    Identifier,
    NonNegativeFloat,
    NonNegativeInt,
    PositiveInt,
    Sha256,
    StrictModel,
)

SNAPSHOT_REPOSITORY: Final = "Qwen/Qwen2.5-0.5B-Instruct"
SNAPSHOT_REVISION: Final = "7ae557604adf67be50417f59c2c2f167def9a775"
SNAPSHOT_SOURCE_URL: Final = (
    "https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct/tree/"
    "7ae557604adf67be50417f59c2c2f167def9a775"
)
SNAPSHOT_NETWORK_SOURCE: Final = SNAPSHOT_SOURCE_URL
REQUIRED_SNAPSHOT_FILES: Final = (
    ".gitattributes",
    "LICENSE",
    "README.md",
    "config.json",
    "generation_config.json",
    "merges.txt",
    "model.safetensors",
    "tokenizer.json",
    "tokenizer_config.json",
    "vocab.json",
)
TOKEN_ENVIRONMENT_VARIABLES: Final = (
    "HF_TOKEN",
    "HUGGINGFACEHUB_API_TOKEN",
    "HUGGINGFACE_HUB_TOKEN",
    "HUGGING_FACE_HUB_TOKEN",
)
PROXY_ENVIRONMENT_VARIABLES: Final = (
    "HTTPS_PROXY",
    "HTTP_PROXY",
    "https_proxy",
    "http_proxy",
)
LAUNCH_ABSENT_ENVIRONMENT_VARIABLES: Final = tuple(
    sorted(
        (
            *TOKEN_ENVIRONMENT_VARIABLES,
            *PROXY_ENVIRONMENT_VARIABLES,
            "VLLM_DISABLE_REQUEST_ID_RANDOMIZATION",
        )
    )
)

ONLINE_ENVIRONMENT: Final = {
    "DO_NOT_TRACK": "1",
    "HF_HUB_DISABLE_IMPLICIT_TOKEN": "1",
    "HF_HUB_DISABLE_TELEMETRY": "1",
}
OFFLINE_TOKENIZER_ENVIRONMENT: Final = {
    **ONLINE_ENVIRONMENT,
    "HF_HUB_OFFLINE": "1",
    "TRANSFORMERS_OFFLINE": "1",
}
OFFLINE_RUNTIME_ENVIRONMENT: Final = {
    **OFFLINE_TOKENIZER_ENVIRONMENT,
    "CUDA_VISIBLE_DEVICES": "0",
    "NO_PROXY": "127.0.0.1,localhost",
    "TOKENIZERS_PARALLELISM": "false",
    "VLLM_DO_NOT_TRACK": "1",
    "VLLM_LOGGING_LEVEL": "INFO",
    "VLLM_NO_USAGE_STATS": "1",
}


def _absolute_public_snapshot_path(value: str) -> PurePosixPath:
    path = PurePosixPath(value)
    if (
        value != path.as_posix()
        or not path.is_absolute()
        or any(part in {"", ".", ".."} for part in path.parts)
        or not value.startswith("/kaggle/working/lis/")
    ):
        raise ValueError("path must be a normalized public-safe absolute Stage 2 path")
    return path


def _relative_inventory_path(value: str) -> PurePosixPath:
    path = PurePosixPath(value)
    if (
        value != path.as_posix()
        or path.is_absolute()
        or not path.parts
        or any(part in {"", ".", ".."} for part in path.parts)
    ):
        raise ValueError("snapshot inventory path must be normalized and relative")
    return path


class Stage2ProcessKind(StrEnum):
    ONLINE_SNAPSHOT_DOWNLOAD = "ONLINE_SNAPSHOT_DOWNLOAD"
    OFFLINE_TOKENIZER_VERIFICATION = "OFFLINE_TOKENIZER_VERIFICATION"
    OFFLINE_VLLM_RESTART_1 = "OFFLINE_VLLM_RESTART_1"
    OFFLINE_VLLM_RESTART_2 = "OFFLINE_VLLM_RESTART_2"
    OFFLINE_VLLM_RESTART_3 = "OFFLINE_VLLM_RESTART_3"


class Stage2ProcessOperation(StrictModel):
    token_false: bool
    local_files_only: bool
    repository_id: Literal["Qwen/Qwen2.5-0.5B-Instruct"] | None
    revision: Literal["7ae557604adf67be50417f59c2c2f167def9a775"] | None
    exact_network_source: str | None
    verified_local_snapshot_path: str
    exits_after_operation: bool

    @model_validator(mode="after")
    def validate_snapshot_path(self) -> Self:
        _absolute_public_snapshot_path(self.verified_local_snapshot_path)
        return self


class Stage2ProcessRecord(StrictModel):
    process_kind: Stage2ProcessKind
    process_identity: Identifier
    process_start_offset_ns: NonNegativeInt
    environment_capture_offset_ns: NonNegativeInt
    first_relevant_import_offset_ns: PositiveInt
    environment: dict[str, str]
    absent_environment_variables: tuple[str, ...]
    operation: Stage2ProcessOperation

    @model_validator(mode="after")
    def validate_process_contract(self) -> Self:
        if not (
            self.process_start_offset_ns
            <= self.environment_capture_offset_ns
            < self.first_relevant_import_offset_ns
        ):
            raise ValueError("process environment must be captured before every relevant import")
        if self.absent_environment_variables != tuple(
            sorted(self.absent_environment_variables)
        ) or len(self.absent_environment_variables) != len(set(self.absent_environment_variables)):
            raise ValueError("absent environment variables must be sorted and unique")

        online = self.process_kind is Stage2ProcessKind.ONLINE_SNAPSHOT_DOWNLOAD
        tokenizer = self.process_kind is Stage2ProcessKind.OFFLINE_TOKENIZER_VERIFICATION
        expected = (
            ONLINE_ENVIRONMENT
            if online
            else OFFLINE_TOKENIZER_ENVIRONMENT
            if tokenizer
            else OFFLINE_RUNTIME_ENVIRONMENT
        )
        dynamic_environment = dict(self.environment)
        hf_home = dynamic_environment.pop("HF_HOME", None)
        vllm_root = dynamic_environment.pop("VLLM_CONFIG_ROOT", None)
        if dynamic_environment != expected:
            raise ValueError("pre-import environment differs from its process-specific contract")
        if online or tokenizer:
            if hf_home is not None or vllm_root is not None:
                raise ValueError("isolated runtime roots apply only to offline vLLM restarts")
        else:
            if hf_home is None or vllm_root is None or hf_home == vllm_root:
                raise ValueError("offline vLLM restart requires distinct isolated runtime roots")
            _absolute_public_snapshot_path(hf_home)
            _absolute_public_snapshot_path(vllm_root)

        required_absent = (
            TOKEN_ENVIRONMENT_VARIABLES
            if online or tokenizer
            else LAUNCH_ABSENT_ENVIRONMENT_VARIABLES
        )
        if self.absent_environment_variables != tuple(sorted(required_absent)):
            raise ValueError("required token/proxy variables are not proven absent")

        operation = self.operation
        if online:
            if (
                not operation.token_false
                or operation.local_files_only
                or operation.repository_id != SNAPSHOT_REPOSITORY
                or operation.revision != SNAPSHOT_REVISION
                or operation.exact_network_source != SNAPSHOT_NETWORK_SOURCE
                or not operation.exits_after_operation
            ):
                raise ValueError("online snapshot operation differs from the pinned contract")
        elif (
            operation.token_false
            or not operation.local_files_only
            or operation.repository_id is not None
            or operation.revision is not None
            or operation.exact_network_source is not None
            or not operation.exits_after_operation
        ):
            raise ValueError("offline process operation may use only the verified local snapshot")
        return self


class Stage2LaunchEnvironment(StrictModel):
    cuda_visible_devices: Literal["0"]
    vllm_logging_level: Literal["INFO"]
    hf_hub_offline: Literal["1"]
    transformers_offline: Literal["1"]
    hf_hub_disable_telemetry: Literal["1"]
    hf_hub_disable_implicit_token: Literal["1"]
    do_not_track: Literal["1"]
    vllm_do_not_track: Literal["1"]
    vllm_no_usage_stats: Literal["1"]
    tokenizers_parallelism: Literal["false"]
    no_proxy: Literal["127.0.0.1,localhost"]
    hf_home: str
    vllm_config_root: str
    absent_variables: tuple[str, ...]

    @model_validator(mode="after")
    def validate_environment(self) -> Self:
        hf_home = _absolute_public_snapshot_path(self.hf_home)
        vllm_root = _absolute_public_snapshot_path(self.vllm_config_root)
        if hf_home == vllm_root:
            raise ValueError("HF_HOME and VLLM_CONFIG_ROOT must be isolated")
        if self.absent_variables != tuple(sorted(LAUNCH_ABSENT_ENVIRONMENT_VARIABLES)):
            raise ValueError("launch environment absence proof differs from the contract")
        return self


class Stage2LaunchSpec(StrictModel):
    schema_version: Literal["0.3.0"]
    executable: Literal["vllm"]
    subcommand: Literal["serve"]
    model_path: str
    tokenizer_path: str
    served_model_name: Literal["qwen2.5-0.5b-instruct-stage2"]
    host: Literal["127.0.0.1"]
    port: Literal[8000]
    dtype: Literal["half"]
    seed: Literal[0]
    max_model_len: Literal[2048]
    gpu_memory_utilization: NonNegativeFloat
    max_num_seqs: Literal[8]
    tensor_parallel_size: Literal[1]
    enforce_eager: Literal[True]
    optimization_level: Literal[0]
    jit_monitor_mode: Literal["error"]
    prefix_caching_enabled: Literal[False]
    generation_config: Literal["vllm"]
    stream_interval: Literal[1]
    request_id_headers_enabled: Literal[True]
    request_logging_enabled: Literal[True]
    per_request_metrics_enabled: Literal[True]
    trust_remote_code: Literal[False]
    quantization: Literal["none"]
    speculative_decoding: Literal["none"]
    lora: Literal["disabled"]
    profiler: Literal["disabled"]
    environment: Stage2LaunchEnvironment
    argv: tuple[str, ...]

    @model_validator(mode="after")
    def validate_launch_identity(self) -> Self:
        model_path = _absolute_public_snapshot_path(self.model_path)
        tokenizer_path = _absolute_public_snapshot_path(self.tokenizer_path)
        if model_path != tokenizer_path:
            raise ValueError("model and tokenizer paths must be the same verified snapshot")
        if self.gpu_memory_utilization != 0.8:
            raise ValueError("GPU-memory utilization differs from the exact 0.80 contract")
        expected_argv = (
            "vllm",
            "serve",
            self.model_path,
            "--tokenizer",
            self.tokenizer_path,
            "--served-model-name",
            "qwen2.5-0.5b-instruct-stage2",
            "--host",
            "127.0.0.1",
            "--port",
            "8000",
            "--dtype",
            "half",
            "--seed",
            "0",
            "--max-model-len",
            "2048",
            "--gpu-memory-utilization",
            "0.80",
            "--max-num-seqs",
            "8",
            "--tensor-parallel-size",
            "1",
            "--enforce-eager",
            "--optimization-level",
            "0",
            "--jit-monitor-mode",
            "error",
            "--no-enable-prefix-caching",
            "--generation-config",
            "vllm",
            "--stream-interval",
            "1",
            "--enable-request-id-headers",
            "--enable-log-requests",
            "--enable-per-request-metrics",
        )
        if self.argv != expected_argv:
            raise ValueError("ordered launch argv differs, conflicts, duplicates, or is incomplete")
        return self


def stage2_launch_identity(spec: Stage2LaunchSpec) -> str:
    return sha256_identity(spec)


class SnapshotFileEntry(StrictModel):
    relative_path: str
    entry_type: Literal["regular_file"]
    byte_size: NonNegativeInt
    observed_byte_size: NonNegativeInt
    sha256: Sha256
    observed_sha256: Sha256

    @model_validator(mode="after")
    def validate_entry(self) -> Self:
        _relative_inventory_path(self.relative_path)
        if self.byte_size != self.observed_byte_size:
            raise ValueError("snapshot file size does not match the observed regular file")
        if self.sha256 != self.observed_sha256:
            raise ValueError("snapshot file hash does not match the observed regular file")
        return self


class SnapshotReadOnlyTransition(StrictModel):
    started_offset_ns: NonNegativeInt
    completed_offset_ns: PositiveInt
    writable_before: Literal[True]
    writable_after: Literal[False]
    verified_regular_file_count: Literal[10]
    verification_evidence_sha256: Sha256

    @model_validator(mode="after")
    def validate_transition(self) -> Self:
        if self.completed_offset_ns <= self.started_offset_ns:
            raise ValueError("read-only transition must have a positive interval")
        return self


class SnapshotSpecialTokenIds(StrictModel):
    bos_token_id: Literal[151643]
    eos_token_id: Literal[151645]
    pad_token_id: Literal[151643]
    unk_token_id: None


class ModelTokenizerSnapshotManifest(StrictModel):
    schema_version: Literal["0.3.0"]
    manifest_name: Literal["model-tokenizer-snapshot-manifest-v0.3.0"]
    repository: Literal["Qwen/Qwen2.5-0.5B-Instruct"]
    revision: Literal["7ae557604adf67be50417f59c2c2f167def9a775"]
    source_url: Literal[
        "https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct/tree/"
        "7ae557604adf67be50417f59c2c2f167def9a775"
    ]
    license: Literal["Apache-2.0"]
    download_process: Stage2ProcessRecord
    download_process_identity_sha256: Sha256
    required_file_allowlist: tuple[str, ...]
    model_content_inventory: tuple[SnapshotFileEntry, ...] = Field(min_length=10, max_length=10)
    hf_local_metadata_inventory: tuple[SnapshotFileEntry, ...] = Field(min_length=1)
    content_identity_sha256: Sha256
    tokenizer_class: Literal["Qwen2TokenizerFast"]
    config_sha256: Sha256
    tokenizer_json_sha256: Sha256
    tokenizer_config_sha256: Sha256
    vocabulary_sha256: Sha256
    merges_sha256: Sha256
    special_token_ids: SnapshotSpecialTokenIds
    chat_template_sha256: Sha256 | None
    chat_template_absent: bool
    read_only_transition: SnapshotReadOnlyTransition
    offline_tokenizer_verification_process: Stage2ProcessRecord
    offline_tokenizer_verification_process_identity_sha256: Sha256
    snapshot_root_path: str
    snapshot_root_identity_sha256: Sha256

    @model_validator(mode="after")
    def validate_manifest(self) -> Self:
        _absolute_public_snapshot_path(self.snapshot_root_path)
        if (
            self.download_process.process_kind is not Stage2ProcessKind.ONLINE_SNAPSHOT_DOWNLOAD
            or self.offline_tokenizer_verification_process.process_kind
            is not Stage2ProcessKind.OFFLINE_TOKENIZER_VERIFICATION
        ):
            raise ValueError("snapshot manifest process kinds differ from the required sequence")
        if (
            self.download_process.operation.verified_local_snapshot_path != self.snapshot_root_path
            or self.offline_tokenizer_verification_process.operation.verified_local_snapshot_path
            != self.snapshot_root_path
        ):
            raise ValueError("snapshot processes are not bound to the exact snapshot root")
        if self.download_process_identity_sha256 != sha256_identity(self.download_process):
            raise ValueError("download-process identity does not reconstruct")
        if self.offline_tokenizer_verification_process_identity_sha256 != sha256_identity(
            self.offline_tokenizer_verification_process
        ):
            raise ValueError("offline tokenizer process identity does not reconstruct")
        if self.required_file_allowlist != REQUIRED_SNAPSHOT_FILES:
            raise ValueError("required snapshot allowlist differs from pinned read-only metadata")
        content_paths = tuple(entry.relative_path for entry in self.model_content_inventory)
        metadata_paths = tuple(entry.relative_path for entry in self.hf_local_metadata_inventory)
        for paths in (content_paths, metadata_paths):
            if paths != tuple(sorted(paths)) or len(paths) != len(set(paths)):
                raise ValueError("snapshot inventories must be sorted and unique")
            if len({path.casefold() for path in paths}) != len(paths):
                raise ValueError("snapshot inventory contains a case-collision ambiguity")
        if content_paths != REQUIRED_SNAPSHOT_FILES:
            raise ValueError("model content inventory differs from the required-file allowlist")
        if any(path.startswith(".cache/huggingface/") for path in content_paths) or any(
            not path.startswith(".cache/huggingface/") for path in metadata_paths
        ):
            raise ValueError("Hugging Face local metadata and model content are mixed")
        all_paths = (*content_paths, *metadata_paths)
        if len({path.casefold() for path in all_paths}) != len(all_paths):
            raise ValueError("snapshot inventories collide across content and metadata")

        content_by_path = {entry.relative_path: entry for entry in self.model_content_inventory}
        expected_hashes = {
            "config.json": self.config_sha256,
            "tokenizer.json": self.tokenizer_json_sha256,
            "tokenizer_config.json": self.tokenizer_config_sha256,
            "vocab.json": self.vocabulary_sha256,
            "merges.txt": self.merges_sha256,
        }
        if any(content_by_path[path].sha256 != digest for path, digest in expected_hashes.items()):
            raise ValueError("tokenizer/config/vocabulary hashes differ from content inventory")
        expected_content_identity = sha256_identity(self.model_content_inventory)
        if self.content_identity_sha256 != expected_content_identity:
            raise ValueError("snapshot content identity does not reconstruct")
        expected_root_identity = sha256_identity(
            {
                "content_identity_sha256": self.content_identity_sha256,
                "snapshot_root_path": self.snapshot_root_path,
            }
        )
        if self.snapshot_root_identity_sha256 != expected_root_identity:
            raise ValueError("snapshot-root identity does not reconstruct")
        if self.download_process_identity_sha256 == (
            self.offline_tokenizer_verification_process_identity_sha256
        ):
            raise ValueError("download and offline tokenizer verification require fresh processes")
        if self.chat_template_absent != (self.chat_template_sha256 is None):
            raise ValueError("chat-template hash or explicit absence is required")
        return self


def snapshot_content_identity(entries: tuple[SnapshotFileEntry, ...]) -> str:
    return sha256_identity(entries)


def snapshot_root_identity(path: str, content_identity_sha256: str) -> str:
    return sha256_identity(
        {
            "content_identity_sha256": content_identity_sha256,
            "snapshot_root_path": path,
        }
    )


def validate_process_sequence(records: tuple[Stage2ProcessRecord, ...]) -> None:
    if tuple(record.process_kind for record in records) != tuple(Stage2ProcessKind):
        raise ValueError("exact online, tokenizer, and three runtime process order is required")
    identities = tuple(record.process_identity for record in records)
    if len(identities) != len(set(identities)):
        raise ValueError("Stage 2 process identities must be fresh and distinct")
    snapshot_paths = tuple(record.operation.verified_local_snapshot_path for record in records)
    if len(set(snapshot_paths)) != 1:
        raise ValueError("every process must bind the same verified local snapshot path")


class GpuMemorySample(StrictModel):
    observation_offset_ns: NonNegativeInt
    allocated_bytes: NonNegativeInt


def gpu_memory_tolerance_bytes(samples: tuple[GpuMemorySample, ...]) -> int:
    if len(samples) != 5:
        raise ValueError("GPU-memory stability requires exactly five samples")
    first = samples[0].allocated_bytes
    return max((first + 99) // 100, 67_108_864)


def validate_gpu_memory_stability(samples: tuple[GpuMemorySample, ...]) -> int:
    tolerance = gpu_memory_tolerance_bytes(samples)
    if any(
        right.observation_offset_ns - left.observation_offset_ns < 200_000_000
        for left, right in pairwise(samples)
    ):
        raise ValueError("GPU-memory samples require at least 200-ms spacing")
    values = tuple(sample.allocated_bytes for sample in samples)
    if max(values) - min(values) > tolerance:
        raise ValueError("GPU-memory samples exceed the documented stability tolerance")
    return tolerance
