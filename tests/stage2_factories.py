from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from typing import Literal, cast

from pydantic import BaseModel

from llm_inference_systems.canonical import sha256_identity
from llm_inference_systems.stage2_attestation import (
    COMPONENT_ORDER,
    AttestedComponentIdentity,
    CudaBackedExecutionAttestation,
    FutureRealRuntimeAttestation,
    LinuxEnvironmentManifest,
    NvidiaT4DeviceIdentity,
    NvidiaT4ResourceAttestation,
    PerRequestMetricsAttestation,
    PrometheusMeasurementAttestation,
    PublicSafetyAttestation,
    RequestIdentityAttestation,
    RuntimePackageExecutionLockAttestation,
    ServerRestartAttestation,
    ServerRestartIdentity,
)
from llm_inference_systems.stage2_contracts import (
    BundleFileEntry,
    BundleState,
    RequestIdentityChain,
    RuntimePhaseRecord,
    Stage2BundleManifest,
    Stage2EvidenceScope,
    Stage2RequestEvidence,
)
from llm_inference_systems.stage2_control import (
    PHASE_EVIDENCE_KINDS,
    CancellationProbe,
    FirstGenerationTokenEvidence,
    ProcessExitEvidence,
    ResidualStateEvidence,
    Stage2RuntimeControlEvidence,
    bundle_manifest_sha256,
    evaluate_cancellation,
)
from llm_inference_systems.stage2_prometheus import (
    PrometheusSnapshot,
    derive_counter_delta,
    parse_prometheus_snapshot,
)
from llm_inference_systems.stage2_protocol import (
    Stage2StreamValidator,
    correlate_request_logs,
    retain_raw_log_records,
)
from llm_inference_systems.stage2_runtime import (
    LAUNCH_ABSENT_ENVIRONMENT_VARIABLES,
    OFFLINE_RUNTIME_ENVIRONMENT,
    OFFLINE_TOKENIZER_ENVIRONMENT,
    ONLINE_ENVIRONMENT,
    REQUIRED_SNAPSHOT_FILES,
    SNAPSHOT_NETWORK_SOURCE,
    TOKEN_ENVIRONMENT_VARIABLES,
    GpuMemorySample,
    ModelTokenizerSnapshotManifest,
    SnapshotFileEntry,
    SnapshotReadOnlyTransition,
    SnapshotSpecialTokenIds,
    Stage2LaunchEnvironment,
    Stage2LaunchSpec,
    Stage2ProcessKind,
    Stage2ProcessOperation,
    Stage2ProcessRecord,
    snapshot_content_identity,
    snapshot_root_identity,
    stage2_launch_identity,
)

PROMPT = tuple(range(64))
SNAPSHOT_PATH = "/kaggle/working/lis/model-snapshot"
FIXTURE_IDENTITY = "f" * 64


def _data(value: dict[str, object]) -> bytes:
    return f"data: {json.dumps(value, sort_keys=True, separators=(',', ':'))}\n\n".encode()


def make_log_chain(
    external_id: str = "stage2-fixture-001",
    *,
    cancellation: bool = False,
    first_observation_offset_ns: int = 0,
    abort_observation_offset_ns: int | None = None,
    fixture_marked: bool = True,
) -> RequestIdentityChain:
    internal = f"cmpl-{external_id}-0-deadbeef"
    provenance_marker = "TEST_FIXTURE_ONLY" if fixture_marked else "SYNTHETIC_FUTURE_SHAPE_ONLY"
    lines = [
        f"Received request cmpl-{external_id}-0: params: {provenance_marker}.",
        f"Added request {internal}.",
    ]
    if cancellation:
        lines.extend(
            (
                f"Request cmpl-{external_id}-0 aborted.",
                f"Aborted request(s) {internal}.",
            )
        )
    records = retain_raw_log_records(
        tuple(lines),
        source_stream_id=("stage2-fixture-log" if fixture_marked else "synthetic-future-shape-log"),
        first_observation_offset_ns=first_observation_offset_ns,
        observation_offsets_ns=(
            (
                first_observation_offset_ns,
                first_observation_offset_ns + 1,
                abort_observation_offset_ns,
                abort_observation_offset_ns + 1,
            )
            if cancellation and abort_observation_offset_ns is not None
            else None
        ),
    )
    return correlate_request_logs(external_id, records, cancellation=cancellation)


def make_request_evidence(
    *,
    fixture_identity_sha256: str | None,
    external_id: str = "stage2-fixture-001",
    base_offset_ns: int = 0,
    output_token_ids: tuple[int, ...] = tuple(range(32)),
) -> Stage2RequestEvidence:
    if len(output_token_ids) != 32:
        raise ValueError("Stage 2 fixture request requires exactly 32 output token IDs")
    fixture_marked = fixture_identity_sha256 is not None
    validator = Stage2StreamValidator(
        external_base_id=external_id,
        sent_prompt_token_ids=PROMPT,
        dispatch_offset_ns=base_offset_ns,
        fixture_identity_sha256=fixture_identity_sha256,
    )
    validator.accept_response_headers(external_id, base_offset_ns + 10)
    for event_index, token in enumerate(output_token_ids):
        choice: dict[str, object] = {
            "index": 0,
            "text": (
                f"<fixture-{token}>" if fixture_marked else f"<synthetic-future-shape-{token}>"
            ),
            "token_ids": [token],
            "finish_reason": "length" if event_index == 31 else None,
        }
        if event_index == 0:
            choice["prompt_token_ids"] = list(PROMPT)
        validator.feed(
            _data({"id": f"cmpl-{external_id}", "choices": [choice]}),
            base_offset_ns + 20 + event_index,
        )
    validator.feed(
        _data(
            {
                "id": f"cmpl-{external_id}",
                "choices": [],
                "usage": {
                    "prompt_tokens": 64,
                    "completion_tokens": 32,
                    "total_tokens": 96,
                },
                "metrics": {
                    "time_to_first_token_ms": 1.0,
                    "generation_time_ms": 2.0,
                    "queue_time_ms": 0.0,
                    "mean_itl_ms": 0.031,
                    "tokens_per_second": 16.0,
                },
            }
        ),
        base_offset_ns + 60,
    )
    validator.feed(b"data: [DONE]\n\n", base_offset_ns + 70)
    return validator.close_transport(
        base_offset_ns + 80,
        identity_chain=make_log_chain(
            external_id,
            first_observation_offset_ns=base_offset_ns + 1,
            fixture_marked=fixture_marked,
        ),
    )


def prometheus_exposition(
    *,
    prompt: int = 0,
    generation: int = 0,
    abort: int = 0,
    length: int = 0,
    stop: int = 0,
    error: int = 0,
    repetition: int = 0,
    running: int = 0,
    waiting: int = 0,
    prefix_queries: int = 0,
    prefix_hits: int = 0,
) -> str:
    labels = 'engine="0",model_name="qwen2.5-0.5b-instruct-stage2"'
    lines = [
        f"vllm:num_requests_running{{{labels}}} {running}.0",
        f"vllm:num_requests_waiting{{{labels}}} {waiting}.0",
        f"vllm:kv_cache_usage_perc{{{labels}}} 0.25",
        f"vllm:prompt_tokens_total{{{labels}}} {prompt}.0",
        f"vllm:generation_tokens_total{{{labels}}} {generation}.0",
        f"vllm:num_preemptions_total{{{labels}}} 0.0",
        f"vllm:prefix_cache_queries_total{{{labels}}} {prefix_queries}.0",
        f"vllm:prefix_cache_hits_total{{{labels}}} {prefix_hits}.0",
    ]
    for reason, value in (
        ("abort", abort),
        ("error", error),
        ("length", length),
        ("repetition", repetition),
        ("stop", stop),
    ):
        success_labels = (
            f'engine="0",finished_reason="{reason}",model_name="qwen2.5-0.5b-instruct-stage2"'
        )
        lines.append(f"vllm:request_success_total{{{success_labels}}} {value}.0")
    return "\n".join(lines) + "\n"


def make_snapshot(
    offset_ns: int,
    *,
    process_start_id: str = "fixture-process",
    prompt: int = 0,
    generation: int = 0,
    abort: int = 0,
    length: int = 0,
    stop: int = 0,
    error: int = 0,
    repetition: int = 0,
    running: int = 0,
    waiting: int = 0,
    prefix_queries: int = 0,
    prefix_hits: int = 0,
) -> PrometheusSnapshot:
    return parse_prometheus_snapshot(
        prometheus_exposition(
            prompt=prompt,
            generation=generation,
            abort=abort,
            length=length,
            stop=stop,
            error=error,
            repetition=repetition,
            running=running,
            waiting=waiting,
            prefix_queries=prefix_queries,
            prefix_hits=prefix_hits,
        ),
        process_start_id=process_start_id,
        scrape_wall_clock_utc=datetime(2026, 8, 28, tzinfo=UTC),
        scrape_monotonic_offset_ns=offset_ns,
    )


def make_cancellation_probe(
    *,
    base_offset_ns: int = 0,
    abort_delta: int = 1,
    process_start_id: str = "fixture-process",
    fixture_marked: bool = True,
    external_id: str = "E_cancel",
) -> CancellationProbe:
    second = 1_000_000_000
    pre = tuple(
        make_snapshot(
            base_offset_ns + index * 100_000_000,
            process_start_id=process_start_id,
        )
        for index in range(10)
    )
    drain = tuple(
        make_snapshot(
            base_offset_ns + 1_300_000_000 + index * 100_000_000,
            process_start_id=process_start_id,
            prompt=64,
            generation=1,
            abort=abort_delta,
        )
        for index in range(10)
    )
    stable = tuple(
        make_snapshot(
            base_offset_ns + 2_200_000_000 + index * 100_000_000,
            process_start_id=process_start_id,
            prompt=64,
            generation=1,
            abort=abort_delta,
        )
        for index in range(11)
    )
    cooldown = tuple(
        make_snapshot(
            base_offset_ns + 3_200_000_000 + index * 100_000_000,
            process_start_id=process_start_id,
            prompt=64,
            generation=1,
            abort=abort_delta,
        )
        for index in range(21)
    )
    raw_inventory = "project_processes=[]\nactive_requests=[]\n"
    return CancellationProbe(
        identity_chain=make_log_chain(
            external_id,
            cancellation=True,
            fixture_marked=fixture_marked,
            first_observation_offset_ns=base_offset_ns + second + 1,
            abort_observation_offset_ns=base_offset_ns + 1_200_000_001,
        ),
        raw_log_start_byte_offset=0,
        dispatch_offset_ns=base_offset_ns + second,
        first_generation_token=FirstGenerationTokenEvidence(
            external_request_id=external_id,
            response_body_id=f"cmpl-{external_id}",
            observation_offset_ns=base_offset_ns + 1_100_000_000,
            output_token_ids=(1000,),
        ),
        client_close_offset_ns=base_offset_ns + 1_200_000_000,
        pre_dispatch_snapshots=pre,
        drain_snapshots=drain,
        stable_generation_snapshots=stable,
        cooldown_snapshots=cooldown,
        later_retained_snapshots=(),
        residual_state=ResidualStateEvidence(
            observation_offset_ns=base_offset_ns + 5_300_000_000,
            raw_process_inventory=raw_inventory,
            raw_process_inventory_sha256=hashlib.sha256(raw_inventory.encode()).hexdigest(),
            active_request_ids=(),
            project_process_ids=(),
        ),
    )


def make_process_records() -> tuple[Stage2ProcessRecord, ...]:
    records: list[Stage2ProcessRecord] = []
    for index, process_kind in enumerate(Stage2ProcessKind):
        online = process_kind is Stage2ProcessKind.ONLINE_SNAPSHOT_DOWNLOAD
        tokenizer = process_kind is Stage2ProcessKind.OFFLINE_TOKENIZER_VERIFICATION
        environment = dict(
            ONLINE_ENVIRONMENT
            if online
            else OFFLINE_TOKENIZER_ENVIRONMENT
            if tokenizer
            else OFFLINE_RUNTIME_ENVIRONMENT
        )
        if not online and not tokenizer:
            environment["HF_HOME"] = "/kaggle/working/lis/hf-home-launch"
            environment["VLLM_CONFIG_ROOT"] = "/kaggle/working/lis/vllm-root-launch"
        process_identity = (
            f"server-process-{index - 1}" if not online and not tokenizer else f"process-{index}"
        )
        records.append(
            Stage2ProcessRecord(
                process_kind=process_kind,
                process_identity=process_identity,
                process_start_offset_ns=index * 10,
                environment_capture_offset_ns=index * 10 + 1,
                first_relevant_import_offset_ns=index * 10 + 2,
                environment=environment,
                absent_environment_variables=tuple(
                    sorted(
                        TOKEN_ENVIRONMENT_VARIABLES
                        if online or tokenizer
                        else LAUNCH_ABSENT_ENVIRONMENT_VARIABLES
                    )
                ),
                operation=Stage2ProcessOperation(
                    token_false=online,
                    local_files_only=not online,
                    repository_id="Qwen/Qwen2.5-0.5B-Instruct" if online else None,
                    revision=("7ae557604adf67be50417f59c2c2f167def9a775" if online else None),
                    exact_network_source=SNAPSHOT_NETWORK_SOURCE if online else None,
                    verified_local_snapshot_path=SNAPSHOT_PATH,
                    exits_after_operation=True,
                ),
            )
        )
    return tuple(records)


def make_runtime_phases(
    evidence_identities: dict[str, str] | None = None,
) -> tuple[RuntimePhaseRecord, ...]:
    identities = evidence_identities or {}
    return tuple(
        RuntimePhaseRecord(
            phase=phase,
            started_offset_ns=index * 10_000_000_000,
            ended_offset_ns=index * 10_000_000_000 + 9_000_000_000,
            passed=True,
            evidence_kind=PHASE_EVIDENCE_KINDS[phase],
            evidence_identity_sha256=identities.get(
                phase.value,
                hashlib.sha256(phase.value.encode()).hexdigest(),
            ),
            evidence_references=(f"raw/phases/{phase.value.casefold()}.json",),
        )
        for index, phase in enumerate(PHASE_EVIDENCE_KINDS)
    )


def make_runtime_control(
    *,
    repetition_index: int = 1,
    future_shape: bool = False,
) -> Stage2RuntimeControlEvidence:
    process_records = make_process_records()
    server_process_identity = f"server-process-{repetition_index}"
    cancellation_probe = make_cancellation_probe(
        base_offset_ns=110_000_000_000,
        process_start_id=server_process_identity,
        fixture_marked=not future_shape,
        external_id=f"E_r{repetition_index}_cancel",
    )
    cancellation_result = evaluate_cancellation(cancellation_probe)
    steady = tuple(
        make_snapshot(
            120_000_000_000 + index * 100_000_000,
            process_start_id=server_process_identity,
        )
        for index in range(10)
    )
    memory_samples = tuple(
        GpuMemorySample(
            observation_offset_ns=80_000_000_000 + index * 200_000_000,
            allocated_bytes=8_000_000_000 + index * 1_000_000,
        )
        for index in range(5)
    )
    stabilization_request_ids = tuple(
        f"E_r{repetition_index}_stabilize_{index}" for index in range(3)
    )
    warmup_request_ids = tuple(f"E_r{repetition_index}_shape_warmup_{index}" for index in range(4))
    measured_request_ids = tuple(
        f"E_r{repetition_index}_measure_{index:02d}" for index in range(16)
    )
    measured_client_slots: tuple[Literal[0, 1], ...] = (0, 1) * 8
    prefix_query_delta = derive_counter_delta(
        steady[0], steady[-1], "vllm:prefix_cache_queries_total"
    )
    prefix_hit_delta = derive_counter_delta(steady[0], steady[-1], "vllm:prefix_cache_hits_total")
    final_scrape = make_snapshot(
        140_100_000_000,
        process_start_id=server_process_identity,
        prompt=1024,
        generation=512,
        length=16,
    )
    shutdown_processes = tuple(
        ProcessExitEvidence(
            process_identity=process_identity,
            exit_code=0,
            observed_offset_ns=158_000_000_000 + index,
            raw_evidence_sha256=hashlib.sha256(process_identity.encode()).hexdigest(),
        )
        for index, process_identity in enumerate(
            (
                server_process_identity,
                f"worker-process-{repetition_index}",
            )
        )
    )
    residual_evidence_sha256 = "b" * 64
    phase_identities = {
        "OFFLINE_SNAPSHOT_VERIFICATION": sha256_identity(process_records[:2]),
        "RUNTIME_PROCESS_START": sha256_identity(process_records[2:]),
        "JIT_COMPILATION_STATE": sha256_identity(()),
        "ALLOCATOR_KV_STABILIZATION": sha256_identity(
            {
                "gpu_memory_samples": memory_samples,
                "gpu_memory_tolerance_bytes": 80_000_000,
            }
        ),
        "EXCLUDED_STABILIZATION_REQUESTS": sha256_identity(stabilization_request_ids),
        "EXCLUDED_SHAPE_WARMUPS": sha256_identity(warmup_request_ids),
        "CANCELLATION_PROBE_DRAIN": sha256_identity(
            {"probe": cancellation_probe, "result": cancellation_result}
        ),
        "STEADY_STATE_GATE": sha256_identity(
            {
                "prefix_cache_hit_delta": prefix_hit_delta,
                "prefix_cache_query_delta": prefix_query_delta,
                "quiet_interval_end_offset_ns": 123_000_000_000,
                "quiet_interval_start_offset_ns": 121_000_000_000,
                "steady_state_snapshots": steady,
            }
        ),
        "MEASURED_WINDOW": sha256_identity(
            {
                "measured_client_slot_assignments": measured_client_slots,
                "measured_request_ids": measured_request_ids,
                "requested_client_concurrency": 2,
            }
        ),
        "FINAL_METRICS_DRAIN": sha256_identity(final_scrape),
        "SHUTDOWN": sha256_identity(shutdown_processes),
        "NO_RESIDUAL_PROCESS_VERIFICATION": sha256_identity(
            {
                "residual_active_request_ids": (),
                "residual_process_ids": (),
                "residual_verification_evidence_sha256": residual_evidence_sha256,
                "residual_verification_offset_ns": 168_000_000_000,
            }
        ),
    }
    return Stage2RuntimeControlEvidence(
        repetition_index=repetition_index,
        phases=make_runtime_phases(phase_identities),
        process_records=process_records,
        gpu_memory_samples=memory_samples,
        gpu_memory_tolerance_bytes=80_000_000,
        stabilization_request_count=3,
        stabilization_request_ids=stabilization_request_ids,
        workload_shape_warmup_count=4,
        workload_shape_warmup_request_ids=warmup_request_ids,
        cancellation_probe=cancellation_probe,
        cancellation_result=cancellation_result,
        steady_state_snapshots=steady,
        prefix_cache_query_delta=prefix_query_delta,
        prefix_cache_hit_delta=prefix_hit_delta,
        post_warmup_jit_event_hashes=(),
        quiet_interval_start_offset_ns=121_000_000_000,
        quiet_interval_end_offset_ns=123_000_000_000,
        measured_request_count=16,
        measured_request_ids=measured_request_ids,
        requested_client_concurrency=2,
        measured_client_slot_assignments=measured_client_slots,
        final_metric_scrape=final_scrape,
        shutdown_processes=shutdown_processes,
        residual_process_ids=(),
        residual_active_request_ids=(),
        residual_verification_offset_ns=168_000_000_000,
        residual_verification_evidence_sha256=residual_evidence_sha256,
    )


def make_launch_spec() -> Stage2LaunchSpec:
    environment = Stage2LaunchEnvironment(
        cuda_visible_devices="0",
        vllm_logging_level="INFO",
        hf_hub_offline="1",
        transformers_offline="1",
        hf_hub_disable_telemetry="1",
        hf_hub_disable_implicit_token="1",
        do_not_track="1",
        vllm_do_not_track="1",
        vllm_no_usage_stats="1",
        tokenizers_parallelism="false",
        no_proxy="127.0.0.1,localhost",
        hf_home="/kaggle/working/lis/hf-home-launch",
        vllm_config_root="/kaggle/working/lis/vllm-root-launch",
        absent_variables=LAUNCH_ABSENT_ENVIRONMENT_VARIABLES,
    )
    argv = (
        "vllm",
        "serve",
        SNAPSHOT_PATH,
        "--tokenizer",
        SNAPSHOT_PATH,
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
    return Stage2LaunchSpec(
        schema_version="0.3.0",
        executable="vllm",
        subcommand="serve",
        model_path=SNAPSHOT_PATH,
        tokenizer_path=SNAPSHOT_PATH,
        served_model_name="qwen2.5-0.5b-instruct-stage2",
        host="127.0.0.1",
        port=8000,
        dtype="half",
        seed=0,
        max_model_len=2048,
        gpu_memory_utilization=0.8,
        max_num_seqs=8,
        tensor_parallel_size=1,
        enforce_eager=True,
        optimization_level=0,
        jit_monitor_mode="error",
        prefix_caching_enabled=False,
        generation_config="vllm",
        stream_interval=1,
        request_id_headers_enabled=True,
        request_logging_enabled=True,
        per_request_metrics_enabled=True,
        trust_remote_code=False,
        quantization="none",
        speculative_decoding="none",
        lora="disabled",
        profiler="disabled",
        environment=environment,
        argv=argv,
    )


def _snapshot_entry(path: str, index: int) -> SnapshotFileEntry:
    digest = hashlib.sha256(path.encode()).hexdigest()
    return SnapshotFileEntry(
        relative_path=path,
        entry_type="regular_file",
        byte_size=index + 1,
        observed_byte_size=index + 1,
        sha256=digest,
        observed_sha256=digest,
    )


def make_snapshot_manifest() -> ModelTokenizerSnapshotManifest:
    content = tuple(
        _snapshot_entry(path, index) for index, path in enumerate(REQUIRED_SNAPSHOT_FILES)
    )
    metadata = (_snapshot_entry(".cache/huggingface/download/metadata.json", 100),)
    by_path = {entry.relative_path: entry.sha256 for entry in content}
    content_identity = snapshot_content_identity(content)
    download_process, offline_tokenizer_process, *_ = make_process_records()
    return ModelTokenizerSnapshotManifest(
        schema_version="0.3.0",
        manifest_name="model-tokenizer-snapshot-manifest-v0.3.0",
        repository="Qwen/Qwen2.5-0.5B-Instruct",
        revision="7ae557604adf67be50417f59c2c2f167def9a775",
        source_url=(
            "https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct/tree/"
            "7ae557604adf67be50417f59c2c2f167def9a775"
        ),
        license="Apache-2.0",
        download_process=download_process,
        download_process_identity_sha256=sha256_identity(download_process),
        required_file_allowlist=REQUIRED_SNAPSHOT_FILES,
        model_content_inventory=content,
        hf_local_metadata_inventory=metadata,
        content_identity_sha256=content_identity,
        tokenizer_class="Qwen2TokenizerFast",
        config_sha256=by_path["config.json"],
        tokenizer_json_sha256=by_path["tokenizer.json"],
        tokenizer_config_sha256=by_path["tokenizer_config.json"],
        vocabulary_sha256=by_path["vocab.json"],
        merges_sha256=by_path["merges.txt"],
        special_token_ids=SnapshotSpecialTokenIds(
            bos_token_id=151643,
            eos_token_id=151645,
            pad_token_id=151643,
            unk_token_id=None,
        ),
        chat_template_sha256="c" * 64,
        chat_template_absent=False,
        read_only_transition=SnapshotReadOnlyTransition(
            started_offset_ns=1,
            completed_offset_ns=2,
            writable_before=True,
            writable_after=False,
            verified_regular_file_count=10,
            verification_evidence_sha256="d" * 64,
        ),
        offline_tokenizer_verification_process=offline_tokenizer_process,
        offline_tokenizer_verification_process_identity_sha256=sha256_identity(
            offline_tokenizer_process
        ),
        snapshot_root_path=SNAPSHOT_PATH,
        snapshot_root_identity_sha256=snapshot_root_identity(SNAPSHOT_PATH, content_identity),
    )


def _identity_model[ModelT: BaseModel](
    model_type: type[ModelT], values: dict[str, object]
) -> ModelT:
    values["identity_sha256"] = sha256_identity(values)
    return model_type.model_validate(values)


def make_real_runtime_attestation() -> FutureRealRuntimeAttestation:
    parsed = make_request_evidence(
        fixture_identity_sha256=None,
        external_id="E_r1_measure_00",
        base_offset_ns=130_100_000_000,
    )
    identity_chain = make_log_chain(
        parsed.external_request_id,
        first_observation_offset_ns=130_100_000_001,
        fixture_marked=False,
    )
    request_identity = RequestIdentityAttestation(
        identity_chain=identity_chain,
        identity_sha256=sha256_identity(identity_chain),
    )
    metrics = PerRequestMetricsAttestation(
        request_evidence_sha256=sha256_identity(parsed),
        metrics=parsed.server_per_request_metrics,
        metrics_sha256=sha256_identity(parsed.server_per_request_metrics),
    )
    runtime_controls = cast(
        tuple[
            Stage2RuntimeControlEvidence,
            Stage2RuntimeControlEvidence,
            Stage2RuntimeControlEvidence,
        ],
        tuple(
            make_runtime_control(repetition_index=index, future_shape=True) for index in (1, 2, 3)
        ),
    )
    prometheus_measurements = []
    for control in runtime_controls:
        baseline = make_snapshot(
            130_000_000_000,
            process_start_id=f"server-process-{control.repetition_index}",
        )
        final = control.final_metric_scrape
        deltas = (
            derive_counter_delta(baseline, final, "vllm:prompt_tokens_total"),
            derive_counter_delta(baseline, final, "vllm:generation_tokens_total"),
            derive_counter_delta(
                baseline, final, "vllm:request_success_total", finished_reason="length"
            ),
            derive_counter_delta(baseline, final, "vllm:num_preemptions_total"),
            derive_counter_delta(baseline, final, "vllm:prefix_cache_queries_total"),
            derive_counter_delta(baseline, final, "vllm:prefix_cache_hits_total"),
        )
        prometheus_measurements.append(
            PrometheusMeasurementAttestation(
                baseline_snapshot=baseline,
                final_snapshot=final,
                counter_deltas=deltas,
                evidence_sha256=sha256_identity(
                    {
                        "baseline_snapshot": baseline,
                        "counter_deltas": deltas,
                        "final_snapshot": final,
                    }
                ),
            )
        )
    prometheus_measurements_tuple = cast(
        tuple[
            PrometheusMeasurementAttestation,
            PrometheusMeasurementAttestation,
            PrometheusMeasurementAttestation,
        ],
        tuple(prometheus_measurements),
    )
    launch = make_launch_spec()
    snapshot_manifest = make_snapshot_manifest()
    execution_lock_values: dict[str, object] = {
        "schema_version": "0.3.0",
        "status": "COMPLETE",
        "python_version": "3.13.15",
        "uv_version": "0.12.5",
        "vllm_version": "0.28.0",
        "vllm_git_revision": "2cf0a6915ce544dc493a0990f2ea38d81601128a",
        "pytorch_index": "https://download.pytorch.org/whl/cu129",
        "flashinfer_index": "https://flashinfer.ai/whl/cu129",
        "artifacts": (
            {
                "package": "vllm",
                "version": "0.28.0",
                "filename": "vllm-0.28.0+cu129-cp38-abi3-manylinux_2_28_x86_64.whl",
                "source_url": (
                    "https://github.com/vllm-project/vllm/releases/download/v0.28.0/"
                    "vllm-0.28.0%2Bcu129-cp38-abi3-manylinux_2_28_x86_64.whl"
                ),
                "sha256": ("8ec943b66a0c6b4351d0778e99d7bacfca5788dd8eedd49425092bacb61c4397"),
                "hash_source": "CONTROLLER_AUTHORIZED_SPEC",
            },
            {
                "package": "torch",
                "version": "2.13.0+cu129",
                "filename": "torch-2.13.0+cu129-cp313-cp313-manylinux_2_28_x86_64.whl",
                "source_url": (
                    "https://download-r2.pytorch.org/whl/cu129/"
                    "torch-2.13.0%2Bcu129-cp313-cp313-manylinux_2_28_x86_64.whl"
                ),
                "sha256": ("6e3bcf183e3096db45bf539dc21f820963074986ece7a56550714f12863c76af"),
                "hash_source": "OFFICIAL_INDEX_METADATA",
            },
            {
                "package": "torchaudio",
                "version": "2.11.0+cu129",
                "filename": ("torchaudio-2.11.0+cu129-cp313-cp313-manylinux_2_28_x86_64.whl"),
                "source_url": (
                    "https://download-r2.pytorch.org/whl/cu129/"
                    "torchaudio-2.11.0%2Bcu129-cp313-cp313-manylinux_2_28_x86_64.whl"
                ),
                "sha256": ("45103fac849ffee337976ff19eac81725b3396e2c18e3f48ed92ba7669cb32d7"),
                "hash_source": "OFFICIAL_INDEX_METADATA",
            },
            {
                "package": "torchvision",
                "version": "0.28.0+cu129",
                "filename": ("torchvision-0.28.0+cu129-cp313-cp313-manylinux_2_28_x86_64.whl"),
                "source_url": (
                    "https://download-r2.pytorch.org/whl/cu129/"
                    "torchvision-0.28.0%2Bcu129-cp313-cp313-manylinux_2_28_x86_64.whl"
                ),
                "sha256": "3" * 64,
                "hash_source": "AUTHORIZED_BINARY_SHA256_VERIFICATION",
            },
        ),
        "resolver_lock_sha256": "1" * 64,
        "installed_distribution_inventory_sha256": "2" * 64,
        "resolver_lock_claimed_complete": True,
        "reviewed_protocol_lock_sha256": (
            "b4dc9dd99a2531132423055cd342da785d3271ab48d642a0af737e6b9e760e5d"
        ),
        "installed": True,
        "executed": True,
    }
    execution_lock = _identity_model(RuntimePackageExecutionLockAttestation, execution_lock_values)
    linux = _identity_model(
        LinuxEnvironmentManifest,
        {
            "operating_system": "Linux",
            "architecture": "x86_64",
            "kernel_release": "synthetic-shape-kernel",
            "glibc_version": "synthetic-shape-glibc",
            "cpu_model": "synthetic-shape-cpu",
            "logical_cpu_count": 4,
            "memory_total_bytes": 28_000_000_000,
            "filesystem_total_bytes": 19_000_000_000,
            "environment_evidence_sha256": "5" * 64,
        },
    )
    devices = (
        NvidiaT4DeviceIdentity(
            physical_index=0,
            model="NVIDIA T4",
            memory_mib=15_360,
            compute_capability="7.5",
            pci_identity="synthetic-shape-pci-0",
            gpu_uuid_retained=False,
        ),
        NvidiaT4DeviceIdentity(
            physical_index=1,
            model="NVIDIA T4",
            memory_mib=15_360,
            compute_capability="7.5",
            pci_identity="synthetic-shape-pci-1",
            gpu_uuid_retained=False,
        ),
    )
    nvidia = _identity_model(
        NvidiaT4ResourceAttestation,
        {
            "devices": devices,
            "runtime_visible_physical_indexes": (0,),
            "gpu_one_unused": True,
            "isolation_evidence_sha256": "6" * 64,
        },
    )
    cuda = _identity_model(
        CudaBackedExecutionAttestation,
        {
            "torch_version": "2.13.0+cu129",
            "vllm_version": "0.28.0",
            "cuda_available": True,
            "runtime_visible_gpu_count": 1,
            "cuda_device_index": 0,
            "server_process_identity": "server-process-1",
            "raw_execution_evidence_sha256": "7" * 64,
        },
    )
    launch_identity = stage2_launch_identity(launch)
    raw_scan_evidence_sha256 = "8" * 64
    bundle_manifests = cast(
        tuple[Stage2BundleManifest, Stage2BundleManifest, Stage2BundleManifest],
        tuple(
            Stage2BundleManifest(
                schema_version="0.3.0",
                measurement_protocol_version="0.3.0",
                state=BundleState.COMMITTED,
                repetition_index=index,
                source_commit="a" * 40,
                created_at_utc=datetime(2026, 8, 28, tzinfo=UTC),
                files=tuple(
                    sorted(
                        (
                            *(
                                BundleFileEntry(
                                    path=phase.evidence_references[0],
                                    sha256=phase.evidence_identity_sha256,
                                    size=1,
                                )
                                for phase in runtime_controls[index - 1].phases
                            ),
                            BundleFileEntry(
                                path="raw/installed-distribution-inventory.json",
                                sha256=execution_lock.installed_distribution_inventory_sha256,
                                size=1,
                            ),
                            BundleFileEntry(
                                path="raw/public-safety-scan.json",
                                sha256=raw_scan_evidence_sha256,
                                size=1,
                            ),
                            BundleFileEntry(
                                path="raw/resolver-lock.json",
                                sha256=execution_lock.resolver_lock_sha256,
                                size=1,
                            ),
                            BundleFileEntry(
                                path="raw/reviewed-stage2-execution-lock.json",
                                sha256=execution_lock.reviewed_protocol_lock_sha256,
                                size=1,
                            ),
                        ),
                        key=lambda entry: entry.path,
                    )
                ),
                reconstruction_sha256=str(index) * 64,
            )
            for index in (1, 2, 3)
        ),
    )
    restarts = tuple(
        ServerRestartIdentity(
            repetition_index=index,
            server_process_identity=f"server-process-{index}",
            worker_process_identities=(f"worker-process-{index}",),
            launch_spec_identity_sha256=launch_identity,
            output_bundle_identity_sha256=bundle_manifest_sha256(bundle_manifests[index - 1]),
        )
        for index in (1, 2, 3)
    )
    server_restarts = _identity_model(
        ServerRestartAttestation,
        {"restarts": restarts},
    )
    scan_inventory_sha256 = sha256_identity(tuple(manifest.files for manifest in bundle_manifests))
    public_safety = PublicSafetyAttestation(
        passed=True,
        finding_count=0,
        scan_inventory_sha256=scan_inventory_sha256,
        raw_scan_evidence_sha256=raw_scan_evidence_sha256,
        scan_result_sha256=sha256_identity(
            {
                "finding_count": 0,
                "passed": True,
                "raw_scan_evidence_sha256": raw_scan_evidence_sha256,
                "scan_inventory_sha256": scan_inventory_sha256,
            }
        ),
    )
    components = {
        "parsed_reconciled_stream_evidence": sha256_identity(parsed),
        "request_identity_and_raw_log_chain": sha256_identity(request_identity),
        "strict_per_request_metrics": sha256_identity(metrics),
        "prometheus_evidence_and_counter_deltas": sha256_identity(prometheus_measurements_tuple),
        "accepted_cancellation_and_drain": sha256_identity(
            tuple(
                {"probe": control.cancellation_probe, "result": control.cancellation_result}
                for control in runtime_controls
            )
        ),
        "complete_runtime_phase_controls": sha256_identity(runtime_controls),
        "exact_stage2_launch_spec": launch_identity,
        "exact_model_tokenizer_snapshot_manifest": sha256_identity(snapshot_manifest),
        "runtime_package_execution_lock": sha256_identity(execution_lock),
        "linux_environment_manifest": sha256_identity(linux),
        "nvidia_t4_resource_and_isolation": sha256_identity(nvidia),
        "cuda_backed_execution": sha256_identity(cuda),
        "server_process_restart_identity": sha256_identity(server_restarts),
        "committed_bundle": sha256_identity(bundle_manifests),
        "public_safety_pass": sha256_identity(public_safety),
    }
    return FutureRealRuntimeAttestation(
        schema_version="0.3.0",
        evidence_scope=Stage2EvidenceScope.TEST_FIXTURE_ONLY,
        parsed_stream_evidence=parsed,
        request_identity=request_identity,
        per_request_metrics=metrics,
        prometheus_measurements=prometheus_measurements_tuple,
        runtime_controls=runtime_controls,
        launch_spec=launch,
        snapshot_manifest=snapshot_manifest,
        execution_lock=execution_lock,
        linux_environment=linux,
        nvidia_resources=nvidia,
        cuda_execution=cuda,
        server_restarts=server_restarts,
        bundle_manifests=bundle_manifests,
        public_safety=public_safety,
        component_identities=tuple(
            AttestedComponentIdentity(component=name, identity_sha256=components[name])
            for name in COMPONENT_ORDER
        ),
    )
