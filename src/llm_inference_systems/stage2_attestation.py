"""Separate fixture and future real-runtime attestation boundaries for Stage 2."""

from __future__ import annotations

import base64
import binascii
import hashlib
from datetime import timedelta
from typing import Final, Literal, Self

from pydantic import AwareDatetime, Field, model_validator

from llm_inference_systems.canonical import canonical_json_bytes, sha256_identity
from llm_inference_systems.contracts import (
    Identifier,
    NonNegativeInt,
    Sha256,
    StrictModel,
)
from llm_inference_systems.stage2_contracts import (
    BundleState,
    RequestIdentityChain,
    Stage2BundleManifest,
    Stage2EvidenceScope,
    Stage2ManifestBoundFile,
    Stage2PerRequestMetrics,
    Stage2RequestEvidence,
)
from llm_inference_systems.stage2_control import (
    Stage2RuntimeControlEvidence,
    bundle_manifest_sha256,
    evaluate_cancellation,
)
from llm_inference_systems.stage2_prometheus import (
    CounterDelta,
    PrometheusProtocolError,
    PrometheusSnapshot,
    derive_counter_delta,
    require_quiescent,
    validate_measured_window_deltas,
)
from llm_inference_systems.stage2_runtime import (
    LAUNCH_ABSENT_ENVIRONMENT_VARIABLES,
    OFFLINE_RUNTIME_ENVIRONMENT,
    ModelTokenizerSnapshotManifest,
    Stage2LaunchSpec,
    stage2_launch_identity,
)

COMPONENT_ORDER: Final = (
    "parsed_reconciled_stream_evidence",
    "request_identity_and_raw_log_chain",
    "strict_per_request_metrics",
    "prometheus_evidence_and_counter_deltas",
    "accepted_cancellation_and_drain",
    "complete_runtime_phase_controls",
    "exact_stage2_launch_spec",
    "exact_model_tokenizer_snapshot_manifest",
    "runtime_package_execution_lock",
    "linux_environment_manifest",
    "nvidia_t4_resource_and_isolation",
    "cuda_backed_execution",
    "server_process_restart_identity",
    "committed_bundle",
    "public_safety_pass",
)

MAX_MEASURED_WINDOW_SCRAPE_GATE_DISTANCE_NS: Final = 1_000_000_000


class FixtureAttestation(StrictModel):
    schema_version: Literal["0.3.0"]
    evidence_scope: Literal[Stage2EvidenceScope.TEST_FIXTURE_ONLY]
    fixture_identity_sha256: Sha256
    parsed_stream_evidence: Stage2RequestEvidence
    parsed_stream_evidence_sha256: Sha256

    @model_validator(mode="after")
    def validate_fixture(self) -> Self:
        if self.parsed_stream_evidence.fixture_identity_sha256 != self.fixture_identity_sha256:
            raise ValueError("fixture attestation requires the parser's fixture identity")
        if self.parsed_stream_evidence_sha256 != sha256_identity(self.parsed_stream_evidence):
            raise ValueError("fixture parsed-stream identity does not reconstruct")
        return self


class RequestIdentityAttestation(StrictModel):
    identity_chain: RequestIdentityChain
    identity_sha256: Sha256

    @model_validator(mode="after")
    def validate_identity(self) -> Self:
        if self.identity_sha256 != sha256_identity(self.identity_chain):
            raise ValueError("request identity attestation does not reconstruct")
        return self


class PerRequestMetricsAttestation(StrictModel):
    request_evidence_sha256: Sha256
    metrics: Stage2PerRequestMetrics
    metrics_sha256: Sha256

    @model_validator(mode="after")
    def validate_metrics(self) -> Self:
        if self.metrics_sha256 != sha256_identity(self.metrics):
            raise ValueError("per-request metric identity does not reconstruct")
        return self


class PrometheusRawScrapeCapture(StrictModel):
    """Lossless collector boundary for one raw Prometheus exposition scrape."""

    schema_version: Literal["0.3.0"]
    evidence_scope: Stage2EvidenceScope
    repetition_index: Literal[1, 2, 3]
    process_start_id: Identifier
    scrape_wall_clock_utc: AwareDatetime
    scrape_monotonic_offset_ns: NonNegativeInt
    raw_exposition_base64: str
    decoded_byte_count: NonNegativeInt
    raw_exposition_sha256: Sha256
    capture_source: Literal[
        "TEST_FIXTURE_ONLY_CPU_SCRAPE",
        "FUTURE_RUNTIME_PROMETHEUS_COLLECTOR",
    ]
    identity_sha256: Sha256

    @model_validator(mode="after")
    def validate_capture(self) -> Self:
        if self.scrape_wall_clock_utc.utcoffset() != timedelta(0):
            raise ValueError("Prometheus capture wall clock must use UTC")
        try:
            raw = base64.b64decode(self.raw_exposition_base64, validate=True)
        except (binascii.Error, ValueError) as error:
            raise ValueError("Prometheus exposition Base64 is invalid") from error
        if base64.b64encode(raw).decode("ascii") != self.raw_exposition_base64:
            raise ValueError("Prometheus exposition Base64 is not canonical")
        if (
            len(raw) != self.decoded_byte_count
            or hashlib.sha256(raw).hexdigest() != self.raw_exposition_sha256
        ):
            raise ValueError("Prometheus exposition bytes do not match count or SHA-256")
        try:
            raw.decode("utf-8")
        except UnicodeDecodeError as error:
            raise ValueError("Prometheus exposition is not UTF-8") from error
        fixture = self.evidence_scope is Stage2EvidenceScope.TEST_FIXTURE_ONLY
        if fixture != (self.capture_source == "TEST_FIXTURE_ONLY_CPU_SCRAPE"):
            raise ValueError("Prometheus capture source and evidence scope differ")
        if self.identity_sha256 != sha256_identity(
            self, omit_fields=frozenset({"identity_sha256"})
        ):
            raise ValueError("Prometheus raw-capture identity does not reconstruct")
        return self

    def raw_exposition(self) -> str:
        return base64.b64decode(self.raw_exposition_base64, validate=True).decode("utf-8")


class PrometheusMeasurementAttestation(StrictModel):
    schema_version: Literal["0.3.0"]
    repetition_index: Literal[1, 2, 3]
    repetition_manifest_sha256: Sha256
    server_process_identity: Identifier
    served_model_label: Literal["qwen2.5-0.5b-instruct-stage2"]
    engine_label: Literal["0"]
    baseline_raw_exposition_file: Stage2ManifestBoundFile
    baseline_parsed_snapshot_file: Stage2ManifestBoundFile
    final_raw_exposition_file: Stage2ManifestBoundFile
    final_parsed_snapshot_file: Stage2ManifestBoundFile
    baseline_snapshot: PrometheusSnapshot
    final_snapshot: PrometheusSnapshot
    measured_phase_identity_sha256: Sha256
    measured_phase_start_offset_ns: NonNegativeInt
    measured_phase_end_offset_ns: NonNegativeInt
    first_measured_request_dispatch_offset_ns: NonNegativeInt
    last_measured_request_terminal_offset_ns: NonNegativeInt
    final_drain_boundary_offset_ns: NonNegativeInt
    counter_deltas: tuple[CounterDelta, ...] = Field(min_length=6, max_length=6)
    evidence_sha256: Sha256

    @model_validator(mode="after")
    def validate_prometheus(self) -> Self:
        expected_paths = (
            "raw/prometheus/measured-window-baseline.json",
            "derived/prometheus/measured-window-baseline-snapshot.json",
            "raw/prometheus/measured-window-final.json",
            "derived/prometheus/measured-window-final-snapshot.json",
        )
        references = (
            self.baseline_raw_exposition_file,
            self.baseline_parsed_snapshot_file,
            self.final_raw_exposition_file,
            self.final_parsed_snapshot_file,
        )
        if tuple(item.path for item in references) != expected_paths:
            raise ValueError("Prometheus evidence paths differ from the fixed repetition layout")
        if len({item.path.casefold() for item in references}) != 4:
            raise ValueError("Prometheus evidence paths are duplicated or ambiguous")
        if (
            self.baseline_snapshot.process_start_id != self.server_process_identity
            or self.final_snapshot.process_start_id != self.server_process_identity
        ):
            raise ValueError("Prometheus measurement crosses a process or restart")
        if not (
            self.measured_phase_start_offset_ns
            <= self.baseline_snapshot.scrape_monotonic_offset_ns
            < self.first_measured_request_dispatch_offset_ns
            <= self.last_measured_request_terminal_offset_ns
            <= self.measured_phase_end_offset_ns
            < self.final_drain_boundary_offset_ns
            <= self.final_snapshot.scrape_monotonic_offset_ns
        ):
            raise ValueError("Prometheus scrape placement differs from the measured-window gates")
        if (
            self.final_snapshot.scrape_wall_clock_utc
            <= self.baseline_snapshot.scrape_wall_clock_utc
        ):
            raise ValueError("Prometheus scrape wall clocks are stale or reordered")
        if (
            self.first_measured_request_dispatch_offset_ns
            - self.baseline_snapshot.scrape_monotonic_offset_ns
            > MAX_MEASURED_WINDOW_SCRAPE_GATE_DISTANCE_NS
            or self.final_snapshot.scrape_monotonic_offset_ns - self.final_drain_boundary_offset_ns
            > MAX_MEASURED_WINDOW_SCRAPE_GATE_DISTANCE_NS
        ):
            raise ValueError("Prometheus scrape is stale relative to its accepted gate")
        if (
            self.baseline_parsed_snapshot_file.sha256
            != hashlib.sha256(canonical_json_bytes(self.baseline_snapshot) + b"\n").hexdigest()
            or self.final_parsed_snapshot_file.sha256
            != hashlib.sha256(canonical_json_bytes(self.final_snapshot) + b"\n").hexdigest()
        ):
            raise ValueError("Prometheus parsed snapshot file identity does not reconstruct")
        try:
            require_quiescent(self.baseline_snapshot)
            require_quiescent(self.final_snapshot)
            reconstructed = (
                derive_counter_delta(
                    self.baseline_snapshot,
                    self.final_snapshot,
                    "vllm:prompt_tokens_total",
                ),
                derive_counter_delta(
                    self.baseline_snapshot,
                    self.final_snapshot,
                    "vllm:generation_tokens_total",
                ),
                derive_counter_delta(
                    self.baseline_snapshot,
                    self.final_snapshot,
                    "vllm:request_success_total",
                    finished_reason="length",
                ),
                derive_counter_delta(
                    self.baseline_snapshot,
                    self.final_snapshot,
                    "vllm:num_preemptions_total",
                ),
                derive_counter_delta(
                    self.baseline_snapshot,
                    self.final_snapshot,
                    "vllm:prefix_cache_queries_total",
                ),
                derive_counter_delta(
                    self.baseline_snapshot,
                    self.final_snapshot,
                    "vllm:prefix_cache_hits_total",
                ),
            )
            validate_measured_window_deltas(reconstructed)
        except PrometheusProtocolError as error:
            raise ValueError("Prometheus measurement attestation is invalid") from error
        if self.counter_deltas != reconstructed:
            raise ValueError("Prometheus counter deltas do not reconstruct")
        expected_identity = sha256_identity(self, omit_fields=frozenset({"evidence_sha256"}))
        if self.evidence_sha256 != expected_identity:
            raise ValueError("Prometheus evidence identity does not reconstruct")
        return self


class ResolvedVllmRuntimeArtifact(StrictModel):
    package: Literal["vllm"]
    version: Literal["0.28.0"]
    filename: Literal["vllm-0.28.0+cu129-cp38-abi3-manylinux_2_28_x86_64.whl"]
    source_url: Literal[
        "https://github.com/vllm-project/vllm/releases/download/v0.28.0/"
        "vllm-0.28.0%2Bcu129-cp38-abi3-manylinux_2_28_x86_64.whl"
    ]
    sha256: Literal["8ec943b66a0c6b4351d0778e99d7bacfca5788dd8eedd49425092bacb61c4397"]
    hash_source: Literal["CONTROLLER_AUTHORIZED_SPEC"]


class ResolvedTorchRuntimeArtifact(StrictModel):
    package: Literal["torch"]
    version: Literal["2.13.0+cu129"]
    filename: Literal["torch-2.13.0+cu129-cp313-cp313-manylinux_2_28_x86_64.whl"]
    source_url: Literal[
        "https://download-r2.pytorch.org/whl/cu129/"
        "torch-2.13.0%2Bcu129-cp313-cp313-manylinux_2_28_x86_64.whl"
    ]
    sha256: Literal["6e3bcf183e3096db45bf539dc21f820963074986ece7a56550714f12863c76af"]
    hash_source: Literal["OFFICIAL_INDEX_METADATA"]


class ResolvedTorchaudioRuntimeArtifact(StrictModel):
    package: Literal["torchaudio"]
    version: Literal["2.11.0+cu129"]
    filename: Literal["torchaudio-2.11.0+cu129-cp313-cp313-manylinux_2_28_x86_64.whl"]
    source_url: Literal[
        "https://download-r2.pytorch.org/whl/cu129/"
        "torchaudio-2.11.0%2Bcu129-cp313-cp313-manylinux_2_28_x86_64.whl"
    ]
    sha256: Literal["45103fac849ffee337976ff19eac81725b3396e2c18e3f48ed92ba7669cb32d7"]
    hash_source: Literal["OFFICIAL_INDEX_METADATA"]


class ResolvedTorchvisionRuntimeArtifact(StrictModel):
    package: Literal["torchvision"]
    version: Literal["0.28.0+cu129"]
    filename: Literal["torchvision-0.28.0+cu129-cp313-cp313-manylinux_2_28_x86_64.whl"]
    source_url: Literal[
        "https://download-r2.pytorch.org/whl/cu129/"
        "torchvision-0.28.0%2Bcu129-cp313-cp313-manylinux_2_28_x86_64.whl"
    ]
    sha256: Sha256
    hash_source: Literal["AUTHORIZED_BINARY_SHA256_VERIFICATION"]


class RuntimePackageExecutionLockAttestation(StrictModel):
    schema_version: Literal["0.3.0"]
    status: Literal["COMPLETE"]
    python_version: Literal["3.13.15"]
    uv_version: Literal["0.12.5"]
    vllm_version: Literal["0.28.0"]
    vllm_git_revision: Literal["2cf0a6915ce544dc493a0990f2ea38d81601128a"]
    pytorch_index: Literal["https://download.pytorch.org/whl/cu129"]
    flashinfer_index: Literal["https://flashinfer.ai/whl/cu129"]
    artifacts: tuple[
        ResolvedVllmRuntimeArtifact,
        ResolvedTorchRuntimeArtifact,
        ResolvedTorchaudioRuntimeArtifact,
        ResolvedTorchvisionRuntimeArtifact,
    ]
    resolver_lock_sha256: Sha256
    installed_distribution_inventory_sha256: Sha256
    resolver_lock_claimed_complete: Literal[True]
    reviewed_protocol_lock_sha256: Literal[
        "b4dc9dd99a2531132423055cd342da785d3271ab48d642a0af737e6b9e760e5d"
    ]
    installed: Literal[True]
    executed: Literal[True]
    identity_sha256: Sha256

    @model_validator(mode="after")
    def validate_lock_identity(self) -> Self:
        if tuple(artifact.package for artifact in self.artifacts) != (
            "vllm",
            "torch",
            "torchaudio",
            "torchvision",
        ):
            raise ValueError("runtime package inventory is missing, duplicated, or reordered")
        if self.identity_sha256 != sha256_identity(
            self, omit_fields=frozenset({"identity_sha256"})
        ):
            raise ValueError("runtime package execution-lock identity does not reconstruct")
        return self


class LinuxEnvironmentManifest(StrictModel):
    operating_system: Literal["Linux"]
    architecture: Literal["x86_64"]
    kernel_release: Identifier
    glibc_version: Identifier
    cpu_model: Identifier
    logical_cpu_count: NonNegativeInt
    memory_total_bytes: NonNegativeInt
    filesystem_total_bytes: NonNegativeInt
    environment_evidence_sha256: Sha256
    identity_sha256: Sha256

    @model_validator(mode="after")
    def validate_linux(self) -> Self:
        if self.logical_cpu_count < 4 or self.memory_total_bytes < 28_000_000_000:
            raise ValueError("Linux environment does not satisfy the fixed resource gate")
        if self.filesystem_total_bytes < 19_000_000_000:
            raise ValueError("Linux filesystem does not satisfy the fixed resource gate")
        if self.identity_sha256 != sha256_identity(
            self, omit_fields=frozenset({"identity_sha256"})
        ):
            raise ValueError("Linux environment identity does not reconstruct")
        return self


class NvidiaT4DeviceIdentity(StrictModel):
    physical_index: Literal[0, 1]
    model: Literal["NVIDIA T4"]
    memory_mib: NonNegativeInt
    compute_capability: Literal["7.5"]
    pci_identity: Identifier
    gpu_uuid_retained: Literal[False]

    @model_validator(mode="after")
    def validate_device(self) -> Self:
        if self.memory_mib < 15_000:
            raise ValueError("NVIDIA T4 memory evidence is below the fixed threshold")
        return self


class NvidiaT4ResourceAttestation(StrictModel):
    devices: tuple[NvidiaT4DeviceIdentity, NvidiaT4DeviceIdentity]
    runtime_visible_physical_indexes: tuple[Literal[0]]
    gpu_one_unused: Literal[True]
    isolation_evidence_sha256: Sha256
    identity_sha256: Sha256

    @model_validator(mode="after")
    def validate_resources(self) -> Self:
        if tuple(device.physical_index for device in self.devices) != (0, 1):
            raise ValueError("NVIDIA resource evidence requires exact T4 physical indexes")
        if self.identity_sha256 != sha256_identity(
            self, omit_fields=frozenset({"identity_sha256"})
        ):
            raise ValueError("NVIDIA resource identity does not reconstruct")
        return self


class CudaBackedExecutionAttestation(StrictModel):
    torch_version: Literal["2.13.0+cu129"]
    vllm_version: Literal["0.28.0"]
    cuda_available: Literal[True]
    runtime_visible_gpu_count: Literal[1]
    cuda_device_index: Literal[0]
    server_process_identity: Identifier
    raw_execution_evidence_sha256: Sha256
    identity_sha256: Sha256

    @model_validator(mode="after")
    def validate_cuda_identity(self) -> Self:
        if self.identity_sha256 != sha256_identity(
            self, omit_fields=frozenset({"identity_sha256"})
        ):
            raise ValueError("CUDA-backed execution identity does not reconstruct")
        return self


class ServerRestartIdentity(StrictModel):
    repetition_index: Literal[1, 2, 3]
    server_process_identity: Identifier
    worker_process_identities: tuple[Identifier, ...] = Field(min_length=1)
    launch_spec_identity_sha256: Sha256
    output_bundle_identity_sha256: Sha256


class ServerRestartAttestation(StrictModel):
    restarts: tuple[ServerRestartIdentity, ServerRestartIdentity, ServerRestartIdentity]
    identity_sha256: Sha256

    @model_validator(mode="after")
    def validate_restarts(self) -> Self:
        if tuple(item.repetition_index for item in self.restarts) != (1, 2, 3):
            raise ValueError("server restart attestation requires exact ordered repetitions")
        identities = tuple(item.server_process_identity for item in self.restarts)
        if len(set(identities)) != 3:
            raise ValueError("server restart process identities must be distinct")
        if self.identity_sha256 != sha256_identity(
            self, omit_fields=frozenset({"identity_sha256"})
        ):
            raise ValueError("server restart identity does not reconstruct")
        return self


class PublicSafetyAttestation(StrictModel):
    passed: Literal[True]
    finding_count: Literal[0]
    scan_inventory_sha256: Sha256
    raw_scan_evidence_sha256: Sha256
    scan_result_sha256: Sha256

    @model_validator(mode="after")
    def validate_scan_result(self) -> Self:
        if self.scan_result_sha256 != sha256_identity(
            {
                "finding_count": self.finding_count,
                "passed": self.passed,
                "raw_scan_evidence_sha256": self.raw_scan_evidence_sha256,
                "scan_inventory_sha256": self.scan_inventory_sha256,
            }
        ):
            raise ValueError("public-safety result does not reconstruct")
        return self


class AttestedComponentIdentity(StrictModel):
    component: Identifier
    identity_sha256: Sha256


class FutureRealRuntimeAttestation(StrictModel):
    """Legacy request/component-shape graph retained at fixture scope only.

    Final future-runtime classification is owned exclusively by the cardinality-complete
    ``Stage2ExperimentAttestation`` boundary.
    """

    schema_version: Literal["0.3.0"]
    evidence_scope: Literal[Stage2EvidenceScope.TEST_FIXTURE_ONLY]
    parsed_stream_evidence: Stage2RequestEvidence
    request_identity: RequestIdentityAttestation
    per_request_metrics: PerRequestMetricsAttestation
    prometheus_measurements: tuple[
        PrometheusMeasurementAttestation,
        PrometheusMeasurementAttestation,
        PrometheusMeasurementAttestation,
    ]
    runtime_controls: tuple[
        Stage2RuntimeControlEvidence,
        Stage2RuntimeControlEvidence,
        Stage2RuntimeControlEvidence,
    ]
    launch_spec: Stage2LaunchSpec
    snapshot_manifest: ModelTokenizerSnapshotManifest
    execution_lock: RuntimePackageExecutionLockAttestation
    linux_environment: LinuxEnvironmentManifest
    nvidia_resources: NvidiaT4ResourceAttestation
    cuda_execution: CudaBackedExecutionAttestation
    server_restarts: ServerRestartAttestation
    bundle_manifests: tuple[
        Stage2BundleManifest,
        Stage2BundleManifest,
        Stage2BundleManifest,
    ]
    public_safety: PublicSafetyAttestation
    component_identities: tuple[AttestedComponentIdentity, ...] = Field(
        min_length=15,
        max_length=15,
    )

    @model_validator(mode="after")
    def validate_real_runtime_boundary(self) -> Self:
        parsed = self.parsed_stream_evidence
        if parsed.fixture_identity_sha256 is not None:
            raise ValueError("parsed fixture evidence cannot receive a real-runtime boundary")
        chain = self.request_identity.identity_chain
        raw_records = tuple(
            record
            for record in (
                chain.request_received_log,
                chain.request_add_log,
                chain.external_abort_log,
                chain.internal_abort_log,
            )
            if record is not None
        )
        if (
            "fixture" in parsed.external_request_id.casefold()
            or "<fixture-" in parsed.output_text.casefold()
            or any(
                "fixture" in record.source_stream_id.casefold()
                or "TEST_FIXTURE_ONLY" in record.raw_record
                for record in raw_records
            )
        ):
            raise ValueError("fixture-marked evidence cannot receive a real-runtime boundary")
        if (
            chain.external_base_id != parsed.external_request_id
            or chain.response_body_id != parsed.response_request_id
            or chain.serving_item_id != parsed.serving_item_request_id
            or chain.internal_engine_id != parsed.internal_engine_request_id
            or self.request_identity.identity_sha256 != parsed.request_identity_chain_sha256
        ):
            raise ValueError("parsed request and raw request-log identity chain differ")
        parsed_identity = sha256_identity(parsed)
        if (
            self.per_request_metrics.request_evidence_sha256 != parsed_identity
            or self.per_request_metrics.metrics != parsed.server_per_request_metrics
        ):
            raise ValueError("strict per-request metrics are not bound to parsed stream evidence")
        if tuple(control.repetition_index for control in self.runtime_controls) != (1, 2, 3):
            raise ValueError("runtime controls require exact ordered repetition identities")
        if tuple(manifest.repetition_index for manifest in self.bundle_manifests) != (1, 2, 3):
            raise ValueError("bundle manifests require exact ordered repetition identities")
        launch_identity = stage2_launch_identity(self.launch_spec)
        if self.launch_spec.model_path != self.snapshot_manifest.snapshot_root_path:
            raise ValueError("launch spec is not bound to the verified snapshot root")
        process_records = self.runtime_controls[0].process_records
        if any(control.process_records != process_records for control in self.runtime_controls[1:]):
            raise ValueError("repetition controls disagree on the retained process sequence")
        if (
            self.snapshot_manifest.download_process != process_records[0]
            or self.snapshot_manifest.offline_tokenizer_verification_process != process_records[1]
        ):
            raise ValueError("snapshot manifest is not bound to the runtime process sequence")
        launch_environment = self.launch_spec.environment
        expected_runtime_environment = {
            **OFFLINE_RUNTIME_ENVIRONMENT,
            "HF_HOME": launch_environment.hf_home,
            "VLLM_CONFIG_ROOT": launch_environment.vllm_config_root,
        }
        if any(
            process.environment != expected_runtime_environment for process in process_records[2:]
        ):
            raise ValueError("runtime processes are not bound to the exact launch environment")
        if launch_environment.absent_variables != LAUNCH_ABSENT_ENVIRONMENT_VARIABLES:
            raise ValueError("launch environment absence proof differs from the exact contract")
        if any(
            restart.launch_spec_identity_sha256 != launch_identity
            for restart in self.server_restarts.restarts
        ):
            raise ValueError("server restart identity is not bound to the exact launch spec")
        if tuple(process.process_identity for process in process_records[2:]) != tuple(
            restart.server_process_identity for restart in self.server_restarts.restarts
        ):
            raise ValueError("server restart identities differ from the runtime process sequence")
        server_processes = {
            restart.server_process_identity for restart in self.server_restarts.restarts
        }
        if self.cuda_execution.server_process_identity not in server_processes:
            raise ValueError("CUDA execution is not bound to a retained server process")
        for control, prometheus, restart, manifest in zip(
            self.runtime_controls,
            self.prometheus_measurements,
            self.server_restarts.restarts,
            self.bundle_manifests,
            strict=True,
        ):
            cancellation_chain = control.cancellation_probe.identity_chain
            cancellation_records = (
                cancellation_chain.request_received_log,
                cancellation_chain.request_add_log,
                cancellation_chain.external_abort_log,
                cancellation_chain.internal_abort_log,
            )
            if any(
                record is not None
                and (
                    "fixture" in record.source_stream_id.casefold()
                    or "TEST_FIXTURE_ONLY" in record.raw_record
                )
                for record in cancellation_records
            ):
                raise ValueError("fixture-marked cancellation evidence cannot be promoted")
            if evaluate_cancellation(control.cancellation_probe) != control.cancellation_result:
                raise ValueError("cancellation attestation does not reconstruct from raw evidence")
            expected_shutdown = {
                restart.server_process_identity,
                *restart.worker_process_identities,
            }
            if {item.process_identity for item in control.shutdown_processes} != expected_shutdown:
                raise ValueError("server and worker shutdown evidence is incomplete")
            if manifest.state is not BundleState.COMMITTED:
                raise ValueError("future real-runtime classification requires committed bundles")
            if restart.output_bundle_identity_sha256 != bundle_manifest_sha256(manifest):
                raise ValueError("restart output identity is not bound to its manifest bytes")
            measured_phase = next(
                phase for phase in control.phases if phase.phase.value == "MEASURED_WINDOW"
            )
            if (
                prometheus.baseline_snapshot.process_start_id != restart.server_process_identity
                or prometheus.final_snapshot != control.final_metric_scrape
                or not (
                    measured_phase.started_offset_ns
                    <= prometheus.baseline_snapshot.scrape_monotonic_offset_ns
                    <= measured_phase.ended_offset_ns
                )
            ):
                raise ValueError("Prometheus evidence is detached from its repetition control")
            bundle_files = {entry.path: entry for entry in manifest.files}
            if any(
                len(phase.evidence_references) != 1
                or phase.evidence_references[0] not in bundle_files
                or bundle_files[phase.evidence_references[0]].sha256
                != phase.evidence_identity_sha256
                for phase in control.phases
            ):
                raise ValueError("phase evidence is not retained in its committed bundle")
            required_bundle_hashes = {
                "raw/installed-distribution-inventory.json": (
                    self.execution_lock.installed_distribution_inventory_sha256
                ),
                "raw/public-safety-scan.json": self.public_safety.raw_scan_evidence_sha256,
                "raw/resolver-lock.json": self.execution_lock.resolver_lock_sha256,
                "raw/reviewed-stage2-execution-lock.json": (
                    self.execution_lock.reviewed_protocol_lock_sha256
                ),
            }
            if any(
                path not in bundle_files or bundle_files[path].sha256 != expected_hash
                for path, expected_hash in required_bundle_hashes.items()
            ):
                raise ValueError("required lock or safety evidence is absent from a bundle")
        first_measured_phase = next(
            phase
            for phase in self.runtime_controls[0].phases
            if phase.phase.value == "MEASURED_WINDOW"
        )
        first_baseline = self.prometheus_measurements[0].baseline_snapshot
        if not (
            first_baseline.scrape_monotonic_offset_ns <= parsed.timing.dispatch_offset_ns
            and first_measured_phase.started_offset_ns <= parsed.timing.dispatch_offset_ns
            and parsed.timing.transport_terminal_offset_ns <= first_measured_phase.ended_offset_ns
        ):
            raise ValueError("parsed request evidence is detached from the measured window")
        if self.public_safety.scan_inventory_sha256 != sha256_identity(
            tuple(manifest.files for manifest in self.bundle_manifests)
        ):
            raise ValueError("public-safety pass is not bound to all committed inventories")

        actual = {
            "parsed_reconciled_stream_evidence": parsed_identity,
            "request_identity_and_raw_log_chain": sha256_identity(self.request_identity),
            "strict_per_request_metrics": sha256_identity(self.per_request_metrics),
            "prometheus_evidence_and_counter_deltas": sha256_identity(self.prometheus_measurements),
            "accepted_cancellation_and_drain": sha256_identity(
                tuple(
                    {"probe": control.cancellation_probe, "result": control.cancellation_result}
                    for control in self.runtime_controls
                )
            ),
            "complete_runtime_phase_controls": sha256_identity(self.runtime_controls),
            "exact_stage2_launch_spec": launch_identity,
            "exact_model_tokenizer_snapshot_manifest": sha256_identity(self.snapshot_manifest),
            "runtime_package_execution_lock": sha256_identity(self.execution_lock),
            "linux_environment_manifest": sha256_identity(self.linux_environment),
            "nvidia_t4_resource_and_isolation": sha256_identity(self.nvidia_resources),
            "cuda_backed_execution": sha256_identity(self.cuda_execution),
            "server_process_restart_identity": sha256_identity(self.server_restarts),
            "committed_bundle": sha256_identity(self.bundle_manifests),
            "public_safety_pass": sha256_identity(self.public_safety),
        }
        observed_names = tuple(item.component for item in self.component_identities)
        if observed_names != COMPONENT_ORDER or len(set(observed_names)) != len(observed_names):
            raise ValueError("final attestation component identities are missing or reordered")
        if any(
            item.identity_sha256 != actual[item.component] for item in self.component_identities
        ):
            raise ValueError("final attestation component identity does not reconstruct")
        return self
