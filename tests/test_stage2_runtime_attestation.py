from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from llm_inference_systems.canonical import sha256_identity
from llm_inference_systems.stage2_attestation import (
    COMPONENT_ORDER,
    FixtureAttestation,
    FutureRealRuntimeAttestation,
    PrometheusMeasurementAttestation,
    RuntimePackageExecutionLockAttestation,
)
from llm_inference_systems.stage2_contracts import (
    RuntimePhaseRecord,
    Stage2EvidenceScope,
    Stage2ExecutionLock,
)
from llm_inference_systems.stage2_control import (
    CancellationClassification,
    CancellationProbe,
    Stage2RuntimeControlEvidence,
    evaluate_cancellation,
)
from llm_inference_systems.stage2_prometheus import parse_prometheus_snapshot
from llm_inference_systems.stage2_runtime import (
    ModelTokenizerSnapshotManifest,
    SnapshotFileEntry,
    SnapshotReadOnlyTransition,
    Stage2LaunchEnvironment,
    Stage2LaunchSpec,
    Stage2ProcessRecord,
)
from tests.stage2_factories import (
    FIXTURE_IDENTITY,
    make_cancellation_probe,
    make_launch_spec,
    make_process_records,
    make_real_runtime_attestation,
    make_request_evidence,
    make_runtime_control,
    make_snapshot,
    make_snapshot_manifest,
)

ROOT = Path(__file__).resolve().parents[1]


def test_online_snapshot_process_requires_preimport_controls_and_remains_online() -> None:
    online = make_process_records()[0]
    assert online.operation.token_false is True
    assert online.operation.local_files_only is False
    assert "HF_HUB_OFFLINE" not in online.environment
    assert "TRANSFORMERS_OFFLINE" not in online.environment
    Stage2ProcessRecord.model_validate(online.model_dump(mode="python"))

    missing_controls = online.model_dump(mode="python")
    missing_controls["environment"] = {}
    with pytest.raises(ValidationError, match="process-specific contract"):
        Stage2ProcessRecord.model_validate(missing_controls)

    incorrectly_offline = online.model_dump(mode="python")
    incorrectly_offline["environment"] = {
        **online.environment,
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
    }
    with pytest.raises(ValidationError, match="process-specific contract"):
        Stage2ProcessRecord.model_validate(incorrectly_offline)


def test_runtime_phase_records_require_positive_duration_and_phase_evidence() -> None:
    record = make_runtime_control().phases[0]
    value = record.model_dump(mode="python")
    value["ended_offset_ns"] = value["started_offset_ns"]
    with pytest.raises(ValidationError, match="strictly after"):
        RuntimePhaseRecord.model_validate(value)
    value = record.model_dump(mode="python")
    value["evidence_references"] = ()
    with pytest.raises(ValidationError):
        RuntimePhaseRecord.model_validate(value)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("stabilization_request_count", 2),
        ("workload_shape_warmup_count", 3),
        ("quiet_interval_end_offset_ns", 122_000_000_000),
        ("residual_process_ids", (99,)),
        ("post_warmup_jit_event_hashes", ("a" * 64,)),
    ],
)
def test_runtime_controls_reject_counts_quiet_jit_or_residual_drift(
    field: str,
    value: object,
) -> None:
    control = make_runtime_control().model_dump(mode="python")
    control[field] = value
    with pytest.raises(ValidationError):
        Stage2RuntimeControlEvidence.model_validate(control)


def test_runtime_controls_reject_incomplete_or_unstable_gpu_memory() -> None:
    control = make_runtime_control().model_dump(mode="python")
    control["gpu_memory_samples"] = control["gpu_memory_samples"][:4]
    with pytest.raises(ValidationError):
        Stage2RuntimeControlEvidence.model_validate(control)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("shutdown_processes", ()),
        ("residual_active_request_ids", ("E-residual",)),
        ("residual_verification_offset_ns", 159_000_000_000),
    ],
)
def test_runtime_controls_reject_missing_shutdown_or_residual_verification(
    field: str,
    value: object,
) -> None:
    control = make_runtime_control().model_dump(mode="python")
    control[field] = value
    with pytest.raises(ValidationError):
        Stage2RuntimeControlEvidence.model_validate(control)

    control = make_runtime_control().model_dump(mode="python")
    control["final_metric_scrape"]["scrape_monotonic_offset_ns"] = 150_000_000_000
    with pytest.raises(ValidationError, match="final-drain phase"):
        Stage2RuntimeControlEvidence.model_validate(control)

    control = make_runtime_control().model_dump(mode="python")
    control["gpu_memory_samples"][-1]["allocated_bytes"] = 8_200_000_000
    with pytest.raises(ValidationError, match="stability tolerance"):
        Stage2RuntimeControlEvidence.model_validate(control)


def test_cancellation_rejects_bare_booleans_and_missing_predispatch_evidence() -> None:
    value = make_cancellation_probe().model_dump(mode="python")
    value["log_matched"] = True
    value["running"] = 0
    value["delta"] = 0
    with pytest.raises(ValidationError):
        CancellationProbe.model_validate(value)

    value = make_cancellation_probe().model_dump(mode="python")
    value["pre_dispatch_snapshots"] = value["pre_dispatch_snapshots"][:9]
    with pytest.raises(ValidationError):
        CancellationProbe.model_validate(value)


def test_cancellation_rejects_nonzero_baseline_ambiguous_labels_and_counter_reset() -> None:
    probe = make_cancellation_probe()
    baseline = probe.pre_dispatch_snapshots[-1]
    nonzero = make_snapshot(
        baseline.scrape_monotonic_offset_ns,
        running=1,
    )
    pre = (*probe.pre_dispatch_snapshots[:-1], nonzero)
    result = evaluate_cancellation(probe.model_copy(update={"pre_dispatch_snapshots": pre}))
    assert result.classification is CancellationClassification.RESIDUAL_WORK_TIMEOUT

    duplicate_raw = baseline.raw_exposition + (
        'vllm:num_requests_running{engine="0",model_name="qwen2.5-0.5b-instruct-stage2"} 0.0\n'
    )
    ambiguous = parse_prometheus_snapshot(
        duplicate_raw,
        process_start_id=baseline.process_start_id,
        scrape_wall_clock_utc=baseline.scrape_wall_clock_utc,
        scrape_monotonic_offset_ns=baseline.scrape_monotonic_offset_ns,
    )
    pre = (*probe.pre_dispatch_snapshots[:-1], ambiguous)
    result = evaluate_cancellation(probe.model_copy(update={"pre_dispatch_snapshots": pre}))
    assert result.classification is CancellationClassification.RESIDUAL_WORK_TIMEOUT

    reset_pre = tuple(
        make_snapshot(snapshot.scrape_monotonic_offset_ns, prompt=100)
        for snapshot in probe.pre_dispatch_snapshots
    )
    assert (
        evaluate_cancellation(
            probe.model_copy(update={"pre_dispatch_snapshots": reset_pre})
        ).classification
        is CancellationClassification.RESIDUAL_WORK_TIMEOUT
    )


def test_cancellation_rejects_abort_over_one_growth_short_cooldown_and_deadline() -> None:
    assert (
        evaluate_cancellation(make_cancellation_probe(abort_delta=2)).classification
        is CancellationClassification.LATER_COMPLETION
    )
    probe = make_cancellation_probe()
    short = probe.model_copy(update={"cooldown_snapshots": probe.cooldown_snapshots[:11]})
    assert (
        evaluate_cancellation(short).classification
        is CancellationClassification.RESIDUAL_WORK_TIMEOUT
    )
    growing = make_snapshot(
        probe.cooldown_snapshots[-1].scrape_monotonic_offset_ns,
        prompt=64,
        generation=2,
        abort=1,
    )
    growth_probe = probe.model_copy(
        update={"cooldown_snapshots": (*probe.cooldown_snapshots[:-1], growing)}
    )
    assert (
        evaluate_cancellation(growth_probe).classification
        is CancellationClassification.RESIDUAL_WORK_TIMEOUT
    )
    long_cooldown = tuple(
        make_snapshot(
            3_200_000_000 + index * 100_000_000,
            prompt=64,
            generation=1,
            abort=1,
        )
        for index in range(101)
    )
    deadline_probe = probe.model_copy(update={"cooldown_snapshots": long_cooldown})
    assert (
        evaluate_cancellation(deadline_probe).classification
        is CancellationClassification.RESIDUAL_WORK_TIMEOUT
    )


def test_launch_spec_rejects_every_missing_top_level_or_environment_field() -> None:
    spec = make_launch_spec()
    data = spec.model_dump(mode="python")
    for field in Stage2LaunchSpec.model_fields:
        missing = dict(data)
        missing.pop(field)
        with pytest.raises(ValidationError):
            Stage2LaunchSpec.model_validate(missing)
    environment = spec.environment.model_dump(mode="python")
    for field in Stage2LaunchEnvironment.model_fields:
        missing_environment = dict(environment)
        missing_environment.pop(field)
        with pytest.raises(ValidationError):
            Stage2LaunchEnvironment.model_validate(missing_environment)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("model_path", "Qwen/Qwen2.5-0.5B-Instruct"),
        ("tokenizer_path", "/kaggle/working/lis/other-snapshot"),
        ("host", "0.0.0.0"),
        ("port", 8001),
        ("gpu_memory_utilization", 0.81),
        ("tensor_parallel_size", 2),
        ("trust_remote_code", True),
        ("quantization", "int8"),
    ],
)
def test_launch_spec_rejects_fixed_field_drift(field: str, value: object) -> None:
    data = make_launch_spec().model_dump(mode="python")
    data[field] = value
    with pytest.raises(ValidationError):
        Stage2LaunchSpec.model_validate(data)


def test_launch_spec_rejects_unknown_duplicated_or_conflicting_argv() -> None:
    data = make_launch_spec().model_dump(mode="python")
    data["argv"] = (*data["argv"], "--host", "0.0.0.0")
    with pytest.raises(ValidationError, match="ordered launch argv"):
        Stage2LaunchSpec.model_validate(data)
    data = make_launch_spec().model_dump(mode="python")
    data["unknown_flag"] = True
    with pytest.raises(ValidationError):
        Stage2LaunchSpec.model_validate(data)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("repository", "substituted/model"),
        ("revision", "0" * 40),
        ("license", "unknown"),
        ("tokenizer_class", "SubstitutedTokenizer"),
        ("offline_tokenizer_verification_process_identity_sha256", None),
    ],
)
def test_snapshot_manifest_rejects_identity_or_offline_verification_drift(
    field: str,
    value: object,
) -> None:
    data = make_snapshot_manifest().model_dump(mode="python")
    data[field] = value
    with pytest.raises(ValidationError):
        ModelTokenizerSnapshotManifest.model_validate(data)


@pytest.mark.parametrize(
    "mutation",
    [
        "missing",
        "duplicate",
        "unsorted",
        "traversal",
        "absolute",
        "symlink",
        "size",
        "hash",
        "metadata-mixing",
        "readonly-failure",
        "special-token",
    ],
)
def test_snapshot_manifest_rejects_inventory_and_verification_bypasses(mutation: str) -> None:
    data = make_snapshot_manifest().model_dump(mode="python")
    inventory = list(data["model_content_inventory"])
    if mutation == "missing":
        inventory.pop()
    elif mutation == "duplicate":
        inventory[-1] = inventory[-2]
    elif mutation == "unsorted":
        inventory[0], inventory[1] = inventory[1], inventory[0]
    elif mutation == "traversal":
        inventory[0]["relative_path"] = "../escape"
    elif mutation == "absolute":
        inventory[0]["relative_path"] = "/absolute/file"
    elif mutation == "symlink":
        inventory[0]["entry_type"] = "symlink"
    elif mutation == "size":
        inventory[0]["observed_byte_size"] += 1
    elif mutation == "hash":
        inventory[0]["observed_sha256"] = "0" * 64
    elif mutation == "metadata-mixing":
        inventory[0]["relative_path"] = ".cache/huggingface/content.json"
    elif mutation == "readonly-failure":
        data["read_only_transition"]["writable_after"] = True
    else:
        data["special_token_ids"]["eos_token_id"] = 1
    data["model_content_inventory"] = inventory
    with pytest.raises(ValidationError):
        ModelTokenizerSnapshotManifest.model_validate(data)


def test_snapshot_file_entry_rejects_nonfile_and_readonly_zero_duration() -> None:
    entry = make_snapshot_manifest().model_content_inventory[0].model_dump(mode="python")
    entry["entry_type"] = "directory"
    with pytest.raises(ValidationError):
        SnapshotFileEntry.model_validate(entry)
    transition = make_snapshot_manifest().read_only_transition.model_dump(mode="python")
    transition["completed_offset_ns"] = transition["started_offset_ns"]
    with pytest.raises(ValidationError):
        SnapshotReadOnlyTransition.model_validate(transition)


def test_snapshot_manifest_rejects_case_collision_alternate_url_and_unsorted_metadata() -> None:
    manifest = make_snapshot_manifest()
    data = manifest.model_dump(mode="python")
    metadata = data["hf_local_metadata_inventory"][0]
    collision = dict(metadata)
    collision["relative_path"] = metadata["relative_path"].upper()
    data["hf_local_metadata_inventory"] = tuple(
        sorted((metadata, collision), key=lambda entry: entry["relative_path"])
    )
    with pytest.raises(ValidationError, match="case-collision"):
        ModelTokenizerSnapshotManifest.model_validate(data)

    data = manifest.model_dump(mode="python")
    data["source_url"] = "https://huggingface.co/substituted/model"
    with pytest.raises(ValidationError):
        ModelTokenizerSnapshotManifest.model_validate(data)

    data = manifest.model_dump(mode="python")
    second = dict(data["hf_local_metadata_inventory"][0])
    second["relative_path"] = ".cache/huggingface/download/a.json"
    data["hf_local_metadata_inventory"] = (
        data["hf_local_metadata_inventory"][0],
        second,
    )
    with pytest.raises(ValidationError, match="sorted and unique"):
        ModelTokenizerSnapshotManifest.model_validate(data)


def test_execution_lock_rejects_alternate_vllm_url_and_retains_blocked_state() -> None:
    path = ROOT / "execution-lock/stage2-execution-lock.json"
    lock = Stage2ExecutionLock.model_validate_json(path.read_bytes())
    assert lock.resolver_lock_claimed_complete is False
    data = lock.model_dump(mode="python")
    artifacts = list(data["artifacts"])
    artifacts[0]["source_url"] = (
        "https://wheels.vllm.ai/2cf0a6915ce544dc493a0990f2ea38d81601128a/"
        "vllm-0.28.0%2Bcu129-cp38-abi3-manylinux_2_28_x86_64.whl"
    )
    data["artifacts"] = artifacts
    with pytest.raises(ValidationError):
        Stage2ExecutionLock.model_validate(data)


def test_fixture_attestation_can_produce_only_test_fixture_boundary() -> None:
    evidence = make_request_evidence(fixture_identity_sha256=FIXTURE_IDENTITY)
    attestation = FixtureAttestation(
        schema_version="0.3.0",
        evidence_scope=Stage2EvidenceScope.TEST_FIXTURE_ONLY,
        fixture_identity_sha256=FIXTURE_IDENTITY,
        parsed_stream_evidence=evidence,
        parsed_stream_evidence_sha256=sha256_identity(evidence),
    )
    assert attestation.evidence_scope is Stage2EvidenceScope.TEST_FIXTURE_ONLY
    value = attestation.model_dump(mode="python")
    value["evidence_scope"] = Stage2EvidenceScope.FUTURE_REAL_RUNTIME
    with pytest.raises(ValidationError):
        FixtureAttestation.model_validate(value)


def test_complete_synthetic_real_runtime_attestation_validates_structurally() -> None:
    attestation = make_real_runtime_attestation()
    assert attestation.evidence_scope is Stage2EvidenceScope.FUTURE_REAL_RUNTIME
    assert tuple(item.component for item in attestation.component_identities) == COMPONENT_ORDER
    assert all(len(item.identity_sha256) == 64 for item in attestation.component_identities)
    assert tuple(control.repetition_index for control in attestation.runtime_controls) == (1, 2, 3)
    assert tuple(bundle.repetition_index for bundle in attestation.bundle_manifests) == (1, 2, 3)


def test_measured_prometheus_attestation_requires_quiescent_raw_snapshots() -> None:
    prometheus = make_real_runtime_attestation().prometheus_measurements[0]
    value = prometheus.model_dump(mode="python")
    value["baseline_snapshot"] = make_snapshot(
        130_000_000_000,
        process_start_id="server-process-1",
        running=1,
    ).model_dump(mode="python")
    with pytest.raises(ValidationError, match="Prometheus measurement attestation"):
        PrometheusMeasurementAttestation.model_validate(value)


@pytest.mark.parametrize("mutation", ["missing-artifact", "alternate-source", "reviewed-lock"])
def test_future_complete_execution_lock_requires_exact_resolved_inventory(
    mutation: str,
) -> None:
    value = make_real_runtime_attestation().execution_lock.model_dump(mode="python")
    if mutation == "missing-artifact":
        value["artifacts"] = value["artifacts"][:-1]
    elif mutation == "alternate-source":
        value["artifacts"][0]["source_url"] = "https://packages.invalid/vllm.whl"
    else:
        value["reviewed_protocol_lock_sha256"] = "0" * 64
    value["identity_sha256"] = sha256_identity(value, omit_fields=frozenset({"identity_sha256"}))
    with pytest.raises(ValidationError):
        RuntimePackageExecutionLockAttestation.model_validate(value)


def test_parsed_fixture_evidence_cannot_become_real_runtime_evidence() -> None:
    attestation = make_real_runtime_attestation()
    data = attestation.model_dump(mode="python")
    data["parsed_stream_evidence"] = make_request_evidence(
        fixture_identity_sha256=FIXTURE_IDENTITY
    ).model_dump(mode="python")
    with pytest.raises(ValidationError, match="parsed fixture evidence"):
        FutureRealRuntimeAttestation.model_validate(data)


def test_fixture_marked_cancellation_evidence_cannot_be_promoted() -> None:
    data = make_real_runtime_attestation().model_dump(mode="python")
    controls = list(data["runtime_controls"])
    controls[0] = make_runtime_control(
        repetition_index=1,
        future_shape=False,
    ).model_dump(mode="python")
    data["runtime_controls"] = tuple(controls)
    with pytest.raises(ValidationError, match="fixture-marked cancellation"):
        FutureRealRuntimeAttestation.model_validate(data)


@pytest.mark.parametrize(
    "mutation",
    [
        "snapshot-process",
        "runtime-environment",
        "restart-process",
        "shutdown-process",
    ],
)
def test_real_runtime_attestation_rejects_cross_binding_drift(mutation: str) -> None:
    data = make_real_runtime_attestation().model_dump(mode="python")
    if mutation == "snapshot-process":
        data["snapshot_manifest"]["download_process"]["process_identity"] = "other-process"
        data["snapshot_manifest"]["download_process_identity_sha256"] = sha256_identity(
            data["snapshot_manifest"]["download_process"]
        )
    elif mutation == "runtime-environment":
        data["runtime_controls"][0]["process_records"][2]["environment"]["HF_HOME"] = (
            "/kaggle/working/lis/other-hf-home"
        )
    elif mutation == "restart-process":
        data["server_restarts"]["restarts"][0]["server_process_identity"] = "other-server-process"
        restart_values = data["server_restarts"]
        restart_values["identity_sha256"] = sha256_identity(
            restart_values, omit_fields=frozenset({"identity_sha256"})
        )
    else:
        data["runtime_controls"][0]["shutdown_processes"] = data["runtime_controls"][0][
            "shutdown_processes"
        ][:-1]
    with pytest.raises(ValidationError):
        FutureRealRuntimeAttestation.model_validate(data)


@pytest.mark.parametrize(
    "mutation",
    ["control-order", "prometheus-process", "phase-file", "bundle-identity"],
)
def test_real_runtime_attestation_rejects_repetition_evidence_splicing(
    mutation: str,
) -> None:
    data = make_real_runtime_attestation().model_dump(mode="python")
    if mutation == "control-order":
        controls = list(data["runtime_controls"])
        controls[0], controls[1] = controls[1], controls[0]
        data["runtime_controls"] = tuple(controls)
    elif mutation == "prometheus-process":
        measurement = data["prometheus_measurements"][0]
        measurement["baseline_snapshot"]["process_start_id"] = "server-process-2"
        measurement["final_snapshot"]["process_start_id"] = "server-process-2"
        measurement["evidence_sha256"] = sha256_identity(
            {
                "baseline_snapshot": measurement["baseline_snapshot"],
                "counter_deltas": measurement["counter_deltas"],
                "final_snapshot": measurement["final_snapshot"],
            }
        )
    elif mutation == "phase-file":
        phase_path = data["runtime_controls"][0]["phases"][0]["evidence_references"][0]
        entry = next(
            item for item in data["bundle_manifests"][0]["files"] if item["path"] == phase_path
        )
        entry["sha256"] = "0" * 64
    else:
        data["server_restarts"]["restarts"][0]["output_bundle_identity_sha256"] = "0" * 64
        restart_values = data["server_restarts"]
        restart_values["identity_sha256"] = sha256_identity(
            restart_values, omit_fields=frozenset({"identity_sha256"})
        )
    with pytest.raises(ValidationError):
        FutureRealRuntimeAttestation.model_validate(data)


@pytest.mark.parametrize(
    "field",
    [
        "parsed_stream_evidence",
        "request_identity",
        "per_request_metrics",
        "prometheus_measurements",
        "runtime_controls",
        "launch_spec",
        "snapshot_manifest",
        "execution_lock",
        "linux_environment",
        "nvidia_resources",
        "cuda_execution",
        "server_restarts",
        "bundle_manifests",
        "public_safety",
    ],
)
def test_real_runtime_classification_rejects_each_missing_attestation(field: str) -> None:
    data = make_real_runtime_attestation().model_dump(mode="python")
    data.pop(field)
    with pytest.raises(ValidationError):
        FutureRealRuntimeAttestation.model_validate(data)


def test_real_runtime_classification_rejects_missing_component_identity() -> None:
    data = make_real_runtime_attestation().model_dump(mode="python")
    data["component_identities"] = data["component_identities"][:-1]
    with pytest.raises(ValidationError):
        FutureRealRuntimeAttestation.model_validate(data)


def test_generated_snapshot_schema_encodes_exact_manifest_name_and_repository() -> None:
    schema_path = ROOT / "schemas/model-tokenizer-snapshot-manifest-v0.3.0.schema.json"
    if not schema_path.exists():
        pytest.skip("generated schemas are written by the schema-generation gate")
    schema_text = schema_path.read_text()
    assert "model-tokenizer-snapshot-manifest-v0.3.0" in schema_text
    assert "Qwen/Qwen2.5-0.5B-Instruct" in schema_text
    assert "7ae557604adf67be50417f59c2c2f167def9a775" in schema_text


def test_snapshot_metadata_allowlist_matches_read_only_pinned_api_inventory() -> None:
    manifest = make_snapshot_manifest()
    expected = (
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
    assert manifest.required_file_allowlist == expected
    metadata_identity = hashlib.sha256(
        json.dumps(expected, separators=(",", ":")).encode()
    ).hexdigest()
    assert len(metadata_identity) == 64
