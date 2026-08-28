"""Composition root for one complete Stage 1 loopback fixture run."""

from __future__ import annotations

import hashlib
import platform
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

import httpx

from llm_inference_systems.artifact_io import (
    ValidatedBundle,
    validate_execution_bundle,
    validate_stage1_inputs,
    write_execution_bundle,
)
from llm_inference_systems.canonical import sha256_identity
from llm_inference_systems.fixture_server import LOOPBACK_HOST, FixtureServer
from llm_inference_systems.loadgen import execute_closed_loop_workload
from llm_inference_systems.stage1_contracts import (
    EvidenceBoundary,
    FixtureDefinition,
    Stage1ExecutionManifest,
    Stage1RunConfiguration,
    Stage1WorkloadDefinition,
)
from llm_inference_systems.stage1_metrics import derive_stage1_summary, semantic_fingerprint
from llm_inference_systems.streaming import EvidenceCollector


def repository_root() -> Path:
    return Path(__file__).resolve().parents[2]


def determine_source_commit(root: Path) -> str:
    """Return a commit only for clean source; checked evidence files may be untracked."""

    if not (root / ".git").exists():
        return "ARCHIVE_NO_GIT"
    tracked = subprocess.run(
        ["git", "diff", "--quiet", "HEAD", "--"],
        cwd=root,
        check=False,
    )
    untracked = subprocess.run(
        ["git", "ls-files", "--others", "--exclude-standard", "-z"],
        cwd=root,
        check=True,
        capture_output=True,
    )
    untracked_names = tuple(name for name in untracked.stdout.decode().split("\0") if name)
    only_evidence_untracked = all(
        name.startswith("artifacts/stage1-fixture/") for name in untracked_names
    )
    if tracked.returncode != 0 or not only_evidence_untracked:
        return "WORKTREE_DIRTY"
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def load_stage1_inputs(
    workload_path: Path,
    configuration_path: Path,
    fixture_path: Path,
) -> tuple[Stage1WorkloadDefinition, Stage1RunConfiguration, FixtureDefinition]:
    workload = Stage1WorkloadDefinition.model_validate_json(workload_path.read_bytes())
    configuration = Stage1RunConfiguration.model_validate_json(configuration_path.read_bytes())
    fixture = FixtureDefinition.model_validate_json(fixture_path.read_bytes())
    validate_stage1_inputs(workload, configuration, fixture)
    return workload, configuration, fixture


async def run_fixture_to_directory(
    *,
    workload_path: Path,
    configuration_path: Path,
    fixture_path: Path,
    output_directory: Path,
    source_commit: str | None = None,
) -> ValidatedBundle:
    root = repository_root()
    workload, configuration, fixture = load_stage1_inputs(
        workload_path,
        configuration_path,
        fixture_path,
    )
    run_origin_ns = time.monotonic_ns()
    started_at = datetime.now(UTC)
    collector = EvidenceCollector(run_origin_ns)
    server = FixtureServer(fixture, collector)
    await server.start()
    try:
        timeout = httpx.Timeout(
            connect=configuration.timeout_policy.connect_timeout_seconds,
            write=configuration.timeout_policy.write_timeout_seconds,
            pool=configuration.timeout_policy.pool_timeout_seconds,
            read=configuration.timeout_policy.read_timeout_seconds,
        )
        concurrency = configuration.load_shape.requested_client_concurrency
        limits = httpx.Limits(
            max_connections=concurrency,
            max_keepalive_connections=concurrency,
        )
        async with httpx.AsyncClient(
            base_url=f"http://{LOOPBACK_HOST}:{server.port}",
            timeout=timeout,
            limits=limits,
            trust_env=False,
            follow_redirects=False,
            http2=False,
        ) as client:
            requests = await execute_closed_loop_workload(
                client,
                workload,
                configuration,
                fixture,
                collector,
            )
    finally:
        await server.stop()
    ended_at = datetime.now(UTC)
    run_duration_ns = time.monotonic_ns() - run_origin_ns
    stream_events = tuple(collector.stream_events)
    server_events = tuple(collector.server_events)
    summary = derive_stage1_summary(configuration, requests, stream_events)
    workload_hash = sha256_identity(workload)
    configuration_hash = sha256_identity(configuration)
    fixture_hash = sha256_identity(fixture)
    fingerprint = semantic_fingerprint(
        workload_sha256=workload_hash,
        configuration_sha256=configuration_hash,
        fixture_sha256=fixture_hash,
        requests=requests,
        stream_events=stream_events,
        summary=summary,
    )
    lock_hash = hashlib.sha256((root / "uv.lock").read_bytes()).hexdigest()
    environment_identity = (
        f"{platform.system().casefold()}-{platform.machine().casefold()}-"
        f"python-{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
    )
    run_id = (
        "run-"
        + hashlib.sha256(
            f"{started_at.isoformat()}:{run_origin_ns}:{server.port}".encode()
        ).hexdigest()[:24]
    )
    manifest = Stage1ExecutionManifest(
        boundary=EvidenceBoundary(),
        schema_version="0.2.0",
        measurement_contract_version="0.2.0",
        fixture_protocol=fixture.protocol,
        fixture_protocol_version=fixture.protocol_version,
        run_id=run_id,
        source_commit=source_commit or determine_source_commit(root),
        package_version="0.2.0",
        python_version=(
            f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
        ),
        package_lock_sha256=lock_hash,
        environment_identity=environment_identity,
        runtime_identity="deterministic-loopback-http-fixture",
        model_identity="fixture-no-model",
        tokenizer_identity="not-executed-fixture-exact-markers",
        started_at_utc=started_at,
        ended_at_utc=ended_at,
        run_duration_ns=run_duration_ns,
        loopback_host=LOOPBACK_HOST,
        loopback_port=server.port,
        workload=workload,
        configuration=configuration,
        fixture=fixture,
        workload_sha256=workload_hash,
        configuration_sha256=configuration_hash,
        fixture_sha256=fixture_hash,
        raw_file_sha256={
            "requests.jsonl": "0" * 64,
            "stream-events.jsonl": "0" * 64,
            "server-events.jsonl": "0" * 64,
        },
        summary_sha256="0" * 64,
        semantic_fingerprint=fingerprint,
        content_sha256=None,
    )
    write_execution_bundle(
        output_directory,
        manifest_without_file_hashes=manifest,
        requests=requests,
        stream_events=stream_events,
        server_events=server_events,
        summary=summary,
    )
    return validate_execution_bundle(output_directory)
