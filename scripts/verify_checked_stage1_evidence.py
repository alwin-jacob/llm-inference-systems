"""Verify checked Stage 1 fixture evidence without generating a new run."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import platform
from datetime import date
from pathlib import Path, PurePosixPath
from typing import Literal, cast

from llm_inference_systems import __version__
from llm_inference_systems.artifact_io import (
    ValidatedBundle,
    reconstruct_summary,
    validate_execution_bundle,
)
from llm_inference_systems.canonical import canonical_json
from llm_inference_systems.contracts import PositiveInt, Sha256, StrictModel
from llm_inference_systems.stage1_comparison import validate_comparison_report
from llm_inference_systems.stage1_contracts import (
    EvidenceBoundary,
    Stage1ComparisonPolicy,
    Stage1ComparisonReport,
    Stage1FailureKind,
    Stage1TerminalClass,
    StreamEvidenceKind,
)
from llm_inference_systems.stage1_metrics import derive_stage1_request_metrics

EXPECTED_STAGE0_COMMIT = "77d0ac61b685b3f65edcf43f61899e900eebf5e8"
EXPECTED_STREAMING_COMMIT = "927a0a1c57e7c90aef87f5282093a3076e786b73"
EXPECTED_SOURCE_COMMIT = "66c7f8c6d1c254c89e10c59747d7f957449ba758"


class CheckedEvidenceFile(StrictModel):
    """One non-manifest file retained beneath the checked evidence directory."""

    path: str
    sha256: Sha256
    size: PositiveInt


class CheckedEvidenceManifest(StrictModel):
    """Closed local contract for the checked Stage 1 evidence inventory."""

    schema_version: Literal["1.0.0"]
    evidence_date: str
    evidence_scope: Literal["TEST_FIXTURE_ONLY"]

    stage0_foundation_commit: str
    streaming_implementation_commit: str
    archive_safety_fix_commit: str
    execution_source_commit: str

    package_version: Literal["0.2.0"]
    stage0_contract_version: Literal["0.1.0"]
    stage1_measurement_contract_version: Literal["0.2.0"]
    python_version: Literal["3.13.15"]
    httpx_version: Literal["0.28.1"]

    workload_path: str
    workload_sha256: Sha256
    workload_identity_sha256: Sha256
    configuration_path: str
    configuration_sha256: Sha256
    configuration_identity_sha256: Sha256
    fixture_path: str
    fixture_sha256: Sha256
    fixture_identity_sha256: Sha256
    regression_policy_path: str
    regression_policy_sha256: Sha256

    run_a_run_id: str
    run_b_run_id: str
    run_a_content_hash: Sha256
    run_b_content_hash: Sha256
    run_a_semantic_fingerprint: Sha256
    run_b_semantic_fingerprint: Sha256
    semantic_fingerprints_match: Literal[True]

    comparison_policy_hash: Sha256
    comparison_report_hash: Sha256
    comparison_compatible: Literal[True]
    comparison_policy_passed: Literal[True]
    comparison_performance_interpretation_allowed: Literal[False]

    measured_request_count: Literal[8]
    successful_request_count: Literal[5]
    failed_non_timeout_count: Literal[2]
    timeout_count: Literal[1]
    warmup_record_count: Literal[1]
    warmup_excluded_count: Literal[1]

    failure_rate_numerator: Literal[2]
    failure_rate_denominator: Literal[8]
    failure_rate: float
    timeout_rate_numerator: Literal[1]
    timeout_rate_denominator: Literal[8]
    timeout_rate: float

    requested_client_concurrency: Literal[2]
    observed_max_client_concurrency: Literal[2]
    server_batch_observed: Literal[False]

    synthetic_fixture: Literal[True]
    loopback_http_execution: Literal[True]
    real_runtime_execution: Literal[False]
    model_execution: Literal[False]
    tokenizer_execution: Literal[False]
    gpu_execution: Literal[False]
    cuda_execution: Literal[False]
    performance_claim_allowed: Literal[False]
    historical_authentication_effect: Literal["NONE"]

    files: list[CheckedEvidenceFile]


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _relative_path(value: str) -> PurePosixPath:
    path = PurePosixPath(value)
    _require(value == path.as_posix(), "path is not normalized POSIX relative form")
    _require(not path.is_absolute(), "absolute path is not allowed")
    _require(bool(path.parts) and ".." not in path.parts, "path traversal is not allowed")
    return path


def _repository_file(root: Path, relative: str) -> Path:
    path = _relative_path(relative)
    resolved = root.joinpath(*path.parts)
    _require(resolved.is_file(), f"repository input is missing: {relative}")
    _require(not resolved.is_symlink(), f"repository input is a symlink: {relative}")
    return resolved


def _verify_inventory(
    evidence_directory: Path,
    manifest: CheckedEvidenceManifest,
) -> None:
    _require(not evidence_directory.is_symlink(), "evidence directory cannot be a symlink")
    all_paths = tuple(evidence_directory.rglob("*"))
    _require(not any(path.is_symlink() for path in all_paths), "evidence contains a symlink")

    listed = [entry.path for entry in manifest.files]
    _require(listed == sorted(listed), "evidence file inventory must be sorted")
    _require(len(listed) == len(set(listed)), "evidence file inventory contains duplicates")
    _require("evidence-manifest.json" not in listed, "manifest cannot inventory itself")

    actual = sorted(
        path.relative_to(evidence_directory).as_posix()
        for path in all_paths
        if path.is_file() and path.name != "evidence-manifest.json"
    )
    _require(listed == actual, "evidence file inventory differs from the checked tree")

    for entry in manifest.files:
        relative = _relative_path(entry.path)
        path = evidence_directory.joinpath(*relative.parts)
        _require(path.is_file(), f"checked evidence file is missing: {entry.path}")
        _require(path.stat().st_size == entry.size, f"size differs: {entry.path}")
        _require(_sha256(path) == entry.sha256, f"SHA-256 differs: {entry.path}")


def _verify_boundary(boundary: EvidenceBoundary) -> None:
    expected: dict[str, object] = {
        "evidence_scope": "TEST_FIXTURE_ONLY",
        "synthetic_fixture": True,
        "real_runtime_execution": False,
        "model_execution": False,
        "tokenizer_execution": False,
        "gpu_execution": False,
        "cuda_execution": False,
        "performance_claim_allowed": False,
        "historical_authentication_effect": "NONE",
    }
    _require(boundary.model_dump(mode="json") == expected, "evidence boundary differs")


def _verify_summary(bundle: ValidatedBundle) -> None:
    summary = bundle.summary
    _verify_boundary(bundle.manifest.boundary)
    _verify_boundary(summary.boundary)
    _require(bundle.manifest.source_commit == EXPECTED_SOURCE_COMMIT, "source commit differs")
    _require(bundle.manifest.loopback_host == "127.0.0.1", "run is not IPv4 loopback")
    _require(bundle.manifest.package_version == "0.2.0", "package version differs")
    _require(bundle.manifest.python_version == "3.13.15", "Python version differs")
    _require(summary.attempted_measured_requests == 8, "measured count differs")
    _require(summary.terminal_measured_requests == 8, "terminal count differs")
    _require(summary.successful_measured_requests == 5, "success count differs")
    _require(summary.failed_non_timeout_measured_requests == 2, "failure count differs")
    _require(summary.timed_out_measured_requests == 1, "timeout count differs")
    _require(summary.cancelled_measured_requests == 0, "cancellation count differs")
    _require(summary.warmup_record_count == 1, "warmup count differs")
    _require(summary.warmup_excluded_count == 1, "warmup exclusion differs")
    _require(summary.failure_rate.numerator == 2, "failure numerator differs")
    _require(summary.failure_rate.denominator == 8, "failure denominator differs")
    _require(summary.failure_rate.value == 0.25, "failure rate differs")
    _require(summary.timeout_rate.numerator == 1, "timeout numerator differs")
    _require(summary.timeout_rate.denominator == 8, "timeout denominator differs")
    _require(summary.timeout_rate.value == 0.125, "timeout rate differs")
    _require(summary.requested_client_concurrency == 2, "requested concurrency differs")
    _require(summary.observed_max_client_concurrency == 2, "observed concurrency differs")
    _require(
        summary.configured_server_maximum_batch_size is None,
        "configured server batch size was asserted",
    )
    _require(summary.observed_server_batch_size is None, "server batch observation was asserted")


def _verify_semantic_proofs(bundle: ValidatedBundle) -> None:
    requests = {request.case_id: request for request in bundle.requests}
    first_body = requests["success-first-body-before-token"]
    _require(
        first_body.timing.first_response_body_bytes_offset_ns is not None
        and first_body.timing.first_output_token_offset_ns is not None
        and first_body.timing.first_response_body_bytes_offset_ns
        < first_body.timing.first_output_token_offset_ns,
        "first-body-before-first-token evidence differs",
    )

    multi = requests["success-multi-token-event"]
    _require(
        not derive_stage1_request_metrics(multi).itl_ns,
        "multi-token SSE event fabricated ITL",
    )
    single = requests["success-single-output-token"]
    _require(
        derive_stage1_request_metrics(single).tpot_ns is None,
        "single-token success fabricated TPOT",
    )

    malformed = requests["malformed-after-partial-output"]
    _require(
        malformed.failure is not None
        and malformed.failure.kind is Stage1FailureKind.PROTOCOL_MALFORMED_STREAM
        and malformed.output_token_count == 1,
        "malformed partial-output evidence differs",
    )
    http_error = requests["http-error"]
    _require(
        http_error.failure is not None
        and http_error.failure.kind is Stage1FailureKind.HTTP_STATUS
        and http_error.http_status == 503,
        "HTTP failure evidence differs",
    )
    timeout = requests["timeout-after-partial-body"]
    timeout_raw = tuple(
        event
        for event in bundle.stream_events
        if event.request_id == timeout.request_id
        and event.kind is StreamEvidenceKind.RAW_BODY_CHUNK
    )
    _require(
        timeout.terminal_class is Stage1TerminalClass.TIMEOUT
        and timeout.failure is not None
        and timeout.failure.kind is Stage1FailureKind.TIMEOUT
        and bool(timeout_raw),
        "timeout partial-body evidence differs",
    )


def _verify_inputs(
    root: Path,
    manifest: CheckedEvidenceManifest,
    baseline: ValidatedBundle,
    candidate: ValidatedBundle,
) -> Stage1ComparisonPolicy:
    input_hashes = (
        (manifest.workload_path, manifest.workload_sha256),
        (manifest.configuration_path, manifest.configuration_sha256),
        (manifest.fixture_path, manifest.fixture_sha256),
        (manifest.regression_policy_path, manifest.regression_policy_sha256),
    )
    for relative, expected_hash in input_hashes:
        _require(_sha256(_repository_file(root, relative)) == expected_hash, "input hash differs")

    for bundle in (baseline, candidate):
        _require(
            bundle.manifest.workload_sha256 == manifest.workload_identity_sha256,
            "workload identity differs",
        )
        _require(
            bundle.manifest.configuration_sha256 == manifest.configuration_identity_sha256,
            "configuration identity differs",
        )
        _require(
            bundle.manifest.fixture_sha256 == manifest.fixture_identity_sha256,
            "fixture identity differs",
        )
    return Stage1ComparisonPolicy.model_validate_json(
        _repository_file(root, manifest.regression_policy_path).read_bytes()
    )


def _verify(evidence_directory: Path) -> dict[str, object]:
    _require(evidence_directory.is_dir(), "checked evidence directory does not exist")
    manifest_path = evidence_directory / "evidence-manifest.json"
    manifest = CheckedEvidenceManifest.model_validate_json(manifest_path.read_bytes())
    _require(
        date.fromisoformat(manifest.evidence_date).isoformat() == manifest.evidence_date,
        "date differs",
    )
    _require(evidence_directory.name == manifest.evidence_date, "directory date differs")
    _require(manifest.stage0_foundation_commit == EXPECTED_STAGE0_COMMIT, "Stage 0 commit differs")
    _require(
        manifest.streaming_implementation_commit == EXPECTED_STREAMING_COMMIT,
        "streaming commit differs",
    )
    _require(manifest.archive_safety_fix_commit == EXPECTED_SOURCE_COMMIT, "fix commit differs")
    _require(manifest.execution_source_commit == EXPECTED_SOURCE_COMMIT, "source commit differs")
    _require(manifest.failure_rate == 0.25, "manifest failure rate differs")
    _require(manifest.timeout_rate == 0.125, "manifest timeout rate differs")
    _require(manifest.package_version == __version__, "installed package version differs")
    _require(manifest.python_version == platform.python_version(), "running Python differs")
    _require(
        manifest.httpx_version == importlib.metadata.version("httpx"),
        "installed HTTPX version differs",
    )
    _verify_inventory(evidence_directory, manifest)

    baseline_path = evidence_directory / "run-a"
    candidate_path = evidence_directory / "run-b"
    baseline = validate_execution_bundle(baseline_path)
    candidate = validate_execution_bundle(candidate_path)
    _require(reconstruct_summary(baseline_path) == baseline.summary, "Run A summary differs")
    _require(reconstruct_summary(candidate_path) == candidate.summary, "Run B summary differs")
    _verify_summary(baseline)
    _verify_summary(candidate)
    _verify_semantic_proofs(baseline)
    _verify_semantic_proofs(candidate)

    _require(baseline.manifest.run_id == manifest.run_a_run_id, "Run A ID differs")
    _require(candidate.manifest.run_id == manifest.run_b_run_id, "Run B ID differs")
    _require(baseline.manifest.content_sha256 == manifest.run_a_content_hash, "Run A hash differs")
    _require(candidate.manifest.content_sha256 == manifest.run_b_content_hash, "Run B hash differs")
    _require(
        baseline.manifest.content_sha256 != candidate.manifest.content_sha256,
        "run-specific content identities unexpectedly match",
    )
    _require(
        baseline.manifest.semantic_fingerprint == manifest.run_a_semantic_fingerprint,
        "Run A fingerprint differs",
    )
    _require(
        candidate.manifest.semantic_fingerprint == manifest.run_b_semantic_fingerprint,
        "Run B fingerprint differs",
    )
    _require(
        baseline.manifest.semantic_fingerprint == candidate.manifest.semantic_fingerprint,
        "semantic fingerprints do not reproduce",
    )

    root = Path(__file__).resolve().parents[1]
    policy = _verify_inputs(root, manifest, baseline, candidate)
    report = Stage1ComparisonReport.model_validate_json(
        (evidence_directory / "comparison.json").read_bytes()
    )
    _verify_boundary(report.boundary)
    validate_comparison_report(report, baseline, candidate, policy)
    _require(report.policy_sha256 == manifest.comparison_policy_hash, "policy hash differs")
    _require(report.content_sha256 == manifest.comparison_report_hash, "report hash differs")
    _require(report.compatible, "comparison is incompatible")
    _require(report.policy_passed, "comparison policy did not pass")
    _require(not report.performance_interpretation_allowed, "performance interpretation enabled")

    return {
        "status": "verified",
        "evidence_scope": "TEST_FIXTURE_ONLY",
        "execution_source_commit": EXPECTED_SOURCE_COMMIT,
        "run_count": 2,
        "semantic_reproduction": True,
        "comparison_policy_passed": True,
        "measured_request_count": 8,
        "successful_request_count": 5,
        "failed_non_timeout_count": 2,
        "timeout_count": 1,
        "failure_rate": {"numerator": 2, "denominator": 8},
        "timeout_rate": {"numerator": 1, "denominator": 8},
        "requested_client_concurrency": 2,
        "observed_max_client_concurrency": 2,
        "server_batch_observed": False,
        "real_runtime_execution": False,
        "model_execution": False,
        "tokenizer_execution": False,
        "gpu_execution": False,
        "performance_claim_allowed": False,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="verify already-generated checked Stage 1 fixture evidence"
    )
    parser.add_argument("evidence_directory", metavar="EVIDENCE_DIRECTORY")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    result = _verify(Path(cast(str, args.evidence_directory)))
    print(canonical_json(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
