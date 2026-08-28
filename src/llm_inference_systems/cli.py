"""Argparse CLI for versioned validation and loopback-only Stage 1/Stage 2A fixtures."""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path
from typing import cast

from pydantic import BaseModel, ValidationError

from llm_inference_systems import __version__
from llm_inference_systems.artifact_io import (
    atomic_write,
    reconstruct_summary,
    validate_execution_bundle,
)
from llm_inference_systems.canonical import (
    canonical_json,
    canonical_json_bytes,
    verify_artifact_content_hash,
    verify_report_content_hash,
)
from llm_inference_systems.contracts import (
    ComparisonPolicy,
    ComparisonReport,
    RunArtifact,
    RunConfiguration,
    WorkloadDefinition,
)
from llm_inference_systems.runner import run_fixture_to_directory
from llm_inference_systems.schema_io import SCHEMA_MODELS, schema_sync_mismatches
from llm_inference_systems.stage1_comparison import compare_validated_bundles
from llm_inference_systems.stage1_contracts import (
    FixtureDefinition,
    Stage1ComparisonPolicy,
    Stage1RunConfiguration,
    Stage1WorkloadDefinition,
)
from llm_inference_systems.stage2_attestation import FutureRealRuntimeAttestation
from llm_inference_systems.stage2_contracts import (
    Stage2BundleManifest,
    Stage2CompletionRequest,
    Stage2ExecutionLock,
    Stage2RunConfiguration,
)
from llm_inference_systems.stage2_control import Stage2RuntimeControlEvidence
from llm_inference_systems.stage2_runtime import (
    ModelTokenizerSnapshotManifest,
    Stage2LaunchSpec,
)

MAX_VALIDATION_BYTES = 10 * 1024 * 1024

VALIDATION_MODELS: dict[str, type[BaseModel]] = {
    "validate-artifact": RunArtifact,
    "validate-comparison-policy": ComparisonPolicy,
    "validate-comparison-report": ComparisonReport,
    "validate-fixture": FixtureDefinition,
    "validate-stage2-request": Stage2CompletionRequest,
    "validate-stage2-bundle-manifest": Stage2BundleManifest,
    "validate-stage2-execution-lock": Stage2ExecutionLock,
    "validate-stage2-launch-spec": Stage2LaunchSpec,
    "validate-stage2-snapshot-manifest": ModelTokenizerSnapshotManifest,
    "validate-stage2-runtime-control": Stage2RuntimeControlEvidence,
    "validate-stage2-real-runtime-attestation": FutureRealRuntimeAttestation,
}


def _emit(value: object) -> None:
    sys.stdout.write(canonical_json(value) + "\n")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="llm-inference")
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("version", help="print package and contract versions")
    for command in ("validate-workload", "validate-config", *VALIDATION_MODELS):
        child = subparsers.add_parser(
            command, help=f"validate a {command.removeprefix('validate-')}"
        )
        child.add_argument("path", metavar="PATH")
    subparsers.add_parser("schema-check", help="verify generated schemas are synchronized")

    fixture_run = subparsers.add_parser(
        "fixture-run", help="execute the built-in loopback-only Stage 1 fixture path"
    )
    fixture_run.add_argument("--workload", required=True)
    fixture_run.add_argument("--config", required=True)
    fixture_run.add_argument("--fixture", required=True)
    fixture_run.add_argument("--output-dir", required=True)

    validate_run = subparsers.add_parser(
        "validate-run-dir", help="validate and reconstruct a Stage 1 raw bundle"
    )
    validate_run.add_argument("path", metavar="RUN_DIRECTORY")
    summarize = subparsers.add_parser(
        "summarize-run", help="reconstruct a Stage 1 summary from raw evidence"
    )
    summarize.add_argument("path", metavar="RUN_DIRECTORY")

    compare = subparsers.add_parser(
        "compare-runs", help="compare two validated bundles under a semantic-only policy"
    )
    compare.add_argument("--baseline", required=True)
    compare.add_argument("--candidate", required=True)
    compare.add_argument("--policy", required=True)
    compare.add_argument("--output", required=True)
    return parser


def _validation_error(error: ValidationError) -> dict[str, object]:
    issues = [
        {"location": [str(part) for part in item["loc"]], "type": item["type"]}
        for item in error.errors(include_input=False, include_context=False, include_url=False)[:20]
    ]
    return {"error": "validation_failed", "issue_count": error.error_count(), "issues": issues}


def _read_validation_bytes(path_text: str) -> bytes:
    path = Path(path_text)
    if path.stat().st_size > MAX_VALIDATION_BYTES:
        raise ValueError("input exceeds validation size limit")
    return path.read_bytes()


def _versioned_model(data: bytes, *, kind: str) -> BaseModel:
    decoded = json.loads(data)
    if not isinstance(decoded, dict):
        raise ValueError("versioned input must be an object")
    version = decoded.get("schema_version")
    if kind == "workload":
        model: type[BaseModel] = (
            WorkloadDefinition if version == "0.1.0" else Stage1WorkloadDefinition
        )
    else:
        if version == "0.1.0":
            model = RunConfiguration
        elif version == "0.2.0":
            model = Stage1RunConfiguration
        else:
            model = Stage2RunConfiguration
    return model.model_validate_json(data)


def _validate(path_text: str, command: str) -> int:
    try:
        data = _read_validation_bytes(path_text)
        if command == "validate-workload":
            value = _versioned_model(data, kind="workload")
        elif command == "validate-config":
            value = _versioned_model(data, kind="config")
        else:
            value = VALIDATION_MODELS[command].model_validate_json(data)
        if isinstance(value, RunArtifact) and not verify_artifact_content_hash(value):
            raise ValueError("artifact content hash is missing or invalid")
        if isinstance(value, ComparisonReport) and not verify_report_content_hash(value):
            raise ValueError("comparison report content hash is missing or invalid")
    except ValidationError as error:
        _emit(_validation_error(error))
        return 1
    except (OSError, ValueError, TypeError, json.JSONDecodeError):
        _emit({"error": "validation_failed", "issue_count": 1})
        return 1
    _emit({"model": type(value).__name__, "status": "valid"})
    return 0


def _fixture_run(args: argparse.Namespace) -> int:
    try:
        bundle = asyncio.run(
            run_fixture_to_directory(
                workload_path=Path(cast(str, args.workload)),
                configuration_path=Path(cast(str, args.config)),
                fixture_path=Path(cast(str, args.fixture)),
                output_directory=Path(cast(str, args.output_dir)),
            )
        )
    except (OSError, ValueError, RuntimeError, ValidationError):
        _emit({"error": "fixture_run_failed", "status": "failed"})
        return 1
    _emit(
        {
            "content_sha256": bundle.manifest.content_sha256,
            "evidence_scope": "TEST_FIXTURE_ONLY",
            "semantic_fingerprint": bundle.manifest.semantic_fingerprint,
            "status": "completed",
        }
    )
    return 0


def _validate_run(path_text: str) -> int:
    try:
        bundle = validate_execution_bundle(Path(path_text))
    except (OSError, ValueError, ValidationError):
        _emit({"error": "run_validation_failed", "status": "invalid"})
        return 1
    _emit(
        {
            "content_sha256": bundle.manifest.content_sha256,
            "semantic_fingerprint": bundle.manifest.semantic_fingerprint,
            "status": "valid",
        }
    )
    return 0


def _summarize(path_text: str) -> int:
    try:
        summary = reconstruct_summary(Path(path_text))
    except (OSError, ValueError, ValidationError):
        _emit({"error": "summary_reconstruction_failed", "status": "invalid"})
        return 1
    _emit(summary)
    return 0


def _compare(args: argparse.Namespace) -> int:
    try:
        baseline = validate_execution_bundle(Path(cast(str, args.baseline)))
        candidate = validate_execution_bundle(Path(cast(str, args.candidate)))
        policy = Stage1ComparisonPolicy.model_validate_json(
            _read_validation_bytes(cast(str, args.policy))
        )
        report = compare_validated_bundles(baseline, candidate, policy)
        atomic_write(Path(cast(str, args.output)), canonical_json_bytes(report) + b"\n")
    except (OSError, ValueError, ValidationError):
        _emit({"error": "comparison_failed", "status": "invalid"})
        return 1
    _emit(
        {
            "performance_interpretation_allowed": False,
            "policy_passed": report.policy_passed,
            "report_content_sha256": report.content_sha256,
            "status": "passed" if report.policy_passed else "regression",
        }
    )
    return 0 if report.policy_passed else 3


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    command = cast(str, args.command)
    if command == "version":
        _emit(
            {
                "package_version": __version__,
                "stage0_artifact_schema_version": "0.1.0",
                "stage0_measurement_contract_version": "0.1.0",
                "stage1_measurement_contract_version": "0.2.0",
                "stage2_measurement_protocol_version": "0.3.0",
            }
        )
        return 0
    if command == "schema-check":
        schema_directory = Path(__file__).resolve().parents[2] / "schemas"
        mismatches = schema_sync_mismatches(schema_directory)
        if mismatches:
            _emit({"mismatches": list(mismatches), "status": "out_of_sync"})
            return 1
        _emit({"schema_count": len(SCHEMA_MODELS), "status": "synchronized"})
        return 0
    if command == "fixture-run":
        return _fixture_run(args)
    if command == "validate-run-dir":
        return _validate_run(cast(str, args.path))
    if command == "summarize-run":
        return _summarize(cast(str, args.path))
    if command == "compare-runs":
        return _compare(args)
    return _validate(cast(str, args.path), command)
