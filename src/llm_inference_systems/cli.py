"""Standard-library argparse CLI for Stage 0 validation only."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import cast

from pydantic import BaseModel, ValidationError

from llm_inference_systems import __version__
from llm_inference_systems.canonical import (
    canonical_json,
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
from llm_inference_systems.schema_io import schema_sync_mismatches

MAX_VALIDATION_BYTES = 10 * 1024 * 1024

VALIDATION_MODELS: dict[str, type[BaseModel]] = {
    "validate-workload": WorkloadDefinition,
    "validate-config": RunConfiguration,
    "validate-artifact": RunArtifact,
    "validate-comparison-policy": ComparisonPolicy,
    "validate-comparison-report": ComparisonReport,
}


def _emit(value: object) -> None:
    sys.stdout.write(canonical_json(value) + "\n")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="llm-inference")
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("version", help="print package and contract versions")
    for command in VALIDATION_MODELS:
        child = subparsers.add_parser(
            command, help=f"validate a {command.removeprefix('validate-')}"
        )
        child.add_argument("path", metavar="PATH")
    subparsers.add_parser("schema-check", help="verify generated schemas are synchronized")
    return parser


def _validation_error(error: ValidationError) -> dict[str, object]:
    issues = [
        {"location": [str(part) for part in item["loc"]], "type": item["type"]}
        for item in error.errors(include_input=False, include_context=False, include_url=False)[:20]
    ]
    return {"error": "validation_failed", "issue_count": error.error_count(), "issues": issues}


def _validate(path_text: str, model: type[BaseModel]) -> int:
    try:
        path = Path(path_text)
        if path.stat().st_size > MAX_VALIDATION_BYTES:
            raise ValueError("input exceeds validation size limit")
        value = model.model_validate_json(path.read_bytes())
        if isinstance(value, RunArtifact) and not verify_artifact_content_hash(value):
            raise ValueError("artifact content hash is missing or invalid")
        if isinstance(value, ComparisonReport) and not verify_report_content_hash(value):
            raise ValueError("comparison report content hash is missing or invalid")
    except ValidationError as error:
        _emit(_validation_error(error))
        return 1
    except (OSError, ValueError, json.JSONDecodeError):
        _emit({"error": "validation_failed", "issue_count": 1})
        return 1
    _emit({"model": model.__name__, "status": "valid"})
    return 0


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    command = cast(str, args.command)
    if command == "version":
        _emit(
            {
                "artifact_schema_version": "0.1.0",
                "measurement_contract_version": "0.1.0",
                "version": __version__,
            }
        )
        return 0
    if command == "schema-check":
        schema_directory = Path(__file__).resolve().parents[2] / "schemas"
        mismatches = schema_sync_mismatches(schema_directory)
        if mismatches:
            _emit({"mismatches": list(mismatches), "status": "out_of_sync"})
            return 1
        _emit({"schema_count": 5, "status": "synchronized"})
        return 0
    return _validate(cast(str, args.path), VALIDATION_MODELS[command])
