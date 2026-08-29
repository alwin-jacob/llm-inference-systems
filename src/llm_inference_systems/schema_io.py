"""Canonical JSON Schema generation and synchronization checks."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from jsonschema import validators  # type: ignore[import-untyped]
from pydantic import BaseModel

from llm_inference_systems.contracts import (
    ComparisonPolicy,
    ComparisonReport,
    RunArtifact,
    RunConfiguration,
    WorkloadDefinition,
)
from llm_inference_systems.stage1_contracts import (
    FixtureDefinition,
    Stage1ComparisonPolicy,
    Stage1ComparisonReport,
    Stage1ExecutionManifest,
    Stage1RunConfiguration,
    Stage1WorkloadDefinition,
)
from llm_inference_systems.stage2_contracts import (
    Stage2BundleManifest,
    Stage2CompletionRequest,
    Stage2ExecutionLock,
    Stage2RunConfiguration,
)
from llm_inference_systems.stage2_control import Stage2RuntimeControlEvidence
from llm_inference_systems.stage2_experiment import (
    Stage2AggregateExperimentManifest,
    Stage2ExperimentAttestation,
    Stage2MeasuredRequestAttestation,
    Stage2RepetitionAttestation,
)
from llm_inference_systems.stage2_runtime import (
    ModelTokenizerSnapshotManifest,
    Stage2LaunchSpec,
)

SCHEMA_MODELS: dict[str, type[BaseModel]] = {
    "comparison-policy-v0.1.0.schema.json": ComparisonPolicy,
    "comparison-report-v0.1.0.schema.json": ComparisonReport,
    "run-artifact-v0.1.0.schema.json": RunArtifact,
    "run-configuration-v0.1.0.schema.json": RunConfiguration,
    "workload-definition-v0.1.0.schema.json": WorkloadDefinition,
    "comparison-policy-v0.2.0.schema.json": Stage1ComparisonPolicy,
    "comparison-report-v0.2.0.schema.json": Stage1ComparisonReport,
    "execution-manifest-v0.2.0.schema.json": Stage1ExecutionManifest,
    "fixture-definition-v0.2.0.schema.json": FixtureDefinition,
    "run-configuration-v0.2.0.schema.json": Stage1RunConfiguration,
    "workload-definition-v0.2.0.schema.json": Stage1WorkloadDefinition,
    "bundle-manifest-v0.3.0.schema.json": Stage2BundleManifest,
    "aggregate-experiment-manifest-v0.3.0.schema.json": Stage2AggregateExperimentManifest,
    "completion-request-v0.3.0.schema.json": Stage2CompletionRequest,
    "execution-lock-v0.3.0.schema.json": Stage2ExecutionLock,
    "launch-spec-v0.3.0.schema.json": Stage2LaunchSpec,
    "model-tokenizer-snapshot-manifest-v0.3.0.schema.json": (ModelTokenizerSnapshotManifest),
    "experiment-attestation-v0.3.0.schema.json": Stage2ExperimentAttestation,
    "measured-request-attestation-v0.3.0.schema.json": Stage2MeasuredRequestAttestation,
    "real-runtime-attestation-v0.3.0.schema.json": Stage2ExperimentAttestation,
    "repetition-attestation-v0.3.0.schema.json": Stage2RepetitionAttestation,
    "run-configuration-v0.3.0.schema.json": Stage2RunConfiguration,
    "runtime-control-v0.3.0.schema.json": Stage2RuntimeControlEvidence,
}


def generated_schema_bytes(model: type[BaseModel]) -> bytes:
    schema = model.model_json_schema(mode="validation")
    validator = validators.validator_for(schema)
    validator.check_schema(schema)
    return (
        json.dumps(schema, allow_nan=False, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")


def write_schemas(directory: Path) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    for filename, model in SCHEMA_MODELS.items():
        (directory / filename).write_bytes(generated_schema_bytes(model))


def schema_sync_mismatches(directory: Path) -> tuple[str, ...]:
    mismatches: list[str] = []
    for filename, model in SCHEMA_MODELS.items():
        path = directory / filename
        expected = generated_schema_bytes(model)
        if not path.is_file() or path.read_bytes() != expected:
            mismatches.append(filename)
    return tuple(mismatches)


def schema_digests(directory: Path) -> dict[str, str]:
    return {
        filename: hashlib.sha256((directory / filename).read_bytes()).hexdigest()
        for filename in SCHEMA_MODELS
    }
