"""Deterministic canonical JSON and unkeyed SHA-256 identities."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from enum import Enum
from pathlib import PurePath
from typing import TypeVar

from pydantic import BaseModel

from llm_inference_systems.contracts import (
    ComparisonPolicy,
    ComparisonReport,
    ConfigurationIdentity,
    RunArtifact,
    RunConfiguration,
    SLODefinition,
    WorkloadDefinition,
    WorkloadIdentity,
)

ModelT = TypeVar("ModelT", bound=BaseModel)


def _normalize(value: object) -> object:
    if isinstance(value, BaseModel):
        return _normalize(value.model_dump(mode="python"))
    if isinstance(value, Enum):
        return _normalize(value.value)
    if isinstance(value, datetime):
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("canonical timestamps must be timezone-aware")
        utc_value = value.astimezone(UTC)
        return utc_value.isoformat(timespec="microseconds").replace("+00:00", "Z")
    if isinstance(value, PurePath):
        return value.as_posix()
    if isinstance(value, Mapping):
        normalized: dict[str, object] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise TypeError("canonical JSON object keys must be strings")
            normalized[key] = _normalize(item)
        return normalized
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [_normalize(item) for item in value]
    if value is None or isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("canonical JSON rejects NaN and Infinity")
        return 0.0 if value == 0.0 else value
    raise TypeError(f"unsupported canonical JSON type: {type(value).__name__}")


def canonical_json_bytes(value: object, *, omit_fields: frozenset[str] = frozenset()) -> bytes:
    """Return compact sorted UTF-8 JSON, optionally omitting top-level object fields."""

    normalized = _normalize(value)
    if omit_fields:
        if not isinstance(normalized, dict):
            raise TypeError("omit_fields requires a top-level object")
        normalized = {key: item for key, item in normalized.items() if key not in omit_fields}
    text = json.dumps(
        normalized,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )
    return text.encode("utf-8")


def canonical_json(value: object, *, omit_fields: frozenset[str] = frozenset()) -> str:
    return canonical_json_bytes(value, omit_fields=omit_fields).decode("utf-8")


def sha256_identity(value: object, *, omit_fields: frozenset[str] = frozenset()) -> str:
    return hashlib.sha256(canonical_json_bytes(value, omit_fields=omit_fields)).hexdigest()


def workload_identity(workload: WorkloadDefinition) -> WorkloadIdentity:
    return WorkloadIdentity(
        content_sha256=sha256_identity(workload),
        case_ids=tuple(case.case_id for case in workload.cases),
        ordering_policy=workload.ordering_policy,
    )


def configuration_identity(configuration: RunConfiguration) -> ConfigurationIdentity:
    return ConfigurationIdentity(content_sha256=sha256_identity(configuration))


def slo_policy_identity(slo: SLODefinition) -> str:
    return sha256_identity(slo)


def comparison_policy_identity(policy: ComparisonPolicy) -> str:
    return sha256_identity(policy)


def artifact_content_identity(artifact: RunArtifact) -> str:
    return sha256_identity(artifact, omit_fields=frozenset({"artifact_content_sha256"}))


def report_content_identity(report: ComparisonReport) -> str:
    return sha256_identity(report, omit_fields=frozenset({"report_content_sha256"}))


def with_artifact_content_hash(artifact: RunArtifact) -> RunArtifact:
    digest = artifact_content_identity(artifact)
    return artifact.model_copy(update={"artifact_content_sha256": digest})


def with_report_content_hash(report: ComparisonReport) -> ComparisonReport:
    digest = report_content_identity(report)
    return report.model_copy(update={"report_content_sha256": digest})


def verify_artifact_content_hash(artifact: RunArtifact) -> bool:
    return (
        artifact.artifact_content_sha256 is not None
        and artifact.artifact_content_sha256 == artifact_content_identity(artifact)
    )


def verify_report_content_hash(report: ComparisonReport) -> bool:
    return (
        report.report_content_sha256 is not None
        and report.report_content_sha256 == report_content_identity(report)
    )
