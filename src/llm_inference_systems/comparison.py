"""Deterministic comparison compatibility and delta derivation."""

from __future__ import annotations

from datetime import datetime
from enum import Enum

from pydantic import BaseModel

from llm_inference_systems.canonical import (
    canonical_json,
    comparison_policy_identity,
    verify_artifact_content_hash,
    with_report_content_hash,
)
from llm_inference_systems.contracts import (
    ARTIFACT_SCHEMA_VERSION,
    COMPARISON_CONTRACT_VERSION,
    ComparisonCompatibility,
    ComparisonMismatch,
    ComparisonPolicy,
    ComparisonReport,
    EvidenceScope,
    FailureComparisonPolicy,
    MetricDelta,
    RunArtifact,
)


def _field_value(artifact: RunArtifact, field: str) -> object:
    value: object = artifact
    for part in field.split("."):
        if not isinstance(value, BaseModel):
            raise ValueError(f"comparison field cannot be traversed: {field}")
        if part not in type(value).model_fields:
            raise ValueError(f"unknown comparison field: {field}")
        value = getattr(value, part)
    return value


def _display(value: object) -> str:
    if isinstance(value, Enum):
        return str(value.value)
    if isinstance(value, str):
        return value
    return canonical_json(value)


def _record_mismatch(
    mismatches: list[ComparisonMismatch],
    field: str,
    baseline: object,
    candidate: object,
    *,
    allowed: bool,
) -> None:
    if baseline != candidate:
        mismatches.append(
            ComparisonMismatch(
                field=field,
                baseline_value=_display(baseline),
                candidate_value=_display(candidate),
                allowed=allowed,
            )
        )


def _minimum_samples_met(artifact: RunArtifact, policy: ComparisonPolicy) -> bool:
    distributions = (
        artifact.summary.ttft,
        artifact.summary.end_to_end_success,
        artifact.summary.tpot,
        artifact.summary.itl,
    )
    return artifact.summary.successful_count >= policy.minimum_successful_requests and all(
        item.sample_count >= policy.minimum_metric_samples for item in distributions
    )


def check_compatibility(
    baseline: RunArtifact,
    candidate: RunArtifact,
    policy: ComparisonPolicy,
) -> ComparisonCompatibility:
    """Check every declared identity constraint before any metric delta is produced."""

    mismatches: list[ComparisonMismatch] = []
    _record_mismatch(
        mismatches,
        "baseline_artifact_content_hash_valid",
        True,
        verify_artifact_content_hash(baseline),
        allowed=False,
    )
    _record_mismatch(
        mismatches,
        "candidate_artifact_content_hash_valid",
        True,
        verify_artifact_content_hash(candidate),
        allowed=False,
    )
    _record_mismatch(
        mismatches,
        "baseline_artifact_sha256",
        policy.baseline_artifact_sha256,
        baseline.artifact_content_sha256,
        allowed=False,
    )
    _record_mismatch(
        mismatches,
        "candidate_artifact_sha256",
        policy.candidate_artifact_sha256,
        candidate.artifact_content_sha256,
        allowed=False,
    )
    _record_mismatch(
        mismatches,
        "baseline_slo_policy_sha256",
        policy.slo_policy_sha256,
        baseline.summary.goodput_slo_policy_sha256,
        allowed=False,
    )
    _record_mismatch(
        mismatches,
        "candidate_slo_policy_sha256",
        policy.slo_policy_sha256,
        candidate.summary.goodput_slo_policy_sha256,
        allowed=False,
    )
    for field in policy.fields_required_identical:
        _record_mismatch(
            mismatches,
            field,
            _field_value(baseline, field),
            _field_value(candidate, field),
            allowed=False,
        )
    for field in policy.fields_allowed_to_differ:
        _record_mismatch(
            mismatches,
            field,
            _field_value(baseline, field),
            _field_value(candidate, field),
            allowed=True,
        )
    if policy.failure_policy is FailureComparisonPolicy.INVALIDATE:
        _record_mismatch(
            mismatches,
            "failed_count",
            0,
            baseline.summary.failed_count + candidate.summary.failed_count,
            allowed=False,
        )

    baseline_samples = _minimum_samples_met(baseline, policy)
    candidate_samples = _minimum_samples_met(candidate, policy)
    compatible = (
        baseline_samples
        and candidate_samples
        and not any(not mismatch.allowed for mismatch in mismatches)
    )
    return ComparisonCompatibility(
        compatible=compatible,
        mismatches=tuple(mismatches),
        baseline_sample_requirement_met=baseline_samples,
        candidate_sample_requirement_met=candidate_samples,
    )


def _delta(metric: str, baseline: float, candidate: float) -> MetricDelta:
    absolute = candidate - baseline
    relative = absolute / baseline if baseline != 0 else None
    return MetricDelta(
        metric=metric,
        baseline_value=baseline,
        candidate_value=candidate,
        absolute_delta=absolute,
        relative_delta=relative,
    )


def _metric_deltas(baseline: RunArtifact, candidate: RunArtifact) -> tuple[MetricDelta, ...]:
    scalar_pairs: tuple[tuple[str, float, float], ...] = (
        (
            "successful_request_throughput",
            baseline.summary.successful_request_throughput,
            candidate.summary.successful_request_throughput,
        ),
        ("goodput", baseline.summary.goodput, candidate.summary.goodput),
        ("failure_rate", baseline.summary.failure_rate, candidate.summary.failure_rate),
        ("timeout_rate", baseline.summary.timeout_rate, candidate.summary.timeout_rate),
    )
    deltas = [_delta(name, left, right) for name, left, right in scalar_pairs]
    distributions = (
        ("ttft", baseline.summary.ttft, candidate.summary.ttft),
        (
            "end_to_end_success",
            baseline.summary.end_to_end_success,
            candidate.summary.end_to_end_success,
        ),
        ("tpot", baseline.summary.tpot, candidate.summary.tpot),
        ("itl", baseline.summary.itl, candidate.summary.itl),
    )
    for name, left, right in distributions:
        for percentile in ("p50", "p95", "p99"):
            baseline_value = getattr(left, percentile)
            candidate_value = getattr(right, percentile)
            if baseline_value is not None and candidate_value is not None:
                deltas.append(_delta(f"{name}.{percentile}", baseline_value, candidate_value))
    return tuple(deltas)


def create_comparison_report(
    baseline: RunArtifact,
    candidate: RunArtifact,
    policy: ComparisonPolicy,
    *,
    created_at: datetime,
) -> ComparisonReport:
    compatibility = check_compatibility(baseline, candidate, policy)
    report = ComparisonReport(
        schema_version=ARTIFACT_SCHEMA_VERSION,
        comparison_contract_version=COMPARISON_CONTRACT_VERSION,
        evidence_scope=EvidenceScope.TEST_FIXTURE_ONLY,
        created_at=created_at,
        report_content_sha256=None,
        policy_sha256=comparison_policy_identity(policy),
        baseline_artifact_sha256=policy.baseline_artifact_sha256,
        candidate_artifact_sha256=policy.candidate_artifact_sha256,
        compatibility=compatibility,
        deltas=_metric_deltas(baseline, candidate) if compatibility.compatible else (),
    )
    return with_report_content_hash(report)
