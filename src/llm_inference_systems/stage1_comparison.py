"""Semantic-only Stage 1 fixture comparison with no timing performance gates."""

from __future__ import annotations

from datetime import UTC, datetime

from llm_inference_systems.artifact_io import ValidatedBundle
from llm_inference_systems.canonical import sha256_identity
from llm_inference_systems.stage1_contracts import (
    TIMING_DISCLAIMER,
    EvidenceBoundary,
    SemanticCheck,
    Stage1ComparisonPolicy,
    Stage1ComparisonReport,
)


def report_content_identity(report: Stage1ComparisonReport) -> str:
    return sha256_identity(report, omit_fields=frozenset({"content_sha256"}))


def verify_report_content_hash(report: Stage1ComparisonReport) -> bool:
    return report.content_sha256 is not None and report.content_sha256 == report_content_identity(
        report
    )


def compare_validated_bundles(
    baseline: ValidatedBundle,
    candidate: ValidatedBundle,
    policy: Stage1ComparisonPolicy,
    *,
    created_at_utc: datetime | None = None,
) -> Stage1ComparisonReport:
    left = baseline.manifest
    right = candidate.manifest
    input_identities_match = (
        left.workload_sha256 == right.workload_sha256
        and left.configuration_sha256 == right.configuration_sha256
        and left.fixture_sha256 == right.fixture_sha256
    )
    compatibility_match = (
        left.measurement_contract_version == right.measurement_contract_version
        and left.fixture_protocol == right.fixture_protocol
        and left.fixture_protocol_version == right.fixture_protocol_version
        and left.runtime_identity == right.runtime_identity
        and left.environment_identity == right.environment_identity
        and left.configuration.timeout_policy == right.configuration.timeout_policy
        and left.configuration.slo == right.configuration.slo
        and left.configuration.load_shape == right.configuration.load_shape
    )
    left_taxonomy = tuple(
        (
            request.case_id,
            request.phase,
            request.terminal_class.value,
            request.failure.kind.value if request.failure else None,
        )
        for request in baseline.requests
    )
    right_taxonomy = tuple(
        (
            request.case_id,
            request.phase,
            request.terminal_class.value,
            request.failure.kind.value if request.failure else None,
        )
        for request in candidate.requests
    )
    checks = (
        SemanticCheck(check="input-identities-match", passed=input_identities_match),
        SemanticCheck(check="compatibility-fields-match", passed=compatibility_match),
        SemanticCheck(
            check="semantic-fingerprints-match",
            passed=left.semantic_fingerprint == right.semantic_fingerprint,
        ),
        SemanticCheck(
            check="attempted-count-unchanged",
            passed=(
                baseline.summary.attempted_measured_requests
                == candidate.summary.attempted_measured_requests
            ),
        ),
        SemanticCheck(
            check="successful-count-unchanged",
            passed=(
                baseline.summary.successful_measured_requests
                == candidate.summary.successful_measured_requests
            ),
        ),
        SemanticCheck(
            check="failure-count-not-increased",
            passed=(
                candidate.summary.failed_non_timeout_measured_requests
                <= baseline.summary.failed_non_timeout_measured_requests
            ),
        ),
        SemanticCheck(
            check="timeout-count-not-increased",
            passed=(
                candidate.summary.timed_out_measured_requests
                <= baseline.summary.timed_out_measured_requests
            ),
        ),
        SemanticCheck(check="terminal-taxonomy-unchanged", passed=left_taxonomy == right_taxonomy),
    )
    compatible = input_identities_match and compatibility_match
    report = Stage1ComparisonReport(
        boundary=EvidenceBoundary(),
        schema_version="0.2.0",
        comparison_contract_version="0.2.0",
        timing_disclaimer=TIMING_DISCLAIMER,
        created_at_utc=created_at_utc or datetime.now(UTC),
        policy_sha256=sha256_identity(policy),
        baseline_content_sha256=left.content_sha256 or "0" * 64,
        candidate_content_sha256=right.content_sha256 or "0" * 64,
        baseline_semantic_fingerprint=left.semantic_fingerprint,
        candidate_semantic_fingerprint=right.semantic_fingerprint,
        semantic_fingerprints_match=(left.semantic_fingerprint == right.semantic_fingerprint),
        compatible=compatible,
        policy_passed=compatible and all(check.passed for check in checks),
        performance_interpretation_allowed=False,
        checks=checks,
        content_sha256=None,
    )
    return report.model_copy(update={"content_sha256": report_content_identity(report)})


def validate_comparison_report(
    report: Stage1ComparisonReport,
    baseline: ValidatedBundle,
    candidate: ValidatedBundle,
    policy: Stage1ComparisonPolicy,
) -> None:
    if not verify_report_content_hash(report):
        raise ValueError("comparison report content hash is invalid")
    rebuilt = compare_validated_bundles(
        baseline,
        candidate,
        policy,
        created_at_utc=report.created_at_utc,
    )
    if rebuilt != report:
        raise ValueError("comparison report differs from reconstructed semantic comparison")
