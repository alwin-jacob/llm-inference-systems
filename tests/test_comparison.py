"""Compatibility must be established before any comparison delta exists."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from llm_inference_systems.canonical import with_artifact_content_hash
from llm_inference_systems.comparison import check_compatibility, create_comparison_report
from llm_inference_systems.contracts import (
    ComparisonKind,
    ComparisonPolicy,
    FailureComparisonPolicy,
    ModelIdentity,
    RunArtifact,
    SamplingConfiguration,
    SLODefinition,
    TimeoutPolicy,
    TokenizerIdentity,
    WorkloadIdentity,
)
from tests.factories import artifact, comparison_policy, load_configuration


def _candidate_with_difference(difference: str) -> RunArtifact:
    configuration = load_configuration()
    if difference == "workload_hash":
        identity = WorkloadIdentity(
            content_sha256="1" * 64,
            case_ids=configuration.workload_identity.case_ids,
            ordering_policy=configuration.workload_identity.ordering_policy,
        )
        return artifact(configuration.model_copy(update={"workload_identity": identity}))
    if difference == "model_revision":
        model = ModelIdentity(
            model_id=configuration.model_identity.model_id,
            exact_revision="different-fixture-revision",
            prompt_template_sha256=configuration.model_identity.prompt_template_sha256,
            identity_source=configuration.model_identity.identity_source,
        )
        return artifact(configuration.model_copy(update={"model_identity": model}))
    if difference == "tokenizer_revision":
        tokenizer = TokenizerIdentity(
            tokenizer_id=configuration.tokenizer_identity.tokenizer_id,
            exact_revision="different-fixture-revision",
            identity_source=configuration.tokenizer_identity.identity_source,
        )
        return artifact(configuration.model_copy(update={"tokenizer_identity": tokenizer}))
    if difference == "sampling":
        sampling = configuration.sampling.model_copy(update={"temperature": 0.5})
        return artifact(configuration.model_copy(update={"sampling": sampling}))
    if difference == "output_limit":
        sampling = SamplingConfiguration(
            seed=configuration.sampling.seed,
            temperature=configuration.sampling.temperature,
            top_p=configuration.sampling.top_p,
            maximum_output_tokens=4,
            stop_sequences=configuration.sampling.stop_sequences,
        )
        return artifact(configuration.model_copy(update={"sampling": sampling}))
    if difference == "timeout":
        timeout = TimeoutPolicy(
            connect_timeout_ns=configuration.timeout_policy.connect_timeout_ns,
            first_output_token_timeout_ns=(
                configuration.timeout_policy.first_output_token_timeout_ns
            ),
            request_timeout_ns=250_000_000,
        )
        return artifact(configuration.model_copy(update={"timeout_policy": timeout}))
    if difference == "hardware":
        return artifact(processor="different-synthetic-processor")
    if difference == "slo":
        slo = SLODefinition(policy_name="different-slo", ttft_threshold_ns=60_000_000)
        return artifact(configuration.model_copy(update={"slo": slo}))
    if difference == "measurement_contract":
        candidate = artifact().model_copy(update={"measurement_contract_version": "0.2.0"})
        return with_artifact_content_hash(candidate)
    if difference == "runtime":
        return artifact(runtime_name="different-synthetic-runtime")
    raise AssertionError(f"unknown difference: {difference}")


@pytest.mark.parametrize(
    "difference",
    [
        "workload_hash",
        "model_revision",
        "tokenizer_revision",
        "sampling",
        "output_limit",
        "timeout",
        "hardware",
        "slo",
        "measurement_contract",
        "runtime",
    ],
)
def test_undeclared_incompatibility_rejected(difference: str) -> None:
    baseline = artifact()
    candidate = _candidate_with_difference(difference)
    compatibility = check_compatibility(
        baseline,
        candidate,
        comparison_policy(baseline, candidate),
    )
    assert not compatibility.compatible
    assert any(not mismatch.allowed for mismatch in compatibility.mismatches)


def test_cross_runtime_change_requires_and_accepts_explicit_policy() -> None:
    baseline = artifact()
    candidate = artifact(runtime_name="different-synthetic-runtime")
    implicit = comparison_policy(baseline, candidate)
    explicit = comparison_policy(
        baseline,
        candidate,
        kind=ComparisonKind.CROSS_RUNTIME,
    )
    assert not check_compatibility(baseline, candidate, implicit).compatible
    accepted = check_compatibility(baseline, candidate, explicit)
    assert accepted.compatible
    assert any(
        mismatch.field == "runtime_identity" and mismatch.allowed
        for mismatch in accepted.mismatches
    )


def test_cross_hardware_change_requires_and_accepts_explicit_policy() -> None:
    baseline = artifact()
    candidate = artifact(processor="different-synthetic-processor")
    implicit = comparison_policy(baseline, candidate)
    explicit = comparison_policy(
        baseline,
        candidate,
        kind=ComparisonKind.CROSS_HARDWARE,
    )
    assert not check_compatibility(baseline, candidate, implicit).compatible
    assert check_compatibility(baseline, candidate, explicit).compatible


def test_invalid_artifact_content_hash_prevents_compatibility() -> None:
    baseline = artifact()
    candidate = artifact()
    tampered_summary = candidate.summary.model_copy(update={"goodput": 999.0})
    tampered = candidate.model_copy(update={"summary": tampered_summary})
    policy = comparison_policy(baseline, tampered)
    compatibility = check_compatibility(baseline, tampered, policy)
    assert not compatibility.compatible
    assert any(
        mismatch.field == "candidate_artifact_content_hash_valid"
        for mismatch in compatibility.mismatches
    )


def test_incompatible_report_has_no_deltas() -> None:
    baseline = artifact()
    candidate = artifact(processor="different-synthetic-processor")
    report = create_comparison_report(
        baseline,
        candidate,
        comparison_policy(baseline, candidate),
        created_at=datetime(2026, 8, 27, 12, 0, tzinfo=UTC),
    )
    assert not report.compatibility.compatible
    assert report.deltas == ()


def test_compatible_report_computes_deltas_after_compatibility() -> None:
    baseline = artifact()
    candidate = artifact()
    report = create_comparison_report(
        baseline,
        candidate,
        comparison_policy(baseline, candidate),
        created_at=datetime(2026, 8, 27, 12, 0, tzinfo=UTC),
    )
    assert report.compatibility.compatible
    assert report.deltas


def test_failure_invalidation_policy_rejects_retained_failures() -> None:
    baseline = artifact()
    candidate = artifact()
    policy = comparison_policy(
        baseline,
        candidate,
        failure_policy=FailureComparisonPolicy.INVALIDATE,
    )
    assert not check_compatibility(baseline, candidate, policy).compatible


def test_sample_minimum_prevents_comparison() -> None:
    baseline = artifact()
    candidate = artifact()
    policy = comparison_policy(
        baseline,
        candidate,
        minimum_successful_requests=3,
    )
    compatibility = check_compatibility(baseline, candidate, policy)
    assert not compatibility.compatible
    assert not compatibility.baseline_sample_requirement_met
    assert not compatibility.candidate_sample_requirement_met


def test_policy_cannot_omit_essential_compatibility_field() -> None:
    baseline = artifact()
    candidate = artifact()
    valid = comparison_policy(baseline, candidate)
    required = tuple(
        field
        for field in valid.fields_required_identical
        if field != "configuration.timeout_policy"
    )
    data = valid.model_dump(mode="python")
    data["fields_required_identical"] = required
    with pytest.raises(ValidationError, match="missing"):
        ComparisonPolicy.model_validate(data)


def test_cross_runtime_policy_must_declare_runtime_exception() -> None:
    baseline = artifact()
    candidate = artifact(runtime_name="different-synthetic-runtime")
    valid = comparison_policy(baseline, candidate)
    data = valid.model_dump(mode="python")
    data["comparison_kind"] = ComparisonKind.CROSS_RUNTIME
    data["fields_required_identical"] = tuple(
        field for field in valid.fields_required_identical if field != "runtime_identity"
    )
    with pytest.raises(ValidationError, match="allow runtime_identity"):
        ComparisonPolicy.model_validate(data)
