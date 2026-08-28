"""Canonicalization and unkeyed content identity tests."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta, timezone

import pytest

from llm_inference_systems.canonical import (
    artifact_content_identity,
    canonical_json,
    canonical_json_bytes,
    report_content_identity,
    sha256_identity,
    verify_artifact_content_hash,
    verify_report_content_hash,
    workload_identity,
)
from llm_inference_systems.comparison import create_comparison_report
from llm_inference_systems.contracts import ComparisonReport, RunArtifact
from tests.factories import (
    FIXED_TIME,
    artifact,
    comparison_policy,
    load_configuration,
    load_workload,
)


def test_canonical_hash_is_stable_across_dict_ordering() -> None:
    left = {"alpha": 1, "nested": {"x": 2, "y": 3}}
    right = {"nested": {"y": 3, "x": 2}, "alpha": 1}
    assert canonical_json_bytes(left) == canonical_json_bytes(right)
    assert sha256_identity(left) == sha256_identity(right)


def test_canonical_sequence_order_is_preserved() -> None:
    assert canonical_json([3, 2, 1]) != canonical_json([1, 2, 3])


def test_canonical_datetime_normalizes_to_utc() -> None:
    local = datetime(2026, 8, 27, 13, 0, tzinfo=timezone(timedelta(hours=1)))
    utc = datetime(2026, 8, 27, 12, 0, tzinfo=UTC)
    assert canonical_json(local) == canonical_json(utc)
    assert canonical_json(local) == '"2026-08-27T12:00:00.000000Z"'


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
def test_canonical_json_rejects_nonfinite_values(value: float) -> None:
    with pytest.raises(ValueError, match="NaN and Infinity"):
        canonical_json({"value": value})


def test_canonical_json_rejects_naive_timestamp() -> None:
    with pytest.raises(ValueError, match="timezone-aware"):
        canonical_json(datetime.fromisoformat("2026-08-27T12:00:00"))


def test_canonical_json_rejects_non_string_keys() -> None:
    with pytest.raises(TypeError, match="keys must be strings"):
        canonical_json({1: "not-json"})


def test_negative_zero_has_single_canonical_form() -> None:
    assert canonical_json(-0.0) == canonical_json(0.0) == "0.0"


def test_workload_identity_matches_checked_in_configuration() -> None:
    identity = workload_identity(load_workload())
    assert identity == load_configuration().workload_identity
    assert identity.content_sha256 == (
        "1fc5f409205a9d0827b6074a6161b82a418e22c41a748c3f7e8f5b2c41423581"
    )


def test_artifact_self_hash_omits_only_self_hash_field() -> None:
    value = artifact()
    assert verify_artifact_content_hash(value)
    assert value.artifact_content_sha256 == artifact_content_identity(value)
    changed_hash_field = value.model_copy(update={"artifact_content_sha256": "f" * 64})
    assert artifact_content_identity(changed_hash_field) == artifact_content_identity(value)
    changed_content = value.model_copy(update={"created_at": FIXED_TIME + timedelta(seconds=1)})
    assert artifact_content_identity(changed_content) != artifact_content_identity(value)
    assert not verify_artifact_content_hash(changed_content)


def test_report_self_hash_omits_only_self_hash_field() -> None:
    baseline = artifact()
    candidate = artifact()
    report = create_comparison_report(
        baseline,
        candidate,
        comparison_policy(baseline, candidate),
        created_at=FIXED_TIME,
    )
    assert verify_report_content_hash(report)
    changed_hash_field = report.model_copy(update={"report_content_sha256": "f" * 64})
    assert report_content_identity(changed_hash_field) == report_content_identity(report)
    changed_content = report.model_copy(update={"policy_sha256": "e" * 64})
    assert report_content_identity(changed_content) != report_content_identity(report)
    assert not verify_report_content_hash(changed_content)


def test_unkeyed_hash_is_never_exposed_as_signature() -> None:
    assert all("signature" not in field.casefold() for field in RunArtifact.model_fields)
    assert all("signature" not in field.casefold() for field in ComparisonReport.model_fields)
    assert len(sha256_identity({"synthetic": True})) == 64
