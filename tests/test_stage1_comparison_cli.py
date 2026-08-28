"""Stage 1 semantic comparison and CLI exit-contract tests."""

from __future__ import annotations

import inspect
import json
from pathlib import Path

import pytest

import llm_inference_systems.cli as cli_module
from llm_inference_systems.artifact_io import ValidatedBundle
from llm_inference_systems.cli import main as cli_main
from llm_inference_systems.runner import run_fixture_to_directory
from llm_inference_systems.stage1_comparison import (
    compare_validated_bundles,
    validate_comparison_report,
    verify_report_content_hash,
)
from llm_inference_systems.stage1_contracts import Stage1ComparisonPolicy
from tests.factories import ROOT


def _policy() -> Stage1ComparisonPolicy:
    return Stage1ComparisonPolicy.model_validate_json(
        (ROOT / "examples/configs/stage1-regression-policy-v1.json").read_bytes()
    )


def _with_manifest(
    bundle: ValidatedBundle,
    **updates: object,
) -> ValidatedBundle:
    return ValidatedBundle(
        manifest=bundle.manifest.model_copy(update=updates),
        requests=bundle.requests,
        stream_events=bundle.stream_events,
        server_events=bundle.server_events,
        summary=bundle.summary,
    )


def test_compatible_repeat_runs_compare_without_performance_gates(
    stage1_bundle_pair: tuple[Path, ValidatedBundle, Path, ValidatedBundle],
) -> None:
    _, baseline, _, candidate = stage1_bundle_pair
    report = compare_validated_bundles(baseline, candidate, _policy())
    assert report.compatible
    assert report.policy_passed
    assert report.performance_interpretation_allowed is False
    assert verify_report_content_hash(report)
    validate_comparison_report(report, baseline, candidate, _policy())


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("workload_sha256", "1" * 64),
        ("configuration_sha256", "2" * 64),
        ("fixture_sha256", "3" * 64),
        ("measurement_contract_version", "9.9.9"),
    ],
)
def test_incompatible_input_or_contract_identity_is_rejected(
    stage1_bundle_pair: tuple[Path, ValidatedBundle, Path, ValidatedBundle],
    field: str,
    value: str,
) -> None:
    _, baseline, _, candidate = stage1_bundle_pair
    changed = _with_manifest(candidate, **{field: value})
    report = compare_validated_bundles(baseline, changed, _policy())
    assert not report.compatible
    assert not report.policy_passed


def test_cli_fixture_run_executes_real_loopback_and_validates(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    output = tmp_path / "cli-run"
    assert (
        cli_main(
            [
                "fixture-run",
                "--workload",
                str(ROOT / "examples/workloads/streaming-fixture-v1.json"),
                "--config",
                str(ROOT / "examples/configs/stage1-streaming-v1.json"),
                "--fixture",
                str(ROOT / "examples/fixtures/streaming-fixture-v1.json"),
                "--output-dir",
                str(output),
            ]
        )
        == 0
    )
    assert json.loads(capsys.readouterr().out)["status"] == "completed"
    assert cli_main(["validate-run-dir", str(output)]) == 0
    assert json.loads(capsys.readouterr().out)["status"] == "valid"
    assert cli_main(["summarize-run", str(output)]) == 0
    assert json.loads(capsys.readouterr().out)["attempted_measured_requests"] == 8


def test_cli_invalid_run_directory_returns_one(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    invalid = tmp_path / "invalid"
    invalid.mkdir()
    assert cli_main(["validate-run-dir", str(invalid)]) == 1
    assert json.loads(capsys.readouterr().out)["status"] == "invalid"


def test_cli_comparison_pass_returns_zero(
    stage1_bundle_pair: tuple[Path, ValidatedBundle, Path, ValidatedBundle],
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    baseline_path, _, candidate_path, _ = stage1_bundle_pair
    output = tmp_path / "comparison.json"
    result = cli_main(
        [
            "compare-runs",
            "--baseline",
            str(baseline_path),
            "--candidate",
            str(candidate_path),
            "--policy",
            str(ROOT / "examples/configs/stage1-regression-policy-v1.json"),
            "--output",
            str(output),
        ]
    )
    assert result == 0
    assert json.loads(capsys.readouterr().out)["status"] == "passed"
    assert output.is_file()


def test_cli_semantic_regression_returns_three(
    stage1_bundle_pair: tuple[Path, ValidatedBundle, Path, ValidatedBundle],
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, baseline, _, candidate = stage1_bundle_pair
    regressed_summary = candidate.summary.model_copy(
        update={"failed_non_timeout_measured_requests": 3}
    )
    regressed = ValidatedBundle(
        manifest=candidate.manifest,
        requests=candidate.requests,
        stream_events=candidate.stream_events,
        server_events=candidate.server_events,
        summary=regressed_summary,
    )
    bundles = iter((baseline, regressed))
    monkeypatch.setattr(cli_module, "validate_execution_bundle", lambda _path: next(bundles))
    result = cli_main(
        [
            "compare-runs",
            "--baseline",
            "synthetic-baseline",
            "--candidate",
            "synthetic-candidate",
            "--policy",
            str(ROOT / "examples/configs/stage1-regression-policy-v1.json"),
            "--output",
            str(tmp_path / "regression.json"),
        ]
    )
    assert result == 3
    assert json.loads(capsys.readouterr().out)["status"] == "regression"


def test_fixture_run_has_no_arbitrary_network_destination_option() -> None:
    with pytest.raises(SystemExit) as error:
        cli_main(
            [
                "fixture-run",
                "--workload",
                "workload.json",
                "--config",
                "config.json",
                "--fixture",
                "fixture.json",
                "--output-dir",
                "output",
                "--host",
                "example.invalid",
            ]
        )
    assert error.value.code == 2


def test_httpx_client_security_options_are_structural() -> None:
    source = inspect.getsource(run_fixture_to_directory)
    assert "trust_env=False" in source
    assert "follow_redirects=False" in source
    assert "http2=False" in source
    assert "127.0.0.1" not in source
    assert "LOOPBACK_HOST" in source
