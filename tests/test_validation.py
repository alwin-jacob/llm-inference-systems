"""Schema, CLI, safety, and deterministic verifier tests."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from jsonschema import validators  # type: ignore[import-untyped]

from llm_inference_systems import __version__
from llm_inference_systems.cli import main as cli_main
from llm_inference_systems.schema_io import (
    SCHEMA_MODELS,
    generated_schema_bytes,
    schema_sync_mismatches,
)
from scripts.check_public_safety import _patterns, scan_repository
from scripts.verify_stage0 import main as verify_main
from scripts.verify_stage1 import main as verify_stage1_main
from tests.factories import ROOT, artifact


def test_all_generated_schemas_are_byte_synchronized() -> None:
    assert schema_sync_mismatches(ROOT / "schemas") == ()
    for filename, model in SCHEMA_MODELS.items():
        assert (ROOT / "schemas" / filename).read_bytes() == generated_schema_bytes(model)


def test_schema_filenames_contain_contract_version() -> None:
    assert len(SCHEMA_MODELS) == 25
    assert all(
        any(f"v{version}.schema.json" in filename for version in ("0.1.0", "0.2.0", "0.3.0"))
        for filename in SCHEMA_MODELS
    )


def test_each_generated_schema_is_valid_json_schema() -> None:
    for filename in SCHEMA_MODELS:
        schema = json.loads((ROOT / "schemas" / filename).read_bytes())
        validator = validators.validator_for(schema)
        validator.check_schema(schema)


def test_cli_version_is_deterministic(capsys: pytest.CaptureFixture[str]) -> None:
    assert cli_main(["version"]) == 0
    first = capsys.readouterr().out
    assert cli_main(["version"]) == 0
    second = capsys.readouterr().out
    assert first == second
    assert json.loads(first) == {
        "package_version": __version__,
        "stage0_artifact_schema_version": "0.1.0",
        "stage0_measurement_contract_version": "0.1.0",
        "stage1_measurement_contract_version": "0.2.0",
        "stage2_measurement_protocol_version": "0.3.0",
    }


def test_cli_validates_checked_in_workload_and_configuration(
    capsys: pytest.CaptureFixture[str],
) -> None:
    workload_path = ROOT / "examples/workloads/deterministic-smoke-v1.json"
    config_path = ROOT / "examples/configs/stage0-contract-v1.json"
    assert cli_main(["validate-workload", str(workload_path)]) == 0
    assert json.loads(capsys.readouterr().out)["status"] == "valid"
    assert cli_main(["validate-config", str(config_path)]) == 0
    assert json.loads(capsys.readouterr().out)["status"] == "valid"


def test_cli_rejects_unknown_fields_without_echoing_input(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    marker = "private-marker-must-not-be-echoed"
    path = tmp_path / "invalid.json"
    path.write_text(
        json.dumps(
            {
                "schema_version": "0.1.0",
                "name": "invalid",
                "description": "invalid synthetic fixture",
                "ordering_policy": "DECLARED",
                "prompt_transformation": "literal",
                "cases": [{"case_id": "case", "prompt": "fixture"}],
                "unexpected": marker,
            }
        ),
        encoding="utf-8",
    )
    assert cli_main(["validate-workload", str(path)]) == 1
    output = capsys.readouterr().out
    assert marker not in output
    assert json.loads(output)["error"] == "validation_failed"


def test_cli_requires_valid_artifact_self_hash(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    value = artifact()
    tampered = value.model_copy(
        update={"summary": value.summary.model_copy(update={"goodput": 100.0})}
    )
    path = tmp_path / "tampered-artifact.json"
    path.write_text(tampered.model_dump_json(), encoding="utf-8")
    assert cli_main(["validate-artifact", str(path)]) == 1
    assert json.loads(capsys.readouterr().out)["error"] == "validation_failed"


def test_cli_schema_check_uses_committed_generated_files(
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert cli_main(["schema-check"]) == 0
    assert json.loads(capsys.readouterr().out) == {
        "schema_count": 25,
        "status": "synchronized",
    }


def test_public_safety_scan_has_no_repository_findings() -> None:
    assert scan_repository(ROOT) == ()


def test_public_safety_rules_detect_private_key_header() -> None:
    synthetic_header = "-----BEGIN " + "PRIVATE KEY-----"
    matches = [name for name, pattern in _patterns() if pattern.search(synthetic_header)]
    assert "private-key-header" in matches


def test_stage0_verifier_exercises_real_contract_logic(
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert verify_main() == 0
    result = json.loads(capsys.readouterr().out)
    assert result["evidence_scope"] == "TEST_FIXTURE_ONLY"
    assert result["real_runtime_execution"] is False
    assert result["gpu_execution"] is False
    assert result["performance_claim_allowed"] is False
    assert all(result["verified"].values())


def test_cli_validates_checked_in_stage1_inputs(
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert (
        cli_main(["validate-workload", str(ROOT / "examples/workloads/streaming-fixture-v1.json")])
        == 0
    )
    assert json.loads(capsys.readouterr().out)["model"] == "Stage1WorkloadDefinition"
    assert (
        cli_main(["validate-config", str(ROOT / "examples/configs/stage1-streaming-v1.json")]) == 0
    )
    assert json.loads(capsys.readouterr().out)["model"] == "Stage1RunConfiguration"
    assert (
        cli_main(["validate-fixture", str(ROOT / "examples/fixtures/streaming-fixture-v1.json")])
        == 0
    )
    assert json.loads(capsys.readouterr().out)["model"] == "FixtureDefinition"


def test_stage1_verifier_executes_real_loopback_fixture(
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert verify_stage1_main() == 0
    result = json.loads(capsys.readouterr().out)
    assert result["loopback_http_execution"] is True
    assert result["measured_request_count"] == 8
    assert result["failed_non_timeout_count"] == 2
    assert result["timeout_count"] == 1
