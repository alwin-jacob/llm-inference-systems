from __future__ import annotations

import asyncio
import inspect
import json
import time
from datetime import UTC, datetime
from pathlib import Path

import httpx
import pytest
from jsonschema import ValidationError as JsonSchemaValidationError  # type: ignore[import-untyped]
from jsonschema import validate as validate_json_schema
from pydantic import ValidationError

from llm_inference_systems import __version__
from llm_inference_systems.stage2_contracts import (
    ExecutionLockStatus,
    LoopbackEndpoint,
    Stage2ExecutionLock,
    Stage2RequestEvidence,
    Stage2RunConfiguration,
)
from llm_inference_systems.stage2_fixture_server import Stage2FixtureServer
from llm_inference_systems.stage2_prometheus import (
    parse_prometheus_snapshot,
    select_exact_series,
)
from llm_inference_systems.stage2_protocol import (
    Stage2StreamValidator,
    build_completion_request,
    correlate_request_logs,
)
from scripts.verify_checked_stage1_evidence import (
    HISTORICAL_STAGE1_PACKAGE_VERSION,
    _verify,
)
from scripts.verify_stage2a import FROZEN_HASHES, HISTORICAL_STAGE1_UV_LOCK_SHA256
from scripts.verify_stage2a import main as verify_stage2a

ROOT = Path(__file__).resolve().parents[1]


def test_loopback_endpoint_rejects_arbitrary_host_or_url() -> None:
    endpoint = LoopbackEndpoint(host="127.0.0.1", port=8000)
    assert endpoint.completions_url == "http://127.0.0.1:8000/v1/completions"
    with pytest.raises(ValidationError):
        LoopbackEndpoint.model_validate({"host": "0.0.0.0", "port": 8000})
    with pytest.raises(ValidationError):
        LoopbackEndpoint.model_validate(
            {"host": "127.0.0.1", "port": 8000, "url": "https://example.invalid"}
        )


def test_stage2_config_unknown_fields_and_launch_drift_are_rejected() -> None:
    path = ROOT / "examples/configs/stage2a-protocol-fixture-v1.json"
    value = json.loads(path.read_bytes())
    Stage2RunConfiguration.model_validate_json(path.read_bytes())
    value["launch_arguments"] = tuple(value["launch_arguments"])
    with pytest.raises(ValidationError):
        Stage2RunConfiguration.model_validate({**value, "unknown": True})
    value["launch_arguments"] = value["launch_arguments"][:-1]
    with pytest.raises(ValidationError, match="launch arguments"):
        Stage2RunConfiguration.model_validate(value)


async def _exercise_fixture(
    *, finish_only: bool, grouped: bool
) -> tuple[bytes, Stage2FixtureServer, Stage2RequestEvidence]:
    server = Stage2FixtureServer(
        finish_only_terminal=finish_only,
        grouped_tokens=grouped,
    )
    await server.start()
    try:
        envelope = build_completion_request("fixture-http-001", tuple(range(64)))
        origin_ns = time.monotonic_ns()

        def offset_ns() -> int:
            return time.monotonic_ns() - origin_ns

        validator = Stage2StreamValidator(
            external_base_id=envelope.x_request_id,
            sent_prompt_token_ids=envelope.body.prompt,
            dispatch_offset_ns=0,
            frame_clock=offset_ns,
        )
        async with httpx.AsyncClient(
            base_url=f"http://127.0.0.1:{server.port}",
            trust_env=False,
            follow_redirects=False,
            http2=False,
        ) as client:
            async with client.stream(
                "POST",
                "/v1/completions",
                headers={"X-Request-Id": envelope.x_request_id},
                json=envelope.body.model_dump(mode="json"),
            ) as response:
                assert response.status_code == 200
                assert response.headers["X-Request-Id"] == envelope.x_request_id
                validator.accept_response_headers(response.headers["X-Request-Id"], offset_ns())
                async for chunk in response.aiter_bytes():
                    validator.feed(chunk, offset_ns())
            chain = correlate_request_logs(
                envelope.x_request_id, tuple(server.logs), cancellation=False
            )
            evidence = validator.close_transport(offset_ns(), identity_chain=chain)
            metrics = await client.get("/metrics")
            assert metrics.status_code == 200
            snapshot = parse_prometheus_snapshot(
                metrics.text,
                process_start_id="stage2-fixture-process",
                scrape_wall_clock_utc=datetime(2026, 8, 28, tzinfo=UTC),
                scrape_monotonic_offset_ns=1,
            )
            assert select_exact_series(snapshot, "vllm:prompt_tokens_total").value == 64
            assert select_exact_series(snapshot, "vllm:generation_tokens_total").value == 32
            body = b"".join(chunk.data for chunk in validator.retained_raw_body_chunks)
    finally:
        await server.stop()
    return body, server, evidence


@pytest.mark.parametrize(
    ("finish_only", "grouped"),
    [(False, False), (True, False), (False, True)],
)
def test_cpu_fixture_server_stream_logs_and_metrics(finish_only: bool, grouped: bool) -> None:
    body, server, evidence = asyncio.run(
        _exercise_fixture(finish_only=finish_only, grouped=grouped)
    )
    assert b"data: [DONE]\n\n" in body
    assert b'"total_tokens":96' in body
    if finish_only:
        assert b'"finish_reason":"length","index":0,"text":"","token_ids":[]' in body
        assert evidence.terminal_event_carried_token_ids is False
    else:
        assert evidence.terminal_event_carried_token_ids is True
    if grouped:
        assert evidence.client_generation_tpot.unavailable_reason == "GROUPED_TOKEN_EVENT"
    assert evidence.final_output_token_ids == tuple(range(1000, 1032))
    chain = correlate_request_logs("fixture-http-001", tuple(server.logs), cancellation=False)
    assert chain.internal_engine_id.endswith("deadbeef")


def test_cpu_fixture_rejects_header_body_mismatch() -> None:
    async def exercise() -> None:
        async with Stage2FixtureServer() as server:
            envelope = build_completion_request("fixture-http-001", tuple(range(64)))
            async with httpx.AsyncClient(
                base_url=f"http://127.0.0.1:{server.port}", trust_env=False
            ) as client:
                response = await client.post(
                    "/v1/completions",
                    headers={"X-Request-Id": "different"},
                    json=envelope.body.model_dump(mode="json"),
                )
                assert response.status_code == 400

    asyncio.run(exercise())


def test_fixture_server_has_no_host_or_endpoint_constructor_parameter() -> None:
    parameters = inspect.signature(Stage2FixtureServer).parameters
    assert "host" not in parameters
    assert "endpoint" not in parameters
    assert "url" not in parameters


def test_execution_lock_is_separate_uninstalled_and_explicitly_blocked() -> None:
    lock = Stage2ExecutionLock.model_validate_json(
        (ROOT / "execution-lock/stage2-execution-lock.json").read_bytes()
    )
    assert lock.status is ExecutionLockStatus.BLOCKED_BINARY_RETRIEVAL_AUTHORIZATION_REQUIRED
    assert lock.installed is False
    assert lock.executed is False
    assert lock.vllm_git_revision == "2cf0a6915ce544dc493a0990f2ea38d81601128a"
    assert lock.qwen_model_repository == "Qwen/Qwen2.5-0.5B-Instruct"
    assert lock.qwen_snapshot_source_url.endswith(lock.qwen_snapshot_revision)
    vllm = next(item for item in lock.artifacts if item.package == "vllm")
    assert vllm.sha256 == "8ec943b66a0c6b4351d0778e99d7bacfca5788dd8eedd49425092bacb61c4397"
    torchvision = next(item for item in lock.artifacts if item.package == "torchvision")
    assert torchvision.sha256 is None


@pytest.mark.parametrize(
    "mutation",
    ["duplicate", "source", "hash", "model-repository", "model-source", "false-complete"],
)
def test_execution_lock_rejects_supply_chain_or_status_drift(mutation: str) -> None:
    path = ROOT / "execution-lock/stage2-execution-lock.json"
    lock = Stage2ExecutionLock.model_validate_json(path.read_bytes())
    value = lock.model_dump(mode="python")
    artifacts = list(lock.artifacts)
    if mutation == "duplicate":
        artifacts[3] = artifacts[0]
    elif mutation == "source":
        artifacts[0] = artifacts[0].model_copy(
            update={"source_url": "https://packages.invalid/vllm.whl"}
        )
    elif mutation == "hash":
        artifacts[1] = artifacts[1].model_copy(update={"sha256": "0" * 64})
    elif mutation == "model-repository":
        value["qwen_model_repository"] = "substituted/model"
    elif mutation == "model-source":
        value["qwen_snapshot_source_url"] = "https://models.invalid/substituted"
    else:
        value["status"] = "COMPLETE"
    value["artifacts"] = tuple(artifacts)
    with pytest.raises(ValidationError):
        Stage2ExecutionLock.model_validate(value)


@pytest.mark.parametrize("mutation", ["duplicate", "artifact-source", "model-source"])
def test_execution_lock_schema_encodes_exact_supply_chain_allowlist(mutation: str) -> None:
    value = json.loads((ROOT / "execution-lock/stage2-execution-lock.json").read_bytes())
    schema = json.loads((ROOT / "schemas/execution-lock-v0.3.0.schema.json").read_bytes())
    if mutation == "duplicate":
        value["artifacts"][3] = value["artifacts"][0]
    elif mutation == "artifact-source":
        value["artifacts"][0]["source_url"] = "https://packages.invalid/substituted"
    else:
        value["qwen_snapshot_source_url"] = "https://models.invalid/substituted"
    with pytest.raises(JsonSchemaValidationError):
        validate_json_schema(value, schema)


def test_historical_stage1_verifies_under_current_package_0_3_0() -> None:
    assert __version__ == "0.3.0"
    assert HISTORICAL_STAGE1_PACKAGE_VERSION == "0.2.0"
    result = _verify(ROOT / "artifacts/stage1-fixture/2026-08-27")
    assert result["status"] == "verified"


def test_stage2_verifier_covers_frozen_bytes_and_dependency_boundary(
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert len(FROZEN_HASHES) == 30
    assert verify_stage2a() == 0
    output = json.loads(capsys.readouterr().out)
    assert output["package_version"] == "0.3.0"
    assert output["forbidden_runtime_imports"] is False
    assert HISTORICAL_STAGE1_UV_LOCK_SHA256 == (
        "748fd114d05ea6e96c058f41b8a1ee0736d30339f100179e3ee7c47c7e6c59e6"
    )


def test_ordinary_lock_has_no_runtime_gpu_or_model_dependency() -> None:
    lock = (ROOT / "uv.lock").read_text().casefold()
    forbidden = ('name = "vllm"', 'name = "torch"', 'name = "transformers"')
    assert not any(name in lock for name in forbidden)


def test_no_ordinary_source_or_test_imports_runtime_packages() -> None:
    forbidden = (
        "import " + "vllm",
        "import " + "torch",
        "import " + "transformers",
        "import " + "huggingface_hub",
    )
    paths = tuple((ROOT / "src").rglob("*.py")) + tuple((ROOT / "tests").rglob("*.py"))
    for path in paths:
        text = path.read_text()
        assert not any(value in text for value in forbidden)
