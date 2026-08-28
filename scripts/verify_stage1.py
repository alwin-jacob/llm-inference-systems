"""Execute and verify two independent real-loopback Stage 1 fixture runs."""

from __future__ import annotations

import asyncio
import tempfile
from pathlib import Path

from llm_inference_systems.canonical import canonical_json
from llm_inference_systems.runner import run_fixture_to_directory
from llm_inference_systems.stage1_comparison import compare_validated_bundles
from llm_inference_systems.stage1_contracts import (
    Stage1ComparisonPolicy,
    Stage1FailureKind,
    Stage1TerminalClass,
    StreamEvidenceKind,
)
from llm_inference_systems.stage1_metrics import derive_stage1_request_metrics


async def _verify(root: Path, temporary_root: Path) -> dict[str, object]:
    workload_path = root / "examples/workloads/streaming-fixture-v1.json"
    configuration_path = root / "examples/configs/stage1-streaming-v1.json"
    fixture_path = root / "examples/fixtures/streaming-fixture-v1.json"
    baseline = await run_fixture_to_directory(
        workload_path=workload_path,
        configuration_path=configuration_path,
        fixture_path=fixture_path,
        output_directory=temporary_root / "run-a",
    )
    candidate = await run_fixture_to_directory(
        workload_path=workload_path,
        configuration_path=configuration_path,
        fixture_path=fixture_path,
        output_directory=temporary_root / "run-b",
    )
    if baseline.manifest.loopback_host != "127.0.0.1":
        raise AssertionError("fixture server did not use IPv4 loopback")
    if baseline.manifest.semantic_fingerprint != candidate.manifest.semantic_fingerprint:
        raise AssertionError("repeat fixture semantic fingerprints differ")
    if baseline.manifest.content_sha256 == candidate.manifest.content_sha256:
        raise AssertionError("run-specific content identities unexpectedly match")

    requests = {request.case_id: request for request in baseline.requests}
    first_body = requests["success-first-body-before-token"]
    if not (
        first_body.timing.first_response_body_bytes_offset_ns is not None
        and first_body.timing.first_output_token_offset_ns is not None
        and first_body.timing.first_response_body_bytes_offset_ns
        < first_body.timing.first_output_token_offset_ns
    ):
        raise AssertionError("first-body-before-first-token ordering proof failed")
    multi = requests["success-multi-token-event"]
    if derive_stage1_request_metrics(multi).itl_ns:
        raise AssertionError("multi-token SSE event fabricated ITL")
    single = requests["success-single-output-token"]
    if derive_stage1_request_metrics(single).tpot_ns is not None:
        raise AssertionError("one-token success fabricated TPOT")
    malformed = requests["malformed-after-partial-output"]
    if (
        malformed.failure is None
        or malformed.failure.kind is not Stage1FailureKind.PROTOCOL_MALFORMED_STREAM
        or malformed.output_token_count != 1
    ):
        raise AssertionError("malformed partial-output evidence was not retained")
    http_error = requests["http-error"]
    if (
        http_error.failure is None
        or http_error.failure.kind is not Stage1FailureKind.HTTP_STATUS
        or http_error.http_status != 503
    ):
        raise AssertionError("HTTP failure evidence was not retained")
    timeout = requests["timeout-after-partial-body"]
    timeout_raw = [
        event
        for event in baseline.stream_events
        if event.request_id == timeout.request_id
        and event.kind is StreamEvidenceKind.RAW_BODY_CHUNK
    ]
    if (
        timeout.terminal_class is not Stage1TerminalClass.TIMEOUT
        or not timeout_raw
        or timeout.failure is None
        or timeout.failure.kind is not Stage1FailureKind.TIMEOUT
    ):
        raise AssertionError("timeout partial-body evidence was not retained")
    summary = baseline.summary
    if (summary.failure_rate.numerator, summary.failure_rate.denominator) != (2, 8):
        raise AssertionError("failure-rate numerator/denominator proof failed")
    if (summary.timeout_rate.numerator, summary.timeout_rate.denominator) != (1, 8):
        raise AssertionError("timeout-rate numerator/denominator proof failed")
    if summary.warmup_record_count != 1 or summary.warmup_excluded_count != 1:
        raise AssertionError("warmup retention/exclusion proof failed")
    if summary.observed_max_client_concurrency != 2:
        raise AssertionError("observed client concurrency proof failed")
    if summary.observed_server_batch_size is not None:
        raise AssertionError("server batch size was fabricated")

    policy = Stage1ComparisonPolicy.model_validate_json(
        (root / "examples/configs/stage1-regression-policy-v1.json").read_bytes()
    )
    report = compare_validated_bundles(baseline, candidate, policy)
    if not report.policy_passed:
        raise AssertionError("semantic-only repeat comparison did not pass")
    return {
        "stage": 1,
        "evidence_scope": "TEST_FIXTURE_ONLY",
        "loopback_http_execution": True,
        "loopback_host": "127.0.0.1",
        "real_runtime_execution": False,
        "model_execution": False,
        "tokenizer_execution": False,
        "gpu_execution": False,
        "cuda_execution": False,
        "performance_claim_allowed": False,
        "historical_authentication_effect": "NONE",
        "measured_request_count": summary.attempted_measured_requests,
        "successful_request_count": summary.successful_measured_requests,
        "failed_non_timeout_count": summary.failed_non_timeout_measured_requests,
        "timeout_count": summary.timed_out_measured_requests,
        "observed_max_client_concurrency": summary.observed_max_client_concurrency,
        "server_batch_observed": False,
        "semantic_reproduction": True,
        "comparison_policy_passed": True,
    }


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    with tempfile.TemporaryDirectory(prefix="lis-stage1-verify-") as temporary:
        result = asyncio.run(_verify(root, Path(temporary)))
    print(canonical_json(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
