from __future__ import annotations

import hashlib
import json

import pytest
from pydantic import ValidationError

from llm_inference_systems.canonical import sha256_identity
from llm_inference_systems.stage2_contracts import (
    RawLogRecord,
    RequestIdentityChain,
    Stage2CompletionRequest,
    Stage2RequestEnvelope,
    Stage2RequestEvidence,
)
from llm_inference_systems.stage2_protocol import (
    Stage2ProtocolError,
    Stage2StreamValidator,
    build_cancellation_request,
    build_completion_request,
    correlate_request_logs,
    retain_raw_log_records,
    validate_effective_request,
)

PROMPT = tuple(range(64))
EXTERNAL_ID = "stage2-fixture-001"
INTERNAL_ID = "cmpl-stage2-fixture-001-0-deadbeef"
FIXTURE_IDENTITY = "f" * 64


def _data(value: dict[str, object]) -> bytes:
    return f"data: {json.dumps(value, sort_keys=True, separators=(',', ':'))}\n\n".encode()


def _log_records(lines: tuple[str, ...]) -> tuple[RawLogRecord, ...]:
    return retain_raw_log_records(lines, source_stream_id="fixture-log")


def _choice(
    token_ids: tuple[int, ...],
    *,
    finish_reason: str | None,
    prompt: tuple[int, ...] | None = None,
    body_id: str = f"cmpl-{EXTERNAL_ID}",
) -> bytes:
    choice: dict[str, object] = {
        "index": 0,
        "text": "".join(f"<fixture-{token}>" for token in token_ids),
        "token_ids": list(token_ids),
        "finish_reason": finish_reason,
    }
    if prompt is not None:
        choice["prompt_token_ids"] = list(prompt)
    return _data({"id": body_id, "choices": [choice]})


def _usage(
    *,
    prompt_tokens: int = 64,
    completion_tokens: int = 32,
    total_tokens: int = 96,
) -> bytes:
    return _data(
        {
            "id": f"cmpl-{EXTERNAL_ID}",
            "choices": [],
            "usage": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": total_tokens,
            },
            "metrics": {
                "time_to_first_token_ms": 1.25,
                "generation_time_ms": 2.5,
                "queue_time_ms": 0.0,
                "mean_itl_ms": None,
                "tokens_per_second": 12.8,
            },
        }
    )


def _validator() -> Stage2StreamValidator:
    validator = Stage2StreamValidator(
        external_base_id=EXTERNAL_ID,
        sent_prompt_token_ids=PROMPT,
        dispatch_offset_ns=0,
        fixture_identity_sha256=FIXTURE_IDENTITY,
    )
    validator.accept_response_headers(EXTERNAL_ID, 10)
    return validator


def _identity_chain() -> RequestIdentityChain:
    return correlate_request_logs(
        EXTERNAL_ID,
        retain_raw_log_records(
            (
                f"Received request cmpl-{EXTERNAL_ID}-0: params: TEST_FIXTURE_ONLY.",
                f"Added request {INTERNAL_ID}.",
            ),
            source_stream_id="fixture-log",
        ),
        cancellation=False,
    )


def _complete(
    *,
    finish_only: bool = False,
    grouped: bool = False,
    identity_chain: RequestIdentityChain | None = None,
) -> Stage2RequestEvidence:
    validator = _validator()
    width = 2 if grouped else 1
    groups = tuple(tuple(range(index, index + width)) for index in range(0, 32, width))
    for index, group in enumerate(groups):
        final_content = index == len(groups) - 1 and not finish_only
        validator.feed(
            _choice(
                group,
                finish_reason="length" if final_content else None,
                prompt=PROMPT if index == 0 else None,
            ),
            20 + index,
        )
    terminal_offset = 20 + len(groups) + 1
    if finish_only:
        validator.feed(_choice((), finish_reason="length"), terminal_offset)
    validator.feed(_usage(), terminal_offset + 10)
    validator.feed(b"data: [DONE]\n\n", terminal_offset + 20)
    return validator.close_transport(
        terminal_offset + 30,
        identity_chain=identity_chain or _identity_chain(),
    )


def _validator_ready_for_usage() -> tuple[Stage2StreamValidator, int]:
    validator = _validator()
    for token in range(32):
        validator.feed(
            _choice(
                (token,),
                finish_reason="length" if token == 31 else None,
                prompt=PROMPT if token == 0 else None,
            ),
            20 + token,
        )
    return validator, 60


def _usage_with_metrics(metrics: object, *, include_metrics: bool = True) -> bytes:
    value: dict[str, object] = {
        "id": f"cmpl-{EXTERNAL_ID}",
        "choices": [],
        "usage": {"prompt_tokens": 64, "completion_tokens": 32, "total_tokens": 96},
    }
    if include_metrics:
        value["metrics"] = metrics
    return _data(value)


def test_exact_completion_request_and_cancellation_shape() -> None:
    envelope = build_completion_request(EXTERNAL_ID, PROMPT)
    assert envelope.x_request_id == envelope.body.request_id == EXTERNAL_ID
    assert envelope.body.prompt == PROMPT
    assert envelope.body.max_tokens == envelope.body.min_tokens == 32
    cancellation = build_cancellation_request("stage2-cancel", PROMPT)
    assert cancellation.max_tokens == cancellation.min_tokens == 512
    assert cancellation.ignore_eos is True


def test_request_unknown_field_and_prompt_length_are_rejected() -> None:
    value = build_completion_request(EXTERNAL_ID, PROMPT).body.model_dump(mode="json")
    value["arbitrary"] = True
    with pytest.raises(ValidationError):
        Stage2CompletionRequest.model_validate(value)
    value.pop("arbitrary")
    value["prompt"] = list(range(63))
    with pytest.raises(ValidationError):
        Stage2CompletionRequest.model_validate(value)


def test_header_body_request_id_mismatch_is_rejected() -> None:
    body = build_completion_request(EXTERNAL_ID, PROMPT).body
    with pytest.raises(ValidationError, match="X-Request-Id"):
        Stage2RequestEnvelope(x_request_id="different-id", body=body)


def test_effective_request_cannot_change_required_field() -> None:
    requested = build_completion_request(EXTERNAL_ID, PROMPT).body
    changed = requested.model_copy(update={"prompt": tuple(reversed(PROMPT))})
    with pytest.raises(Stage2ProtocolError, match="effective request"):
        validate_effective_request(requested, changed)


def test_response_header_mismatch_and_duplicate_are_rejected() -> None:
    validator = Stage2StreamValidator(
        external_base_id=EXTERNAL_ID,
        sent_prompt_token_ids=PROMPT,
        dispatch_offset_ns=0,
    )
    with pytest.raises(Stage2ProtocolError, match="X-Request-Id"):
        validator.accept_response_headers("different", 1)
    validator.accept_response_headers(EXTERNAL_ID, 2)
    with pytest.raises(Stage2ProtocolError, match="duplicate"):
        validator.accept_response_headers(EXTERNAL_ID, 3)


def test_token_bearing_generation_terminal_reconciles_exactly() -> None:
    evidence = _complete()
    assert evidence.final_output_token_ids == tuple(range(32))
    assert evidence.returned_prompt_token_ids == PROMPT
    assert evidence.terminal_event_carried_token_ids is True
    assert evidence.usage.total_tokens == 96
    assert evidence.client_generation_tpot.value_ns is not None
    assert evidence.server_per_request_metrics.time_to_first_token_ms == 1.25
    assert evidence.server_per_request_metrics.mean_itl_ms is None
    assert evidence.local_prompt_token_count is None


def test_finish_only_generation_terminal_is_accepted() -> None:
    evidence = _complete(finish_only=True)
    assert evidence.terminal_event_carried_token_ids is False
    assert len(evidence.final_output_token_ids) == 32
    assert evidence.token_events[-1].output_token_ids == ()
    assert evidence.token_events[-1].finish_reason == "length"


def test_grouped_token_event_disables_only_token_observation_metrics() -> None:
    evidence = _complete(grouped=True)
    assert evidence.client_generation_tpot.value_ns is None
    assert evidence.client_generation_tpot.unavailable_reason == "GROUPED_TOKEN_EVENT"
    assert evidence.token_observation_itl.unavailable_reason == "GROUPED_TOKEN_EVENT"
    assert evidence.stream_output_gap_ns
    assert evidence.usage.total_tokens == 96
    assert len(evidence.final_output_token_ids) == 32


def test_durable_request_evidence_reconstructs_text_timing_metrics_and_disagreements() -> None:
    evidence = _complete()
    value = evidence.model_dump(mode="python")
    with pytest.raises(ValidationError, match="does not reconstruct"):
        Stage2RequestEvidence.model_validate({**value, "output_text_sha256": "0" * 64})
    with pytest.raises(ValidationError, match="disagreements"):
        Stage2RequestEvidence.model_validate({**value, "local_prompt_token_count": 63})
    events = tuple(
        event.model_copy(update={"prompt_token_ids": None}) for event in evidence.token_events
    )
    with pytest.raises(ValidationError, match="matching first token event"):
        Stage2RequestEvidence.model_validate({**value, "token_events": events})
    reconciled = Stage2RequestEvidence.model_validate(
        {
            **value,
            "local_prompt_token_count": 63,
            "disagreements": ("LOCAL_PROMPT_COUNT_DIFFERS_FROM_SERVER_USAGE",),
        }
    )
    assert reconciled.local_prompt_token_count == 63


def test_split_and_coalesced_sse_input_are_accepted() -> None:
    validator = _validator()
    first = _choice((0,), finish_reason=None, prompt=PROMPT)
    validator.feed(first[:7], 20)
    validator.feed(first[7:], 21)
    second = _choice((1,), finish_reason=None)
    third = _choice((2,), finish_reason=None)
    validator.feed(second + third, 22)
    for token in range(3, 32):
        validator.feed(
            _choice((token,), finish_reason="length" if token == 31 else None),
            30 + token,
        )
    validator.feed(_usage(), 80)
    validator.feed(b"data: [DONE]\n\n", 90)
    evidence = validator.close_transport(100, identity_chain=_identity_chain())
    assert evidence.final_output_token_ids == tuple(range(32))
    assert b"".join(chunk.data for chunk in validator.retained_raw_body_chunks).startswith(first)


def test_coalesced_generation_usage_and_done_capture_distinct_frame_times() -> None:
    frame_offsets = iter((61, 62))
    validator = Stage2StreamValidator(
        external_base_id=EXTERNAL_ID,
        sent_prompt_token_ids=PROMPT,
        dispatch_offset_ns=0,
        fixture_identity_sha256=FIXTURE_IDENTITY,
        frame_clock=frame_offsets.__next__,
    )
    validator.accept_response_headers(EXTERNAL_ID, 10)
    for token in range(31):
        validator.feed(
            _choice(
                (token,),
                finish_reason=None,
                prompt=PROMPT if token == 0 else None,
            ),
            20 + token,
        )
    validator.feed(
        _choice((31,), finish_reason="length") + _usage() + b"data: [DONE]\n\n",
        60,
    )
    evidence = validator.close_transport(63, identity_chain=_identity_chain())
    assert evidence.timing.generation_terminal_offset_ns == 60
    assert evidence.timing.usage_terminal_offset_ns == 61
    assert evidence.timing.protocol_terminal_offset_ns == 62


def test_malformed_stream_retains_exact_raw_failure_bytes() -> None:
    validator = _validator()
    malformed = b'data: {"unterminated":\n\n'
    with pytest.raises(Stage2ProtocolError, match="malformed SSE JSON"):
        validator.feed(malformed, 20)
    retained = validator.retained_raw_body_chunks
    assert len(retained) == 1
    assert retained[0].data == malformed


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("id", '"cmpl-stage2-fixture-001"'),
        ("choices", "[]"),
        ("usage", "{}"),
        ("metrics", "{}"),
    ],
)
def test_sse_response_json_rejects_duplicate_fields(field: str, value: str) -> None:
    validator = _validator()
    duplicated = (
        'data: {"id":"cmpl-stage2-fixture-001","choices":[],'
        f'"{field}":{value},"{field}":{value}'
        "}\n\n"
    ).encode("ascii")
    with pytest.raises(Stage2ProtocolError, match="duplicate field"):
        validator.feed(duplicated, 20)


@pytest.mark.parametrize(
    ("body_id", "prompt"),
    [
        ("cmpl-wrong", PROMPT),
        (f"cmpl-{EXTERNAL_ID}", tuple(range(1, 65))),
    ],
)
def test_body_id_or_returned_prompt_mismatch_is_rejected(
    body_id: str,
    prompt: tuple[int, ...],
) -> None:
    validator = _validator()
    if body_id != f"cmpl-{EXTERNAL_ID}":
        with pytest.raises(Stage2ProtocolError, match="body ID"):
            validator.feed(_choice((0,), finish_reason=None, prompt=prompt, body_id=body_id), 20)
        return
    for token in range(32):
        validator.feed(
            _choice(
                (token,),
                finish_reason="length" if token == 31 else None,
                prompt=prompt if token == 0 else None,
            ),
            20 + token,
        )
    validator.feed(_usage(), 60)
    validator.feed(b"data: [DONE]\n\n", 70)
    with pytest.raises(Stage2ProtocolError, match="prompt token IDs"):
        validator.close_transport(80, identity_chain=_identity_chain())


def test_usage_mismatch_is_rejected() -> None:
    validator = _validator()
    for token in range(32):
        validator.feed(
            _choice(
                (token,),
                finish_reason="length" if token == 31 else None,
                prompt=PROMPT if token == 0 else None,
            ),
            20 + token,
        )
    with pytest.raises(Stage2ProtocolError, match="usage reconciliation"):
        validator.feed(_usage(total_tokens=95), 60)


def test_usage_terminal_omitting_metrics_object_is_rejected() -> None:
    validator, offset = _validator_ready_for_usage()
    with pytest.raises(Stage2ProtocolError, match="requires per-request metrics"):
        validator.feed(_usage_with_metrics({}, include_metrics=False), offset)


@pytest.mark.parametrize(
    "missing",
    [
        "time_to_first_token_ms",
        "generation_time_ms",
        "queue_time_ms",
        "mean_itl_ms",
        "tokens_per_second",
    ],
)
def test_usage_metrics_omitting_each_required_key_is_rejected(missing: str) -> None:
    metrics: dict[str, object] = {
        "time_to_first_token_ms": 1.0,
        "generation_time_ms": 2.0,
        "queue_time_ms": 0.0,
        "mean_itl_ms": 0.1,
        "tokens_per_second": 10.0,
    }
    metrics.pop(missing)
    validator, offset = _validator_ready_for_usage()
    with pytest.raises(Stage2ProtocolError, match="exact five-field shape"):
        validator.feed(_usage_with_metrics(metrics), offset)


def test_usage_metrics_arbitrary_key_is_rejected() -> None:
    metrics = {
        "time_to_first_token_ms": 1.0,
        "generation_time_ms": 2.0,
        "queue_time_ms": 0.0,
        "mean_itl_ms": 0.1,
        "tokens_per_second": 10.0,
        "client_latency_ms": 3.0,
    }
    validator, offset = _validator_ready_for_usage()
    with pytest.raises(Stage2ProtocolError, match="exact five-field shape"):
        validator.feed(_usage_with_metrics(metrics), offset)


@pytest.mark.parametrize("invalid", [-1.0, float("nan"), float("inf"), "1.0", True])
def test_usage_metrics_invalid_values_are_rejected(invalid: object) -> None:
    metrics: dict[str, object] = {
        "time_to_first_token_ms": invalid,
        "generation_time_ms": 2.0,
        "queue_time_ms": 0.0,
        "mean_itl_ms": 0.1,
        "tokens_per_second": 10.0,
    }
    validator, offset = _validator_ready_for_usage()
    with pytest.raises(Stage2ProtocolError, match="finite nonnegative"):
        validator.feed(_usage_with_metrics(metrics), offset)


def test_usage_metrics_complete_finite_and_explicit_null_shapes_are_retained() -> None:
    finite = _complete().server_per_request_metrics
    assert finite.model_dump(mode="python") == {
        "time_to_first_token_ms": 1.25,
        "generation_time_ms": 2.5,
        "queue_time_ms": 0.0,
        "mean_itl_ms": None,
        "tokens_per_second": 12.8,
    }
    validator, offset = _validator_ready_for_usage()
    metrics = dict.fromkeys(type(finite).model_fields)
    validator.feed(_usage_with_metrics(metrics), offset)
    validator.feed(b"data: [DONE]\n\n", offset + 10)
    evidence = validator.close_transport(offset + 20, identity_chain=_identity_chain())
    assert all(value is None for value in evidence.server_per_request_metrics.model_dump().values())


def test_terminal_requires_exactly_32_output_ids() -> None:
    validator = _validator()
    validator.feed(_choice((0,), finish_reason=None, prompt=PROMPT), 20)
    with pytest.raises(Stage2ProtocolError, match="exactly 32"):
        validator.feed(_choice((1,), finish_reason="length"), 21)


@pytest.mark.parametrize("failure", ["usage-before-generation", "done-before-usage"])
def test_terminal_reordering_is_rejected(failure: str) -> None:
    validator = _validator()
    if failure == "usage-before-generation":
        with pytest.raises(Stage2ProtocolError, match="before generation"):
            validator.feed(_usage(), 20)
    else:
        with pytest.raises(Stage2ProtocolError, match="before usage"):
            validator.feed(b"data: [DONE]\n\n", 20)


def test_duplicate_usage_done_and_post_terminal_data_are_rejected() -> None:
    validator = _validator()
    for token in range(32):
        validator.feed(
            _choice(
                (token,),
                finish_reason="length" if token == 31 else None,
                prompt=PROMPT if token == 0 else None,
            ),
            20 + token,
        )
    validator.feed(_usage(), 60)
    with pytest.raises(Stage2ProtocolError, match="duplicate usage"):
        validator.feed(_usage(), 61)
    validator.feed(b"data: [DONE]\n\n", 70)
    with pytest.raises(Stage2ProtocolError, match="post-protocol"):
        validator.feed(b": late\n\n", 71)


def test_transport_close_requires_done_and_strictly_later_offset() -> None:
    validator = _validator()
    with pytest.raises(Stage2ProtocolError, match=r"without \[DONE\]"):
        validator.close_transport(20, identity_chain=_identity_chain())


def test_identity_chain_requires_exact_single_log_chain() -> None:
    lines = (
        f"Received request cmpl-{EXTERNAL_ID}-0: params: TEST_FIXTURE_ONLY.",
        f"Added request {INTERNAL_ID}.",
    )
    chain = correlate_request_logs(EXTERNAL_ID, _log_records(lines), cancellation=False)
    assert chain.internal_engine_id == INTERNAL_ID
    assert chain.request_add_log.raw_record_sha256 == hashlib.sha256(lines[1].encode()).hexdigest()
    with pytest.raises(Stage2ProtocolError, match="ambiguous"):
        correlate_request_logs(
            EXTERNAL_ID,
            _log_records((*lines, lines[1])),
            cancellation=False,
        )


def test_request_evidence_requires_matching_validated_log_chain() -> None:
    other = "other-fixture"
    wrong_chain = correlate_request_logs(
        other,
        _log_records(
            (
                f"Received request cmpl-{other}-0: params: TEST_FIXTURE_ONLY.",
                f"Added request cmpl-{other}-0-cafebabe.",
            )
        ),
        cancellation=False,
    )
    with pytest.raises(Stage2ProtocolError, match="identity chain differs"):
        _complete(identity_chain=wrong_chain)
    evidence = _complete()
    assert evidence.request_identity_chain_sha256 == sha256_identity(_identity_chain())


def test_cancellation_identity_requires_both_abort_logs() -> None:
    base = (
        f"Received request cmpl-{EXTERNAL_ID}-0: params: TEST_FIXTURE_ONLY.",
        f"Added request {INTERNAL_ID}.",
    )
    aborts = (
        f"Request cmpl-{EXTERNAL_ID}-0 aborted.",
        f"Aborted request(s) {INTERNAL_ID}.",
    )
    chain = correlate_request_logs(
        EXTERNAL_ID,
        _log_records((*base, *aborts)),
        cancellation=True,
    )
    assert chain.external_abort_log is not None
    assert chain.internal_abort_log is not None
    with pytest.raises(Stage2ProtocolError, match="both external and internal"):
        correlate_request_logs(
            EXTERNAL_ID,
            _log_records((*base, aborts[0])),
            cancellation=True,
        )


def test_cross_request_log_correlation_is_rejected() -> None:
    lines = (
        f"Received request cmpl-{EXTERNAL_ID}-0: params: TEST_FIXTURE_ONLY.",
        f"Added request {INTERNAL_ID}.",
        "Added request cmpl-other-0-cafebabe.",
    )
    with pytest.raises(Stage2ProtocolError, match="cross-request"):
        correlate_request_logs(EXTERNAL_ID, _log_records(lines), cancellation=False)
