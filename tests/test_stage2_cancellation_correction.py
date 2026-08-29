from __future__ import annotations

import asyncio
import base64
import hashlib
import time
from datetime import UTC, datetime
from typing import Literal, cast

import httpx
import pytest
from pydantic import ValidationError

from llm_inference_systems.canonical import canonical_json_bytes, sha256_identity
from llm_inference_systems.stage2_control import (
    CancellationClassification,
    CancellationProbe,
    CancellationScrapeObservationEvidence,
    FirstGenerationDeliveryEvidence,
    ResidualStateEvidence,
    evaluate_cancellation,
)
from llm_inference_systems.stage2_experiment import (
    Stage2CancellationWireCapture,
    Stage2RawResponseBodyChunk,
    replay_stage2_cancellation_wire_capture,
)
from llm_inference_systems.stage2_fixture_server import (
    Stage2CancellationFixtureScenario,
    Stage2FixtureServer,
)
from llm_inference_systems.stage2_prometheus import PrometheusSnapshot, parse_prometheus_snapshot
from llm_inference_systems.stage2_protocol import (
    Stage2CancellationStreamCapture,
    Stage2ProtocolError,
    build_cancellation_request,
    correlate_request_logs,
    retain_raw_log_capture,
    retain_raw_log_records,
)
from tests.stage2_experiment_factories import make_experiment_attestation
from tests.stage2_factories import make_cancellation_probe, make_runtime_control, make_snapshot

PROMPT = tuple(range(64))
EXTERNAL_ID = "fixture-cancellation-correction"
SERVER_PROCESS_ID = "fixture-process"


def _rehash(values: dict[str, object], *, field: str = "identity_sha256") -> None:
    values[field] = sha256_identity(values, omit_fields=frozenset({field}))


def _generation_frame(
    wire: Stage2CancellationWireCapture,
    token_ids: tuple[int, ...],
    *,
    prompt: bool,
) -> bytes:
    choice: dict[str, object] = {
        "index": 0,
        "text": "".join(f"<fixture-{token_id}>" for token_id in token_ids),
        "token_ids": list(token_ids),
        "finish_reason": None,
    }
    if prompt:
        choice["prompt_token_ids"] = list(wire.request_body.canonical_request.prompt)
    return (
        b"data: "
        + canonical_json_bytes({"id": f"cmpl-{wire.external_request_id}", "choices": [choice]})
        + b"\n\n"
    )


def _wire_with_body(
    wire: Stage2CancellationWireCapture,
    body: bytes,
    *,
    complete_frame_count: int,
) -> Stage2CancellationWireCapture:
    values = wire.model_dump(mode="python")
    chunk_values = values["response_body_chunks"][0]
    assert isinstance(chunk_values, dict)
    observation_offset = cast(int, chunk_values["observation_offset_ns"])
    chunk_values["completed_sse_frame_observation_offsets_ns"] = tuple(
        observation_offset for _ in range(complete_frame_count)
    )
    chunk_values["exact_bytes_base64"] = base64.b64encode(body).decode("ascii")
    chunk_values["decoded_byte_count"] = len(body)
    chunk_values["sha256"] = hashlib.sha256(body).hexdigest()
    _rehash(chunk_values)
    typed_chunk = Stage2RawResponseBodyChunk.model_validate(chunk_values)
    temporary = wire.model_copy(update={"response_body_chunks": (typed_chunk,)})
    replay = replay_stage2_cancellation_wire_capture(temporary)
    chunks = (typed_chunk,)
    inventory_sha256 = sha256_identity(chunks)
    values["response_body_chunks"] = tuple(chunk.model_dump(mode="python") for chunk in chunks)
    values["parser_replay"] = replay.model_dump(mode="python")
    exchange = values["http_exchange"]
    assert isinstance(exchange, dict)
    exchange["response_body_byte_count"] = len(body)
    exchange["response_body_sha256"] = hashlib.sha256(body).hexdigest()
    exchange["response_body_inventory_sha256"] = inventory_sha256
    exchange["response_body_completion_observation_offset_ns"] = observation_offset
    _rehash(exchange)
    close = values["intentional_client_close"]
    assert isinstance(close, dict)
    close["raw_response_body_inventory_sha256"] = inventory_sha256
    close["parser_replay_identity_sha256"] = replay.identity_sha256
    _rehash(close)
    _rehash(values)
    return Stage2CancellationWireCapture.model_validate(values)


@pytest.fixture(scope="module")
def cancellation_wire() -> Stage2CancellationWireCapture:
    return make_experiment_attestation()[0].repetitions[0].cancellation_wire


async def _capture_loopback(
    scenario: Stage2CancellationFixtureScenario,
) -> tuple[
    Stage2CancellationStreamCapture,
    int,
    tuple[str, ...],
    tuple[int, ...],
]:
    async with Stage2FixtureServer(cancellation_scenario=scenario) as server:
        request = build_cancellation_request(EXTERNAL_ID, PROMPT)
        dispatch_offset_ns = time.monotonic_ns()
        capture = Stage2CancellationStreamCapture(
            external_base_id=EXTERNAL_ID,
            sent_prompt_token_ids=PROMPT,
            dispatch_offset_ns=dispatch_offset_ns,
        )
        async with (
            httpx.AsyncClient(
                base_url=f"http://127.0.0.1:{server.port}",
                trust_env=False,
                follow_redirects=False,
                http2=False,
                timeout=2.0,
            ) as client,
            client.stream(
                "POST",
                "/v1/completions",
                headers={"X-Request-Id": EXTERNAL_ID},
                json=request.model_dump(mode="json"),
            ) as response,
        ):
            assert response.status_code == 200
            capture.accept_response_headers(response.headers["X-Request-Id"], time.monotonic_ns())
            async for chunk in response.aiter_raw():
                if capture.feed(chunk, time.monotonic_ns()):
                    capture.close(time.monotonic_ns())
                    await response.aclose()
                    capture.complete_transport_close(time.monotonic_ns())
                    break
            else:
                capture.observe_clean_eof(time.monotonic_ns())
        for _ in range(100):
            if len(server.logs) >= 4:
                break
            await asyncio.sleep(0.01)
        return (
            capture,
            dispatch_offset_ns,
            tuple(server.logs),
            tuple(server.log_observation_offsets_ns),
        )


async def _capture_accepted_loopback_probe() -> tuple[
    CancellationProbe,
    Stage2CancellationStreamCapture,
]:
    """Build accepted evidence from real fixture HTTP streams and metric responses."""

    async with (
        Stage2FixtureServer(
            cancellation_scenario=(
                Stage2CancellationFixtureScenario.COMPLETE_FRAME_WITH_TRAILING_FRAGMENT
            )
        ) as server,
        httpx.AsyncClient(
            base_url=f"http://127.0.0.1:{server.port}",
            trust_env=False,
            follow_redirects=False,
            http2=False,
            timeout=2.0,
        ) as client,
    ):
        last_scrape_completion_offset_ns: int | None = None

        async def scrape(
            target_offset_ns: int,
            phase: Literal["PRE_DISPATCH", "DRAIN", "STABLE_GENERATION", "COOLDOWN", "LATER"],
            phase_ordinal: int,
        ) -> tuple[PrometheusSnapshot, CancellationScrapeObservationEvidence]:
            nonlocal last_scrape_completion_offset_ns
            earliest_offset_ns = target_offset_ns
            if last_scrape_completion_offset_ns is not None:
                earliest_offset_ns = max(
                    earliest_offset_ns,
                    last_scrape_completion_offset_ns + 100_000_000,
                )
            delay_ns = earliest_offset_ns - time.monotonic_ns()
            if delay_ns > 0:
                await asyncio.sleep(delay_ns / 1_000_000_000)
            dispatch_offset_ns = time.monotonic_ns()
            response = await client.get("/metrics")
            completion_offset_ns = time.monotonic_ns()
            last_scrape_completion_offset_ns = completion_offset_ns
            response.raise_for_status()
            snapshot = parse_prometheus_snapshot(
                response.text,
                process_start_id=SERVER_PROCESS_ID,
                scrape_wall_clock_utc=datetime.now(UTC),
                scrape_monotonic_offset_ns=completion_offset_ns,
            )
            return snapshot, CancellationScrapeObservationEvidence(
                phase=phase,
                phase_ordinal=phase_ordinal,
                scheduled_offset_ns=target_offset_ns,
                request_dispatch_offset_ns=dispatch_offset_ns,
                response_completion_offset_ns=completion_offset_ns,
                snapshot_identity_sha256=sha256_identity(snapshot),
            )

        pre_start = time.monotonic_ns() + 10_000_000
        pre_pairs = tuple(
            [
                await scrape(pre_start + index * 100_000_000, "PRE_DISPATCH", index)
                for index in range(10)
            ]
        )
        pre = tuple(snapshot for snapshot, _ in pre_pairs)
        pre_observations = tuple(observation for _, observation in pre_pairs)
        request = build_cancellation_request(EXTERNAL_ID, PROMPT)
        dispatch_offset_ns = time.monotonic_ns()
        capture = Stage2CancellationStreamCapture(
            external_base_id=EXTERNAL_ID,
            sent_prompt_token_ids=PROMPT,
            dispatch_offset_ns=dispatch_offset_ns,
        )
        async with client.stream(
            "POST",
            "/v1/completions",
            headers={"X-Request-Id": EXTERNAL_ID},
            json=request.model_dump(mode="json"),
        ) as response:
            assert response.status_code == 200
            capture.accept_response_headers(response.headers["X-Request-Id"], time.monotonic_ns())
            async for chunk in response.aiter_raw():
                if capture.feed(chunk, time.monotonic_ns()):
                    capture.close(time.monotonic_ns())
                    await response.aclose()
                    capture.complete_transport_close(time.monotonic_ns())
                    break
            else:
                capture.observe_clean_eof(time.monotonic_ns())
        for _ in range(100):
            if len(server.logs) >= 4 and server.running_requests == 0:
                break
            await asyncio.sleep(0.01)
        assert len(server.logs) == 4
        assert server.running_requests == 0
        raw_log_capture = retain_raw_log_capture(
            tuple(server.logs),
            source_stream_id=f"{SERVER_PROCESS_ID}-raw-log",
            observation_offsets_ns=tuple(server.log_observation_offsets_ns),
        )
        chain = correlate_request_logs(EXTERNAL_ID, raw_log_capture.records, cancellation=True)
        close_offset = capture.close_offset_ns
        delivery = capture.first_generation_delivery
        external_abort = chain.external_abort_log
        assert close_offset is not None
        assert delivery is not None
        assert external_abort is not None
        post_start = max(time.monotonic_ns(), external_abort.observation_offset_ns) + 10_000_000
        drain_pairs = tuple(
            [await scrape(post_start + index * 100_000_000, "DRAIN", index) for index in range(10)]
        )
        drain = tuple(snapshot for snapshot, _ in drain_pairs)
        drain_observations = tuple(observation for _, observation in drain_pairs)
        stable_tail_pairs = tuple(
            [
                await scrape(
                    post_start + (10 + index) * 100_000_000,
                    "STABLE_GENERATION",
                    index + 1,
                )
                for index in range(10)
            ]
        )
        stable_tail = tuple(snapshot for snapshot, _ in stable_tail_pairs)
        stable = (drain[-1], *stable_tail)
        stable_observations = (
            drain_observations[-1].model_copy(
                update={"phase": "STABLE_GENERATION", "phase_ordinal": 0}
            ),
            *(observation for _, observation in stable_tail_pairs),
        )
        cooldown_tail_pairs = tuple(
            [
                await scrape(
                    post_start + (20 + index) * 100_000_000,
                    "COOLDOWN",
                    index + 1,
                )
                for index in range(20)
            ]
        )
        cooldown_tail = tuple(snapshot for snapshot, _ in cooldown_tail_pairs)
        cooldown = (stable[-1], *cooldown_tail)
        cooldown_observations = (
            stable_observations[-1].model_copy(update={"phase": "COOLDOWN", "phase_ordinal": 0}),
            *(observation for _, observation in cooldown_tail_pairs),
        )
        raw_inventory = "project_processes=[]\nactive_requests=[]\n"
        probe = CancellationProbe(
            repetition_index=1,
            server_process_identity=SERVER_PROCESS_ID,
            identity_chain=chain,
            raw_log_capture=raw_log_capture,
            raw_log_capture_sha256=raw_log_capture.raw_bytes_sha256,
            raw_log_start_byte_offset=0,
            dispatch_offset_ns=dispatch_offset_ns,
            first_generation_delivery=FirstGenerationDeliveryEvidence(
                external_request_id=EXTERNAL_ID,
                response_body_id=f"cmpl-{EXTERNAL_ID}",
                generation_event_ordinal=delivery.generation_event_ordinal,
                body_chunk_ordinal=delivery.body_chunk_ordinal,
                observation_offset_ns=delivery.observation_offset_ns,
                output_token_ids=delivery.output_token_ids,
            ),
            client_close_offset_ns=close_offset,
            pre_dispatch_snapshots=pre,
            drain_snapshots=drain,
            stable_generation_snapshots=stable,
            cooldown_snapshots=cooldown,
            later_retained_snapshots=(),
            scrape_observations=(
                *pre_observations,
                *drain_observations,
                *stable_observations,
                *cooldown_observations,
            ),
            residual_state=ResidualStateEvidence(
                observation_offset_ns=max(
                    time.monotonic_ns(), cooldown[-1].scrape_monotonic_offset_ns
                ),
                raw_process_inventory=raw_inventory,
                raw_process_inventory_sha256=hashlib.sha256(raw_inventory.encode()).hexdigest(),
                active_request_ids=(),
                project_process_ids=(),
            ),
        )
        return probe, capture


@pytest.mark.parametrize(
    ("scenario", "first_ids", "event_count", "pending"),
    [
        pytest.param(
            Stage2CancellationFixtureScenario.SINGLE_TOKEN,
            (1000,),
            1,
            False,
            id="one-token-first-generation-event",
        ),
        pytest.param(
            Stage2CancellationFixtureScenario.GROUPED_TOKENS,
            (1000, 1001),
            1,
            False,
            id="multiple-token-first-generation-event",
        ),
        pytest.param(
            Stage2CancellationFixtureScenario.COALESCED_FRAMES,
            (1000,),
            2,
            False,
            id="two-complete-frames-in-one-body-read",
        ),
        pytest.param(
            Stage2CancellationFixtureScenario.COMPLETE_FRAME_WITH_TRAILING_FRAGMENT,
            (1000,),
            1,
            True,
            id="complete-frame-plus-incomplete-trailing-bytes",
        ),
    ],
)
def test_actual_loopback_accepts_every_first_generation_delivery_shape(
    scenario: Stage2CancellationFixtureScenario,
    first_ids: tuple[int, ...],
    event_count: int,
    pending: bool,
) -> None:
    capture, _, logs, _ = asyncio.run(_capture_loopback(scenario))
    delivery = capture.first_generation_delivery
    assert delivery is not None
    assert delivery.output_token_ids == first_ids
    assert len(capture.generation_events) == event_count
    assert bool(capture.pending_bytes) is pending
    assert len(capture.retained_raw_body_chunks) == 1
    assert all(
        event.observation_offset_ns == delivery.observation_offset_ns
        for event in capture.generation_events
    )
    assert logs[-2].startswith("Aborted request(s) cmpl-")
    assert logs[-1] == f"Request cmpl-{EXTERNAL_ID}-0 aborted."


def test_actual_loopback_pinned_order_reconstructs_accepted_drain_probe() -> None:
    probe, capture = asyncio.run(_capture_accepted_loopback_probe())
    result = evaluate_cancellation(probe)
    assert result.accepted
    assert result.classification is CancellationClassification.SERVER_ABORT_ACKNOWLEDGED_AND_DRAINED
    internal_abort = probe.identity_chain.internal_abort_log
    external_abort = probe.identity_chain.external_abort_log
    assert internal_abort is not None
    assert external_abort is not None
    assert probe.client_close_offset_ns <= internal_abort.observation_offset_ns
    assert internal_abort.observation_offset_ns <= external_abort.observation_offset_ns
    close_offset = capture.close_offset_ns
    close_completion_offset = capture.transport_close_completion_offset_ns
    assert close_offset is not None
    assert close_completion_offset is not None
    assert close_offset <= close_completion_offset
    assert all(
        observation.scheduled_offset_ns
        <= observation.request_dispatch_offset_ns
        < observation.response_completion_offset_ns
        for observation in probe.scrape_observations
    )
    assert all(
        observation.response_completion_offset_ns != observation.scheduled_offset_ns
        for observation in probe.scrape_observations
    )
    assert capture.pending_bytes
    assert hashlib.sha256(capture.pending_bytes).hexdigest()


@pytest.mark.parametrize(
    ("scenario", "message"),
    [
        pytest.param(
            Stage2CancellationFixtureScenario.GENERATION_TERMINAL,
            "generation terminal",
            id="generation-terminal-before-close",
        ),
        pytest.param(
            Stage2CancellationFixtureScenario.USAGE_TERMINAL,
            "usage terminal",
            id="usage-terminal-before-close",
        ),
        pytest.param(
            Stage2CancellationFixtureScenario.DONE_TERMINAL,
            r"\[DONE\]",
            id="done-before-close",
        ),
        pytest.param(
            Stage2CancellationFixtureScenario.CLEAN_EOF,
            "clean EOF",
            id="clean-eof-before-close",
        ),
    ],
)
def test_actual_loopback_rejects_every_terminal_before_close(
    scenario: Stage2CancellationFixtureScenario,
    message: str,
) -> None:
    with pytest.raises(Stage2ProtocolError, match=message):
        asyncio.run(_capture_loopback(scenario))


def test_same_frame_usage_terminal_before_close_is_rejected() -> None:
    capture = Stage2CancellationStreamCapture(
        external_base_id=EXTERNAL_ID,
        sent_prompt_token_ids=PROMPT,
        dispatch_offset_ns=0,
    )
    capture.accept_response_headers(EXTERNAL_ID, 1)
    body = {
        "id": f"cmpl-{EXTERNAL_ID}",
        "choices": [
            {
                "index": 0,
                "text": "<fixture-1000>",
                "token_ids": [1000],
                "finish_reason": None,
                "prompt_token_ids": list(PROMPT),
            }
        ],
        "usage": {"prompt_tokens": 64, "completion_tokens": 1, "total_tokens": 65},
    }
    with pytest.raises(Stage2ProtocolError, match="usage terminal"):
        capture.feed(b"data: " + canonical_json_bytes(body) + b"\n\n", 2)


def test_client_close_before_any_generation_event_is_rejected() -> None:
    capture = Stage2CancellationStreamCapture(
        external_base_id=EXTERNAL_ID,
        sent_prompt_token_ids=PROMPT,
        dispatch_offset_ns=0,
    )
    capture.accept_response_headers(EXTERNAL_ID, 1)
    with pytest.raises(Stage2ProtocolError, match="before a generation event"):
        capture.close(2)


def test_close_offset_before_triggering_read_is_rejected(
    cancellation_wire: Stage2CancellationWireCapture,
) -> None:
    capture = Stage2CancellationStreamCapture(
        external_base_id=cancellation_wire.external_request_id,
        sent_prompt_token_ids=cancellation_wire.request_body.canonical_request.prompt,
        dispatch_offset_ns=0,
    )
    capture.accept_response_headers(cancellation_wire.external_request_id, 1)
    body = _generation_frame(cancellation_wire, (1000,), prompt=True)
    assert capture.feed(body, 3)
    with pytest.raises(Stage2ProtocolError, match=r"not monotonic|precedes"):
        capture.close(2)


def test_post_close_response_byte_attribution_is_rejected(
    cancellation_wire: Stage2CancellationWireCapture,
) -> None:
    capture, _, _, _ = asyncio.run(
        _capture_loopback(Stage2CancellationFixtureScenario.POST_CLOSE_DATA)
    )
    with pytest.raises(Stage2ProtocolError, match="after client close"):
        capture.feed(
            _generation_frame(cancellation_wire, (1001,), prompt=False),
            time.monotonic_ns(),
        )


def test_grouped_first_event_is_accepted_and_all_ids_retained(
    cancellation_wire: Stage2CancellationWireCapture,
) -> None:
    wire = _wire_with_body(
        cancellation_wire,
        _generation_frame(cancellation_wire, (1000, 1001), prompt=True),
        complete_frame_count=1,
    )
    assert wire.parser_replay.first_generation_delivery.output_token_ids == (1000, 1001)
    assert wire.parser_replay.all_output_token_ids == (1000, 1001)


def test_two_coalesced_frames_are_accepted_and_both_replayed(
    cancellation_wire: Stage2CancellationWireCapture,
) -> None:
    body = _generation_frame(cancellation_wire, (1000,), prompt=True) + _generation_frame(
        cancellation_wire, (1001, 1002), prompt=False
    )
    wire = _wire_with_body(cancellation_wire, body, complete_frame_count=2)
    assert len(wire.parser_replay.generation_events) == 2
    assert wire.parser_replay.all_output_token_ids == (1000, 1001, 1002)
    assert len({event.observation_offset_ns for event in wire.parser_replay.generation_events}) == 1


def test_incomplete_trailing_bytes_are_accepted_with_exact_pending_identity(
    cancellation_wire: Stage2CancellationWireCapture,
) -> None:
    pending = b'data: {"id":"incomplete'
    wire = _wire_with_body(
        cancellation_wire,
        _generation_frame(cancellation_wire, (1000,), prompt=True) + pending,
        complete_frame_count=1,
    )
    replay = wire.parser_replay
    assert base64.b64decode(replay.pending_bytes_base64) == pending
    assert base64.b64decode(replay.parser_pending_bytes_base64) == pending
    assert replay.pending_byte_count == len(pending)
    assert replay.pending_bytes_sha256 == hashlib.sha256(pending).hexdigest()
    assert replay.parser_state_at_close == "INCOMPLETE_TRAILING_SSE_BYTES"


def test_crlf_pending_bytes_retain_raw_and_normalized_parser_state(
    cancellation_wire: Stage2CancellationWireCapture,
) -> None:
    raw_pending = b"data: partial\r\ndata: more"
    parser_pending = b"data: partial\ndata: more"
    wire = _wire_with_body(
        cancellation_wire,
        _generation_frame(cancellation_wire, (1000,), prompt=True) + raw_pending,
        complete_frame_count=1,
    )
    replay = wire.parser_replay
    assert base64.b64decode(replay.pending_bytes_base64) == raw_pending
    assert replay.pending_byte_count == len(raw_pending)
    assert base64.b64decode(replay.parser_pending_bytes_base64) == parser_pending
    assert replay.parser_pending_byte_count == len(parser_pending)
    assert replay.parser_pending_bytes_sha256 == hashlib.sha256(parser_pending).hexdigest()


def test_split_crlf_boundary_replays_through_actual_incremental_parser(
    cancellation_wire: Stage2CancellationWireCapture,
) -> None:
    capture = Stage2CancellationStreamCapture(
        external_base_id=cancellation_wire.external_request_id,
        sent_prompt_token_ids=cancellation_wire.request_body.canonical_request.prompt,
        dispatch_offset_ns=0,
    )
    capture.accept_response_headers(cancellation_wire.external_request_id, 1)
    assert not capture.feed(b": TEST_FIXTURE_ONLY split-CRLF\r", 2)
    assert capture.feed(b"\n\r\n" + _generation_frame(cancellation_wire, (1000,), prompt=True), 3)
    capture.close(4)
    capture.complete_transport_close(5)
    assert tuple(event.kind for event in capture.parsed_sse_events) == ("comment", "data")
    assert capture.pending_bytes == b""
    assert capture.parser_pending_bytes == b""


def test_pending_bytes_altered_without_updating_evidence_are_rejected(
    cancellation_wire: Stage2CancellationWireCapture,
) -> None:
    pending = b"data: partial"
    wire = _wire_with_body(
        cancellation_wire,
        _generation_frame(cancellation_wire, (1000,), prompt=True) + pending,
        complete_frame_count=1,
    )
    values = wire.parser_replay.model_dump(mode="python")
    values["pending_bytes_base64"] = base64.b64encode(pending + b"-drift").decode("ascii")
    with pytest.raises(ValidationError, match="pending-byte replay differs"):
        type(wire.parser_replay).model_validate(values)


@pytest.mark.parametrize(
    ("field", "replacement"),
    [
        pytest.param("pending_byte_count", 0, id="pending-count-disagrees"),
        pytest.param(
            "pending_bytes_sha256",
            hashlib.sha256(b"").hexdigest(),
            id="pending-hash-disagrees",
        ),
        pytest.param(
            "parser_state_at_close",
            "AT_SSE_FRAME_BOUNDARY",
            id="pending-parser-state-disagrees",
        ),
    ],
)
def test_pending_count_hash_or_parser_state_disagreement_is_rejected(
    cancellation_wire: Stage2CancellationWireCapture,
    field: str,
    replacement: object,
) -> None:
    wire = _wire_with_body(
        cancellation_wire,
        _generation_frame(cancellation_wire, (1000,), prompt=True) + b"data: partial",
        complete_frame_count=1,
    )
    values = wire.parser_replay.model_dump(mode="python")
    values[field] = replacement
    with pytest.raises(ValidationError, match="pending-byte replay differs"):
        type(wire.parser_replay).model_validate(values)


def test_old_external_then_internal_abort_order_is_rejected_as_pinned_runtime_drift() -> None:
    internal_id = f"cmpl-{EXTERNAL_ID}-0-deadbeef"
    lines = (
        f"Received request cmpl-{EXTERNAL_ID}-0: params: TEST_FIXTURE_ONLY.",
        f"Added request {internal_id}.",
        f"Request cmpl-{EXTERNAL_ID}-0 aborted.",
        f"Aborted request(s) {internal_id}.",
    )
    with pytest.raises(Stage2ProtocolError, match="pinned internal-before-external"):
        correlate_request_logs(
            EXTERNAL_ID,
            retain_raw_log_records(lines, source_stream_id=f"{SERVER_PROCESS_ID}-raw-log"),
            cancellation=True,
        )


@pytest.mark.parametrize(
    ("internal_offset_delta", "external_offset_delta", "case"),
    [
        pytest.param(-1, 1, "internal", id="internal-abort-before-close"),
        pytest.param(-2, -1, "external", id="external-abort-before-close"),
    ],
)
def test_abort_log_before_intentional_close_is_rejected(
    internal_offset_delta: int,
    external_offset_delta: int,
    case: str,
) -> None:
    probe = make_cancellation_probe()
    close = probe.client_close_offset_ns
    internal_id = probe.identity_chain.internal_engine_id
    lines = (
        (
            f"Received request cmpl-{probe.identity_chain.external_base_id}-0: "
            "params: TEST_FIXTURE_ONLY."
        ),
        f"Added request {internal_id}.",
        f"Aborted request(s) {internal_id}.",
        f"Request cmpl-{probe.identity_chain.external_base_id}-0 aborted.",
    )
    offsets = (
        probe.dispatch_offset_ns + 1,
        probe.dispatch_offset_ns + 2,
        close + internal_offset_delta,
        close + external_offset_delta,
    )
    raw_log_capture = retain_raw_log_capture(
        lines,
        source_stream_id=f"{probe.server_process_identity}-raw-log",
        observation_offsets_ns=offsets,
    )
    chain = correlate_request_logs(
        probe.identity_chain.external_base_id, raw_log_capture.records, cancellation=True
    )
    changed = probe.model_copy(
        update={
            "identity_chain": chain,
            "raw_log_capture": raw_log_capture,
            "raw_log_capture_sha256": raw_log_capture.raw_bytes_sha256,
        }
    )
    assert case in {"internal", "external"}
    assert (
        evaluate_cancellation(changed).classification is CancellationClassification.TERMINAL_UNKNOWN
    )


@pytest.mark.parametrize(
    "missing",
    [
        pytest.param("internal_abort_log", id="internal-abort-absent"),
        pytest.param("external_abort_log", id="external-abort-absent"),
    ],
)
def test_missing_internal_or_external_abort_is_rejected(missing: str) -> None:
    probe = make_cancellation_probe()
    chain = probe.identity_chain.model_copy(update={missing: None})
    changed = probe.model_copy(update={"identity_chain": chain})
    assert (
        evaluate_cancellation(changed).classification
        is CancellationClassification.UNKNOWN_ACKNOWLEDGEMENT
    )


@pytest.mark.parametrize(
    "duplicate",
    [
        pytest.param("internal", id="duplicate-internal-abort"),
        pytest.param("external", id="duplicate-external-abort"),
    ],
)
def test_duplicate_internal_or_external_abort_is_rejected(duplicate: str) -> None:
    probe = make_cancellation_probe()
    chain = probe.identity_chain
    internal_abort = chain.internal_abort_log
    external_abort = chain.external_abort_log
    assert internal_abort is not None
    assert external_abort is not None
    lines = [
        chain.request_received_log.raw_record,
        chain.request_add_log.raw_record,
        internal_abort.raw_record,
        external_abort.raw_record,
    ]
    lines.append(lines[2 if duplicate == "internal" else 3])
    with pytest.raises(Stage2ProtocolError, match="both external and internal"):
        correlate_request_logs(
            chain.external_base_id,
            retain_raw_log_records(
                tuple(lines), source_stream_id=f"{probe.server_process_identity}-raw-log"
            ),
            cancellation=True,
        )


def test_duplicate_abort_in_complete_raw_log_capture_invalidates_probe() -> None:
    probe = make_cancellation_probe()
    lines = tuple(record.raw_record for record in probe.raw_log_capture.records)
    raw_log_capture = retain_raw_log_capture(
        (*lines, lines[-1]),
        source_stream_id=probe.raw_log_capture.source_stream_id,
    )
    changed = probe.model_copy(
        update={
            "raw_log_capture": raw_log_capture,
            "raw_log_capture_sha256": raw_log_capture.raw_bytes_sha256,
        }
    )
    assert (
        evaluate_cancellation(changed).classification
        is CancellationClassification.ID_CORRELATION_FAILURE
    )


def test_scrape_schedule_cannot_replace_actual_http_observation_clock() -> None:
    probe = make_cancellation_probe()
    observations = list(probe.scrape_observations)
    stable = next(
        index
        for index, observation in enumerate(observations)
        if observation.phase == "STABLE_GENERATION" and observation.phase_ordinal == 1
    )
    observations[stable] = observations[stable].model_copy(
        update={"scheduled_offset_ns": observations[stable].scheduled_offset_ns + 1}
    )
    changed = probe.model_copy(update={"scrape_observations": tuple(observations)})
    assert (
        evaluate_cancellation(changed).classification
        is CancellationClassification.RESIDUAL_WORK_TIMEOUT
    )


@pytest.mark.parametrize("phase", ["PRE_DISPATCH", "DRAIN"])
def test_scrape_schedule_cannot_hide_compressed_actual_observations(phase: str) -> None:
    probe = make_cancellation_probe()
    observations = list(probe.scrape_observations)
    if phase == "PRE_DISPATCH":
        start = probe.dispatch_offset_ns - 1_000
        snapshots = tuple(make_snapshot(start + ordinal) for ordinal in range(10))
        snapshot_update = {"pre_dispatch_snapshots": snapshots}
        observation_start = 0
    else:
        start = probe.drain_snapshots[-1].scrape_monotonic_offset_ns
        snapshots = tuple(
            make_snapshot(start + ordinal, prompt=64, generation=1, abort=1)
            for ordinal in range(10)
        )
        stable = (snapshots[-1], *probe.stable_generation_snapshots[1:])
        snapshot_update = {
            "drain_snapshots": snapshots,
            "stable_generation_snapshots": stable,
        }
        observation_start = 10
        observations[20] = observations[20].model_copy(
            update={
                "request_dispatch_offset_ns": snapshots[-1].scrape_monotonic_offset_ns,
                "response_completion_offset_ns": snapshots[-1].scrape_monotonic_offset_ns,
                "snapshot_identity_sha256": sha256_identity(snapshots[-1]),
            }
        )
    for ordinal, snapshot in enumerate(snapshots):
        index = observation_start + ordinal
        observations[index] = observations[index].model_copy(
            update={
                "request_dispatch_offset_ns": snapshot.scrape_monotonic_offset_ns,
                "response_completion_offset_ns": snapshot.scrape_monotonic_offset_ns,
                "snapshot_identity_sha256": sha256_identity(snapshot),
            }
        )
    changed = probe.model_copy(
        update={**snapshot_update, "scrape_observations": tuple(observations)}
    )
    assert (
        evaluate_cancellation(changed).classification
        is CancellationClassification.RESIDUAL_WORK_TIMEOUT
    )


def test_cancellation_probe_rejects_different_server_process_binding() -> None:
    probe = make_cancellation_probe()
    changed = probe.model_copy(update={"server_process_identity": "different-process"})
    assert (
        evaluate_cancellation(changed).classification
        is CancellationClassification.ID_CORRELATION_FAILURE
    )


def test_runtime_control_rejects_different_cancellation_repetition_binding() -> None:
    control = make_runtime_control(repetition_index=1)
    changed_probe = control.cancellation_probe.model_copy(update={"repetition_index": 2})
    values = control.model_dump(mode="python")
    values["cancellation_probe"] = changed_probe.model_dump(mode="python")
    values["cancellation_result"] = evaluate_cancellation(changed_probe).model_dump(mode="python")
    with pytest.raises(ValidationError, match="repetition or server process"):
        type(control).model_validate(values)
