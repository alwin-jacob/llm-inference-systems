from __future__ import annotations

import base64
import hashlib
import json
from datetime import timedelta
from pathlib import Path

import pytest
from pydantic import ValidationError

from llm_inference_systems.canonical import canonical_json_bytes, sha256_identity
from llm_inference_systems.stage2_attestation import PrometheusRawScrapeCapture
from llm_inference_systems.stage2_experiment import (
    Stage2CancellationWireCapture,
    Stage2ExperimentAttestation,
    Stage2ExperimentError,
    Stage2RawResponseBodyChunk,
    Stage2RequestWireCapture,
    reconstruct_experiment_attestation,
    replay_stage2_cancellation_wire_capture,
)
from llm_inference_systems.stage2_prometheus import (
    PrometheusProtocolError,
    derive_measured_window_deltas,
    parse_prometheus_snapshot,
)
from llm_inference_systems.stage2_transport import (
    CollectorWireCaptureProvenance,
    Stage2HTTPExchangeCapture,
    Stage2LosslessHeaderField,
)
from tests.stage2_experiment_factories import (
    make_experiment_attestation,
    write_synthetic_experiment_directory,
)
from tests.stage2_factories import make_launch_spec, make_snapshot, prometheus_exposition


@pytest.fixture(scope="module")
def experiment() -> Stage2ExperimentAttestation:
    return make_experiment_attestation()[0]


def _rehash(values: dict[str, object], field: str = "identity_sha256") -> None:
    values[field] = sha256_identity(values, omit_fields=frozenset({field}))


@pytest.mark.parametrize(
    ("field", "invalid"),
    [
        ("method", "GET"),
        ("scheme", "https"),
        ("host", "localhost"),
        ("port", 8001),
        ("request_target", "/v1/chat/completions"),
        ("requested_http_version", "HTTP/2"),
        ("observed_response_http_version", "HTTP/2"),
        ("response_status", 201),
        ("normalized_response_media_type", "application/json"),
    ],
)
def test_measured_exchange_rejects_endpoint_protocol_or_media_drift(
    experiment: Stage2ExperimentAttestation,
    field: str,
    invalid: object,
) -> None:
    exchange = experiment.repetitions[0].measured_requests[0].wire_capture.http_exchange
    values = exchange.model_dump(mode="python")
    values[field] = invalid
    _rehash(values)
    with pytest.raises(ValidationError):
        Stage2HTTPExchangeCapture.model_validate(values)


@pytest.mark.parametrize("mutation", ["count", "complete", "omit", "duplicate-singleton"])
def test_complete_httpx_header_capture_rejects_bypass(
    experiment: Stage2ExperimentAttestation,
    mutation: str,
) -> None:
    exchange = experiment.repetitions[0].measured_requests[0].wire_capture.http_exchange
    values = exchange.model_dump(mode="python")
    request_headers = values["request_headers"]
    assert isinstance(request_headers, dict)
    if mutation == "count":
        request_headers["observed_field_count"] = 7
    elif mutation == "complete":
        request_headers["capture_complete_at_declared_layer"] = False
    elif mutation == "omit":
        fields = list(request_headers["fields"])
        del fields[2]
        for ordinal, field in enumerate(fields):
            field["ordinal"] = ordinal
            _rehash(field)
        request_headers["observed_field_count"] = len(fields)
        request_headers["fields"] = tuple(fields)
        request_headers["normalized_view"] = tuple(
            (field["normalized_name"], field["normalized_value"]) for field in fields
        )
    else:
        fields = list(request_headers["fields"])
        duplicate = dict(fields[0])
        duplicate["ordinal"] = len(fields)
        _rehash(duplicate)
        fields.append(duplicate)
        request_headers["fields"] = tuple(fields)
        request_headers["observed_field_count"] = len(fields)
        request_headers["normalized_view"] = (
            *request_headers["normalized_view"],
            (duplicate["normalized_name"], duplicate["normalized_value"]),
        )
    _rehash(request_headers)
    values["request_header_count"] = request_headers["observed_field_count"]
    _rehash(values)
    with pytest.raises(ValidationError):
        Stage2HTTPExchangeCapture.model_validate(values)


@pytest.mark.parametrize(
    "name",
    [
        b"Authorization",
        b"Proxy-Authorization",
        b"Cookie",
        b"Set-Cookie",
        b"X-Api-Key",
        b"Private-Token",
        b"X-Amz-Security-Token",
        b"X-Auth-Token",
        b"Authentication",
        b"X-Access-Key",
        b"X-Session-Token",
        b"X-Api-Token",
        b"X-Api-Secret",
        b"X-Credential",
        b"X-Password",
    ],
)
def test_secret_headers_are_rejected_before_complete_capture_commit(name: bytes) -> None:
    value = b"credential"
    values: dict[str, object] = {
        "ordinal": 0,
        "name_base64": base64.b64encode(name).decode("ascii"),
        "value_base64": base64.b64encode(value).decode("ascii"),
        "name_byte_count": len(name),
        "value_byte_count": len(value),
        "name_sha256": hashlib.sha256(name).hexdigest(),
        "value_sha256": hashlib.sha256(value).hexdigest(),
        "normalized_name": name.decode("ascii").casefold(),
        "normalized_value": value.decode("ascii"),
    }
    _rehash(values)
    with pytest.raises(ValidationError, match="secret-bearing"):
        Stage2LosslessHeaderField.model_validate(values)


def test_exchange_is_bound_to_launch_process_and_measured_case(
    experiment: Stage2ExperimentAttestation,
) -> None:
    request = experiment.repetitions[0].measured_requests[0]
    exchange_values = request.wire_capture.http_exchange.model_dump(mode="python")
    exchange_values["launch_spec_identity_sha256"] = "0" * 64
    _rehash(exchange_values)
    changed_exchange = Stage2HTTPExchangeCapture.model_validate(exchange_values)
    with pytest.raises(ValueError, match="launch or server process"):
        changed_exchange.require_launch_and_process(
            make_launch_spec(), server_process_identity="server-process-1"
        )

    exchange_values = request.wire_capture.http_exchange.model_dump(mode="python")
    exchange_values["evidence_unit_id"] = "stage2-case-v1-16"
    _rehash(exchange_values)
    wire_values = request.wire_capture.model_dump(mode="python")
    wire_values["http_exchange"] = exchange_values
    _rehash(wire_values)
    with pytest.raises(ValidationError, match="detached"):
        Stage2RequestWireCapture.model_validate(wire_values)


@pytest.mark.parametrize(
    ("collection", "omitted_index", "message"),
    [
        ("request_headers", 2, "ordinary client field"),
        ("response_headers", 4, "ordinary server field"),
    ],
)
def test_future_collector_cannot_claim_complete_after_ordinary_header_omission(
    experiment: Stage2ExperimentAttestation,
    collection: str,
    omitted_index: int,
    message: str,
) -> None:
    exchange = experiment.repetitions[0].measured_requests[0].wire_capture.http_exchange
    values = exchange.model_dump(mode="python")
    provenance: dict[str, object] = {
        "capture_kind": "COLLECTOR_CAPTURE",
        "evidence_scope": "FUTURE_REAL_RUNTIME",
        "classification": "FUTURE_REAL_RUNTIME",
        "collector_identity_sha256": "1" * 64,
        "server_process_identity": exchange.server_process_identity,
        "model_snapshot_identity_sha256": "2" * 64,
        "environment_identity_sha256": "3" * 64,
    }
    _rehash(provenance)
    CollectorWireCaptureProvenance.model_validate(provenance)
    values["provenance"] = provenance
    headers = values[collection]
    fields = list(headers["fields"])
    del fields[omitted_index]
    for ordinal, field in enumerate(fields):
        field["ordinal"] = ordinal
        _rehash(field)
    headers["fields"] = tuple(fields)
    headers["observed_field_count"] = len(fields)
    headers["normalized_view"] = tuple(
        (field["normalized_name"], field["normalized_value"]) for field in fields
    )
    _rehash(headers)
    count_field = (
        "request_header_count" if collection == "request_headers" else "response_header_count"
    )
    values[count_field] = len(fields)
    _rehash(values)
    with pytest.raises(ValidationError, match=message):
        Stage2HTTPExchangeCapture.model_validate(values)


@pytest.mark.parametrize(
    ("field", "replacement"),
    [
        ("prompt", list(range(63))),
        ("max_tokens", 511),
        ("min_tokens", 511),
        ("ignore_eos", False),
    ],
)
def test_cancellation_exact_request_bytes_reject_fixed_field_drift(
    experiment: Stage2ExperimentAttestation,
    field: str,
    replacement: object,
) -> None:
    wire = experiment.repetitions[0].cancellation_wire
    values = wire.request_body.model_dump(mode="python")
    decoded = json.loads(base64.b64decode(values["exact_bytes_base64"]))
    decoded[field] = replacement
    raw = canonical_json_bytes(decoded)
    values["exact_bytes_base64"] = base64.b64encode(raw).decode("ascii")
    values["byte_count"] = len(raw)
    values["sha256"] = hashlib.sha256(raw).hexdigest()
    _rehash(values)
    with pytest.raises(ValidationError):
        type(wire.request_body).model_validate(values)


def test_minimal_cancellation_boolean_summary_is_structurally_rejected() -> None:
    with pytest.raises(ValidationError):
        Stage2CancellationWireCapture.model_validate(
            {
                "schema_version": "0.3.0",
                "repetition_index": 1,
                "external_request_id": "E_cancel",
                "transport_closed": True,
            }
        )


@pytest.mark.parametrize(
    "mutation",
    [
        "missing-request-body",
        "body-header-id",
        "missing-response-headers",
        "status",
        "missing-chunks",
        "reordered-chunk",
        "duplicated-chunk",
        "truncated-chunk",
        "appended-chunk",
        "appended-empty-frame",
        "altered-chunk",
        "parser-token-count",
        "close-before-token",
        "close-request-id",
        "identity-chain",
    ],
)
def test_cancellation_wire_rejects_raw_replay_and_identity_bypasses(
    experiment: Stage2ExperimentAttestation,
    mutation: str,
) -> None:
    wire = experiment.repetitions[0].cancellation_wire
    values = wire.model_dump(mode="python")
    if mutation == "missing-request-body":
        values.pop("request_body")
    elif mutation == "body-header-id":
        values["request_body"]["canonical_request"]["request_id"] = "different-cancel"
    elif mutation == "missing-response-headers":
        values["http_exchange"].pop("response_headers")
    elif mutation == "status":
        values["http_exchange"]["response_status"] = 204
    elif mutation == "missing-chunks":
        values["response_body_chunks"] = []
    elif mutation == "reordered-chunk":
        values["response_body_chunks"][0]["ordinal"] = 1
    elif mutation == "duplicated-chunk":
        values["response_body_chunks"] = (
            *values["response_body_chunks"],
            values["response_body_chunks"][0],
        )
    elif mutation in {
        "truncated-chunk",
        "appended-chunk",
        "appended-empty-frame",
        "altered-chunk",
    }:
        chunk = values["response_body_chunks"][0]
        raw = base64.b64decode(chunk["exact_bytes_base64"])
        if mutation == "truncated-chunk":
            raw = raw[:-2]
        elif mutation == "appended-chunk":
            raw += b"data: partial"
        elif mutation == "appended-empty-frame":
            raw += b"\n\n"
        else:
            raw = raw.replace(b"1000", b"1001")
        chunk["exact_bytes_base64"] = base64.b64encode(raw).decode("ascii")
        chunk["decoded_byte_count"] = len(raw)
        chunk["sha256"] = hashlib.sha256(raw).hexdigest()
        _rehash(chunk)
    elif mutation == "parser-token-count":
        values["parser_replay"]["first_generation_token"]["output_token_ids"] = [1000, 1001]
    elif mutation == "close-before-token":
        values["intentional_client_close"]["close_observation_offset_ns"] = (
            wire.parser_replay.first_generation_token.observation_offset_ns - 1
        )
    elif mutation == "close-request-id":
        values["intentional_client_close"]["external_request_id"] = "different-close"
        _rehash(values["intentional_client_close"])
    else:
        values["request_identity"]["identity_chain"]["external_base_id"] = "other-cancel"
    with pytest.raises(ValidationError):
        Stage2CancellationWireCapture.model_validate(values)


def test_repetition_rejects_cancellation_dispatch_detached_from_probe(
    experiment: Stage2ExperimentAttestation,
) -> None:
    repetition = experiment.repetitions[0]
    values = repetition.model_dump(mode="python")
    values["cancellation_wire"]["request_body"]["transmission_offset_ns"] += 1
    values["cancellation_wire"]["http_exchange"][
        "request_body_transmission_observation_offset_ns"
    ] += 1
    values["cancellation_wire"]["http_exchange"]["request_headers"]["observation_offset_ns"] += 1
    _rehash(values["cancellation_wire"]["request_body"])
    _rehash(values["cancellation_wire"]["http_exchange"]["request_headers"])
    _rehash(values["cancellation_wire"]["http_exchange"])
    _rehash(values["cancellation_wire"])
    _rehash(values)
    with pytest.raises(ValidationError):
        type(repetition).model_validate(values)


def test_cancellation_rejects_fully_rehashed_appended_empty_sse_frame(
    experiment: Stage2ExperimentAttestation,
) -> None:
    wire = experiment.repetitions[0].cancellation_wire
    values = wire.model_dump(mode="python")
    chunk = values["response_body_chunks"][0]
    raw = base64.b64decode(chunk["exact_bytes_base64"]) + b"\n\n"
    chunk["exact_bytes_base64"] = base64.b64encode(raw).decode("ascii")
    chunk["decoded_byte_count"] = len(raw)
    chunk["sha256"] = hashlib.sha256(raw).hexdigest()
    _rehash(chunk)
    inventory_sha = sha256_identity(values["response_body_chunks"])
    exchange = values["http_exchange"]
    exchange["response_body_byte_count"] = len(raw)
    exchange["response_body_sha256"] = hashlib.sha256(raw).hexdigest()
    exchange["response_body_inventory_sha256"] = inventory_sha
    _rehash(exchange)
    values["parser_replay"]["raw_response_body_inventory_sha256"] = inventory_sha
    _rehash(values["parser_replay"])
    close = values["intentional_client_close"]
    close["raw_response_body_inventory_sha256"] = inventory_sha
    close["parser_replay_identity_sha256"] = values["parser_replay"]["identity_sha256"]
    _rehash(close)
    _rehash(values)
    with pytest.raises(ValidationError, match="empty or unobserved SSE frame"):
        Stage2CancellationWireCapture.model_validate(values)


def test_cancellation_rejects_chunk_observation_overlapping_prior_frame(
    experiment: Stage2ExperimentAttestation,
) -> None:
    wire = experiment.repetitions[0].cancellation_wire
    values = wire.model_dump(mode="python")
    first_token_offset = wire.parser_replay.first_generation_token.observation_offset_ns
    source_chunk = values["response_body_chunks"][0]
    token_bytes = base64.b64decode(source_chunk["exact_bytes_base64"])
    comment_bytes = b": first\n\n: second\n\n"
    comment_chunk = dict(source_chunk)
    comment_chunk.update(
        {
            "ordinal": 0,
            "observation_offset_ns": first_token_offset - 1,
            "completed_sse_frame_observation_offsets_ns": (
                first_token_offset - 1,
                first_token_offset + 1,
            ),
            "exact_bytes_base64": base64.b64encode(comment_bytes).decode("ascii"),
            "decoded_byte_count": len(comment_bytes),
            "sha256": hashlib.sha256(comment_bytes).hexdigest(),
        }
    )
    _rehash(comment_chunk)
    token_chunk = dict(source_chunk)
    token_chunk.update(
        {
            "ordinal": 1,
            "observation_offset_ns": first_token_offset,
            "completed_sse_frame_observation_offsets_ns": (first_token_offset,),
        }
    )
    _rehash(token_chunk)
    typed_chunks = (
        Stage2RawResponseBodyChunk.model_validate(comment_chunk),
        Stage2RawResponseBodyChunk.model_validate(token_chunk),
    )
    inventory_sha = sha256_identity(typed_chunks)
    temporary = wire.model_copy(update={"response_body_chunks": typed_chunks})
    replay = replay_stage2_cancellation_wire_capture(temporary)
    values["response_body_chunks"] = tuple(
        chunk.model_dump(mode="python") for chunk in typed_chunks
    )
    values["parser_replay"] = replay.model_dump(mode="python")
    raw_body = comment_bytes + token_bytes
    exchange = values["http_exchange"]
    exchange["response_body_byte_count"] = len(raw_body)
    exchange["response_body_sha256"] = hashlib.sha256(raw_body).hexdigest()
    exchange["response_body_inventory_sha256"] = inventory_sha
    exchange["response_body_completion_observation_offset_ns"] = first_token_offset + 1
    _rehash(exchange)
    close = values["intentional_client_close"]
    close["raw_response_body_inventory_sha256"] = inventory_sha
    close["parser_replay_identity_sha256"] = replay.identity_sha256
    _rehash(close)
    _rehash(values)
    with pytest.raises(ValidationError, match="overlap prior SSE-frame"):
        Stage2CancellationWireCapture.model_validate(values)


@pytest.mark.parametrize(
    "terminal", ["generation", "usage", "same-frame-usage", "done", "clean-eof"]
)
def test_cancellation_rejects_completion_terminals_before_intentional_close(
    experiment: Stage2ExperimentAttestation,
    terminal: str,
) -> None:
    wire = experiment.repetitions[0].cancellation_wire
    values = wire.model_dump(mode="python")
    if terminal == "clean-eof":
        values["http_exchange"]["transport_terminal_classification"] = "CLEAN_EOF"
    else:
        chunk = values["response_body_chunks"][0]
        if terminal == "done":
            raw = b"data: [DONE]\n\n"
        elif terminal == "usage":
            raw = canonical_json_bytes(
                {"id": f"cmpl-{wire.external_request_id}", "choices": [], "usage": {}}
            )
            raw = b"data: " + raw + b"\n\n"
        else:
            decoded = json.loads(base64.b64decode(chunk["exact_bytes_base64"])[6:-2])
            if terminal == "same-frame-usage":
                decoded["usage"] = {"prompt_tokens": 64, "completion_tokens": 1}
            else:
                decoded["choices"][0]["finish_reason"] = "length"
            raw = b"data: " + canonical_json_bytes(decoded) + b"\n\n"
        chunk["exact_bytes_base64"] = base64.b64encode(raw).decode("ascii")
        chunk["decoded_byte_count"] = len(raw)
        chunk["sha256"] = hashlib.sha256(raw).hexdigest()
        _rehash(chunk)
        inventory_sha = sha256_identity(values["response_body_chunks"])
        exchange = values["http_exchange"]
        exchange["response_body_byte_count"] = len(raw)
        exchange["response_body_sha256"] = hashlib.sha256(raw).hexdigest()
        exchange["response_body_inventory_sha256"] = inventory_sha
        _rehash(exchange)
        values["parser_replay"]["raw_response_body_inventory_sha256"] = inventory_sha
        _rehash(values["parser_replay"])
        close = values["intentional_client_close"]
        close["raw_response_body_inventory_sha256"] = inventory_sha
        close["parser_replay_identity_sha256"] = values["parser_replay"]["identity_sha256"]
        _rehash(close)
        _rehash(values)
    with pytest.raises(ValidationError):
        Stage2CancellationWireCapture.model_validate(values)


def test_cancellation_wire_file_is_required_by_repetition_and_aggregate(tmp_path: Path) -> None:
    root = tmp_path / "missing-cancellation-wire"
    write_synthetic_experiment_directory(root)
    (root / "repetition-01/raw/cancellation/client-wire.json").unlink()
    with pytest.raises(Stage2ExperimentError, match="inventory"):
        reconstruct_experiment_attestation(root)


@pytest.mark.parametrize("reason", ["abort", "stop", "error", "repetition"])
def test_measured_window_rejects_every_nonlength_finish_reason(reason: str) -> None:
    before = make_snapshot(1)
    reason_values = {name: int(name == reason) for name in ("abort", "stop", "error", "repetition")}
    after = make_snapshot(
        2,
        prompt=1024,
        generation=512,
        length=16,
        abort=reason_values["abort"],
        stop=reason_values["stop"],
        error=reason_values["error"],
        repetition=reason_values["repetition"],
    )
    with pytest.raises(PrometheusProtocolError, match="deltas differ"):
        derive_measured_window_deltas(before, after)


@pytest.mark.parametrize(
    "mutation",
    ["absent", "duplicated", "ambiguous", "cross-process", "reset", "label-drift", "extra"],
)
def test_complete_finish_reason_family_rejects_raw_exposition_bypasses(mutation: str) -> None:
    before_raw = prometheus_exposition()
    after_raw = prometheus_exposition(prompt=1024, generation=512, length=16)
    if mutation == "absent":
        after_raw = (
            "\n".join(
                line for line in after_raw.splitlines() if 'finished_reason="abort"' not in line
            )
            + "\n"
        )
    elif mutation == "duplicated":
        line = next(line for line in after_raw.splitlines() if 'finished_reason="abort"' in line)
        after_raw += line + "\n"
    elif mutation == "ambiguous":
        after_raw += (
            'vllm:request_success_total{engine="0",extra="x",finished_reason="abort",'
            'model_name="qwen2.5-0.5b-instruct-stage2"} 0.0\n'
        )
    elif mutation == "label-drift":
        after_raw = after_raw.replace(
            'finished_reason="stop",model_name="qwen2.5-0.5b-instruct-stage2"',
            'finished_reason="stop",model_name="different-model"',
        )
    elif mutation == "extra":
        after_raw += (
            'vllm:request_success_total{engine="0",finished_reason="cancelled",'
            'model_name="qwen2.5-0.5b-instruct-stage2"} 1.0\n'
        )
    before = parse_prometheus_snapshot(
        prometheus_exposition(prompt=1) if mutation == "reset" else before_raw,
        process_start_id="process-a",
        scrape_wall_clock_utc=make_snapshot(1).scrape_wall_clock_utc,
        scrape_monotonic_offset_ns=1,
    )
    after = parse_prometheus_snapshot(
        prometheus_exposition(length=16) if mutation == "reset" else after_raw,
        process_start_id="process-b" if mutation == "cross-process" else "process-a",
        scrape_wall_clock_utc=make_snapshot(2).scrape_wall_clock_utc,
        scrape_monotonic_offset_ns=2,
    )
    with pytest.raises(PrometheusProtocolError):
        derive_measured_window_deltas(before, after)


@pytest.mark.parametrize(
    ("field", "invalid"),
    [
        ("method", "POST"),
        ("request_target", "/wrong"),
        ("host", "localhost"),
        ("port", 8001),
        ("observed_response_http_version", "HTTP/2"),
        ("response_status", 500),
        ("normalized_response_media_type", "application/json"),
    ],
)
def test_prometheus_scrape_exchange_rejects_transport_drift(
    experiment: Stage2ExperimentAttestation,
    field: str,
    invalid: object,
) -> None:
    capture = experiment.repetitions[0].prometheus_measurement.baseline_capture
    exchange_values = capture.http_exchange.model_dump(mode="python")
    exchange_values[field] = invalid
    _rehash(exchange_values)
    with pytest.raises(ValidationError):
        Stage2HTTPExchangeCapture.model_validate(exchange_values)


@pytest.mark.parametrize("mutation", ["missing-content-type", "malformed", "incomplete", "process"])
def test_prometheus_scrape_capture_rejects_header_and_process_bypasses(
    experiment: Stage2ExperimentAttestation,
    mutation: str,
) -> None:
    capture = experiment.repetitions[0].prometheus_measurement.baseline_capture
    values = capture.model_dump(mode="python")
    if mutation == "missing-content-type":
        values["http_exchange"]["response_headers"]["fields"] = values["http_exchange"][
            "response_headers"
        ]["fields"][1:]
    elif mutation == "malformed":
        values["http_exchange"]["full_response_content_type"] = "malformed"
    elif mutation == "incomplete":
        values["http_exchange"]["response_header_capture_complete_at_layer"] = False
    else:
        values["http_exchange"]["server_process_identity"] = "different-process"
        _rehash(values["http_exchange"])
        _rehash(values)
    with pytest.raises(ValidationError):
        PrometheusRawScrapeCapture.model_validate(values)


@pytest.mark.parametrize(
    "malformed",
    [
        "text/plain; charset=bad value",
        'text/plain; charset="unterminated',
        'text/plain; charset="bad\x7fvalue"',
    ],
)
def test_prometheus_scrape_rejects_malformed_content_type_bytes(
    experiment: Stage2ExperimentAttestation,
    malformed: str,
) -> None:
    capture = experiment.repetitions[0].prometheus_measurement.baseline_capture
    values = capture.http_exchange.model_dump(mode="python")
    headers = values["response_headers"]
    field = headers["fields"][0]
    raw = malformed.encode("ascii")
    field["value_base64"] = base64.b64encode(raw).decode("ascii")
    field["value_byte_count"] = len(raw)
    field["value_sha256"] = hashlib.sha256(raw).hexdigest()
    field["normalized_value"] = malformed
    _rehash(field)
    headers["normalized_view"] = tuple(
        (item["normalized_name"], item["normalized_value"]) for item in headers["fields"]
    )
    _rehash(headers)
    values["full_response_content_type"] = malformed
    _rehash(values)
    with pytest.raises(ValidationError):
        Stage2HTTPExchangeCapture.model_validate(values)


def test_prometheus_measurement_rejects_capture_snapshot_wall_clock_detachment(
    experiment: Stage2ExperimentAttestation,
) -> None:
    measurement = experiment.repetitions[0].prometheus_measurement
    values = measurement.model_dump(mode="python")
    capture = values["baseline_capture"]
    capture["scrape_wall_clock_utc"] += timedelta(seconds=1)
    _rehash(capture)
    changed_capture = PrometheusRawScrapeCapture.model_validate(capture)
    raw_capture_bytes = canonical_json_bytes(changed_capture) + b"\n"
    values["baseline_raw_exposition_file"]["size"] = len(raw_capture_bytes)
    values["baseline_raw_exposition_file"]["sha256"] = hashlib.sha256(raw_capture_bytes).hexdigest()
    _rehash(values, field="evidence_sha256")
    with pytest.raises(ValidationError, match="process or restart"):
        type(measurement).model_validate(values)


def test_positive_fixture_transport_shape_is_complete_and_fixture_only(
    experiment: Stage2ExperimentAttestation,
) -> None:
    assert experiment.evidence_scope.value == "TEST_FIXTURE_ONLY"
    assert experiment.classification.value == "SYNTHETIC_PROTOCOL_SHAPE_ONLY"
    assert sum(len(repetition.measured_requests) for repetition in experiment.repetitions) == 48
    for repetition in experiment.repetitions:
        assert len(repetition.prometheus_measurement.counter_deltas) == 10
        assert (
            repetition.cancellation_wire.intentional_client_close.close_classification
            == "INTENTIONAL_CLIENT_CLOSE_AFTER_FIRST_GENERATION_TOKEN"
        )
        assert repetition.cancellation_wire.parser_replay.first_generation_token.output_token_ids
        request_names = tuple(
            name
            for name, _ in repetition.measured_requests[
                0
            ].wire_capture.http_exchange.request_headers.normalized_view
        )
        assert request_names == (
            "host",
            "accept",
            "accept-encoding",
            "connection",
            "user-agent",
            "x-request-id",
            "content-type",
            "content-length",
        )
        media_types = {
            repetition.prometheus_measurement.baseline_capture.http_exchange.normalized_response_media_type,
            repetition.prometheus_measurement.final_capture.http_exchange.normalized_response_media_type,
        }
        assert media_types == {"text/plain", "application/openmetrics-text"}
