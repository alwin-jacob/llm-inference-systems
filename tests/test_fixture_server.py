"""Real-socket tests for the deliberately small loopback fixture server."""

from __future__ import annotations

import asyncio
import json
import socket
import time
from pathlib import Path

import pytest

from llm_inference_systems.fixture_server import LOOPBACK_HOST, FixtureServer
from llm_inference_systems.stage1_contracts import FixtureDefinition
from llm_inference_systems.streaming import EvidenceCollector
from tests.factories import ROOT


def _fixture() -> FixtureDefinition:
    return FixtureDefinition.model_validate_json(
        (ROOT / "examples/fixtures/streaming-fixture-v1.json").read_bytes()
    )


async def _raw_exchange(request: bytes) -> bytes:
    recorder = EvidenceCollector(time.monotonic_ns())
    async with FixtureServer(_fixture(), recorder) as server:
        reader, writer = await asyncio.open_connection(
            server.host, server.port, family=socket.AF_INET
        )
        writer.write(request)
        await writer.drain()
        response = await reader.read()
        writer.close()
        await writer.wait_closed()
        return response


def _valid_request(case_id: str = "warmup-success") -> bytes:
    case = next(case for case in _fixture().cases if case.case_id == case_id)
    body = json.dumps(
        {
            "model": "fixture-no-model",
            "prompt": case.input_text,
            "max_tokens": case.maximum_output_tokens,
            "temperature": 0,
            "stream": True,
        },
        separators=(",", ":"),
    ).encode()
    return (
        b"POST /v1/completions HTTP/1.1\r\n"
        b"Host: 127.0.0.1\r\n"
        b"X-LIS-Request-ID: raw-test-001\r\n"
        + f"X-LIS-Fixture-Case-ID: {case_id}\r\n".encode()
        + f"Content-Length: {len(body)}\r\n\r\n".encode()
        + body
    )


def test_fixture_server_binds_only_ipv4_loopback_and_ephemeral_port() -> None:
    async def check() -> None:
        recorder = EvidenceCollector(time.monotonic_ns())
        async with FixtureServer(_fixture(), recorder) as server:
            assert server.host == LOOPBACK_HOST == "127.0.0.1"
            assert 0 < server.port <= 65_535

    asyncio.run(check())


@pytest.mark.parametrize(
    ("request_bytes", "status"),
    [
        (b"GET /v1/completions HTTP/1.1\r\nContent-Length: 0\r\n\r\n", b"405"),
        (b"POST /other HTTP/1.1\r\nContent-Length: 0\r\n\r\n", b"404"),
    ],
)
def test_fixture_server_rejects_unsupported_method_or_path(
    request_bytes: bytes, status: bytes
) -> None:
    response = asyncio.run(_raw_exchange(request_bytes))
    assert response.startswith(b"HTTP/1.1 " + status)


def test_fixture_server_enforces_header_size_limit() -> None:
    request = (
        b"POST /v1/completions HTTP/1.1\r\nX-Fill: "
        + b"x" * 9_000
        + b"\r\nContent-Length: 0\r\n\r\n"
    )
    response = asyncio.run(_raw_exchange(request))
    assert response.startswith(b"HTTP/1.1 431")


def test_fixture_server_enforces_body_size_limit_before_reading_body() -> None:
    request = b"POST /v1/completions HTTP/1.1\r\nContent-Length: 20000\r\n\r\n"
    response = asyncio.run(_raw_exchange(request))
    assert response.startswith(b"HTTP/1.1 413")


def test_fixture_server_rejects_malformed_json() -> None:
    body = b"{not-json"
    request = (
        b"POST /v1/completions HTTP/1.1\r\n"
        b"X-LIS-Request-ID: malformed\r\n"
        b"X-LIS-Fixture-Case-ID: warmup-success\r\n"
        + f"Content-Length: {len(body)}\r\n\r\n".encode()
        + body
    )
    response = asyncio.run(_raw_exchange(request))
    assert response.startswith(b"HTTP/1.1 400")


def test_fixture_server_emits_correct_http_chunk_framing() -> None:
    response = asyncio.run(_raw_exchange(_valid_request()))
    headers, body = response.split(b"\r\n\r\n", 1)
    assert response.startswith(b"HTTP/1.1 200")
    assert b"Transfer-Encoding: chunked" in headers
    assert b"data: " in body
    assert body.endswith(b"0\r\n\r\n")


def test_fixture_server_validates_request_values() -> None:
    request = _valid_request().replace(b"fixture-no-model", b"not-a-fixture-model")
    response = asyncio.run(_raw_exchange(request))
    assert response.startswith(b"HTTP/1.1 400")


def test_fixture_server_shutdown_cleans_incomplete_disconnect() -> None:
    async def check() -> None:
        recorder = EvidenceCollector(time.monotonic_ns())
        server = FixtureServer(_fixture(), recorder)
        await server.start()
        _reader, writer = await asyncio.open_connection(server.host, server.port)
        writer.write(b"POST /v1/completions HTTP/1.1\r\n")
        await writer.drain()
        writer.close()
        await writer.wait_closed()
        await server.stop()

    asyncio.run(check())


def test_fixture_server_has_no_configurable_host_parameter() -> None:
    assert "host" not in FixtureServer.__init__.__annotations__
    assert Path(ROOT / "src/llm_inference_systems/fixture_server.py").is_file()
