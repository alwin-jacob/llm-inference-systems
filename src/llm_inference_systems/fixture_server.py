"""Deterministic asyncio HTTP/1.1 fixture bound only to IPv4 loopback."""

from __future__ import annotations

import asyncio
import json
import re
import socket
from collections.abc import Mapping
from contextlib import suppress
from typing import Final, Protocol

from llm_inference_systems.stage1_contracts import (
    FixtureActionKind,
    FixtureCaseDefinition,
    FixtureDefinition,
    ServerEventKind,
)

LOOPBACK_HOST: Final = "127.0.0.1"
FIXTURE_PATH: Final = "/v1/completions"
MAX_REQUEST_HEADER_BYTES: Final = 8 * 1024
MAX_REQUEST_BODY_BYTES: Final = 16 * 1024
_SAFE_ID = re.compile(r"^[A-Za-z0-9._-]{1,160}$")


class ServerEventRecorder(Protocol):
    async def record_server_event(
        self,
        *,
        request_id: str,
        case_id: str,
        kind: ServerEventKind,
        action_index: int | None = None,
        token_delta_count: int | None = None,
        http_status: int | None = None,
    ) -> None: ...


class FixtureHTTPError(ValueError):
    def __init__(self, status: int, message: str) -> None:
        super().__init__(message)
        self.status = status


class FixtureServer:
    """A deliberately small fixture server with no configurable bind address."""

    def __init__(self, fixture: FixtureDefinition, recorder: ServerEventRecorder) -> None:
        self._fixture = fixture
        self._cases = {case.case_id: case for case in fixture.cases}
        self._recorder = recorder
        self._server: asyncio.AbstractServer | None = None
        self._tasks: set[asyncio.Task[None]] = set()
        self._port: int | None = None

    @property
    def host(self) -> str:
        return LOOPBACK_HOST

    @property
    def port(self) -> int:
        if self._port is None:
            raise RuntimeError("fixture server has not started")
        return self._port

    async def start(self) -> None:
        if self._server is not None:
            raise RuntimeError("fixture server is already started")
        self._server = await asyncio.start_server(
            self._handle_client,
            host=LOOPBACK_HOST,
            port=0,
            family=socket.AF_INET,
            limit=MAX_REQUEST_HEADER_BYTES + 1,
            start_serving=True,
        )
        sockets = self._server.sockets or ()
        if len(sockets) != 1:
            await self.stop()
            raise RuntimeError("fixture server expected one IPv4 loopback socket")
        host, port = sockets[0].getsockname()[:2]
        if host != LOOPBACK_HOST or not isinstance(port, int) or port <= 0:
            await self.stop()
            raise RuntimeError("fixture server did not bind the required loopback endpoint")
        self._port = port

    async def stop(self) -> None:
        server, self._server = self._server, None
        if server is not None:
            server.close()
            await server.wait_closed()
        current = asyncio.current_task()
        tasks = tuple(task for task in self._tasks if task is not current and not task.done())
        for task in tasks:
            task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        self._tasks.clear()

    async def __aenter__(self) -> FixtureServer:
        await self.start()
        return self

    async def __aexit__(self, *args: object) -> None:
        await self.stop()

    async def _handle_client(
        self,
        reader: asyncio.StreamReader,
        writer: asyncio.StreamWriter,
    ) -> None:
        task = asyncio.current_task()
        if task is not None:
            self._tasks.add(task)
        request_id: str | None = None
        case_id: str | None = None
        try:
            try:
                request_id, case_id, case = await self._read_request(reader)
            except FixtureHTTPError as error:
                await self._send_plain_response(writer, error.status, "fixture request rejected")
                return
            except (asyncio.IncompleteReadError, ConnectionError):
                return
            await self._recorder.record_server_event(
                request_id=request_id,
                case_id=case_id,
                kind=ServerEventKind.REQUEST_ACCEPTED,
            )
            await self._execute_case(writer, request_id, case)
        except (BrokenPipeError, ConnectionError):
            if request_id is not None and case_id is not None:
                await self._recorder.record_server_event(
                    request_id=request_id,
                    case_id=case_id,
                    kind=ServerEventKind.CLIENT_DISCONNECTED,
                )
        finally:
            if request_id is not None and case_id is not None:
                await self._recorder.record_server_event(
                    request_id=request_id,
                    case_id=case_id,
                    kind=ServerEventKind.REQUEST_HANDLER_ENDED,
                )
            writer.close()
            with suppress(BrokenPipeError, ConnectionError):
                await writer.wait_closed()
            if task is not None:
                self._tasks.discard(task)

    async def _read_request(
        self,
        reader: asyncio.StreamReader,
    ) -> tuple[str, str, FixtureCaseDefinition]:
        try:
            header_block = await reader.readuntil(b"\r\n\r\n")
        except asyncio.LimitOverrunError as error:
            raise FixtureHTTPError(431, "request headers exceed fixture limit") from error
        if len(header_block) > MAX_REQUEST_HEADER_BYTES:
            raise FixtureHTTPError(431, "request headers exceed fixture limit")
        try:
            header_text = header_block[:-4].decode("ascii")
        except UnicodeDecodeError as error:
            raise FixtureHTTPError(400, "request headers must be ASCII") from error
        lines = header_text.split("\r\n")
        request_parts = lines[0].split(" ")
        if len(request_parts) != 3:
            raise FixtureHTTPError(400, "malformed request line")
        method, path, version = request_parts
        if method != "POST":
            raise FixtureHTTPError(405, "fixture endpoint requires POST")
        if path != FIXTURE_PATH:
            raise FixtureHTTPError(404, "fixture endpoint not found")
        if version != "HTTP/1.1":
            raise FixtureHTTPError(400, "fixture endpoint requires HTTP/1.1")
        headers = self._parse_headers(lines[1:])
        length_text = headers.get("content-length")
        if length_text is None:
            raise FixtureHTTPError(411, "Content-Length is required")
        try:
            content_length = int(length_text)
        except ValueError as error:
            raise FixtureHTTPError(400, "invalid Content-Length") from error
        if content_length < 0 or content_length > MAX_REQUEST_BODY_BYTES:
            raise FixtureHTTPError(413, "request body exceeds fixture limit")
        try:
            body_bytes = await reader.readexactly(content_length)
        except asyncio.IncompleteReadError as error:
            raise FixtureHTTPError(400, "incomplete request body") from error
        try:
            body = json.loads(body_bytes)
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise FixtureHTTPError(400, "malformed JSON request") from error
        if not isinstance(body, dict):
            raise FixtureHTTPError(400, "request JSON must be an object")
        request_id = headers.get("x-lis-request-id", "")
        case_id = headers.get("x-lis-fixture-case-id", "")
        if not _SAFE_ID.fullmatch(request_id) or not _SAFE_ID.fullmatch(case_id):
            raise FixtureHTTPError(400, "fixture request identifiers are invalid")
        case = self._cases.get(case_id)
        if case is None:
            raise FixtureHTTPError(400, "unknown fixture case")
        self._validate_body(body, case)
        return request_id, case_id, case

    @staticmethod
    def _parse_headers(lines: list[str]) -> dict[str, str]:
        headers: dict[str, str] = {}
        for line in lines:
            if ":" not in line:
                raise FixtureHTTPError(400, "malformed request header")
            name, value = line.split(":", 1)
            key = name.strip().casefold()
            if not key or key in headers:
                raise FixtureHTTPError(400, "duplicate or empty request header")
            headers[key] = value.strip()
        return headers

    def _validate_body(
        self,
        body: Mapping[str, object],
        case: FixtureCaseDefinition,
    ) -> None:
        if set(body) != {"model", "prompt", "max_tokens", "temperature", "stream"}:
            raise FixtureHTTPError(400, "request JSON fields do not match fixture protocol")
        expected: dict[str, object] = {
            "model": self._fixture.model_sentinel,
            "prompt": case.input_text,
            "max_tokens": case.maximum_output_tokens,
            "temperature": 0,
            "stream": True,
        }
        if body != expected:
            raise FixtureHTTPError(400, "request JSON values do not match fixture case")

    async def _execute_case(
        self,
        writer: asyncio.StreamWriter,
        request_id: str,
        case: FixtureCaseDefinition,
    ) -> None:
        first_action = case.actions[0]
        if first_action.kind is FixtureActionKind.HTTP_ERROR:
            assert first_action.http_status is not None
            assert first_action.text is not None
            await self._recorder.record_server_event(
                request_id=request_id,
                case_id=case.case_id,
                kind=ServerEventKind.HTTP_ERROR_SENT,
                action_index=0,
                http_status=first_action.http_status,
            )
            await self._send_plain_response(writer, first_action.http_status, first_action.text)
            return

        writer.write(
            b"HTTP/1.1 200 OK\r\n"
            b"Content-Type: text/event-stream\r\n"
            b"Transfer-Encoding: chunked\r\n"
            b"Connection: close\r\n\r\n"
        )
        await writer.drain()
        await self._recorder.record_server_event(
            request_id=request_id,
            case_id=case.case_id,
            kind=ServerEventKind.RESPONSE_HEADERS_SENT,
            http_status=200,
        )
        for index, action in enumerate(case.actions):
            if action.delay_ms_before:
                await asyncio.sleep(action.delay_ms_before / 1_000)
            if action.kind is FixtureActionKind.SSE_COMMENT:
                assert action.text is not None
                await self._write_chunk(writer, f": {action.text}\n\n".encode())
                kind = ServerEventKind.SSE_COMMENT_SENT
                token_count = None
            elif action.kind is FixtureActionKind.SSE_TOKEN_EVENT:
                assert action.text is not None
                payload = json.dumps(
                    {"choices": [{"text": action.text}]},
                    ensure_ascii=False,
                    separators=(",", ":"),
                    sort_keys=True,
                )
                await self._write_chunk(writer, f"data: {payload}\n\n".encode())
                kind = ServerEventKind.SSE_TOKEN_EVENT_SENT
                token_count = action.text.count("<t")
            elif action.kind is FixtureActionKind.SSE_MALFORMED_DATA:
                assert action.text is not None
                await self._write_chunk(writer, f"data: {action.text}\n\n".encode())
                kind = ServerEventKind.SSE_MALFORMED_DATA_SENT
                token_count = None
            elif action.kind is FixtureActionKind.SSE_DONE:
                await self._write_chunk(writer, b"data: [DONE]\n\n")
                kind = ServerEventKind.SSE_DONE_SENT
                token_count = None
            elif action.kind is FixtureActionKind.STALL:
                assert action.stall_seconds is not None
                await self._recorder.record_server_event(
                    request_id=request_id,
                    case_id=case.case_id,
                    kind=ServerEventKind.STALL_STARTED,
                    action_index=index,
                )
                await asyncio.sleep(action.stall_seconds)
                continue
            else:
                raise RuntimeError("HTTP_ERROR must be the sole first fixture action")
            await self._recorder.record_server_event(
                request_id=request_id,
                case_id=case.case_id,
                kind=kind,
                action_index=index,
                token_delta_count=token_count,
            )
        await self._write_final_chunk(writer)

    @staticmethod
    async def _write_chunk(writer: asyncio.StreamWriter, payload: bytes) -> None:
        writer.write(f"{len(payload):X}\r\n".encode("ascii") + payload + b"\r\n")
        await writer.drain()

    @staticmethod
    async def _write_final_chunk(writer: asyncio.StreamWriter) -> None:
        writer.write(b"0\r\n\r\n")
        await writer.drain()

    @staticmethod
    async def _send_plain_response(
        writer: asyncio.StreamWriter,
        status: int,
        text: str,
    ) -> None:
        reasons = {
            400: "Bad Request",
            404: "Not Found",
            405: "Method Not Allowed",
            411: "Length Required",
            413: "Content Too Large",
            431: "Request Header Fields Too Large",
            503: "Service Unavailable",
        }
        payload = text.encode("utf-8")
        reason = reasons.get(status, "Fixture Error")
        writer.write(
            f"HTTP/1.1 {status} {reason}\r\n".encode("ascii")
            + b"Content-Type: text/plain; charset=utf-8\r\n"
            + f"Content-Length: {len(payload)}\r\n".encode("ascii")
            + b"Connection: close\r\n\r\n"
            + payload
        )
        await writer.drain()
