"""CPU-only scripted Stage 2 HTTP/SSE/log/metrics fixture on IPv4 loopback."""

from __future__ import annotations

import asyncio
import json
import socket
from contextlib import suppress
from typing import Final

from pydantic import ValidationError

from llm_inference_systems.stage2_contracts import Stage2CompletionRequest

STAGE2_FIXTURE_HOST: Final = "127.0.0.1"
MAX_HEADER_BYTES: Final = 16 * 1024
MAX_BODY_BYTES: Final = 64 * 1024


class Stage2FixtureHTTPError(ValueError):
    def __init__(self, status: int, message: str) -> None:
        super().__init__(message)
        self.status = status


class Stage2FixtureServer:
    """A deterministic protocol fixture with no configurable bind address."""

    def __init__(self, *, finish_only_terminal: bool = False, grouped_tokens: bool = False) -> None:
        self.finish_only_terminal = finish_only_terminal
        self.grouped_tokens = grouped_tokens
        self._server: asyncio.AbstractServer | None = None
        self._tasks: set[asyncio.Task[None]] = set()
        self._port: int | None = None
        self.logs: list[str] = []
        self.prompt_tokens_total = 0
        self.generation_tokens_total = 0
        self.request_success: dict[str, int] = {
            "length": 0,
            "abort": 0,
            "stop": 0,
            "error": 0,
            "repetition": 0,
        }
        self.num_preemptions_total = 0
        self.prefix_cache_queries_total = 0
        self.prefix_cache_hits_total = 0
        self.running_requests = 0
        self.waiting_requests = 0

    @property
    def host(self) -> str:
        return STAGE2_FIXTURE_HOST

    @property
    def port(self) -> int:
        if self._port is None:
            raise RuntimeError("Stage 2 fixture has not started")
        return self._port

    async def start(self) -> None:
        if self._server is not None:
            raise RuntimeError("Stage 2 fixture is already started")
        self._server = await asyncio.start_server(
            self._handle_client,
            host=STAGE2_FIXTURE_HOST,
            port=0,
            family=socket.AF_INET,
            limit=MAX_HEADER_BYTES + 1,
            start_serving=True,
        )
        sockets = self._server.sockets or ()
        if len(sockets) != 1:
            await self.stop()
            raise RuntimeError("Stage 2 fixture expected one IPv4 socket")
        host, port = sockets[0].getsockname()[:2]
        if host != STAGE2_FIXTURE_HOST or not isinstance(port, int) or port <= 0:
            await self.stop()
            raise RuntimeError("Stage 2 fixture did not bind IPv4 loopback")
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

    async def __aenter__(self) -> Stage2FixtureServer:
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
        external_id: str | None = None
        internal_id: str | None = None
        try:
            try:
                method, path, headers, body = await self._read_request(reader)
                if method == "GET" and path == "/metrics":
                    await self._send_metrics(writer)
                    return
                if method != "POST" or path != "/v1/completions":
                    raise Stage2FixtureHTTPError(404, "fixture endpoint not found")
                request = self._validate_completion(headers, body)
                external_id = request.request_id
                serving_item = f"cmpl-{external_id}-0"
                internal_id = f"{serving_item}-deadbeef"
                self.logs.append(f"Received request {serving_item}: params: TEST_FIXTURE_ONLY.")
                self.logs.append(f"Added request {internal_id}.")
                self.running_requests += 1
                await self._send_completion(writer, request)
                self.prompt_tokens_total += 64
                self.generation_tokens_total += 32
                self.request_success["length"] += 1
            except Stage2FixtureHTTPError as error:
                await self._send_plain(writer, error.status, "Stage 2 fixture request rejected")
        except asyncio.CancelledError:
            if external_id is not None and internal_id is not None:
                serving_item = f"cmpl-{external_id}-0"
                self.logs.append(f"Request {serving_item} aborted.")
                self.logs.append(f"Aborted request(s) {internal_id}.")
                self.request_success["abort"] += 1
            raise
        except (BrokenPipeError, ConnectionError):
            if external_id is not None and internal_id is not None:
                serving_item = f"cmpl-{external_id}-0"
                self.logs.append(f"Request {serving_item} aborted.")
                self.logs.append(f"Aborted request(s) {internal_id}.")
                self.request_success["abort"] += 1
        finally:
            if external_id is not None:
                self.running_requests = max(0, self.running_requests - 1)
            writer.close()
            with suppress(BrokenPipeError, ConnectionError):
                await writer.wait_closed()
            if task is not None:
                self._tasks.discard(task)

    async def _read_request(
        self,
        reader: asyncio.StreamReader,
    ) -> tuple[str, str, dict[str, str], bytes]:
        try:
            block = await reader.readuntil(b"\r\n\r\n")
        except asyncio.LimitOverrunError as error:
            raise Stage2FixtureHTTPError(431, "headers exceed fixture limit") from error
        if len(block) > MAX_HEADER_BYTES:
            raise Stage2FixtureHTTPError(431, "headers exceed fixture limit")
        try:
            text = block[:-4].decode("ascii")
        except UnicodeDecodeError as error:
            raise Stage2FixtureHTTPError(400, "headers must be ASCII") from error
        lines = text.split("\r\n")
        parts = lines[0].split(" ")
        if len(parts) != 3 or parts[2] != "HTTP/1.1":
            raise Stage2FixtureHTTPError(400, "malformed HTTP request line")
        headers: dict[str, str] = {}
        for line in lines[1:]:
            if ":" not in line:
                raise Stage2FixtureHTTPError(400, "malformed HTTP header")
            name, value = line.split(":", 1)
            key = name.strip().casefold()
            if not key or key in headers:
                raise Stage2FixtureHTTPError(400, "duplicate or empty HTTP header")
            headers[key] = value.strip()
        method, path, _ = parts
        if method == "GET":
            return method, path, headers, b""
        length_text = headers.get("content-length")
        if length_text is None:
            raise Stage2FixtureHTTPError(411, "Content-Length is required")
        try:
            length = int(length_text)
        except ValueError as error:
            raise Stage2FixtureHTTPError(400, "Content-Length is invalid") from error
        if length < 0 or length > MAX_BODY_BYTES:
            raise Stage2FixtureHTTPError(413, "body exceeds fixture limit")
        try:
            body = await reader.readexactly(length)
        except asyncio.IncompleteReadError as error:
            raise Stage2FixtureHTTPError(400, "request body is incomplete") from error
        return method, path, headers, body

    @staticmethod
    def _validate_completion(headers: dict[str, str], body: bytes) -> Stage2CompletionRequest:
        try:
            request = Stage2CompletionRequest.model_validate_json(body)
        except ValidationError as error:
            raise Stage2FixtureHTTPError(400, "completion request differs") from error
        if headers.get("x-request-id") != request.request_id:
            raise Stage2FixtureHTTPError(400, "header/body request ID mismatch")
        return request

    def _generation_payloads(self, request: Stage2CompletionRequest) -> tuple[bytes, ...]:
        output_ids = tuple(range(1_000, 1_032))
        width = 2 if self.grouped_tokens else 1
        groups = tuple(output_ids[index : index + width] for index in range(0, 32, width))
        payloads: list[bytes] = []
        for index, group in enumerate(groups):
            is_final_content = index == len(groups) - 1 and not self.finish_only_terminal
            choice: dict[str, object] = {
                "index": 0,
                "text": "".join(f"<fixture-{token}>" for token in group),
                "token_ids": list(group),
                "finish_reason": "length" if is_final_content else None,
            }
            if index == 0:
                choice["prompt_token_ids"] = list(request.prompt)
            body = {
                "id": f"cmpl-{request.request_id}",
                "object": "text_completion",
                "model": request.model,
                "choices": [choice],
            }
            payloads.append(
                f"data: {json.dumps(body, sort_keys=True, separators=(',', ':'))}\n\n".encode()
            )
        if self.finish_only_terminal:
            terminal = {
                "id": f"cmpl-{request.request_id}",
                "object": "text_completion",
                "model": request.model,
                "choices": [{"index": 0, "text": "", "token_ids": [], "finish_reason": "length"}],
            }
            payloads.append(
                f"data: {json.dumps(terminal, sort_keys=True, separators=(',', ':'))}\n\n".encode()
            )
        usage = {
            "id": f"cmpl-{request.request_id}",
            "object": "text_completion",
            "model": request.model,
            "choices": [],
            "usage": {"prompt_tokens": 64, "completion_tokens": 32, "total_tokens": 96},
            "metrics": {
                "time_to_first_token_ms": 1.0,
                "generation_time_ms": 2.0,
                "queue_time_ms": 0.0,
                "mean_itl_ms": 0.031,
                "tokens_per_second": 16.0,
            },
        }
        payloads.append(
            f"data: {json.dumps(usage, sort_keys=True, separators=(',', ':'))}\n\n".encode()
        )
        payloads.append(b"data: [DONE]\n\n")
        return tuple(payloads)

    async def _send_completion(
        self,
        writer: asyncio.StreamWriter,
        request: Stage2CompletionRequest,
    ) -> None:
        writer.write(
            b"HTTP/1.1 200 OK\r\n"
            b"Content-Type: text/event-stream\r\n"
            + f"X-Request-Id: {request.request_id}\r\n".encode("ascii")
            + b"Transfer-Encoding: chunked\r\nConnection: close\r\n\r\n"
        )
        await writer.drain()
        for payload in self._generation_payloads(request):
            writer.write(f"{len(payload):X}\r\n".encode("ascii") + payload + b"\r\n")
            await writer.drain()
        writer.write(b"0\r\n\r\n")
        await writer.drain()

    def metrics_exposition(self) -> str:
        labels = 'engine="0",model_name="qwen2.5-0.5b-instruct-stage2"'
        lines = [
            f"vllm:num_requests_running{{{labels}}} {self.running_requests}.0",
            f"vllm:num_requests_waiting{{{labels}}} {self.waiting_requests}.0",
            f"vllm:kv_cache_usage_perc{{{labels}}} 0.0",
            f"vllm:prompt_tokens_total{{{labels}}} {self.prompt_tokens_total}.0",
            f"vllm:generation_tokens_total{{{labels}}} {self.generation_tokens_total}.0",
            f"vllm:num_preemptions_total{{{labels}}} {self.num_preemptions_total}.0",
            f"vllm:prefix_cache_queries_total{{{labels}}} {self.prefix_cache_queries_total}.0",
            f"vllm:prefix_cache_hits_total{{{labels}}} {self.prefix_cache_hits_total}.0",
        ]
        for reason, value in sorted(self.request_success.items()):
            success_labels = (
                f'engine="0",finished_reason="{reason}",model_name="qwen2.5-0.5b-instruct-stage2"'
            )
            lines.append(f"vllm:request_success_total{{{success_labels}}} {value}.0")
        return "\n".join(lines) + "\n"

    async def _send_metrics(self, writer: asyncio.StreamWriter) -> None:
        payload = self.metrics_exposition().encode()
        writer.write(
            b"HTTP/1.1 200 OK\r\nContent-Type: text/plain; version=0.0.4\r\n"
            + f"Content-Length: {len(payload)}\r\n".encode("ascii")
            + b"Connection: close\r\n\r\n"
            + payload
        )
        await writer.drain()

    @staticmethod
    async def _send_plain(writer: asyncio.StreamWriter, status: int, text: str) -> None:
        payload = text.encode()
        writer.write(
            f"HTTP/1.1 {status} Fixture Error\r\n".encode("ascii")
            + f"Content-Length: {len(payload)}\r\n".encode("ascii")
            + b"Connection: close\r\n\r\n"
            + payload
        )
        await writer.drain()
