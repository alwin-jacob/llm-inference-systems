"""Incremental parser for the documented Stage 1 SSE fixture subset."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

MAX_SSE_BUFFER_BYTES = 64 * 1024


class SSEProtocolError(ValueError):
    """Raised when the controlled fixture stream violates its SSE subset."""


@dataclass(frozen=True, slots=True)
class SSEFrame:
    kind: Literal["comment", "data", "done"]
    data: str | None
    comments: tuple[str, ...]


class IncrementalSSEParser:
    """Parse SSE frames without assuming any HTTP chunk/event correspondence."""

    def __init__(self) -> None:
        self._buffer = b""
        self._done = False

    @property
    def done(self) -> bool:
        return self._done

    def feed(self, chunk: bytes) -> tuple[SSEFrame, ...]:
        if not chunk:
            return ()
        self._buffer += chunk
        if len(self._buffer) > MAX_SSE_BUFFER_BYTES:
            raise SSEProtocolError("SSE parser buffer limit exceeded")
        self._buffer = self._buffer.replace(b"\r\n", b"\n")
        frames: list[SSEFrame] = []
        while b"\n\n" in self._buffer:
            raw_frame, self._buffer = self._buffer.split(b"\n\n", 1)
            if not raw_frame:
                continue
            frames.append(self._parse_frame(raw_frame))
        return tuple(frames)

    def _parse_frame(self, raw_frame: bytes) -> SSEFrame:
        try:
            text = raw_frame.decode("utf-8")
        except UnicodeDecodeError as error:
            raise SSEProtocolError("SSE frame is not valid UTF-8") from error
        if "\r" in text:
            raise SSEProtocolError("SSE fixture accepts only LF or CRLF line endings")
        data_lines: list[str] = []
        comments: list[str] = []
        for line in text.split("\n"):
            if line.startswith(":"):
                comments.append(line[1:].lstrip(" "))
            elif line == "data":
                data_lines.append("")
            elif line.startswith("data:"):
                value = line[5:]
                data_lines.append(value[1:] if value.startswith(" ") else value)
            else:
                raise SSEProtocolError("unsupported SSE field in fixture stream")
        if not data_lines:
            if self._done:
                return SSEFrame(kind="comment", data=None, comments=tuple(comments))
            return SSEFrame(kind="comment", data=None, comments=tuple(comments))
        data = "\n".join(data_lines)
        if data == "[DONE]":
            if self._done:
                raise SSEProtocolError("duplicate [DONE] marker")
            self._done = True
            return SSEFrame(kind="done", data=data, comments=tuple(comments))
        if self._done:
            raise SSEProtocolError("data observed after [DONE]")
        return SSEFrame(kind="data", data=data, comments=tuple(comments))

    def finalize(self) -> None:
        remaining = self._buffer.replace(b"\r\n", b"\n")
        if remaining.strip(b"\n"):
            raise SSEProtocolError("incomplete final SSE frame")
        if not self._done:
            raise SSEProtocolError("stream closed without [DONE]")
