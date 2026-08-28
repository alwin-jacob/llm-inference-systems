"""Incremental SSE and fixture-exact token accounting tests."""

from __future__ import annotations

import pytest

from llm_inference_systems.fixture_tokens import (
    FixtureTokenError,
    parse_input_tokens,
    parse_output_tokens,
)
from llm_inference_systems.sse import IncrementalSSEParser, SSEFrame, SSEProtocolError
from llm_inference_systems.streaming import _payload_text


def test_sse_event_split_across_raw_chunks() -> None:
    parser = IncrementalSSEParser()
    assert parser.feed(b'data: {"choices":[{"te') == ()
    frames = parser.feed(b'xt":"<t000>"}]}\n\n')
    assert len(frames) == 1
    assert _payload_text(frames[0]) == "<t000>"


def test_several_sse_events_in_one_raw_chunk() -> None:
    parser = IncrementalSSEParser()
    frames = parser.feed(
        b'data: {"choices":[{"text":"<t000>"}]}\n\n'
        b'data: {"choices":[{"text":"<t001>"}]}\n\n'
        b"data: [DONE]\n\n"
    )
    assert [frame.kind for frame in frames] == ["data", "data", "done"]
    parser.finalize()


def test_sse_comments_are_semantically_separate() -> None:
    parser = IncrementalSSEParser()
    frames = parser.feed(b": fixture-keepalive\n\n")
    assert frames == (SSEFrame(kind="comment", data=None, comments=("fixture-keepalive",)),)


def test_multiple_data_lines_form_one_event() -> None:
    parser = IncrementalSSEParser()
    frames = parser.feed(b"data: first\ndata: second\n\n")
    assert frames[0].data == "first\nsecond"


def test_crlf_sse_is_accepted_incrementally() -> None:
    parser = IncrementalSSEParser()
    assert parser.feed(b"data: [DONE]\r") == ()
    assert parser.feed(b"\n\r\n")[0].kind == "done"
    parser.finalize()


def test_duplicate_done_is_rejected() -> None:
    parser = IncrementalSSEParser()
    parser.feed(b"data: [DONE]\n\n")
    with pytest.raises(SSEProtocolError, match="duplicate"):
        parser.feed(b"data: [DONE]\n\n")


def test_data_after_done_is_rejected() -> None:
    parser = IncrementalSSEParser()
    parser.feed(b"data: [DONE]\n\n")
    with pytest.raises(SSEProtocolError, match="after"):
        parser.feed(b"data: later\n\n")


def test_missing_done_is_rejected_on_finalize() -> None:
    parser = IncrementalSSEParser()
    parser.feed(b'data: {"choices":[{"text":"<t000>"}]}\n\n')
    with pytest.raises(SSEProtocolError, match="without"):
        parser.finalize()


def test_truncated_event_is_rejected_on_finalize() -> None:
    parser = IncrementalSSEParser()
    parser.feed(b"data: truncated")
    with pytest.raises(SSEProtocolError, match="incomplete"):
        parser.finalize()


@pytest.mark.parametrize(
    "data",
    [
        "not-json",
        "{}",
        '{"choices":[]}',
        '{"choices":[{}]}',
        '{"choices":[{"text":3}]}',
    ],
)
def test_malformed_expected_payload_shape_is_rejected(data: str) -> None:
    frame = SSEFrame(kind="data", data=data, comments=())
    with pytest.raises(SSEProtocolError):
        _payload_text(frame)


def test_exact_input_and_output_fixture_markers() -> None:
    assert parse_input_tokens("<p000><p001>") == ("<p000>", "<p001>")
    assert parse_output_tokens("<t010><t011>") == ("<t010>", "<t011>")


@pytest.mark.parametrize(
    "text",
    ["ordinary text", "<t00>", "<t000>suffix", "<p000>", ""],
)
def test_malformed_output_fixture_markers_are_rejected(text: str) -> None:
    with pytest.raises(FixtureTokenError):
        parse_output_tokens(text)


def test_fixture_token_accounting_has_no_tokenizer_provenance() -> None:
    tokens = parse_output_tokens("<t000><t001>")
    assert len(tokens) == 2
    assert all(token.startswith("<t") for token in tokens)
