"""Exact accounting for unmistakably synthetic Stage 1 token markers."""

from __future__ import annotations

import re

_INPUT_TOKEN = re.compile(r"<p[0-9]{3}>")
_OUTPUT_TOKEN = re.compile(r"<t[0-9]{3}>")


class FixtureTokenError(ValueError):
    """Raised when text is not an exact sequence of fixture token markers."""


def _parse(text: str, pattern: re.Pattern[str], *, kind: str) -> tuple[str, ...]:
    if not text:
        raise FixtureTokenError(f"{kind} fixture token text cannot be empty")
    tokens = tuple(match.group(0) for match in pattern.finditer(text))
    if not tokens or "".join(tokens) != text:
        raise FixtureTokenError(f"malformed {kind} fixture token marker text")
    return tokens


def parse_input_tokens(text: str) -> tuple[str, ...]:
    """Parse only concatenated ``<pNNN>`` input markers."""

    return _parse(text, _INPUT_TOKEN, kind="input")


def parse_output_tokens(text: str) -> tuple[str, ...]:
    """Parse only concatenated ``<tNNN>`` output markers."""

    return _parse(text, _OUTPUT_TOKEN, kind="output")
