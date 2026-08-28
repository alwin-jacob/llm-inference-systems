"""Strict Prometheus exposition parsing and same-process Stage 2 deltas."""

from __future__ import annotations

import json
import math
import re
from datetime import timedelta
from typing import Final, Self

from pydantic import AwareDatetime, model_validator

from llm_inference_systems.contracts import (
    Identifier,
    NonNegativeFloat,
    NonNegativeInt,
    StrictModel,
)

MODEL_LABELS: Final = {
    "model_name": "qwen2.5-0.5b-instruct-stage2",
    "engine": "0",
}
GAUGE_NAMES: Final = frozenset(
    {
        "vllm:num_requests_running",
        "vllm:num_requests_waiting",
        "vllm:kv_cache_usage_perc",
    }
)
COUNTER_NAMES: Final = frozenset(
    {
        "vllm:prompt_tokens_total",
        "vllm:generation_tokens_total",
        "vllm:request_success_total",
        "vllm:num_preemptions_total",
        "vllm:prefix_cache_queries_total",
        "vllm:prefix_cache_hits_total",
    }
)
EXPECTED_MEASURED_DELTAS: Final = {
    "vllm:prompt_tokens_total": 1024.0,
    "vllm:generation_tokens_total": 512.0,
    'vllm:request_success_total{finished_reason="length"}': 16.0,
    "vllm:num_preemptions_total": 0.0,
    "vllm:prefix_cache_queries_total": 0.0,
    "vllm:prefix_cache_hits_total": 0.0,
}

_SAMPLE_RE = re.compile(
    r"^(?P<name>[A-Za-z_:][A-Za-z0-9_:]*)"
    r"(?:\{(?P<labels>.*)\})?\s+"
    r"(?P<value>[^\s]+)$"
)
_LABEL_RE = re.compile(r'(?P<name>[A-Za-z_][A-Za-z0-9_]*)="(?P<value>(?:\\.|[^"\\])*)"')


class PrometheusProtocolError(ValueError):
    """Raised when retained exposition is malformed or ambiguous."""


class PrometheusSample(StrictModel):
    name: Identifier
    labels: tuple[tuple[Identifier, str], ...]
    value: NonNegativeFloat

    @model_validator(mode="after")
    def validate_labels(self) -> Self:
        names = tuple(name for name, _ in self.labels)
        if names != tuple(sorted(names)) or len(names) != len(set(names)):
            raise ValueError("sample labels must be sorted and unique")
        return self

    @property
    def label_map(self) -> dict[str, str]:
        return dict(self.labels)


class PrometheusSnapshot(StrictModel):
    process_start_id: Identifier
    scrape_wall_clock_utc: AwareDatetime
    scrape_monotonic_offset_ns: NonNegativeInt
    raw_exposition: str
    samples: tuple[PrometheusSample, ...]
    label_inventory: dict[str, tuple[tuple[str, ...], ...]]

    @model_validator(mode="after")
    def validate_utc(self) -> Self:
        if self.scrape_wall_clock_utc.utcoffset() != timedelta(0):
            raise ValueError("scrape wall-clock provenance must use UTC")
        return self


class CounterDelta(StrictModel):
    metric: Identifier
    labels: tuple[tuple[Identifier, str], ...]
    before: NonNegativeFloat
    after: NonNegativeFloat
    delta: NonNegativeFloat


def _parse_labels(raw: str | None) -> tuple[tuple[str, str], ...]:
    if raw is None or raw == "":
        return ()
    position = 0
    labels: list[tuple[str, str]] = []
    while position < len(raw):
        match = _LABEL_RE.match(raw, position)
        if match is None:
            raise PrometheusProtocolError("malformed Prometheus label set")
        name = match.group("name")
        try:
            value = json.loads(f'"{match.group("value")}"')
        except json.JSONDecodeError as error:
            raise PrometheusProtocolError("malformed Prometheus label escape") from error
        if not isinstance(value, str):
            raise PrometheusProtocolError("Prometheus label value is not text")
        labels.append((name, value))
        position = match.end()
        if position == len(raw):
            break
        if raw[position] != ",":
            raise PrometheusProtocolError("malformed Prometheus label separator")
        position += 1
    names = [name for name, _ in labels]
    if len(names) != len(set(names)):
        raise PrometheusProtocolError("duplicate Prometheus label")
    return tuple(sorted(labels))


def parse_prometheus_snapshot(
    raw_exposition: str,
    *,
    process_start_id: str,
    scrape_wall_clock_utc: AwareDatetime,
    scrape_monotonic_offset_ns: int,
) -> PrometheusSnapshot:
    if not raw_exposition or not raw_exposition.endswith("\n"):
        raise PrometheusProtocolError(
            "Prometheus exposition must be nonempty and newline terminated"
        )
    samples: list[PrometheusSample] = []
    for line in raw_exposition.splitlines():
        if not line or line.startswith("#"):
            continue
        match = _SAMPLE_RE.fullmatch(line)
        if match is None:
            raise PrometheusProtocolError("malformed Prometheus sample")
        try:
            value = float(match.group("value"))
        except ValueError as error:
            raise PrometheusProtocolError("Prometheus sample value is malformed") from error
        if not math.isfinite(value) or value < 0:
            raise PrometheusProtocolError("Prometheus samples must be finite and nonnegative")
        samples.append(
            PrometheusSample(
                name=match.group("name"),
                labels=_parse_labels(match.group("labels")),
                value=value,
            )
        )
    if not samples:
        raise PrometheusProtocolError("Prometheus exposition contains no samples")
    inventory: dict[str, list[tuple[str, ...]]] = {}
    for sample in samples:
        inventory.setdefault(sample.name, []).append(tuple(name for name, _ in sample.labels))
    label_inventory = {
        name: tuple(sorted(label_sets)) for name, label_sets in sorted(inventory.items())
    }
    return PrometheusSnapshot(
        process_start_id=process_start_id,
        scrape_wall_clock_utc=scrape_wall_clock_utc,
        scrape_monotonic_offset_ns=scrape_monotonic_offset_ns,
        raw_exposition=raw_exposition,
        samples=tuple(samples),
        label_inventory=label_inventory,
    )


def select_exact_series(
    snapshot: PrometheusSnapshot,
    name: str,
    *,
    finished_reason: str | None = None,
) -> PrometheusSample:
    if name not in GAUGE_NAMES | COUNTER_NAMES:
        raise PrometheusProtocolError("metric is outside the Stage 2 exact-series contract")
    labels = dict(MODEL_LABELS)
    if name == "vllm:request_success_total":
        if finished_reason is None:
            raise PrometheusProtocolError("request-success selection requires finished_reason")
        labels["finished_reason"] = finished_reason
    elif finished_reason is not None:
        raise PrometheusProtocolError("finished_reason applies only to request-success series")
    expected = tuple(sorted(labels.items()))
    matches = [
        sample for sample in snapshot.samples if sample.name == name and sample.labels == expected
    ]
    if len(matches) != 1:
        raise PrometheusProtocolError("exact Prometheus series is absent, duplicate, or ambiguous")
    return matches[0]


def require_fresh_snapshot(
    snapshot: PrometheusSnapshot,
    *,
    reference_monotonic_offset_ns: int,
    maximum_age_ns: int,
) -> None:
    age = reference_monotonic_offset_ns - snapshot.scrape_monotonic_offset_ns
    if age < 0 or age > maximum_age_ns:
        raise PrometheusProtocolError("Prometheus snapshot is stale or from the future")


def derive_counter_delta(
    before: PrometheusSnapshot,
    after: PrometheusSnapshot,
    name: str,
    *,
    finished_reason: str | None = None,
) -> CounterDelta:
    if name not in COUNTER_NAMES:
        raise PrometheusProtocolError("same-process deltas apply only to Stage 2 counters")
    if before.process_start_id != after.process_start_id:
        raise PrometheusProtocolError("counter subtraction across restarts is prohibited")
    if after.scrape_monotonic_offset_ns <= before.scrape_monotonic_offset_ns:
        raise PrometheusProtocolError("counter snapshots are stale or reordered")
    if before.label_inventory.get(name) != after.label_inventory.get(name):
        raise PrometheusProtocolError("Prometheus label inventory changed between scrapes")
    first = select_exact_series(before, name, finished_reason=finished_reason)
    second = select_exact_series(after, name, finished_reason=finished_reason)
    if second.value < first.value:
        raise PrometheusProtocolError("counter reset or decrease detected")
    return CounterDelta(
        metric=name,
        labels=first.labels,
        before=first.value,
        after=second.value,
        delta=second.value - first.value,
    )


def validate_measured_window_deltas(deltas: tuple[CounterDelta, ...]) -> None:
    observed: dict[str, float] = {}
    for delta in deltas:
        key = delta.metric
        labels = dict(delta.labels)
        if delta.metric == "vllm:request_success_total":
            reason = labels.get("finished_reason")
            key = f'vllm:request_success_total{{finished_reason="{reason}"}}'
        if key in observed:
            raise PrometheusProtocolError("duplicate measured-window counter delta")
        observed[key] = delta.delta
    if observed != EXPECTED_MEASURED_DELTAS:
        raise PrometheusProtocolError("measured-window counter deltas differ from the contract")


def require_quiescent(snapshot: PrometheusSnapshot) -> None:
    for name in ("vllm:num_requests_running", "vllm:num_requests_waiting"):
        if select_exact_series(snapshot, name).value != 0:
            raise PrometheusProtocolError("pre/post metrics gate is not quiescent")
