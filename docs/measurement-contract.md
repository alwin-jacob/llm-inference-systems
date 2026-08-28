# Measurement contract 0.1.0

## Records and scope

The contract retains warmup and measured records, uses unique request and workload-case IDs, and
requires UTC artifact/report timestamps. Artifact identities, model/tokenizer identities,
hardware identities, and environment identities must all be synthetic at Stage 0. Artifact and
report content hashes omit only their own top-level hash field.

## Timing metrics

- TTFT is `first_output_token_offset_ns - dispatch_offset_ns`; first response byte is recorded
  separately and is never substituted.
- TPOT is `(last_output_token_offset_ns - first_output_token_offset_ns) / (output_tokens - 1)`
  and is unavailable for fewer than two output tokens or missing token counts/timestamps.
- ITL contains adjacent differences only when a true per-token sequence is available. A
  single-token chunk event is itself one token observation; a multi-token chunk without
  individual observations makes ITL unavailable.
- p50, p95, and p99 use Hyndman-Fan Type 7 interpolation. Empty samples are explicitly
  unavailable and nonfinite samples are rejected.

## Rates and denominators

All rates use the positive measured window in seconds. Offered request rate counts measured
attempts; terminal request rate counts measured terminal records; successful request throughput
counts only successes. Output-token throughput counts known successful output tokens. Total-token
throughput requires known input and output counts for every success. Goodput counts measured
requests that succeed and satisfy every applicable predeclared SLO.

## Load and batching

`requested_client_concurrency` controls the closed-loop client load shape. Server maximum or
observed batch size is represented independently and requires its own configured or directly
observed provenance. Neither value implies the other.

# Measurement contract 0.2.0

Stage 1 adds real loopback fixture streaming without changing the historical v0.1.0 definitions or
schema bytes. All intervals use `time.monotonic_ns()`; UTC timestamps are audit provenance only.

## Client-observed boundaries

- `dispatch_offset_ns` is captured immediately before HTTPX begins `send(..., stream=True)` after
  the request has been built and serialized. It is the client dispatch boundary, not a packet
  timestamp.
- `response_headers_offset_ns` is captured after HTTPX returns response headers.
- `first_response_body_bytes_offset_ns` is the first nonempty value yielded by
  `response.aiter_raw()`. It is a client-library raw-body boundary, not a NIC or TCP packet time.
- `parsed_event_offset_ns` is represented by each parsed SSE evidence record's observation offset
  when its complete event becomes available.
- `first_output_token_offset_ns` is the first parsed SSE event containing one or more exact fixture
  output markers. Comments and keepalives do not establish TTFT.
- The successful terminal boundary is captured only after `[DONE]` has been accepted, the response
  has ended without prohibited later data, and parser finalization succeeds.
- E2E is successful terminal minus dispatch. TTFT is first output token minus dispatch.

## TPOT reconciliation

Canonical Stage 1 TPOT is:

```text
(terminal_success_offset_ns - first_output_token_offset_ns)
/ (output_token_count - 1)
```

It is available only for successful requests with at least two exact output tokens and an observed
first token. The v0.1.0 token-observation-span formula remains unchanged historically. Stage 1
retains that distinct concept as `OBSERVED_TOKEN_SPAN_PER_INTERVAL_NS`:

```text
(last_output_token_offset_ns - first_output_token_offset_ns)
/ (output_token_count - 1)
```

It is never labeled TPOT. Tests and the real fixture include a delayed `[DONE]` boundary proving
the two quantities differ.

## ITL and fixture-exact accounting

Input `<pNNN>` and output `<tNNN>` markers are parsed exactly with source `FIXTURE_EXACT`; ordinary
or malformed text is rejected. No tokenizer runs. ITL is the sequence of adjacent observation gaps
only when each output token has its own distinct usable event observation. One event containing two
tokens retains `token_delta_count = 2` and an exact output count but makes ITL unavailable; no
timestamps are fabricated.

## Populations, rates, and distributions

Warmup is retained but excluded. Offered rate counts all started measured attempts; terminal rate
counts every measured terminal; successful request throughput and successful output/total fixture
token throughput include only successes. Goodput counts successful measured requests satisfying
the predeclared successful-E2E fixture SLO. Failure and timeout rates retain their explicit
numerators and the attempted-measured denominator; zero attempts yield null.

E2E, TTFT, TPOT, ITL, and observed-token-span distributions are separately qualified, retain Type
7 p50/p95/p99 and sample count, and exclude requests lacking the metric. Timing-derived values are
serialized deterministically from integer nanoseconds, with Type 7 results rounded to six decimal
places and per-second rates rounded to nine decimal places.

These measurements are loopback fixture measurements used to verify the benchmark harness and
measurement semantics. They are not LLM-serving, model, runtime, GPU, or production-performance
measurements.
