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

# Measurement protocol 0.3.0

Stage 2A preserves every `0.1.0` and `0.2.0` contract and schema byte. It adds a CPU-tested protocol
for future real-runtime evidence without executing a runtime, model, tokenizer, GPU, or CUDA.

## Identity and streaming terminals

For external base ID `E`, the header and JSON request ID are `E`, the response header is `E`, every
response body ID is `cmpl-E`, the serving item is `cmpl-E-0`, and the internal engine ID is
`cmpl-E-0-` followed by exactly eight lowercase hexadecimal characters. Request-add and, for
cancellation, internal engine and external serving-item abort logs must correlate to this single
chain. At pinned vLLM revision `2cf0a6915ce544dc493a0990f2ea38d81601128a`, the internal
`Aborted request(s) cmpl-E-0-<suffix>.` record precedes the external
`Request cmpl-E-0 aborted.` record; the reverse order is pinned-runtime drift.

A successful stream retains client dispatch, response headers, first nonempty body bytes, first
output-token event, generation terminal, usage terminal, protocol terminal, and transport terminal.
The final four occur exactly once in strict order. Generation terminal may carry output IDs or be a
finish-only event, but exactly 32 output IDs must already be accumulated, with `finish_reason` equal
to `length`, before the later usage, `[DONE]`, and clean transport close.

Each request reconciles the exact 64 sent and returned prompt IDs, exact 32 accumulated output IDs,
and server usage `64 + 32 = 96`. Sent IDs, returned IDs, event IDs, final IDs, text and text hash,
server usage, optional future local counts, server per-request metrics, and disagreements remain
separate evidence sources.

Every successful future-runtime usage terminal requires a strict `metrics` object containing
exactly `time_to_first_token_ms`, `generation_time_ms`, `queue_time_ms`, `mean_itl_ms`, and
`tokens_per_second`. Every key is present; each value is either explicit null or a finite,
nonnegative number. Null makes only that server-reported metric unavailable and is never replaced
with a client-derived value.

Client-generation TPOT is:

```text
(generation_terminal_offset_ns - first_output_token_offset_ns)
/ (output_token_count - 1)
```

If any content event groups multiple output IDs, client-generation TPOT and token-observation ITL
are unavailable with reason `GROUPED_TOKEN_EVENT`; stream-output gaps and all non-token-timestamp
evidence remain eligible. No per-token timestamps are synthesized.

## Metrics, cancellation, and repetitions

Prometheus evidence retains raw exposition, parsed samples, full label inventory, wall-clock scrape
provenance, and monotonic scrape offset. Exact `model_name` and `engine` series are selected once;
counters are subtracted only within one process, cannot decrease or reset, and are gated by
quiescent pre/post scrapes. Every retained delta must reconstruct exactly from its before/after
values and carry only the exact expected labels. KV-cache percentage is descriptive and is not
memory utilization.

Every repetition owns exactly one measured-window attestation. Its raw baseline capture follows
the steady-state gate and strictly precedes the first measured dispatch. Its raw final capture
follows the 16th accepted transport terminal and a separately retained final-drain completion
boundary, remains on the same server process, and precedes shutdown. Each scrape must be within one
second of its accepted dispatch/drain gate. Both raw captures and both replay-parsed snapshots are path/size/SHA-256
entries in that repetition's committed manifest. The exact required deltas are `1024` prompt
tokens, `512` generation tokens, `16` length completions, and zero preemptions, prefix-cache
queries, and prefix-cache hits.

The cancellation model retains the complete bounded raw-log bytes, newline delimiters, and every
record with source identity, byte offsets, ordinals, monotonic times, and hashes. Correlation is
rerun across the full record inventory so an omitted or duplicate candidate cannot hide outside a
selected four-record chain. It closes after the first observed generation
delivery: the first HTTPX body read that lets the incremental parser reconstruct at least one valid
nonterminal generation event. The close-triggering read may carry multiple token IDs in one event,
multiple complete nonterminal events, and an incomplete trailing SSE fragment. Every byte, complete
frame, event, and token ID already delivered is retained. Incomplete state retains both the exact
raw trailing transport bytes and the parser's CRLF-normalized pending bytes, with separate counts
and SHA-256 values plus the boundary/incomplete state. The intentional-close invocation is ordered
against abort logs, while successful completion of the awaited HTTP response close is separately
retained. No per-token clock is synthesized from a grouped delivery, and cancellation remains a
non-measured probe with no latency, throughput, ITL, TPOT, or token-rate eligibility.

The probe requires exactly ten pre-dispatch zero running/waiting samples at at least 100-ms
spacing, a baseline for every selected counter, dispatch, first-generation-delivery evidence,
intentional close, pinned internal-then-external abort chronology, and exactly ten post-close
quiescent drain samples. It then requires one continuous second of stable generation count at
100-ms cadence, two seconds of cooldown at 100-ms cadence, and a ten-second hard drain deadline.
Each cancellation scrape separately retains its scheduled cadence offset, actual HTTP request
dispatch offset, and actual response-completion observation offset. Cadence is checked against the
schedule, while snapshot identity, chronological ordering, duration, and the deadline are checked
against the actual response-completion clock; the schedule is never substituted for an observed
scrape time.
Deltas are derived only through exact-label, same-process counter arithmetic. Generation terminal,
usage terminal (including usage in a generation-bearing frame), `[DONE]`, clean EOF, post-close
attribution, or any missing stage, reset, ambiguity, label drift, residual state, or contradictory
later retained sample invalidates the probe. An observed abort success-counter delta of zero or one
is retained; every non-abort delta must be zero.

Each of the three repetition runtime-control records requires all 17 ordered, positive-duration
phases with phase-specific evidence hashes/references retained in that repetition's committed
bundle. It also requires exactly five memory samples at least 200 ms apart and:

```text
tolerance_bytes = max(ceil(first_sample_bytes * 0.01), 67_108_864)
max(samples) - min(samples) <= tolerance_bytes
```

It fixes three excluded stabilization requests, four excluded shape warmups, ten steady-state
quiescent samples, a two-second quiet interval, 16 measured requests at requested client
concurrency two, a final scrape/drain, server-and-worker shutdown, and no residual process or
request. Client concurrency remains distinct from server batch size.

The immutable launch identity includes every fixed launch field, exact ordered argv, and normalized
offline environment. The snapshot identity binds the exact pinned repository/revision, the
metadata-derived ten-file regular-file allowlist, separate Hugging Face local metadata, hashes and
sizes, tokenizer identity, read-only transition, download/offline-verification process records,
and a public-safe absolute local root. These are future evidence requirements, not execution
evidence.

Exactly three fresh, non-replaceable repetition bundles are identified by validated manifests with
indices one through three. Each manifest-byte identity binds its restart, phase controls,
same-process measured Prometheus window, cancellation/drain, and shutdown evidence. Each
reconstructed case record is bound to its manifest hash before prompt IDs, output IDs, finish
reason, usage, and output-text hash are compared. Any mismatch makes semantic reproduction invalid
and prohibits pooled performance interpretation. Stage 2A never calculates or displays p99. P50
and p95 are named descriptive values with exact sample counts and restart grouping; goodput and
capacity advancement remain prohibited.

## Cardinality-complete experiment attestation

The final boundary requires the exact versioned 16-case set in each of repetitions 1, 2, and 3.
Measured-request IDs must equal the runtime-control measured-ID set; three stabilization IDs, four
shape-warmup IDs, the cancellation ID, and 16 measured IDs are locally disjoint, and every external
ID is globally unique across repetitions. Case IDs intentionally repeat only for comparison.

Each measured request binds exact transmitted request-body bytes, the complete frozen parsed
request, ordered lossless request and response headers from a closed public-safe name allowlist,
ordered raw body chunks with per-chunk times/counts/hashes and completed-frame observation clocks,
raw request logs, and manifest-bound transport close. The same incremental SSE
parser derives every stored SSE event, all four terminal boundaries, five-field metrics, lifecycle,
token/usage reconciliation, and typed request evidence. Ten distinct path/hash/size references bind
the five raw and five derived records to the committed repetition manifest; fixture constructors
carry a discriminator that cannot satisfy future collector provenance. Lifecycle intervals are
half-open
`[dispatch, terminal)`. Terminal events are processed
before dispatch events at equal timestamps, so touching intervals do not create overlap. Every
interval lies inside the measured phase, the derived maximum active client count is exactly two,
it never exceeds two, and at least one positive-duration overlap exists. Slot labels are not
concurrency evidence.

Server metric availability is derived independently for each of the exact five fields. Explicit
null disables only that server metric's advancement; client metrics never fill it. Grouped token
events separately disable client-generation TPOT and token-observation ITL. Repetition summaries
require 16 requests and experiment summaries require all 48; a metric is advancement-eligible only
when it is available for the full relevant population.
