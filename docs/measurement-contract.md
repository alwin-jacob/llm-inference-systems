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
