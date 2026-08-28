# Failure semantics

Every measured request terminates as `SUCCESS`, `FAILED`, `TIMEOUT`, or `CANCELLED`. A non-success
Stage 1 record retains a matching failure kind, origin, stable error code, occurrence offset, and,
for a timeout, the complete configured timeout policy. The taxonomy distinguishes `HTTP_STATUS`,
`PROTOCOL_MALFORMED_STREAM`, `TIMEOUT`, `TRANSPORT`, `TOKEN_ACCOUNTING`, `CANCELLED`, and
`UNEXPECTED`. Stage 0 outcome names remain unchanged under v0.1.0.

Failed and timed-out requests:

- remain in the terminal-request denominator;
- do not contribute to successful request or token throughput;
- never satisfy goodput;
- retain partial stream observations when present, without turning them into success metrics.

Warmup failures are retained as warmup records but, like all warmup records, are excluded from
measured summaries. An applicable SLO with an unavailable metric is unsatisfied. All applicable
SLO thresholds must pass for a request to contribute to goodput.

Stage 1 retains each nonempty raw response-body chunk before parsing it. A valid token event
followed by malformed JSON remains a protocol failure with its earlier raw and parsed token
evidence intact. A response comment followed by a read stall remains a timeout with its partial
body intact. Neither population contributes partial output to successful token throughput.

For measured requests, `failure_rate` counts only final `FAILED` terminals and explicitly excludes
timeouts and cancellations; `timeout_rate` counts only final `TIMEOUT` terminals. Both use all
started non-warmup attempts as their denominator and retain numerator and denominator. A zero
denominator produces null/unavailable, never division by zero.
