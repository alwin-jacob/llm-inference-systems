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

Stage 2A adds protocol-lifecycle failures without changing either historical taxonomy. A future
successful response requires one generation, usage, protocol, and transport terminal in strict
order. Malformed, missing, duplicated, reordered, or post-terminal data; token/usage disagreement;
or ambiguous identity correlation invalidates the repetition while raw evidence remains retained.

The cancellation evaluator reports one of
`SERVER_ABORT_ACKNOWLEDGED_AND_DRAINED`, `UNKNOWN_ACKNOWLEDGEMENT`, `LATER_COMPLETION`,
`RESIDUAL_WORK_TIMEOUT`, `TERMINAL_UNKNOWN`, or `ID_CORRELATION_FAILURE`. Acceptance requires the
correlated abort chain, exact drain cadence, stable generation counter, cooldown, deadline, allowed
abort counter, and no residual state. A terminal or counter from a non-abort reason is not accepted
as cancellation success.

Stage 2A bundles expose `INCOMPLETE`, `INVALID`, and `COMMITTED`. A crash before the final manifest
remains inspectably incomplete. Durability-operation failure is recorded as invalid at the visible
staging or final bundle path, including failure after directory rename. Summaries refuse
non-committed bundles, and aggregate commit requires three validated, distinctly indexed bundle
manifests plus semantic records bound to those manifest-file content hashes and a passing
reconstructed comparison.
