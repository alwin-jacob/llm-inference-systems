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
an omitted/incomplete/non-finite five-field metrics object; or ambiguous identity correlation
invalidates the repetition while raw evidence remains retained.

The cancellation evaluator reports one of
`SERVER_ABORT_ACKNOWLEDGED_AND_DRAINED`, `UNKNOWN_ACKNOWLEDGEMENT`, `LATER_COMPLETION`,
`RESIDUAL_WORK_TIMEOUT`, `TERMINAL_UNKNOWN`, or `ID_CORRELATION_FAILURE`. Acceptance requires the
correlated close → internal engine abort → external serving-item abort chain, exact drain cadence,
stable generation counter, cooldown, deadline, allowed abort counter, and no residual state. The
former external-before-internal fixture order is rejected. Missing, duplicate, ambiguous,
cross-request, cross-process, cross-repetition, or pre-close abort records anywhere in the complete
bounded raw-log capture invalidate the probe.

Cancellation closes after first observed generation delivery, not proof of exactly one token.
Grouped token IDs, multiple complete nonterminal frames in the close-triggering read, and exact
incomplete trailing bytes are valid and losslessly replayed in both raw and CRLF-normalized parser
form. A generation terminal, usage terminal,
same-frame usage, `[DONE]`, clean EOF, response bytes/events timestamped after close, or a close
before its triggering delivery invalidates the probe. A terminal or counter from a non-abort reason
is not accepted as cancellation success.

A future real-runtime boundary is rejected unless the complete experiment exists: three ordered
committed repetitions, exactly 16 declared measured requests per repetition, exact measured-ID set
equality, globally disjoint external IDs, one accepted cancellation probe and one restart-specific
CUDA attestation per repetition, positive lifecycle overlap deriving concurrency two, 16 complete
cross-restart comparisons, a reconstructed aggregate-validation result, and a manifest-last
aggregate root. Phase names and `passed=true` without positive boundaries and manifest-bound
phase evidence are invalid; missing final drain, failed/incomplete server-and-worker shutdown, or
residual state is invalid. Synthetic completeness demonstrates validator behavior only.

Stage 2A bundles expose `INCOMPLETE`, `INVALID`, and `COMMITTED`. A crash before the final manifest
remains inspectably incomplete. Durability-operation failure is recorded as invalid at the visible
staging or final bundle path, including failure after directory rename. Summaries refuse
non-committed bundles, and aggregate commit requires three validated, distinctly indexed bundle
manifests plus semantic records bound to those manifest-file content hashes and a passing
reconstructed comparison.

An output-token mismatch is retained in a manifest-last `INVALID` aggregate root with reason
`INVALID_SEMANTIC_NONREPRODUCTION`; it cannot become a committed aggregate, authorize pooled
performance interpretation, select a replacement run, or advance a claim. Missing request files,
orphan raw hashes, null-server-metric eligibility
overrides, non-overlapping lifecycles, concurrency above two, or an aggregate manifest written
before any inventoried byte are terminal validation failures.
