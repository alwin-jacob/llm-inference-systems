# Failure semantics

Every measured request terminates as either `SUCCESS` or an explicit failure outcome. A
non-success record must retain a matching `FailureRecord` with a bounded occurrence offset and
an error code. Timeout, cancellation, transport, HTTP, protocol, malformed-stream,
token-accounting, and unexpected failures cannot silently become successes.

Failed and timed-out requests:

- remain in the terminal-request denominator;
- do not contribute to successful request or token throughput;
- never satisfy goodput;
- retain partial stream observations when present, without turning them into success metrics.

Warmup failures are retained as warmup records but, like all warmup records, are excluded from
measured summaries. An applicable SLO with an unavailable metric is unsatisfied. All applicable
SLO thresholds must pass for a request to contribute to goodput.
