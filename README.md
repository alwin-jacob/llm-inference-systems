# LLM Inference Measurement Infrastructure

A reproducible inference-measurement and evidence-protocol implementation for streaming, concurrency, lifecycle analysis, experiment reconstruction, and regression testing.

The project develops versioned measurement contracts and execution artifacts for inference-system experiments. The current release includes a deterministic streaming harness, raw-to-summary reconstruction, failure-aware metrics, experiment comparison, and a Stage 2A protocol layer for runtime-backed measurement workflows.

## Implemented foundation

* Versioned Stage 0 `0.1.0`, Stage 1 `0.2.0`, and Stage 2A `0.3.0` contracts and generated schemas.
* Canonical JSON and deterministic SHA-256 content identities.
* HTTP/1.1 streaming over a project-owned `asyncio.start_server` SSE endpoint with a shared scoped HTTPX `AsyncClient`.
* Incremental SSE parsing across split and coalesced events, comments, `[DONE]`, malformed streams, and partial response bodies.
* Explicit request lifecycle boundaries measured with `time.monotonic_ns()`.
* TTFT, TPOT, ITL, observed-token-span timing, Type 7 percentile distributions, failure/timeout rates, throughput, and goodput-oriented measurement contracts.
* Atomic JSON/JSONL persistence with raw-file digests and deterministic summary reconstruction.
* Workload, runtime, hardware, sampling, timeout, SLO, and measurement-contract compatibility validation before comparisons are evaluated.
* Semantic repeat comparison with explicit separation between deterministic semantics and timing-sensitive observations.
* Pydantic-generated JSON Schemas checked byte-for-byte for synchronization.
* CLI workflows for fixture execution, artifact validation, raw summary reconstruction, and run comparison.
* Strict request identity, streaming terminal, token/usage, Prometheus, cancellation, runtime-phase, process-environment, launch, snapshot-manifest, resource, and experiment-attestation contracts.
* Manifest-last `INCOMPLETE`, `INVALID`, and `COMMITTED` experiment bundle lifecycle.
* Three-repetition experiment structure with request-level evidence, restart-aware comparisons, cancellation probes, measured-window telemetry, derived metric eligibility, and aggregate reconstruction.
* Exact request and response evidence including serialized request bytes, ordered headers, raw response chunks, SSE replay, lifecycle boundaries, and transport-close provenance.
* Cancellation capture through first observed generation delivery, including grouped-token and coalesced-frame handling without fabricating per-token clocks.
* Prometheus capture with raw exposition retention, exact label selection, process-bound counter deltas, and measured-window reconstruction.
* CPU-backed Stage 2A fixture execution covering normal streaming, grouped delivery, cancellation, incomplete trailing frames, terminal validation, drain behavior, and cooldown behavior.

## Measurement architecture

The project separates measurement into three layers.

### Stage 0 — measurement contracts

Stage 0 defines immutable workload, run, artifact, metric, and comparison contracts.

Core responsibilities include:

* canonical serialization;
* content identity;
* request outcome modeling;
* TTFT, TPOT, ITL, throughput, and goodput derivation;
* percentile computation;
* compatibility checking;
* regression-policy evaluation; and
* generated schema synchronization.

### Stage 1 — streaming execution and reconstruction

Stage 1 adds an executable streaming path around the measurement contracts.

A bounded asynchronous client communicates with a deterministic HTTP/1.1 SSE server over IPv4 loopback. The harness records the complete request lifecycle needed for reconstruction:

```text
dispatch
   ↓
response headers
   ↓
first body bytes
   ↓
first output event
   ↓
stream events
   ↓
terminal boundary
   ↓
artifact persistence
```

Run artifacts retain raw request, client-stream, and server observations. Stored summaries can be reconstructed from those retained records and compared under a versioned compatibility policy.

The checked Stage 1 fixture contains:

* one excluded warmup;
* eight measured requests;
* five successful requests;
* two non-timeout failures;
* one timeout; and
* requested and observed client concurrency of two.

The two retained executions reproduce the same semantic fingerprint while preserving independent run identities.

## Stage 2A protocol

Stage 2A extends the system from individual streaming runs toward repeatable runtime experiment collection.

The protocol covers:

* exact completion-request identity;
* incremental SSE and terminal sequencing;
* token and usage reconciliation;
* request and response header capture;
* raw response-body retention;
* lifecycle-derived concurrency;
* Prometheus measurement windows;
* cancellation and server-drain analysis;
* immutable launch specifications;
* process-specific environments;
* model/tokenizer snapshot identity;
* restart-scoped runtime evidence;
* resource and CUDA attestation structures;
* repetition manifests;
* cross-restart semantic comparison;
* metric availability and eligibility;
* aggregate experiment validation; and
* manifest-last reconstruction.

A complete experiment is represented as three independently committed repetitions. Each repetition contains the declared measured request set, cancellation evidence, telemetry, runtime-control evidence, and restart-specific state required for later aggregate comparison.

### Request evidence

Each measured request binds:

* exact serialized request bytes;
* request identity;
* ordered transmitted headers;
* ordered received headers;
* HTTP status and content type;
* ordered raw body chunks;
* completed-frame observation clocks;
* parsed SSE events;
* token and usage evidence;
* lifecycle boundaries; and
* transport-close state.

Derived records are reconstructed from retained raw evidence rather than treated as independent observations.

### Streaming semantics

The incremental SSE parser supports:

* split events;
* multiple events in one body read;
* grouped token IDs;
* comments and keepalives;
* `[DONE]`;
* explicit generation and usage terminals;
* incomplete trailing fragments; and
* malformed stream termination.

Grouped delivery is retained without synthesizing timestamps that were not directly observed.

### Cancellation

Cancellation is modeled as its own protocol path.

The collector retains the close-triggering body read, every complete nonterminal event delivered within it, grouped token IDs, trailing incomplete bytes, and the subsequent server-side lifecycle evidence.

The protocol distinguishes:

```text
first generation delivery
        ↓
intentional client close
        ↓
server abort evidence
        ↓
drain
        ↓
stable counters
        ↓
cooldown
```

Request-close invocation and actual response-close completion are retained separately.

### Prometheus measurement

Each repetition includes a measured telemetry window around its request population.

Raw Prometheus exposition is retained and parsed under an exact metric/label contract. Counter arithmetic is restricted to compatible samples from the same server process.

The protocol retains:

* baseline scrape;
* final scrape;
* process identity;
* raw exposition bytes;
* parsed samples;
* label inventory;
* request-window boundaries; and
* reconstructed counter deltas.

## Experiment lifecycle

Experiment bundles use three explicit states:

```text
INCOMPLETE
    ↓
INVALID  or  COMMITTED
```

Files are persisted before the terminal manifest. Reconstruction verifies inventory, hashes, request identities, derived records, repetition structure, and comparison inputs before a bundle can be treated as committed.

At the aggregate level, the experiment binds three repetition manifests, request comparisons, telemetry, cancellation results, resource evidence, semantic reproduction, summary records, and final validation into one terminal root.

## Metrics

The measurement model distinguishes client-observed and server-reported quantities.

### Client-observed

* end-to-end latency;
* time to first token;
* generation-boundary TPOT;
* token-observation ITL when individually observable;
* request throughput;
* token throughput;
* lifecycle concurrency;
* failure rate;
* timeout rate; and
* goodput under declared SLOs.

### Server-side telemetry

The Stage 2A protocol can retain typed per-request runtime metrics and process-level Prometheus observations independently from client measurements.

Availability is tracked per metric. Missing server telemetry is represented as unavailable rather than substituted with a client-derived value.

## Evidence and reconstruction

The system is designed around reconstruction rather than summary-only reporting.

For a retained run or experiment, derived values are expected to trace back to raw observations:

```text
raw request/response evidence
          ↓
protocol parsing
          ↓
typed request evidence
          ↓
per-request metrics
          ↓
run summary
          ↓
repetition comparison
          ↓
aggregate experiment
```

This makes parsing, aggregation, compatibility checking, and metric eligibility independently testable.

The checked Stage 1 artifacts use deterministic loopback execution to verify these invariants. Stage 2A extends the same model to the experiment protocol needed for runtime-backed execution.

## Release and verification state

Stage 1 is publicly released at:

```text
40d1ecdc26d1b70f20df42de3e1156e13891cc4d
```

GitHub Actions run `33171272608` passed:

* source verification on Python 3.12;
* source verification on Python 3.13;
* checked Stage 1 evidence verification on Python 3.13.15; and
* metadata-free Git-archive verification on Python 3.13.15.

Every job used `uv 0.12.5`.

The Stage 2A protocol source at:

```text
22e3056dc8e7dbdaaa898ab1b65a358c309529eb
```

passed the corresponding local verification gate with:

* 684 tests;
* 25 synchronized schemas;
* public-safety verification;
* historical Stage 1 verification;
* Stage 2A verification; and
* metadata-free archive verification.

## Local verification

Use the locked development environment and run:

```console
git diff --check
uv --version
uv lock --check
uv sync --python 3.13.15 --frozen --group dev
uv run python --version
uv run ruff check .
uv run ruff format --check .
uv run mypy src tests scripts
uv run pytest -q
uv run python scripts/check_schema_sync.py
uv run python scripts/check_public_safety.py
uv run python scripts/verify_stage0.py
uv run python scripts/verify_stage1.py
uv run python scripts/verify_checked_stage1_evidence.py artifacts/stage1-fixture/2026-08-27
uv run python scripts/verify_stage2a.py
uv run llm-inference version
uv run llm-inference validate-workload examples/workloads/deterministic-smoke-v1.json
uv run llm-inference validate-config examples/configs/stage0-contract-v1.json
uv run llm-inference validate-workload examples/workloads/streaming-fixture-v1.json
uv run llm-inference validate-config examples/configs/stage1-streaming-v1.json
uv run llm-inference validate-config examples/configs/stage2a-protocol-fixture-v1.json
uv run llm-inference validate-stage2-request examples/fixtures/stage2a-completion-request-v1.json
uv run llm-inference validate-stage2-execution-lock execution-lock/stage2-execution-lock.json
uv run llm-inference schema-check
rm -rf /tmp/lis-stage1-run-a /tmp/lis-stage1-run-b /tmp/lis-stage1-comparison.json
uv run llm-inference fixture-run --workload examples/workloads/streaming-fixture-v1.json --config examples/configs/stage1-streaming-v1.json --fixture examples/fixtures/streaming-fixture-v1.json --output-dir /tmp/lis-stage1-run-a
uv run llm-inference validate-run-dir /tmp/lis-stage1-run-a
uv run llm-inference summarize-run /tmp/lis-stage1-run-a
uv run llm-inference fixture-run --workload examples/workloads/streaming-fixture-v1.json --config examples/configs/stage1-streaming-v1.json --fixture examples/fixtures/streaming-fixture-v1.json --output-dir /tmp/lis-stage1-run-b
uv run llm-inference validate-run-dir /tmp/lis-stage1-run-b
uv run llm-inference summarize-run /tmp/lis-stage1-run-b
uv run llm-inference compare-runs --baseline /tmp/lis-stage1-run-a --candidate /tmp/lis-stage1-run-b --policy examples/configs/stage1-regression-policy-v1.json --output /tmp/lis-stage1-comparison.json
uv run llm-inference validate-run-dir artifacts/stage1-fixture/2026-08-27/run-a
uv run llm-inference validate-run-dir artifacts/stage1-fixture/2026-08-27/run-b
git diff --check
git remote -v
git tag --list
git status --short
uv run python scripts/verify_git_archive.py
```

The final command creates a Git archive from `HEAD`, extracts it into a metadata-free environment, performs a frozen installation under Python `3.13.15`, reruns source and checked-evidence verification, verifies historical Stage 0/1 inputs and retained artifacts, and removes the temporary directory.

Schema files are generated from the package models with:

```console
uv run python scripts/check_schema_sync.py --write
```

## Design properties

The implementation emphasizes:

* versioned measurement semantics;
* deterministic artifact reconstruction;
* raw-to-derived traceability;
* failure-aware statistics;
* explicit metric availability;
* lifecycle-derived concurrency;
* restart-aware comparison;
* bounded asynchronous execution;
* tamper detection;
* atomic persistence; and
* reproducible verification.
