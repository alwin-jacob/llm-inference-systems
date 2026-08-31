# LLM Inference Systems

This repository preserves its Stage 0 measurement-contract foundation and Stage 1 deterministic
loopback streaming vertical slice, and adds the Stage 2A protocol layer at package/protocol version
`0.3.0`. Stage 2A uses CPU-only scripted fixtures to test the evidence protocol required by a
future real-runtime execution.

The fixtures execute no model, tokenizer, or LLM-serving runtime. No vLLM, GPU, or CUDA execution
has occurred. Fixture timings are local harness-verification measurements and are not model,
runtime, GPU, capacity, or production-performance evidence.

## Evidence boundary

Every executed Stage 0, Stage 1, and Stage 2A fixture is limited to `TEST_FIXTURE_ONLY`. Stage 1 uses
exact project-authored `<pNNN>` and `<tNNN>` markers rather than a tokenizer. Its retained timing
values verify client-observed boundaries, derivation semantics, and reconstruction; they are not
LLM-serving results.

Stage 0, Stage 1, and Stage 2A do not establish vLLM, SGLang, TensorRT-LLM, LLaMA or Mistral
serving, CUDA, NVIDIA GPU execution, H200/H100/A100 use, paged attention, continuous or
in-flight batching, KV-cache behavior, FP8/INT8, speculative decoding, Nsight,
Prometheus/Grafana, throughput or latency benchmark results, any approximately 30% result, or
historical résumé authentication. The separate Linux/CUDA execution lock is metadata only,
uninstalled, and unexecuted. Stage 2B requires separate controller authorization. No current claim
status or public résumé wording advances.

## Implemented foundation

- Byte-preserved Stage 0 `0.1.0` contracts plus isolated Stage 1 `0.2.0` contracts and generated
  schemas, with Stage 2A additions under `0.3.0`.
- Canonical JSON and unkeyed SHA-256 content identities, including self-hash omission.
- A loopback-only `asyncio.start_server` HTTP/1.1 chunked SSE fixture and one shared scoped HTTPX
  `AsyncClient` with environment trust, redirects, and HTTP/2 disabled.
- Incremental SSE parsing that handles split/coalesced events, comments, `[DONE]`, malformed
  streams, and exact partial raw-body retention.
- One retained warmup plus eight measured cases: five successes, two non-timeout failures, and one
  timeout under requested client concurrency two.
- Client dispatch, response headers, first body bytes, first fixture output token, and terminal
  boundaries measured only with `time.monotonic_ns()`.
- Canonical Stage 1 TPOT from the success-terminal boundary, conditional ITL, separately named
  observed-token-span timing, Type 7 qualified distributions, explicit failure/timeout rates, and
  population-qualified throughput fields.
- Atomic JSON/JSONL final-file persistence, raw-file digests, pure summary reconstruction,
  lifecycle-derived concurrency, run-specific content identity, and stable semantic fingerprint.
- Compatibility policies that reject undeclared workload, model, tokenizer, sampling, output
  limit, timeout, SLO, measurement-contract, runtime, and hardware differences.
- A Stage 1 semantic-only repeat comparison that never gates nondeterministic timing or rate values.
- Canonical JSON Schemas generated from Pydantic models and checked byte-for-byte.
- CLI commands for local fixture execution, run-directory validation, raw summary reconstruction,
  and semantic comparison. The fixture command accepts no host, endpoint, URL, or base URL.
- Strict Stage 2A request, four-terminal SSE, token/usage, request-ID, Prometheus, cancellation,
  runtime-phase, process-specific environment, immutable launch, snapshot-manifest,
  fixture/real-attestation, dynamic-resource, and tiny-sample reporting contracts. A successful
  usage terminal requires the exact five-field typed metrics object, including explicit nulls.
- Manifest-last `INCOMPLETE`/`INVALID`/`COMMITTED` repetition bundles, exact raw reconstruction,
  three-restart semantic comparison, and aggregate-commit gating.
- A cardinality-complete experiment boundary requiring exactly three committed repetitions,
  16 measured-request attestations per repetition, three accepted cancellation probes, three
  manifest-bound measured-window Prometheus attestations, three restart-specific CUDA attestation
  shapes, lifecycle-derived concurrency, 16 cross-restart comparisons, derived metric eligibility,
  raw-to-summary reconstruction, and one manifest-last terminal aggregate root. Each measured
  request starts with exact request bytes, ordered transmitted/received header bytes, ordered raw
  response chunks with completed-frame observation clocks, and transport close; parsed SSE and
  typed evidence are replay-derived. Every measured request, cancellation request, and measured-
  window Prometheus scrape is bound to one strict HTTP exchange identity: loopback endpoint,
  configured and observed HTTP versions, status, full Content-Type, complete ordered HTTPX raw
  header pairs, exact body identity, process/restart, and launch specification. Complete means
  complete only at the declared HTTPX request/response-object and `headers.raw`/`aiter_raw()`
  boundary; it is not packet capture and does not expose Ethernet, IP, TCP, TLS, kernel, proxy, or
  server-parser state. Ordinary HTTPX/server-added fields remain retained, while any credential-
  bearing header rejects the capture before durable commit. Cancellation retains its exact
  64-token/512-output request bytes and every raw body-read chunk through the first observed
  generation delivery. Replay retains every complete nonterminal frame and token ID in that
  close-triggering read plus both the exact raw bytes and the CRLF-normalized parser bytes, counts,
  and SHA-256 values of any incomplete trailing SSE fragment. The bounded raw-log byte stream,
  delimiters, and complete record inventory are retained and uniquely correlated. Grouped token
  IDs and coalesced frames are accepted without fabricating per-token clocks, and the probe remains
  ineligible for performance metrics. The actual HTTP response close is awaited and its completion
  is retained separately from the intentional-close invocation. The close classification is
  `INTENTIONAL_CLIENT_CLOSE_AFTER_FIRST_GENERATION_DELIVERY`; a generation/usage terminal,
  `[DONE]`, or clean EOF before close invalidates the probe. It cannot be represented as clean EOF
  or successful stream completion. Each successful `GET /metrics` capture binds the raw
  exposition body to its response headers and supports only `text/plain` or
  `application/openmetrics-text`. The measured-window attestation derives all ten required deltas,
  including `length=16` and zero `abort`, `stop`, `error`, and `repetition`. Semantic
  mismatch is retained only as an `INVALID` root and can never become `COMMITTED`.
- A CPU-only Stage 2A fixture server fixed to `127.0.0.1:0`, including actual HTTPX cancellation
  scenarios for single/grouped token delivery, coalesced frames, incomplete trailing bytes, all
  prohibited terminals, clean EOF, and post-close attribution. Its accepted integration derives
  prompt, generation, abort, drain, and cooldown snapshots from live fixture `/metrics` responses.
  Each scrape keeps the cadence schedule distinct from actual request-dispatch and
  response-completion clocks; the live snapshot observation is the latter.
  Generated `0.3.0` schemas and a verifier prove historical Stage 0/1 bytes and ordinary dependency
  boundaries remain unchanged.

Parsing and reconciliation are evidence-neutral. Request/component attestation remains fixture
scope; only the complete experiment validator can assign a future real-runtime boundary. Its CPU
fixture constructs the full 3 × 16 graph but classifies it only as
`SYNTHETIC_PROTOCOL_SHAPE_ONLY` under `TEST_FIXTURE_ONLY`. It never establishes runtime, model,
tokenizer, Linux/NVIDIA, CUDA, GPU, Kaggle, or performance evidence.

Client concurrency is a load-generator property. A configured or observed server batch size is
a separate field and is never inferred from client concurrency.

## Release and verification state

Stage 1 is publicly released at `40d1ecdc26d1b70f20df42de3e1156e13891cc4d`. GitHub Actions
run `33171272608` passed four jobs: source verification on Python 3.12 and 3.13, checked Stage 1
evidence on exact Python 3.13.15, and metadata-free Git-archive verification on exact Python
3.13.15. Every job used `uv 0.12.5`.

The Stage 2A protocol source at `22e3056dc8e7dbdaaa898ab1b65a358c309529eb` was accepted after
independent source review. Its exact local Python 3.13.15 / `uv 0.12.5` gate passed 684 tests, 25
synchronized schemas, zero-finding public-safety verification, historical Stage 1 verification,
Stage 2A verification, and metadata-free archive verification.

The current branch visibility, branch HEAD, and workflow status are external GitHub state and must
be verified independently. Stage 2A remains `TEST_FIXTURE_ONLY` and its complete CPU-generated
experiment remains `SYNTHETIC_PROTOCOL_SHAPE_ONLY`. Source review, branch visibility, and remote CI
do not establish model, tokenizer, serving-runtime, GPU, CUDA, production, capacity, historical,
interview-defense, or serving-performance evidence.

## Local verification

Use the locked development environment and run the complete gate in this order:

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
uv run llm-inference compare-runs --baseline /tmp/lis-stage1-run-a --candidate /tmp/lis-stage1-run-b --policy examples/configs/stage1-regression-policy-v1.json --output /tmp/lis-stage1-comparison.json
uv run llm-inference validate-run-dir artifacts/stage1-fixture/2026-08-27/run-a
uv run llm-inference validate-run-dir artifacts/stage1-fixture/2026-08-27/run-b
git diff --check
git remote -v
git tag --list
git status --short
uv run python scripts/verify_git_archive.py
```

The final command creates a Git archive from `HEAD`, extracts it without `.git` or `.venv`,
performs a fresh frozen installation under exact Python `3.13.15`, repeats the complete source and
checked-evidence gate including Stage 2A verification, proves historical Stage 0/1 inputs and the
checked artifacts remain byte-identical, and removes the temporary directory.

Schema files are generated only after the package builds:

```console
uv run python scripts/check_schema_sync.py --write
```

The unkeyed SHA-256 values are deterministic content identities and integrity checks. They are
not signatures, do not authenticate an author or origin, and do not prove that synthetic records
came from a real runtime. Stage 1 final-file replacement is atomic, but a crash before persistence
can lose in-memory evidence; this is not a database, WAL, distributed transaction, or resume
system.
