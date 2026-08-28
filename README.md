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
- A CPU-only Stage 2A fixture server fixed to `127.0.0.1:0`, plus generated `0.3.0` schemas and a
  verifier that proves historical Stage 0/1 bytes and ordinary dependency boundaries remain
  unchanged.

Parsing and reconciliation are evidence-neutral. `FixtureAttestation` alone assigns
`TEST_FIXTURE_ONLY`; the separate future real-runtime attestation cannot validate unless all 15
typed component identities validate together. Complete synthetic attestation fixtures prove only
the contract shape and never establish runtime, model, tokenizer, Linux/NVIDIA, CUDA, GPU, Kaggle,
or performance evidence.

Client concurrency is a load-generator property. A configured or observed server batch size is
a separate field and is never inferred from client concurrency.

## Remote verification

The release-preparation commit `68e64bc814d805464f239c452fa8261fedbfde0b` passed
GitHub Actions run `33164155869` while the repository was private. Four jobs succeeded: source
verification on Python 3.12 and 3.13, checked Stage 1 evidence on exact Python 3.13.15, and
metadata-free Git-archive verification on exact Python 3.13.15. Every job used `uv 0.12.5`.

The current `main` documentation-only release head is eligible for public visibility only after its
own identical four-job CI gate succeeds. Repository visibility and current workflow status are
external GitHub state and must be verified independently.

These checks establish repository source, test, artifact, and reconstruction evidence only. They
do not establish model, tokenizer, serving-runtime, GPU, CUDA, production, capacity, historical,
or serving-performance evidence.

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
