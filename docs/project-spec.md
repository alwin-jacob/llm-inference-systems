# Stage 0, Stage 1, and Stage 2A project specification

## Objective

Preserve the deterministic Stage 0 measurement-contract foundation and add one bounded,
reconstructable Stage 1 loopback streaming fixture without executing an LLM runtime or making
performance claims. Add a Stage 2A CPU-fixture-tested protocol for validating a separately
authorized future real-runtime evidence run, while executing none of that runtime stack.

## Required outputs

- versioned workload, configuration, artifact, comparison-policy, and comparison-report models;
- deterministic canonical JSON, unkeyed SHA-256 content identity, and self-hash verification;
- pure metric derivation with explicit unavailable behavior and failure-aware goodput;
- compatibility checks performed before delta calculation;
- generated versioned JSON Schemas and byte-for-byte synchronization checks;
- validation-only CLI, adversarial tests, public-safety scan, and deterministic fixture verifier.
- fixed-destination HTTPX client and standard-library loopback HTTP/1.1 SSE fixture;
- one warmup and eight measured cases covering success, multi-token events, body/token boundary
  separation, malformed partial output, HTTP error, and partial-body timeout;
- exact fixture-marker accounting, canonical Stage 1 TPOT, conditional ITL, failure/timeout rates,
  lifecycle-derived concurrency, raw JSONL retention, atomic writes, exact reconstruction, and
  semantic-only repeat comparison.
- strict Stage 2A request/identity/SSE/typed-five-metric usage, exact Prometheus-series,
  raw-derived cancellation/drain, 17-phase runtime control, process-specific pre-import
  environment, immutable launch, snapshot-manifest, fixture/future-attestation, dynamic-resource,
  manifest-last bundle, three-restart, and tiny-sample reporting contracts;
- CPU-only Stage 2A response/log/metric fixtures, separate uninstalled execution-lock metadata,
  historical Stage 0/1 byte-preservation verification, and generated `0.3.0` schemas.
- one complete synthetic 16 × 3 experiment graph with manifest-bound per-request evidence,
  lifecycle-derived concurrency, three cancellation and CUDA attestation shapes, 16 semantic
  comparisons, derived metric eligibility, aggregate validation, and manifest-last reconstruction.

## Non-goals

Real runtime execution, arbitrary or external benchmark networking, model or tokenizer acquisition,
GPU work, profiling, containers, deployment, package publication, and real benchmarking remain
non-goals. The only fixture networking is actual TCP traffic to project-owned `127.0.0.1`
ephemeral servers. Stage 2A future-runtime contracts and execution-lock metadata are not executable
runtime paths.

Public source release is a separate governance action. It does not expand the
`TEST_FIXTURE_ONLY` evidence scope or authorize any model, tokenizer, serving-runtime, GPU, CUDA,
profiler, deployment, package-publication, or performance claim.
