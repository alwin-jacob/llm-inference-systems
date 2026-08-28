# Stage 0 and Stage 1 project specification

## Objective

Preserve the deterministic Stage 0 measurement-contract foundation and add one bounded,
reconstructable Stage 1 loopback streaming fixture without executing an LLM runtime or making
performance claims.

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

## Non-goals

Real runtime integration, arbitrary or external benchmark networking, model or tokenizer
acquisition, GPU work, profiling, containers, deployment, package publication, and real
benchmarking remain Stage 1 non-goals. The only benchmark networking is the Stage 1 runner's actual
TCP traffic to its own `127.0.0.1` ephemeral fixture.

Public source release is a separate governance action. It does not expand the
`TEST_FIXTURE_ONLY` evidence scope or authorize any model, tokenizer, serving-runtime, GPU, CUDA,
profiler, deployment, package-publication, or performance claim.
