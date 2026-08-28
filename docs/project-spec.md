# Stage 0 project specification

## Objective

Establish a deterministic, testable measurement-contract foundation without executing an LLM
runtime or making performance claims.

## Required outputs

- versioned workload, configuration, artifact, comparison-policy, and comparison-report models;
- deterministic canonical JSON, unkeyed SHA-256 content identity, and self-hash verification;
- pure metric derivation with explicit unavailable behavior and failure-aware goodput;
- compatibility checks performed before delta calculation;
- generated versioned JSON Schemas and byte-for-byte synchronization checks;
- validation-only CLI, adversarial tests, public-safety scan, and deterministic fixture verifier.

## Non-goals

Runtime integration, networking, model or tokenizer acquisition, GPU work, profiling, deployment,
and real benchmarking are outside Stage 0.
