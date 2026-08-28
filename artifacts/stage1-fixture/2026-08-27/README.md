# Stage 1 checked fixture evidence — 2026-08-27

This directory contains checked deterministic loopback HTTP fixture evidence.

It executes no model and no model-serving runtime.

The retained timing and throughput values are local harness-verification measurements, not
LLM-serving, runtime, GPU, production, or historical benchmark evidence.

## Evidence boundary

- Evidence scope: `TEST_FIXTURE_ONLY`
- Execution source commit: `66c7f8c6d1c254c89e10c59747d7f957449ba758`
- Streaming implementation commit: `927a0a1c57e7c90aef87f5282093a3076e786b73`
- Archive-safety fix commit: `66c7f8c6d1c254c89e10c59747d7f957449ba758`
- Historical authentication effect: `NONE`
- Real runtime, model, tokenizer, GPU, and CUDA execution: `false`
- Performance claim allowed: `false`

The fixture runner created its own HTTP server on IPv4 loopback `127.0.0.1` with an OS-assigned
port. No external endpoint, runtime, model, or tokenizer was used.

## Checked inputs

- Workload: `examples/workloads/streaming-fixture-v1.json`
  - file SHA-256: `2563e8cbe46d637c5f111de5aa11b362da21d6ecf2cda2e439d37a648225dba9`
  - canonical identity: `7515b8dee292044cddcd376815119fc51e664d70938394a181c17016e10bb887`
- Configuration: `examples/configs/stage1-streaming-v1.json`
  - file SHA-256: `fc2f677fe2e6af65562ad563c82b92c55cb16f1dff3dae15b8a9627ddbaaf4ba`
  - canonical identity: `b3c26f68937707b8e140e72ea5c5bd0b8ec32d0cfe0b44012368eef473726fdf`
- Fixture: `examples/fixtures/streaming-fixture-v1.json`
  - file SHA-256: `de6eac7819a824e2ec268b08f1d8dd22773107acddd12f25dd9271937d02842b`
  - canonical identity: `adba849b10c04b9fd86f3934dac66e32b8ca9812a17f119ff920e9a912b820ea`
- Regression policy: `examples/configs/stage1-regression-policy-v1.json`
  - file SHA-256: `0d9ba900ca4957f12af6b98489c3813e5bec3d572d24151ecd280e30221ff849`
  - canonical identity: `3bb04e9eeaccd65341e8d0bb753235458cdd1af96ba23cc0e0710f98b03da18d`

Execution used package version `0.2.0`, Stage 0 contract `0.1.0`, Stage 1 measurement contract
`0.2.0`, Python `3.13.15`, and HTTPX `0.28.1`.

## Stable semantic results

Run A and Run B are separate fixture executions. Their semantic fingerprints both equal
`ac33b44eaa11e320bb12ffaac13fbaf381856b132f311ec408e8828064784719`; their run-specific content
identities differ. The semantic comparison passed with timing performance gates disabled.

- Requested client concurrency: `2`
- Observed maximum client concurrency: `2`
- Server batch observed: `false`
- Retained warmup records: `1` (excluded from measured denominators)
- Measured requests: `8`
- Successful measured requests: `5`
- Failed non-timeout measured requests: `2`
- Timed-out measured requests: `1`
- Failure rate: `2 / 8 = 0.25`
- Timeout rate: `1 / 8 = 0.125`

Any retained distributions are **TEST_FIXTURE_ONLY loopback harness diagnostics** with tiny sample
counts. Token counts use synthetic `FIXTURE_EXACT` markers; they are not tokenizer or
server-reported token evidence. Client concurrency is not server batch size, and concurrency `2`
is not a `1–64` sweep.

## Contents

- `run-a/` and `run-b/`: raw request, client-stream, and server evidence; stored summary; and run
  manifest for each independent execution.
- `comparison.json`: reconstructed semantic comparison under the checked policy.
- `evidence-manifest.json`: deterministic inventory, hashes, lineage, and evidence-boundary values.

Run directories validate and their summaries reconstruct from the retained raw evidence. Hashes
provide content identity and modification detection only; they are not signatures, authorship
proof, or historical authentication.
