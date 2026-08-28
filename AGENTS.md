# Repository agent instructions

- Keep execution and claims within the synthetic-fixture-only Stage 0/Stage 1 and CPU-fixture-only
  Stage 2A boundary. Stage 2A may encode a future runtime protocol but may not execute it.
- Preserve every v0.1.0 contract and schema byte-for-byte; add incompatible semantics under a new
  version.
- Permit networking only through the Stage 1 or Stage 2A fixture paths. Each must create its own
  server on IPv4 loopback `127.0.0.1` with an OS-assigned port and expose no arbitrary destination
  option.
- Never add an executable real-runtime, model, tokenizer, serve, benchmark, profile, deploy,
  model-download, GPU, CUDA, container, cloud, or paid-service path. Keep the separate Stage 2
  Linux/CUDA execution lock uninstalled and unexecuted.
- Do not add or use a Git remote unless the exact external Git operation is explicitly authorized.
- Preserve strict schema versions, deterministic canonicalization, raw failure evidence, atomic
  final-file writes, and exact raw-to-summary reconstruction.
- Generate schemas from Pydantic models; never hand-edit committed schema files.
- Keep every locally executed workload, fixture marker, result, and artifact explicitly synthetic
  and `TEST_FIXTURE_ONLY`. Future-runtime fields are contracts, not execution evidence.
- Never call client concurrency server batch size or infer server batching without direct evidence.
- Run every check documented in `README.md` before a commit.
