# Artifact directory policy

No run artifacts were committed at Stage 0. The checked Stage 1 set generated on `2026-08-27` is
retained under `artifacts/stage1-fixture/2026-08-27/` after the implementation and archive-safety
commits passed the pre-evidence gate. Tests and the Stage 1 implementation verifier otherwise use
temporary directories.

The checked directory contains two independently executed loopback bundles, their semantic-only
comparison, a deterministic evidence manifest, and a boundary README. Run bundles retain raw
request/client/server evidence and exact hashes, and each stored summary reconstructs from the raw
records. `scripts/verify_checked_stage1_evidence.py` verifies the checked set without generating a
run, contacting a network endpoint, or depending on Git metadata.

Every Stage 1 bundle is `TEST_FIXTURE_ONLY`. Fixture timing and throughput are harness-verification
evidence, not model-serving, runtime, GPU, production-performance, or historical benchmark
evidence. Hashes provide content identity and modification detection, not signatures, authorship,
or historical authentication.
