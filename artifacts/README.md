# Artifact directory policy

No run artifacts were committed at Stage 0. Stage 1 authorizes one separately gated checked set
under `artifacts/stage1-fixture/<execution-date>/` only after the implementation commit passes an
archive-without-Git verification. Tests and the Stage 1 verifier otherwise use temporary
directories.

Every Stage 1 bundle is `TEST_FIXTURE_ONLY`, retains raw request/client/server evidence and exact
hashes, and reconstructs its stored summary. Fixture timing is harness-verification evidence, not
model-serving, runtime, GPU, or production-performance evidence.
