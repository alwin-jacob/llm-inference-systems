# Public evidence boundary

The repository provides source, test, generated-schema, and local verification evidence for a
Stage 0 synthetic measurement contract. `TEST_FIXTURE_ONLY` values demonstrate code paths and
invariants; they are not LLM measurements.

Stage 0 does not establish any of the following:

- vLLM;
- SGLang;
- TensorRT-LLM;
- LLaMA or Mistral serving;
- CUDA;
- NVIDIA GPU execution;
- H200, H100, or A100 execution;
- paged attention;
- continuous or in-flight batching;
- KV-cache behavior;
- FP8 or INT8 execution;
- speculative decoding;
- Nsight;
- Prometheus or Grafana;
- throughput or latency benchmark results;
- any approximately 30% result;
- historical résumé authentication;
- remote CI execution.

The checked-in CI YAML is configuration evidence only. It must not be described as executed until
separate remote evidence exists. Unkeyed hashes show deterministic content identity and integrity,
not signature, authorship, provenance authenticity, or real execution.
