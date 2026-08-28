# Public evidence boundary

The repository provides source, test, generated-schema, and local verification evidence for its
Stage 0 contracts and Stage 1 deterministic loopback fixture. `TEST_FIXTURE_ONLY` values
demonstrate code paths, actual local fixture HTTP traffic, and reconstruction invariants; they are
not LLM measurements.

| Evidence category | Current state after Stage 1 implementation |
| --- | --- |
| Implemented source | Present locally |
| Local unit/adversarial tests | Present locally |
| Real loopback `TEST_FIXTURE_ONLY` execution | Present locally |
| Checked fixture artifacts | Added only after the separate evidence gate |
| CI configuration | Present as configuration only |
| Remote CI | Not established |
| Real runtime | Not established |
| Real model/tokenizer | Not established |
| GPU/hardware execution | Not established |
| Profiler evidence | Not established |
| Public repository | Not established |

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
- server batch-size behavior;
- a real-runtime goodput result;
- any approximately 30% result;
- historical résumé authentication;
- remote CI execution.

The checked-in CI YAML is configuration evidence only. It must not be described as executed until
separate remote evidence exists. Loopback fixture timing distributions retain small fixture sample
counts and cannot be interpreted as service-performance estimates. Unkeyed hashes show
deterministic content identity and integrity, not signature, authorship, provenance authenticity,
or real-runtime execution.
