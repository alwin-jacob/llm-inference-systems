# Public evidence boundary

The repository provides source, test, generated-schema, and local verification evidence for its
Stage 0 contracts and Stage 1 deterministic loopback fixture. `TEST_FIXTURE_ONLY` values
demonstrate code paths, actual local fixture HTTP traffic, and reconstruction invariants; they are
not LLM measurements.

| Evidence category | Current local state after checked-evidence generation |
| --- | --- |
| Implemented source | Present locally |
| Local unit/adversarial tests | Present locally; 182 passed before evidence generation |
| Fresh-archive verification | Stage 1 implementation/archive-safety source gate passed; the evidence commit also requires the separate post-commit archive gate |
| Checked loopback `TEST_FIXTURE_ONLY` execution | Present locally as two independent runs dated `2026-08-27` |
| Checked raw artifacts | Present locally with request, client-stream, and server records |
| Checked derived summaries | Present and exactly reconstruct from retained raw evidence |
| Checked semantic comparison | Present; compatible and passing with performance gating disabled |
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
separate remote evidence exists. The checked fixture run uses requested and observed client
concurrency `2`; this is not a `1–64` sweep and is not server batch-size evidence. Loopback fixture
timing distributions retain tiny sample counts and cannot be interpreted as service-performance
estimates. `FIXTURE_EXACT` synthetic token markers are not pinned-tokenizer or server-reported token
evidence. Unkeyed hashes show deterministic content identity and modification detection, not
signature, authorship, provenance authenticity, historical authentication, or real-runtime
execution. This local evidence is not a release and does not promote any canonical claim status.
