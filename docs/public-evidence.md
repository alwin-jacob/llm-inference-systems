# Public evidence boundary

The repository provides source, test, generated-schema, and verification evidence for its Stage 0
contracts and Stage 1 deterministic loopback fixture. `TEST_FIXTURE_ONLY` values demonstrate code
paths, actual local fixture HTTP traffic, and reconstruction invariants; they are not LLM
measurements.

| Evidence category | Current reviewed state |
| --- | --- |
| Implemented source | Present in the repository |
| Local unit/adversarial tests | Included in the complete verification gate |
| Metadata-free Git-archive verification | Repeatable exact-Python-3.13.15 local and CI gate |
| Checked loopback `TEST_FIXTURE_ONLY` execution | Two independent runs dated `2026-08-27` |
| Checked raw artifacts | Request, client-stream, and server records retained |
| Checked derived summaries | Exactly reconstruct from retained raw evidence |
| Checked semantic comparison | Compatible and passing with performance gating disabled |
| CI configuration | Present and executed for repository verification |
| Release-preparation remote CI | Passed at `68e64bc` through run `33164155869` |
| Current release-head CI | External GitHub state; publication requires the same four jobs to pass before visibility changes |
| Real runtime | Not established |
| Real model/tokenizer | Not established |
| GPU/hardware execution | Not established |
| Profiler evidence | Not established |
| Repository visibility | External GitHub setting; source content alone does not establish visibility |

Stage 0 and Stage 1 do not establish any of the following:

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
- historical résumé authentication.

GitHub Actions run `33164155869` executed release-preparation SHA `68e64bc` while the repository
was private. Four jobs passed: the Python 3.12/3.13 source matrix, exact-Python-3.13.15
checked-evidence verification, and exact-Python-3.13.15 metadata-free archive verification. Every
job used `uv 0.12.5`.

The publication procedure requires the documentation-only release head to pass the same four jobs
before visibility changes. Current workflow status and repository visibility must be verified
externally; this source file does not establish either state.

The checked fixture run uses requested and observed client concurrency `2`; this is not a `1–64`
sweep and is not server batch-size evidence. Loopback fixture timing distributions retain tiny
sample counts and cannot be interpreted as service-performance estimates. `FIXTURE_EXACT`
synthetic token markers are not pinned-tokenizer or server-reported token evidence. Unkeyed hashes
show deterministic content identity and modification detection, not signature, authorship,
provenance authenticity, historical authentication, or real-runtime execution. This fixture
evidence does not promote any unsupported runtime, model, hardware, performance, historical, or
interview-defense claim.
