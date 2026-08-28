# Public evidence boundary

The repository provides source, test, generated-schema, and local verification evidence for its
Stage 0 contracts and Stage 1 deterministic loopback fixture. `TEST_FIXTURE_ONLY` values
demonstrate code paths, actual local fixture HTTP traffic, and reconstruction invariants; they are
not LLM measurements.

| Evidence category | Current reviewed state |
| --- | --- |
| Implemented source | Present locally |
| Local unit/adversarial tests | Present locally and included in the complete verification gate |
| Metadata-free Git-archive verification | Present as a repeatable exact-Python-3.13.15 local and CI gate |
| Checked loopback `TEST_FIXTURE_ONLY` execution | Present locally as two independent runs dated `2026-08-27` |
| Checked raw artifacts | Present locally with request, client-stream, and server records |
| Checked derived summaries | Present and exactly reconstruct from retained raw evidence |
| Checked semantic comparison | Present; compatible and passing with performance gating disabled |
| CI configuration | Present and executed in private staging |
| Remote CI | Established for source-reviewed SHA `56a06e7` by private run `33161428242` |
| Current release-head CI | Must be verified separately against the release-preparation HEAD before publication |
| Real runtime | Not established |
| Real model/tokenizer | Not established |
| GPU/hardware execution | Not established |
| Profiler evidence | Not established |
| Repository visibility | External GitHub setting; source content alone does not prove public availability |

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
- historical résumé authentication.

Private GitHub Actions run `33161428242` executed the source-reviewed SHA `56a06e7` and
passed four jobs: the Python 3.12/3.13 source matrix, exact-Python-3.13.15 checked-evidence
verification, and exact-Python-3.13.15 metadata-free archive verification. The
release-preparation descendant must have its own passing private run before a separate publication
decision; current HEAD and workflow status must be verified externally.

Repository visibility is an external GitHub setting and must be independently verified; source
content alone does not prove public availability. The checked fixture run uses requested and
observed client concurrency `2`; this is not a `1–64` sweep and is not server batch-size evidence.
Loopback fixture timing distributions retain tiny sample counts and cannot be interpreted as
service-performance estimates. `FIXTURE_EXACT` synthetic token markers are not pinned-tokenizer or
server-reported token evidence. Unkeyed hashes show deterministic content identity and modification
detection, not signature, authorship, provenance authenticity, historical authentication, or
real-runtime execution. This fixture evidence does not promote any unsupported runtime, model,
hardware, performance, historical, or interview-defense claim.
