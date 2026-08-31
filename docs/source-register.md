# Source register

| Source | Role | Evidence limit |
| --- | --- | --- |
| Pydantic models in `contracts.py` | Normative Stage 0 data contracts | Synthetic validation only |
| Generated files in `schemas/` | Machine-readable contract projection | Generated from Pydantic; no manual edits |
| `metrics.py` | Pure deterministic metric definitions | Synthetic fixture inputs only |
| `comparison.py` | Compatibility and delta rules | No cross-runtime/hardware comparison without explicit policy |
| Example JSON files | Workload/configuration validation fixtures | No runtime or performance evidence |
| Tests and `verify_stage0.py` | Adversarial and end-to-end local proof | `TEST_FIXTURE_ONLY` |
| Stage 2A models, protocol modules, CPU fixture, tests, and `verify_stage2a.py` | Local proof of future-runtime protocol invariants | `TEST_FIXTURE_ONLY`; no runtime/model/tokenizer/GPU execution |
| `stage2_experiment.py`, generated experiment/aggregate/wire/Prometheus schemas, and complete 16 × 3 fixture | Exact-wire replay, per-repetition measured-window binding, final cardinality, manifest-link, comparison, eligibility, and reconstruction boundary | `SYNTHETIC_PROTOCOL_SHAPE_ONLY`; cannot establish or advance runtime/performance claims |
| [Pinned vLLM completion serving source](https://github.com/vllm-project/vllm/blob/2cf0a6915ce544dc493a0990f2ea38d81601128a/vllm/entrypoints/openai/completion/serving.py) | Read-only review of completion response IDs, item IDs, token-ID streaming, external serving-item abort logging, final usage, and per-request metrics | Source-contract review at the exact revision only; no import or execution |
| [Pinned vLLM completion protocol source](https://github.com/vllm-project/vllm/blob/2cf0a6915ce544dc493a0990f2ea38d81601128a/vllm/entrypoints/openai/completion/protocol.py) | Read-only review of request fields including explicit request ID, token IDs, and stream interval | Source-contract review only |
| [Pinned vLLM engine protocol source](https://github.com/vllm-project/vllm/blob/2cf0a6915ce544dc493a0990f2ea38d81601128a/vllm/entrypoints/openai/engine/protocol.py) | Read-only review of final per-request timing metric field names | Source-contract review only |
| [Pinned vLLM input processor source](https://github.com/vllm-project/vllm/blob/2cf0a6915ce544dc493a0990f2ea38d81601128a/vllm/v1/engine/input_processor.py) | Read-only review of the eight-character internal request-ID suffix | Source-contract review only |
| [Pinned vLLM asynchronous engine source](https://github.com/vllm-project/vllm/blob/2cf0a6915ce544dc493a0990f2ea38d81601128a/vllm/v1/engine/async_llm.py) | Read-only review of request-add identity, internal abort completion/logging before the external serving-item abort log, and output-collector delta merging that can group a first delivery | Source-contract review at exact revision `2cf0a6915ce544dc493a0990f2ea38d81601128a` only; no import or execution |
| [Pinned vLLM server middleware source](https://github.com/vllm-project/vllm/blob/2cf0a6915ce544dc493a0990f2ea38d81601128a/vllm/entrypoints/serve/utils/server_utils.py) | Read-only review of response request-ID header behavior | Source-contract review only |
| [Pinned vLLM Prometheus source](https://github.com/vllm-project/vllm/blob/2cf0a6915ce544dc493a0990f2ea38d81601128a/vllm/v1/metrics/prometheus.py) | Read-only review of metric names and labels | Parser contract only; no live metrics observed |
| [Pinned vLLM metrics logger source](https://github.com/vllm-project/vllm/blob/2cf0a6915ce544dc493a0990f2ea38d81601128a/vllm/v1/metrics/loggers.py) | Read-only review of model, engine, and finished-reason labeling | Parser contract only |
| [Exact approved vLLM 0.28.0 CUDA 12.9 release asset](https://github.com/vllm-project/vllm/releases/download/v0.28.0/vllm-0.28.0%2Bcu129-cp38-abi3-manylinux_2_28_x86_64.whl) | Exact source URL and controller-authorized SHA-256 in the separate execution lock | Binary was not requested, downloaded, installed, imported, or executed |
| [Official PyTorch CUDA 12.9 index](https://download.pytorch.org/whl/cu129/) | Read-only artifact filenames and exposed hashes for the separate execution lock | Metadata only; artifacts not downloaded or installed |
| [Pinned Qwen snapshot identity](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct/tree/7ae557604adf67be50417f59c2c2f167def9a775) | Exact future repository/revision identity in the separate execution lock | Identity metadata only; snapshot not downloaded or loaded |
| [Pinned Qwen revision API metadata](https://huggingface.co/api/models/Qwen/Qwen2.5-0.5B-Instruct/revision/7ae557604adf67be50417f59c2c2f167def9a775) | Read-only ten-file tree inventory used for the strict required-file allowlist | Metadata only; no model/tokenizer file content retrieved |
| Stage 2A immutable source-review package | Controller-reviewed protocol source at `22e3056dc8e7dbdaaa898ab1b65a358c309529eb`; exact local gate reported 684 tests and 25 synchronized schemas | Source-review and CPU-fixture evidence only; no model/runtime/GPU/performance or interview-defense evidence |
| `.github/workflows/ci.yml` | Remote source/evidence verification configuration | Stage 1 public release `40d1ecdc` passed run `33171272608`; current branch visibility and branch-specific CI remain external GitHub state |
| [HTTPX 0.28.1 on PyPI](https://pypi.org/project/httpx/0.28.1/) | Official package metadata for the exact stable direct dependency | Released 2024-12-06; accessed 2026-08-27; package metadata only |
| [HTTPX async support](https://www.python-httpx.org/async/) | Official documentation for `AsyncClient`, one scoped client, `send(..., stream=True)`, `aiter_raw()`, and explicit `Response.aclose()` in manual mode | Client API semantics only; no runtime-performance claim |
| [HTTPX developer interface](https://www.python-httpx.org/api/#asyncclient) | Official parameters for `trust_env`, redirects, HTTP/2, timeouts, limits, and shared async clients | API configuration semantics only |

The percentile implementation follows the Hyndman-Fan Type 7 formula directly in source. No
external dataset, model output, historical result, or private source is incorporated.

HTTPX `0.28.1` is the latest stable line in the official metadata and was released on 2024-12-06.
The same metadata listed a newer `1.0.dev*` prerelease line (through `1.0.dev5` on 2026-08-21) at
the 2026-08-27 access date. Stage 1 deliberately pins stable `0.28.1` and does not select the
prerelease redesign. Its locked transitives are AnyIO `4.14.2`, Certifi `2026.7.22`, h11 `0.16.0`,
httpcore `1.0.9`, and idna `3.19`; these are dependency facts, not technology or performance
claims.

The Stage 2A vLLM, pinned Qwen tree metadata, and package-index sources were reviewed read-only on
2026-08-28. The vLLM source revision was
`2cf0a6915ce544dc493a0990f2ea38d81601128a`; the Qwen snapshot revision was
`7ae557604adf67be50417f59c2c2f167def9a775`. The ordinary development environment contains none of
those runtime packages. Official index metadata did not expose the selected torchvision wheel
SHA-256, so the separate execution lock remains
`BLOCKED_BINARY_RETRIEVAL_AUTHORIZATION_REQUIRED`; Stage 2A did not download the binary to derive
that hash.
