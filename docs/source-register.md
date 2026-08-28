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
| [Pinned vLLM completion serving source](https://github.com/vllm-project/vllm/blob/2cf0a6915ce544dc493a0990f2ea38d81601128a/vllm/entrypoints/openai/completion/serving.py) | Read-only review of completion response IDs, item IDs, token-ID streaming, final usage, and per-request metrics | Source-contract review at the exact revision only; no import or execution |
| [Pinned vLLM completion protocol source](https://github.com/vllm-project/vllm/blob/2cf0a6915ce544dc493a0990f2ea38d81601128a/vllm/entrypoints/openai/completion/protocol.py) | Read-only review of request fields including explicit request ID, token IDs, and stream interval | Source-contract review only |
| [Pinned vLLM engine protocol source](https://github.com/vllm-project/vllm/blob/2cf0a6915ce544dc493a0990f2ea38d81601128a/vllm/entrypoints/openai/engine/protocol.py) | Read-only review of final per-request timing metric field names | Source-contract review only |
| [Pinned vLLM input processor source](https://github.com/vllm-project/vllm/blob/2cf0a6915ce544dc493a0990f2ea38d81601128a/vllm/v1/engine/input_processor.py) | Read-only review of the eight-character internal request-ID suffix | Source-contract review only |
| [Pinned vLLM asynchronous engine source](https://github.com/vllm-project/vllm/blob/2cf0a6915ce544dc493a0990f2ea38d81601128a/vllm/v1/engine/async_llm.py) | Read-only review of request-add and abort log identity behavior | Source-contract review only |
| [Pinned vLLM server middleware source](https://github.com/vllm-project/vllm/blob/2cf0a6915ce544dc493a0990f2ea38d81601128a/vllm/entrypoints/serve/utils/server_utils.py) | Read-only review of response request-ID header behavior | Source-contract review only |
| [Pinned vLLM Prometheus source](https://github.com/vllm-project/vllm/blob/2cf0a6915ce544dc493a0990f2ea38d81601128a/vllm/v1/metrics/prometheus.py) | Read-only review of metric names and labels | Parser contract only; no live metrics observed |
| [Pinned vLLM metrics logger source](https://github.com/vllm-project/vllm/blob/2cf0a6915ce544dc493a0990f2ea38d81601128a/vllm/v1/metrics/loggers.py) | Read-only review of model, engine, and finished-reason labeling | Parser contract only |
| [Commit-specific vLLM CUDA 12.9 metadata](https://wheels.vllm.ai/2cf0a6915ce544dc493a0990f2ea38d81601128a/cu129/vllm/metadata.json) | Read-only artifact URL metadata for the separate execution lock | Metadata only; wheel not downloaded or installed |
| [Official PyTorch CUDA 12.9 index](https://download.pytorch.org/whl/cu129/) | Read-only artifact filenames and exposed hashes for the separate execution lock | Metadata only; artifacts not downloaded or installed |
| `.github/workflows/ci.yml` | Remote source/evidence verification configuration | Executed at `68e64bc` by run `33164155869`; repository verification only, with no model/runtime/GPU/performance evidence |
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

The Stage 2A vLLM and package-index sources were reviewed read-only on 2026-08-28 at exact revision
`2cf0a6915ce544dc493a0990f2ea38d81601128a`. The ordinary development environment contains none of
those runtime packages. Official index metadata did not expose the selected torchvision wheel
SHA-256, so the separate execution lock remains
`BLOCKED_BINARY_RETRIEVAL_AUTHORIZATION_REQUIRED`; Stage 2A did not download the binary to derive
that hash.
