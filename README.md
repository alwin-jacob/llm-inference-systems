# LLM Inference Systems

This repository currently implements a Stage 0 measurement-contract foundation for a
future LLM serving benchmark. It validates versioned workload and artifact contracts plus
deterministic metric and comparison logic using synthetic test fixtures. It does not execute
an LLM server or establish runtime, model, GPU, or performance evidence.

## Evidence boundary

Every Stage 0 run artifact and comparison report is limited to
`TEST_FIXTURE_ONLY`. Synthetic values exist to test contract behavior; they are not benchmark
results and cannot be promoted to runtime or hardware evidence.

Stage 0 does not establish vLLM, SGLang, TensorRT-LLM, LLaMA or Mistral serving, CUDA,
NVIDIA GPU execution, H200/H100/A100 use, paged attention, continuous or in-flight batching,
KV-cache behavior, FP8/INT8, speculative decoding, Nsight, Prometheus/Grafana, throughput or
latency benchmark results, any approximately 30% result, historical résumé authentication, or
remote CI execution. The workflow file is configuration evidence only.

## Implemented foundation

- Strict Pydantic contracts with explicit `0.1.0` schema and measurement versions.
- Canonical JSON and unkeyed SHA-256 content identities, including self-hash omission.
- Retained warmup, measured, success, timeout, and failure records.
- Type 7 percentiles; first-output-token TTFT; TPOT; conditional ITL; request, token, and
  failure-aware goodput rates.
- Compatibility policies that reject undeclared workload, model, tokenizer, sampling, output
  limit, timeout, SLO, measurement-contract, runtime, and hardware differences.
- Canonical JSON Schemas generated from Pydantic models and checked byte-for-byte.
- A deterministic validation-only CLI. There is no run, serve, benchmark, profile, deploy,
  model-download, GPU, or network command.

Client concurrency is a load-generator property. A configured or observed server batch size is
a separate field and is never inferred from client concurrency.

## Local verification

Use the locked development environment and run the complete gate in this order:

```console
uv lock --check
uv sync --python 3.13.15 --frozen --group dev
uv run python --version
uv run ruff check .
uv run ruff format --check .
uv run mypy src tests scripts
uv run pytest -q
uv run python scripts/check_schema_sync.py
uv run python scripts/check_public_safety.py
uv run python scripts/verify_stage0.py
uv run llm-inference version
uv run llm-inference validate-workload examples/workloads/deterministic-smoke-v1.json
uv run llm-inference validate-config examples/configs/stage0-contract-v1.json
uv run llm-inference schema-check
git diff --check
git remote -v
git status --short
```

Schema files are generated only after the package builds:

```console
uv run python scripts/check_schema_sync.py --write
```

The unkeyed SHA-256 values are deterministic content identities and integrity checks. They are
not signatures, do not authenticate an author or origin, and do not prove that synthetic
records came from a real runtime.
