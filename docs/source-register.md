# Source register

This document records the repository components and pinned external references used to define measurement semantics, protocol behavior, runtime interfaces, and reproducible implementation decisions.

## Repository components

| Source                            | Role                                                                                                       |
| --------------------------------- | ---------------------------------------------------------------------------------------------------------- |
| Pydantic models in `contracts.py` | Versioned Stage 0 workload, configuration, artifact, and comparison contracts                              |
| Generated files in `schemas/`     | Machine-readable projections of the versioned Pydantic contracts                                           |
| `metrics.py`                      | Deterministic metric definitions and summary derivation                                                    |
| `comparison.py`                   | Compatibility validation and comparison-policy evaluation                                                  |
| Example JSON files                | Versioned workload, configuration, and protocol fixtures                                                   |
| Tests and `verify_stage0.py`      | Adversarial and end-to-end verification of Stage 0 contracts                                               |
| Stage 1 streaming implementation  | HTTP/SSE execution, lifecycle capture, persistence, reconstruction, and comparison                         |
| Stage 2A protocol modules         | Request identity, streaming, telemetry, cancellation, runtime-control, resource, and attestation contracts |
| `stage2_experiment.py`            | Repetition assembly, request evidence, cross-restart comparison, eligibility, and aggregate reconstruction |
| `verify_stage2a.py`               | End-to-end Stage 2A protocol verification                                                                  |
| `.github/workflows/ci.yml`        | Automated repository verification                                                                          |

## vLLM protocol references

Stage 2A protocol semantics are pinned to vLLM revision:

```text
2cf0a6915ce544dc493a0990f2ea38d81601128a
```

The following source files define interfaces and behavior represented by the versioned protocol contracts.

| Source                                                                                                                                                   | Protocol role                                                                                                       |
| -------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------- |
| [Completion serving](https://github.com/vllm-project/vllm/blob/2cf0a6915ce544dc493a0990f2ea38d81601128a/vllm/entrypoints/openai/completion/serving.py)   | Completion response IDs, item IDs, token-ID streaming, serving-item lifecycle, final usage, and per-request metrics |
| [Completion protocol](https://github.com/vllm-project/vllm/blob/2cf0a6915ce544dc493a0990f2ea38d81601128a/vllm/entrypoints/openai/completion/protocol.py) | Request fields, explicit request IDs, token IDs, and stream configuration                                           |
| [Engine protocol](https://github.com/vllm-project/vllm/blob/2cf0a6915ce544dc493a0990f2ea38d81601128a/vllm/entrypoints/openai/engine/protocol.py)         | Per-request timing metric field names                                                                               |
| [Input processor](https://github.com/vllm-project/vllm/blob/2cf0a6915ce544dc493a0990f2ea38d81601128a/vllm/v1/engine/input_processor.py)                  | Internal request-ID structure                                                                                       |
| [Async LLM engine](https://github.com/vllm-project/vllm/blob/2cf0a6915ce544dc493a0990f2ea38d81601128a/vllm/v1/engine/async_llm.py)                       | Request-add identity, cancellation chronology, abort behavior, and grouped output delivery                          |
| [Server utilities](https://github.com/vllm-project/vllm/blob/2cf0a6915ce544dc493a0990f2ea38d81601128a/vllm/entrypoints/serve/utils/server_utils.py)      | Response request-ID header behavior                                                                                 |
| [Prometheus metrics](https://github.com/vllm-project/vllm/blob/2cf0a6915ce544dc493a0990f2ea38d81601128a/vllm/v1/metrics/prometheus.py)                   | Runtime metric names and labels used by the Prometheus protocol                                                     |
| [Metrics loggers](https://github.com/vllm-project/vllm/blob/2cf0a6915ce544dc493a0990f2ea38d81601128a/vllm/v1/metrics/loggers.py)                         | Model, engine, and finished-reason labeling                                                                         |

These pinned references make the protocol behavior reviewable against one exact runtime revision rather than an evolving upstream branch.

## Runtime environment references

The versioned execution-lock structures record package and artifact identities for the corresponding Linux/CUDA environment.

Primary references include:

* [vLLM 0.28.0 CUDA 12.9 release asset](https://github.com/vllm-project/vllm/releases/download/v0.28.0/vllm-0.28.0%2Bcu129-cp38-abi3-manylinux_2_28_x86_64.whl)
* [PyTorch CUDA 12.9 package index](https://download.pytorch.org/whl/cu129/)

Artifact identities, versions, and hashes belong to the execution-lock data rather than being duplicated in application code.

## Model and tokenizer snapshot

The snapshot-manifest contract uses the pinned model revision:

```text
Qwen/Qwen2.5-0.5B-Instruct
revision: 7ae557604adf67be50417f59c2c2f167def9a775
```

References:

* [Pinned Qwen revision](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct/tree/7ae557604adf67be50417f59c2c2f167def9a775)
* [Qwen revision metadata](https://huggingface.co/api/models/Qwen/Qwen2.5-0.5B-Instruct/revision/7ae557604adf67be50417f59c2c2f167def9a775)

The revision metadata defines the strict file inventory represented by the snapshot-manifest contract.

## HTTP client

Stage 1 uses HTTPX `0.28.1`.

References:

* [HTTPX 0.28.1 package metadata](https://pypi.org/project/httpx/0.28.1/)
* [HTTPX async support](https://www.python-httpx.org/async/)
* [HTTPX developer interface](https://www.python-httpx.org/api/#asyncclient)

The streaming implementation uses the documented asynchronous client, explicit streaming-response lifecycle, raw-body iteration, timeout controls, connection limits, redirect controls, and explicit response close behavior.

The locked dependency graph includes:

* AnyIO `4.14.2`
* Certifi `2026.7.22`
* h11 `0.16.0`
* httpcore `1.0.9`
* idna `3.19`

## Measurement and statistical definitions

Metric formulas, denominators, availability rules, lifecycle boundaries, and comparison semantics are versioned in the repository's measurement contracts.

Percentiles use the Hyndman–Fan Type 7 interpolation rule implemented directly in source.

The implementation keeps separately modeled:

* request dispatch;
* response headers;
* first response body bytes;
* first output observation;
* generation and usage terminals;
* transport completion;
* TTFT;
* TPOT;
* ITL;
* request throughput;
* token throughput;
* failure and timeout rates;
* goodput;
* client concurrency; and
* server-side telemetry.

## Verification references

Stage 1 public-release source:

```text
40d1ecdc26d1b70f20df42de3e1156e13891cc4d
```

Associated GitHub Actions run:

```text
33171272608
```

Stage 2A reviewed protocol source:

```text
22e3056dc8e7dbdaaa898ab1b65a358c309529eb
```

The Stage 2A verification gate for that source reported:

* 684 passing tests; and
* 25 synchronized schemas.

Together, the source register, versioned contracts, pinned upstream references, generated schemas, checked artifacts, and verification tooling provide reproducible provenance for the implementation decisions represented in this repository.
