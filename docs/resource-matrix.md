# Authorized resource matrix

The only public-safe host statement for this Stage 0 work is:

- macOS arm64;
- Apple M3 Pro;
- 18 GB reported memory;
- no local NVIDIA GPU;
- no local CUDA toolchain;
- no paid resource authorized.

Stage 1 fixture execution uses only:

- the locked open-source HTTPX `0.28.1` dependency and its locked transitives;
- a standard-library `asyncio` server on `127.0.0.1` with an OS-assigned ephemeral port;
- local automated tests, temporary fixture artifacts, and fresh-archive verification under
  `/tmp`.

Fixture execution uses no external endpoint, model, tokenizer, serving runtime, GPU, CUDA,
container, profiler, credential, paid API, or cloud inference resource. GitHub remote and Actions
execution are repository release-verification infrastructure only; they are not part of the
measured inference environment and do not establish model-serving or hardware evidence. The host
facts describe a local validation context, not an inference or benchmark environment.

Stage 2A remains within the same local CPU-fixture boundary. It adds no installed runtime, model,
tokenizer, CUDA package, GPU package, or external endpoint. Its separate, unexecuted Linux/CUDA lock
describes a future provider gate but does not assert that such a provider was allocated or audited.
The complete three-repetition fixture includes CUDA and NVIDIA attestation *shapes* only; its
aggregate root is explicitly `SYNTHETIC_PROTOCOL_SHAPE_ONLY` and cannot convert configured resource
fields into observed hardware evidence.

The future fixed gate requires Linux x86-64, at least four logical CPUs, at least 28,000,000,000
bytes of memory, a filesystem of at least 19,000,000,000 bytes, at least 14,000,000,000 initially
free bytes, at least 5,000,000,000 post-setup free bytes, exactly two physical NVIDIA T4 devices,
and exactly one runtime-visible device. These are encoded requirements, not observed host facts.

The future dynamic gate retains source-attributed estimates for runtime/CUDA downloads, installed
environment, model/tokenizer snapshot, and temporary extraction/cache space. It calculates:

```text
required_setup_bytes = sum(all four component estimates)
required_free_before_setup = max(
  14_000_000_000,
  ceil(required_setup_bytes * 1.25) + 2_000_000_000
)
```

Both fixed and dynamic gates must pass before any separately authorized Stage 2B execution.

The future runtime-control contract separately requires exactly five nonnegative GPU-memory
samples at least 200 ms apart. Its stability tolerance is the larger of one percent of the first
sample rounded upward and 67,108,864 bytes; the observed range may not exceed that value. Stage 2A
tests this arithmetic with CPU synthetic values and does not observe or assert GPU memory.
