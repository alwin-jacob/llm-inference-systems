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
