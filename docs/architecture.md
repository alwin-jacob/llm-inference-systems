# Architecture

Stage 0 remains a pure validation and derivation layer:

1. `contracts.py` defines closed, immutable, versioned Pydantic models.
2. `canonical.py` normalizes JSON-compatible values and derives unkeyed SHA-256 identities.
3. `metrics.py` derives request metrics and summaries from retained synthetic request records.
4. `comparison.py` checks artifact identity, declared compatibility, sample sufficiency, and only
   then derives deltas.
5. `schema_io.py` generates canonical JSON Schemas from the Pydantic models.
6. `cli.py` exposes validation, version, and schema synchronization checks only.

Stage 1 composes a separate fixture-only execution path:

1. versioned Stage 1 workload, configuration, and fixture inputs;
2. a bounded closed-loop load generator with synchronized client lifecycle evidence;
3. one shared HTTPX `AsyncClient` whose base URL is constructed internally from the fixture;
4. an `asyncio.start_server` endpoint fixed to `127.0.0.1:0`;
5. controlled HTTP/1.1 request parsing and chunked SSE response writes;
6. raw HTTPX `aiter_raw()` body retention and incremental SSE parsing;
7. exact fixture-marker token accounting and explicit terminal failure records;
8. atomic JSON/JSONL persistence with file digests;
9. pure raw-to-summary reconstruction and semantic fingerprinting; and
10. compatibility validation followed by semantic-only comparison.

The client and server communicate through an actual IPv4 loopback TCP socket. No mock transport,
ASGI transport, fake response, arbitrary endpoint, model loader, tokenizer loader, profiler, GPU,
or hardware-runtime path exists. Client concurrency and server batch observations remain distinct
concepts with different provenance; server batch is unobserved and null.
