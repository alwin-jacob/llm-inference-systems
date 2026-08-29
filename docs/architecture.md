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

Stage 2A adds a separate `0.3.0` protocol layer exercised only by CPU fixtures:

1. `stage2_contracts.py` defines strict future-runtime configuration, evidence, phase, resource,
   bundle, and execution-lock models;
2. `stage2_protocol.py` constructs the exact completion request and validates request identity,
   incremental SSE, injected per-frame acceptance time for coalesced transport chunks, four ordered
   terminals, token IDs, usage, and timing without importing a serving runtime;
3. `stage2_prometheus.py` retains raw exposition and provenance, parses exact metric/label series,
   and derives same-process nondecreasing counter deltas;
4. `stage2_runtime.py` defines the immutable launch identity, exact process-kind pre-import
   environments, pinned snapshot inventory/identity, read-only transition, and GPU-memory
   stability contracts;
5. `stage2_control.py` derives cancellation from raw logs and raw Prometheus snapshots and requires
   the 17 ordered phase-specific evidence records, memory/steady-state gates, exact excluded and
   measured counts, final drain, complete shutdown, and no-residual verification;
6. `stage2_attestation.py` keeps parsing evidence-neutral and retains request/component-shaped
   attestations at fixture scope;
7. `stage2_experiment.py` assembles the only final future-runtime boundary: exactly 48 measured
   request attestations in three repetitions, ten distinct manifest-bound raw files per request,
   cancellation and CUDA evidence per restart, lifecycle-derived concurrency, 16 comparisons,
   derived availability/eligibility, aggregate validation, and pure directory reconstruction;
8. `stage2_bundle.py` manages inspectable staging state, durable raw and derived evidence,
   public-safety checks, reconstruction, manifest-last commit, directory durability, and tamper
   validation; and
9. `stage2_fixture_server.py` scripts compatible response, log, and complete five-field metric
   shapes on a server fixed
   to `127.0.0.1:0`.

The Stage 2A code does not expose a runtime launcher, model/tokenizer loader, arbitrary endpoint,
GPU path, or execution-lock installer. Future execution fields remain non-executed protocol
requirements. A resolved-model-implementation record remains distinct from the runtime package
name and requires runtime-reported or directly observed provenance. The separate execution lock
cannot enter the ordinary development dependency graph.

The aggregate experiment root inventories exact canonical bytes for shared environment/resource,
NVIDIA isolation, execution lock plus resolver/install/reviewed-lock raw evidence, snapshot plus
read-only verification evidence, launch, safety, workload, all three repetition
manifests, three CUDA attestations, three cancellation results, all 16 comparisons, availability,
summary, aggregate validation, and final attestation records. Repetition and aggregate manifests
must be written strictly last; reconstruction rejects missing, changed, symlinked, traversing,
case-colliding, or unlisted files. Each of the ten per-request raw records is reconstructed against
the exact request identity, parsed stream, metrics, lifecycle, and token/usage values before the
terminal aggregate manifest can be published.
