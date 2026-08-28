# Architecture

Stage 0 is deliberately a pure validation and derivation layer:

1. `contracts.py` defines closed, immutable, versioned Pydantic models.
2. `canonical.py` normalizes JSON-compatible values and derives unkeyed SHA-256 identities.
3. `metrics.py` derives request metrics and summaries from retained synthetic request records.
4. `comparison.py` checks artifact identity, declared compatibility, sample sufficiency, and only
   then derives deltas.
5. `schema_io.py` generates canonical JSON Schemas from the Pydantic models.
6. `cli.py` exposes validation, version, and schema synchronization checks only.

There is no adapter, server client, model loader, tokenizer loader, scheduler, profiler, or
hardware execution path. Client concurrency and server batch observations are represented as
different concepts with different provenance.
