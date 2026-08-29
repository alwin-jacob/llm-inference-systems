# Security, cost, and provenance

Stage 0 operates on repository-local source and in-memory synthetic fixtures. Stage 1 adds only a
project-authored fixture server fixed to IPv4 `127.0.0.1` and port `0`; the runner constructs the
client base URL internally and exposes no host, endpoint, URL, or base-URL input. HTTPX uses
`trust_env=False`, `follow_redirects=False`, and `http2=False`, so environment proxies and redirects
cannot redirect the fixture command. Connection count is bounded by requested client concurrency.

The controlled server limits headers and bodies, requires `POST /v1/completions`, validates the
exact request shape, closes each connection, reads no credential or cache, executes no subprocess,
and invokes no model. The repository does not call paid services, invoke cloud tools, download a
model/tokenizer/dataset/container, or start a container. CLI validation limits input size and emits
stable errors without echoing private input values.

Canonical JSON uses sorted keys, compact UTF-8 encoding, finite numeric values, UTC-normalized
timestamps, and deterministic handling of negative zero. SHA-256 is unkeyed and is used only as a
content identity and integrity check. It is never a digital signature and provides no authorship
or origin authentication.

The safety scanner examines repository candidates for private paths, local environment labels,
credential-shaped values, and prohibited private-source references. A passing scan is a scoped
repository-content check, not a general security certification.

Raw-body retention is authorized here only because the fixture body is project-authored synthetic
text. Each raw chunk is Base64 encoded with byte count and SHA-256, and validation reconstructs the
exact observed body. This does not establish that arbitrary provider output is safe to retain.

Final artifact files are written through destination-directory temporary files, flushed, fsynced,
and atomically replaced. This prevents a reader from observing a half-written final file, but does
not guarantee that in-memory evidence survives a process crash. The bundle is not a database, WAL,
distributed transaction, or crash-resume design.

Stage 2A adds only another project-authored CPU fixture server fixed to IPv4 `127.0.0.1` and port
`0`. Future endpoint contracts accept only that literal host. No runtime launcher or downloader is
implemented. The separate execution-lock JSON is inert metadata and is neither resolved nor
installed by ordinary verification.

Future process records are fail-closed by kind. The single online snapshot process must set
telemetry and implicit-token controls before import, prove all token variables absent, use
`token=False`, bind only the exact pinned snapshot source, and exit. Offline tokenizer and three
fresh offline vLLM restart records require their stricter offline/no-proxy environments before any
relevant import and bind only a verified local snapshot. These are validated synthetic contracts;
Stage 2A did not launch any such process. The only correction-time external read was exact pinned
Hugging Face tree metadata used to define the ten-file allowlist; no file content was retrieved.

Stage 2A repetition bundles reject path traversal, symlinked parent or evidence paths, non-UTF-8
evidence, sensitive private material, unapproved binary suffixes, replacement of retained files,
incomplete or duplicate inventory, altered hashes, and derived data that cannot be reconstructed
exactly from raw evidence. Staging starts as `INCOMPLETE`; terminal failures retain public-safe raw
evidence and become `INVALID`; only reconstruction- and inventory-validated evidence receives a
manifest written last after durable directory placement. A failed post-rename durability operation
leaves the visible bundle non-committed. A commit state is an integrity/lifecycle fact, not
publication or claim approval.

The public-safety scanner rejects credential-shaped values, authorization and cookie headers,
proxy credentials, private home and cache paths, host/notebook/account identifiers, GPU UUIDs,
sensitive private-file patterns, arbitrary remote URLs in executable example configuration,
repository symlinks, and unreviewed binary/profiler artifacts. Runtime-token environment variables
are modeled as unset without reading; tests do not access their values.

The aggregate reconstruction path applies the same public-safe UTF-8 boundary to every inventoried
shared, repetition, request, comparison, summary, and attestation file. It rejects symlinks in any
component, normalized-path escape, case collisions, missing or extra files, size/hash changes, and
environment, hardware, CUDA, cancellation, or safety raw hashes without retained files. The
aggregate manifest is atomically replaced only after every referenced byte validates, is fsynced,
and is later than each inventoried file. This remains integrity evidence, not authorship or runtime
provenance.
