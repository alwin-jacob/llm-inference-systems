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
