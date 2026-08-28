# Security, cost, and provenance

Stage 0 operates on repository-local source and synthetic fixtures. It does not read credentials,
call paid services, invoke cloud tools, download models/tokenizers/datasets/containers, or start a
container. The CLI limits validation input size and emits stable error shapes without echoing
private input values.

Canonical JSON uses sorted keys, compact UTF-8 encoding, finite numeric values, UTC-normalized
timestamps, and deterministic handling of negative zero. SHA-256 is unkeyed and is used only as a
content identity and integrity check. It is never a digital signature and provides no authorship
or origin authentication.

The safety scanner examines repository candidates for private paths, local environment labels,
credential-shaped values, and prohibited private-source references. A passing scan is a scoped
repository-content check, not a general security certification.
