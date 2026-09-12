# Artifact directory

The repository retains checked Stage 1 execution artifacts under:

```text
artifacts/stage1-fixture/2026-08-27/
```

The checked set contains two independently executed deterministic streaming runs, their semantic comparison, and a versioned evidence manifest.

Each run retains:

* request records;
* client-stream records;
* server records;
* reconstructed summaries;
* content identities; and
* lifecycle and measurement metadata.

Stored summaries reconstruct from the retained raw records.

The checked artifact set can be verified with:

```console
uv run python scripts/verify_checked_stage1_evidence.py artifacts/stage1-fixture/2026-08-27
```

The retained Stage 1 executions use the repository's deterministic fixture scope to exercise streaming measurement, failure handling, lifecycle accounting, persistence, reconstruction, and comparison semantics.

Content hashes bind the checked files to their exact retained bytes and support modification detection during verification.
