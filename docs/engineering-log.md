# Engineering log

## 2026-08-27 — Stage 0 recovery

- Reconciled an existing zero-commit repository on `main` with no remotes.
- Confirmed CPython `3.13.15` and a current lockfile before modification.
- Identified the prior editable-build failure: `pyproject.toml` declared `README.md` before that
  file existed.
- Audited all first-pass source, scripts, metadata, examples, and the lockfile.
- Replaced the placeholder workload identity and strengthened artifact, token-observation, and
  comparison integrity checks.
- Added documentation, adversarial tests, generated schemas, safety verification, and local CI
  configuration while retaining the synthetic-fixture-only boundary.

This log records local repository work only. It is not evidence that the configured CI workflow
ran remotely.

## 2026-08-27 — Stage 1 deterministic loopback implementation

- Passed the exact Stage 1 precondition gate on `main` at accepted Stage 0 commit
  `77d0ac61b685b3f65edcf43f61899e900eebf5e8`, with one clean commit, no remotes, and no tags.
- Audited every required Stage 0 source, test, verifier, example, and committed v0.1.0 schema.
- Confirmed Stage 0 TPOT used last-token minus first-token span. Preserved that definition and all
  v0.1.0 schema bytes, then implemented v0.2.0 TPOT from successful terminal minus first token and
  retained the earlier concept under the non-TPOT observed-token-span name.
- Added stable HTTPX `0.28.1` as the only new direct dependency and retained every previous direct
  pin. Added only its locked transitives.
- Added the fixed IPv4 loopback HTTP/1.1 SSE fixture, incremental parser, fixture-exact marker
  accounting, bounded concurrency, raw evidence, atomic persistence, exact reconstruction,
  semantic fingerprint, and semantic-only comparison path.
- Added one retained warmup and eight declared measured scenarios with expected semantic counts of
  five successes, two non-timeout failures, and one timeout at client concurrency two.
- Added adversarial unit, real-socket integration, bundle-tampering, reconstruction, comparison,
  security, and CLI-exit tests. Actual checked evidence is intentionally deferred until after the
  separate implementation commit and fresh-archive gate.
- The pre-commit adversarial review found and closed three validation-hardening gaps: terminal
  classes now require matching timeout/cancellation failure kinds, fixture actions must reconstruct
  their declared exact marker totals, and server-event case identity must match its request.

This implementation evidence is local and `TEST_FIXTURE_ONLY`. It executes no model, tokenizer,
LLM-serving runtime, GPU, CUDA, profiler, container, paid resource, remote CI, or Git remote.

## 2026-08-27 — Stage 1 archive-safety recovery

- Preserved Stage 1 implementation commit `927a0a1c57e7c90aef87f5282093a3076e786b73` after the
  mandatory fresh-archive verification exposed that `check_public_safety.py` depended on Git
  metadata, which an intentionally metadata-free Git archive does not contain.
- The defect blocked fresh-checkout acceptance. Corrected it with deterministic Git-worktree and
  archive-safe file discovery without weakening any content checks.
- No Stage 1 evidence was accepted or published after the failed gate, and no canonical
  claim ledger status changed.
