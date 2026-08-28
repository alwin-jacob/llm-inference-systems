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

## 2026-08-27 — Stage 1 checked fixture evidence generation

- Passed the checked-evidence precondition at archive-safety commit
  `66c7f8c6d1c254c89e10c59747d7f957449ba758` on clean `main`, with three commits, no
  remotes, no tags, and unchanged `uv.lock` SHA-256
  `748fd114d05ea6e96c058f41b8a1ee0736d30339f100179e3ee7c47c7e6c59e6`.
- Retained lineage to Stage 0 foundation commit
  `77d0ac61b685b3f65edcf43f61899e900eebf5e8`, streaming implementation commit
  `927a0a1c57e7c90aef87f5282093a3076e786b73`, and archive-safety/execution-source commit
  `66c7f8c6d1c254c89e10c59747d7f957449ba758`.
- Executed two independent fixture runs from that exact source commit. Both run directories
  validated and their stored summaries reconstructed exactly from retained raw evidence.
- Run A (`run-49c34aa1e3758b98fcb05afe`) has content identity
  `e7fc8030d8fc9c9a9821eea271e495980278e521c8131b5645549ebecec638c6`; Run B
  (`run-2fc08a7b3fcc0571cad209cc`) has distinct content identity
  `5506283a01ee5f00af95ca177fc537e01f29463773589d6cec05fc221a642898`.
- Both runs reproduce semantic fingerprint
  `ac33b44eaa11e320bb12ffaac13fbaf381856b132f311ec408e8828064784719`. Their checked
  semantic comparison is compatible and passes with timing performance gates disabled.
- Each run retains one excluded warmup plus eight measured requests: five successes, two
  non-timeout failures, and one timeout. Failure rate is exactly `2 / 8 = 0.25`; timeout rate is
  exactly `1 / 8 = 0.125`.
- Requested client concurrency is `2`, observed maximum client concurrency is `2`, and server
  batching remains unobserved. Client concurrency is not described as server batch size, and this
  single controlled point is not a `1–64` sweep.
- Added the checked `TEST_FIXTURE_ONLY` raw bundles, reconstructed summaries, comparison,
  deterministic evidence manifest, boundary README, and a metadata-independent checked-evidence
  verifier under the authorized evidence scope.

This local checked evidence executes no model, tokenizer, serving runtime, GPU, CUDA, profiler,
container, VM, paid resource, cloud CLI, or credential. It uses no Git remote or remote CI and is
not deployed, published, historically authenticated, or accepted into the canonical claim ledger.

## 2026-08-28 — Independent source-review correction

- Retained the immutable four-commit Stage 0/Stage 1 lineage and all checked fixture evidence after
  an independent GitHub and public-engineering source review required a narrow release-preparation
  correction.
- Replaced unrestricted local-username matching in the public-safety scan with structured checks
  for exact absolute home values, user-qualified Unix and Windows home paths, and bounded specific
  hostnames while retaining the existing private-material and credential patterns.
- Added regression coverage for ordinary occurrences of the simulated identities `root`, `runner`,
  `ubuntu`, and `alwinjacob`, plus private paths, hostnames, and secret-like material.
- Retained the Python 3.12/3.13 source matrix and added exact-Python-3.13.15 checked-evidence and
  metadata-free Git-archive jobs. The archive gate performs a frozen installation, repeats the
  complete source and evidence verification, checks immutable bytes, and removes its temporary
  directory.
- Preserved `uv.lock` and every checked artifact byte-for-byte. No external operation, remote, tag,
  publication, deployment, or private staging occurred.

## 2026-08-28 — Authorized private staging and release preparation

- Created `alwin-jacob/llm-inference-systems` as a private GitHub repository and pushed the exact
  five-commit source-reviewed history through
  `56a06e75256fe4b2ed8acc783f5d8e92eb49a9a7`.
- Private GitHub Actions run `33161428242` completed successfully with exactly four jobs:
  `local-style-gate (3.12)`, `local-style-gate (3.13)`, `checked-stage1-evidence`, and
  `metadata-free-git-archive`.
- The source jobs passed Ruff, formatting, strict mypy, all 198 tests, 11 synchronized schemas,
  public-safety verification, Stage 0 verification, and Stage 1 verification. The exact-evidence
  and archive jobs passed on CPython `3.13.15`, including byte-identical checked-artifact
  verification and temporary-directory cleanup.
- The repository remained private. No public release, tag, GitHub Release, package publication,
  issue, pull request, deployment, profile or pin change, outreach, Stage 2, runtime/model/GPU
  execution, or paid-resource action occurred.
- The source-review environment used `uv 0.12.5`, but the first private run resolved unpinned
  `uv 0.12.7`. The release-preparation descendant now pins and requires exact `uv 0.12.5`; a new
  four-job private CI run is required before any publication decision.

## 2026-08-28 — Stage 1 release-preparation CI and public-release record

- Release-preparation commit `68e64bc814d805464f239c452fa8261fedbfde0b` passed GitHub
  Actions run `33164155869` while the repository remained private.
- Exactly four jobs succeeded: `local-style-gate (3.12)`, `local-style-gate (3.13)`,
  `checked-stage1-evidence`, and `metadata-free-git-archive`. Every job used `uv 0.12.5`.
- The source jobs passed Ruff, formatting, strict mypy, all 198 tests, 11 synchronized schemas,
  zero-finding public-safety verification, Stage 0 verification, and Stage 1 verification. The
  checked-evidence and archive jobs used exact CPython `3.13.15`; the source matrix covered Python
  3.12 and 3.13.
- Checked Stage 1 artifacts remained byte-identical, and the metadata-free archive gate completed
  with temporary-directory cleanup.
- This documentation-only commit is the intended Stage 1 public-release head. Its own identical
  four-job CI gate must succeed before repository visibility changes; workflow state and visibility
  remain external GitHub state that must be independently verified.
- Public source availability does not promote any runtime, model, tokenizer, GPU, CUDA, profiler,
  performance, historical-authentication, or interview-defense claim. No tag, GitHub Release,
  package publication, deployment, profile or pin change, outreach, Stage 2, runtime/model/GPU
  execution, or paid-resource action is part of this documentation commit.
