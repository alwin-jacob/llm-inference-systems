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

## 2026-08-28 — Controller-authorized local Stage 2A protocol implementation

- Verified a clean public `main` and unchanged `origin/main` at
  `40d1ecdc26d1b70f20df42de3e1156e13891cc4d`, confirmed the required local and remote Stage 2A
  branch was absent, recorded the frozen schema/artifact/example/lock hashes, passed the complete
  198-test public-release gate, and created local branch `stage2a-runtime-protocol` with no upstream.
- Raised only the current package/protocol to `0.3.0`. The Stage 1 checked-evidence verifier now
  compares its manifest with an explicit historical package version `0.2.0`; all historical
  Stage 0/1 schemas, examples, and checked artifacts remain byte-preserved. The public-release
  `uv.lock` is retained as historical evidence under SHA-256
  `748fd114d05ea6e96c058f41b8a1ee0736d30339f100179e3ee7c47c7e6c59e6`, rather than relabeled as
  the Stage 2A lock.
- Added strict future completion-request, request-chain, four-terminal SSE, exact token/usage,
  grouped-token, Prometheus, cancellation/drain, runtime-phase, offline-process, resource-budget,
  bundle lifecycle/reconstruction, three-restart, aggregate, and tiny-sample reporting contracts.
- Added a CPU-only scripted fixture server fixed to IPv4 loopback and an OS-assigned port. Tests
  cover compatible streams, logs, metrics, failure paths, durability simulation, tamper rejection,
  and public-safety boundaries without importing or executing a serving runtime.
- Reviewed official vLLM source text at pinned revision
  `2cf0a6915ce544dc493a0990f2ea38d81601128a` and official package-index metadata read-only. No
  runtime or model package was installed, imported, or executed.
- Created a separate uninstalled Linux/CUDA execution-lock specification. Official metadata did not
  expose the selected torchvision artifact SHA-256; deriving it would require a binary download,
  so its status remains `BLOCKED_BINARY_RETRIEVAL_AUTHORIZATION_REQUIRED`. No resolver lock is
  represented as complete.
- Extended public-safety scanning for credentials, cookies, authorization material, proxy
  credentials, GPU UUIDs, model-cache paths, notebook/account identifiers, employer control-plane
  names, sensitive personal-file names, arbitrary executable remote URLs, symlinks, and unreviewed
  generated binaries.
- Four independent read-only reviews covered source/test correctness, measurement and pinned vLLM
  consistency, security/privacy/claim boundaries, and the separate execution lock. High-confidence
  fixes bind request evidence to its observed log identity chain, validate exact counter arithmetic
  and labels, capture coalesced terminal-frame times, bind aggregate records to three manifest
  identities, reject contradictory cancellation evidence, and harden offline-process and resource
  controls.
- The same review pass made reconstruction failures and post-rename durability failures visibly
  invalid, safety-gated durable raw/derived evidence before write, rejected symlinked parent paths,
  exercised the actual HTTP fixture stream through the protocol validator, and encoded the exact
  artifact and model-repository allowlist in the generated execution-lock schema. No review finding
  required or authorized a real runtime, model, tokenizer, GPU, remote, or public-claim action.

This Stage 2A work is local source and `TEST_FIXTURE_ONLY` evidence only. It does not execute vLLM,
a model, tokenizer, GPU, CUDA, Kaggle, remote compute, or a paid resource; it does not alter public
claims, résumé wording, historical authentication, interview-defense status, remotes, tags, or
published history. Stage 2B remains separately unauthorized.

## 2026-08-29 — Pinned-vLLM cancellation correction

- Corrected the cancellation chronology to match pinned vLLM revision
  `2cf0a6915ce544dc493a0990f2ea38d81601128a`: intentional client close precedes the internal
  `Aborted request(s) ...` record, which precedes the external serving-item `Request ... aborted.`
  record. The earlier external-before-internal fixture order is now explicit pinned-runtime drift.
- Replaced the exactly-one-token/frame-boundary rule with first-generation-delivery semantics.
  Cancellation replay retains every complete nonterminal frame and token ID in the close-triggering
  body read plus exact incomplete trailing bytes, count, SHA-256, and parser state. Coalesced output
  receives one body-read observation clock; no per-token clocks are fabricated.
- Added actual IPv4-loopback HTTPX coverage for single and grouped first events, two coalesced
  frames, complete frames plus incomplete trailing bytes, generation/usage/same-frame-usage/DONE/EOF
  failures, and post-close attribution rejection. The probe is explicitly non-measured and cannot
  advance latency, throughput, ITL, TPOT, or token-rate claims.
- Closed independent-review gaps by retaining the complete bounded raw-log bytes and every record,
  separating exact raw trailing bytes from CRLF-normalized parser-pending bytes, awaiting and
  recording actual HTTP response-close completion, deriving the accepted probe's counters from live
  fixture `/metrics` responses, and scanning decoded canonical Base64 evidence for private or secret
  material.
- Separated each cancellation scrape's exact cadence schedule from its actual HTTP request-dispatch
  and response-completion clocks. Live `/metrics` snapshots use response completion as their
  observation offset; schedule cadence cannot replace or fabricate the observed timing evidence.
- Kept package and protocol `0.3.0`, the execution lock blocked and byte-identical, all frozen
  Stage 0/1 bytes unchanged, and all fixture evidence at
  `SYNTHETIC_PROTOCOL_SHAPE_ONLY / TEST_FIXTURE_ONLY`. No runtime, model, tokenizer, binary, CUDA,
  GPU, Kaggle, remote compute, public Git, or claim-status action occurred.

## 2026-08-28 — Controller-authorized local Stage 2A source correction

- Preserved the three existing Stage 2A commits and every frozen Stage 0/1 byte, then closed the
  nine source-review gaps in one local-only descendant correction.
- Made the five exact per-request metric fields mandatory, including explicit per-field nulls, and
  made cancellation acceptance reconstruct only from retained raw log identity records and exact
  same-process Prometheus snapshots through the complete pre-dispatch/drain/cooldown lifecycle.
- Added strict process-kind pre-import environments, all 17 runtime-control phases, exact memory
  stability/count/quiet/drain/shutdown gates, the immutable launch identity, and the pinned
  ten-file model/tokenizer snapshot-manifest contract.
- Bound distinct phase controls, cancellation/drain evidence, same-process Prometheus windows,
  shutdown evidence, and manifest-byte identities to each of the three non-replaceable restarts;
  raw Prometheus text now reconstructs every retained parsed sample and label inventory.
- Separated fixture attestation from the then-current 15-component future-runtime shape. The later
  experiment-attestation correction supersedes that request-level boundary with the complete
  16 × 3 graph. Complete synthetic objects remain shape-only fixture tests and establish no
  runtime or hardware facts.
- Restored the exact approved GitHub vLLM release URL and retained its authorized hash while the
  inert execution lock remains blocked, uninstalled, unexecuted, resolver-incomplete, and explicit
  about the unresolved torchvision hash. The future-only complete lock shape requires the exact
  four-artifact ordered inventory, exact known hashes and sources, an authorized observed
  torchvision hash, a complete resolver-lock hash, installed-distribution inventory hash, and the
  reviewed blocked-lock byte identity.
- Independent source, security/public-safety, measurement/vLLM, and supply-chain reviews reported
  no P0 findings. Their in-scope P1 findings drove the repetition cross-binding, raw-derived metric
  reconstruction, fixture-marker exclusion, credential scanning, cancellation counter continuity,
  exact future package inventory, and broader ordinary dependency/import checks. Residual P2
  source-observation and evidence-format risks remain documented as future collector concerns.
- The only external read was revision-specific Qwen API tree metadata used to define the allowlist.
  No binary, model, tokenizer content, runtime, GPU, CUDA, Kaggle, remote Git, paid-service, public
  claim, or claim-status action occurred.

## 2026-08-28 — Controller-authorized experiment-attestation correction

- Preserved the four existing local Stage 2A commits and added one integrated experiment boundary
  requiring exactly 48 measured-request attestations across three full restart records.
- Bound ten distinct raw request/body/header/SSE/log/metric/lifecycle/reconciliation files per
  measured request to its committed repetition manifest, plus manifest-bound cancellation stream
  and CUDA raw evidence per repetition.
- Replaced slot-label inference with half-open lifecycle concurrency derivation, exact maximum two,
  positive overlap, phase containment, and global request-ID disjointness.
- Derived per-request, repetition, and experiment metric availability and advancement eligibility;
  explicit null server metrics cannot be filled or overridden by client metrics.
- Integrated the three-restart comparison and aggregate validator into the final model for all 16
  cases. Semantic mismatch is retained in a manifest-last `INVALID` root with reason
  `INVALID_SEMANTIC_NONREPRODUCTION`; it cannot become `COMMITTED`.
- Added an atomic manifest-last aggregate root and pure directory reconstruction covering shared
  environment, NVIDIA isolation, lock, snapshot, launch, safety, workload, three repetition
  manifests, three CUDA attestations, three cancellation results, 16 comparisons, summaries, the
  aggregate validation result, and final attestation.
- Bound request raw bytes back to their exact parsed request, identity-chain, metrics, lifecycle,
  and token/usage fields; bound resolver-lock, installed-distribution, reviewed-lock, snapshot
  transition, environment, NVIDIA, CUDA, cancellation-stream, and safety raw bytes at the same
  reconstructed boundary. Publication now validates the complete typed graph before writing the
  terminal aggregate manifest.
- Kept the complete CPU-generated 16 × 3 graph at `TEST_FIXTURE_ONLY` with classification
  `SYNTHETIC_PROTOCOL_SHAPE_ONLY`. No runtime, model, tokenizer, CUDA, GPU, Kaggle, remote compute,
  spend, claim, public wording, remote Git, or Stage 2B action occurred.

## 2026-08-28 — Controller-authorized wire and measured-window correction

- Required exactly one manifest-bound measured-window Prometheus attestation in each repetition,
  with lossless baseline/final scrape captures, replay-parsed snapshots, same-process identity,
  phase and request boundaries, a separately evidenced drain-completion boundary, one-second scrape
  freshness gates, quiescent gauges, exact selected labels, and exact fixed-workload counter
  reconciliation.
- Replaced parsed-evidence-to-"raw" construction with exact transmitted request bytes, ordered
  lossless request and response header fields, ordered Base64 response chunks, raw server log
  records, public-safe header-name allowlisting, completed-frame observation clocks, and a
  transport-close inventory. The runtime adapter's incremental SSE parser now owns replay; stored
  events, terminals, metrics, lifecycle, token/usage values, and typed evidence must equal its
  output, including split and coalesced terminal reads.
- Added discriminated fixture-versus-collector provenance. Fixture helpers can emit only
  `TEST_FIXTURE_ONLY / SYNTHETIC_PROTOCOL_SHAPE_ONLY`, while a future collector shape requires
  runtime-process, snapshot, and environment identities.
- Extended the aggregate root with three Prometheus attestation files and added standalone wire and
  Prometheus `0.3.0` schemas. The complete 3 × 16 example remains CPU-fixture-only and does not
  establish runtime, model, tokenizer, GPU, CUDA, Kaggle, or performance evidence.
