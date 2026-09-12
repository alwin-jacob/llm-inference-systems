# Public evidence boundary

This repository provides source, test, generated-schema, artifact, and verification evidence for its Stage 0 measurement contracts, Stage 1 deterministic streaming harness, and Stage 2A experiment protocol.

The current checked execution artifacts exercise the measurement and reconstruction stack through deterministic local fixtures. They validate collection semantics, lifecycle accounting, failure handling, reconstruction, compatibility checks, and experiment structure.

## Current reviewed evidence

| Evidence category                      | Current reviewed state                                                                                                |
| -------------------------------------- | --------------------------------------------------------------------------------------------------------------------- |
| Implemented source                     | Present in the repository                                                                                             |
| Local unit and adversarial tests       | Included in the complete verification gate                                                                            |
| Generated schemas                      | Versioned and synchronized from source models                                                                         |
| Metadata-free Git-archive verification | Repeatable Python 3.13.15 local and CI verification                                                                   |
| Checked Stage 1 streaming execution    | Two independent runs dated `2026-08-27`                                                                               |
| Checked raw artifacts                  | Request, client-stream, and server records retained                                                                   |
| Checked derived summaries              | Reconstruct exactly from retained raw evidence                                                                        |
| Checked semantic comparison            | Compatible and passing                                                                                                |
| Stage 2A protocol source               | Present at package/protocol `0.3.0`                                                                                   |
| Stage 2A fixture coverage              | Request, streaming, cancellation, telemetry, restart, and aggregate paths exercised                                   |
| Complete Stage 2A experiment shape     | 48 measured-request attestations, 16 comparisons, three repetitions, three cancellation paths, and one aggregate root |
| CI configuration                       | Present and executed for repository verification                                                                      |
| Stage 1 public-release CI              | Passed at `40d1ecdc` through run `33171272608`                                                                        |
| Stage 2A source verification           | Passed at `22e3056` with 684 tests and 25 synchronized schemas                                                        |

## Stage 1 checked execution

GitHub Actions run `33171272608` executed Stage 1 public-release SHA:

```text
40d1ecdc26d1b70f20df42de3e1156e13891cc4d
```

The run passed:

* the Python 3.12 source gate;
* the Python 3.13 source gate;
* exact-Python-3.13.15 checked-artifact verification; and
* exact-Python-3.13.15 metadata-free Git-archive verification.

Every job used `uv 0.12.5`.

The retained Stage 1 dataset contains two independent executions. Each run preserves raw request, stream, and server records alongside reconstructed summaries and content identities.

The checked workload uses:

* one excluded warmup;
* eight measured requests;
* five successful requests;
* two non-timeout failures;
* one timeout; and
* requested and observed client concurrency of two.

Both runs reproduce the same semantic fingerprint while retaining independent run identities.

Fixture token markers and timing observations are used to exercise the measurement implementation deterministically. Client concurrency and server batching remain distinct quantities throughout the schema and reconstruction logic.

## Stage 2A protocol verification

The Stage 2A protocol source at:

```text
22e3056dc8e7dbdaaa898ab1b65a358c309529eb
```

passed an exact local Python 3.13.15 / `uv 0.12.5` verification gate containing:

* 684 passing tests;
* 25 synchronized schemas;
* public-safety verification;
* Stage 0 verification;
* Stage 1 artifact verification;
* Stage 2A protocol verification; and
* metadata-free archive verification.

Stage 2A extends the repository from individual streaming runs to complete experiment collection and reconstruction.

Its protocol covers:

* exact request identity;
* SSE parsing and terminal ordering;
* token and usage reconciliation;
* raw request and response evidence;
* lifecycle-derived concurrency;
* cancellation behavior;
* measured-window Prometheus collection;
* process and restart identity;
* launch and snapshot manifests;
* resource attestations;
* repetition manifests;
* cross-restart comparisons;
* metric availability and eligibility;
* aggregate validation; and
* manifest-last reconstruction.

## Experiment structure

The Stage 2A fixture suite constructs the complete three-repetition experiment graph.

Each repetition contains:

* 16 measured-request attestations;
* one cancellation path;
* one measured Prometheus window;
* restart-scoped runtime-control records;
* request lifecycle evidence;
* semantic comparison inputs; and
* a terminal repetition manifest.

Across the complete experiment, the protocol validates:

* 48 measured requests;
* three independently indexed repetitions;
* three cancellation paths;
* three measured telemetry windows;
* 16 cross-restart semantic comparisons;
* request and lifecycle identity consistency;
* metric availability and eligibility;
* aggregate reconstruction; and
* one terminal aggregate root.

Request and component records remain independently validated before they can contribute to the aggregate experiment.

## Request and streaming evidence

Each measured request can retain:

* exact serialized request bytes;
* ordered transmitted headers;
* ordered received headers;
* status and content-type identity;
* ordered raw response chunks;
* completed-frame observation clocks;
* parsed SSE records;
* token and usage evidence;
* request lifecycle boundaries; and
* transport-close state.

Derived request records are reconstructed from the retained raw evidence.

The incremental parser exercises:

* split SSE events;
* coalesced events;
* grouped token delivery;
* comments;
* `[DONE]`;
* explicit generation and usage terminals;
* incomplete trailing bytes; and
* malformed stream termination.

Grouped token delivery is retained without synthesizing per-token timestamps.

## Cancellation evidence

The cancellation path closes after first observed generation delivery and retains the complete close-triggering body read.

This includes:

* all complete nonterminal frames already delivered;
* grouped token IDs;
* incomplete trailing transport bytes;
* parser-pending state;
* intentional-close invocation;
* actual response-close completion;
* server-side abort evidence;
* drain observations;
* stable counter observations; and
* cooldown observations.

The cancellation evaluator reconstructs the full retained record set rather than relying on a selected subset of log events.

## Prometheus evidence

Each repetition includes a measured telemetry window around the declared request population.

The protocol retains:

* baseline scrape;
* final scrape;
* process identity;
* raw exposition bytes;
* parsed samples;
* complete selected label inventory;
* request-window boundaries; and
* reconstructed counter deltas.

Counter arithmetic is evaluated only across compatible observations from the same process.

Raw telemetry and replay-derived records are independently bound into the repetition manifest.

## Reconstruction and integrity

Parsing itself does not assign experiment validity. Validation occurs through the complete request, repetition, and aggregate reconstruction paths.

The repository verifies:

```text
raw evidence
    ↓
protocol parsing
    ↓
typed request evidence
    ↓
request metrics
    ↓
repetition records
    ↓
cross-restart comparison
    ↓
aggregate experiment
```

Repetition bundles use:

```text
INCOMPLETE
    ↓
INVALID  or  COMMITTED
```

The terminal manifest is written only after the retained inventory and reconstructed records satisfy the declared protocol.

The aggregate validator applies the same principle across all three repetitions and their shared experiment records.

## Content identity

Unkeyed SHA-256 values are used for deterministic content identity and modification detection.

They bind retained files and reconstructed records to exact bytes so that verification can detect missing, changed, substituted, or inconsistent evidence.

This integrity layer is combined with versioned schemas, request identity, lifecycle provenance, and reconstruction checks throughout the repository.
