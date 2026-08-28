# Source register

| Source | Role | Evidence limit |
| --- | --- | --- |
| Pydantic models in `contracts.py` | Normative Stage 0 data contracts | Synthetic validation only |
| Generated files in `schemas/` | Machine-readable contract projection | Generated from Pydantic; no manual edits |
| `metrics.py` | Pure deterministic metric definitions | Synthetic fixture inputs only |
| `comparison.py` | Compatibility and delta rules | No cross-runtime/hardware comparison without explicit policy |
| Example JSON files | Workload/configuration validation fixtures | No runtime or performance evidence |
| Tests and `verify_stage0.py` | Adversarial and end-to-end local proof | `TEST_FIXTURE_ONLY` |
| `.github/workflows/ci.yml` | Remote source/evidence verification configuration | Executed at `68e64bc` by run `33164155869`; repository verification only, with no model/runtime/GPU/performance evidence |
| [HTTPX 0.28.1 on PyPI](https://pypi.org/project/httpx/0.28.1/) | Official package metadata for the exact stable direct dependency | Released 2024-12-06; accessed 2026-08-27; package metadata only |
| [HTTPX async support](https://www.python-httpx.org/async/) | Official documentation for `AsyncClient`, one scoped client, `send(..., stream=True)`, `aiter_raw()`, and explicit `Response.aclose()` in manual mode | Client API semantics only; no runtime-performance claim |
| [HTTPX developer interface](https://www.python-httpx.org/api/#asyncclient) | Official parameters for `trust_env`, redirects, HTTP/2, timeouts, limits, and shared async clients | API configuration semantics only |

The percentile implementation follows the Hyndman-Fan Type 7 formula directly in source. No
external dataset, model output, historical result, or private source is incorporated.

HTTPX `0.28.1` is the latest stable line in the official metadata and was released on 2024-12-06.
The same metadata listed a newer `1.0.dev*` prerelease line (through `1.0.dev5` on 2026-08-21) at
the 2026-08-27 access date. Stage 1 deliberately pins stable `0.28.1` and does not select the
prerelease redesign. Its locked transitives are AnyIO `4.14.2`, Certifi `2026.7.22`, h11 `0.16.0`,
httpcore `1.0.9`, and idna `3.19`; these are dependency facts, not technology or performance
claims.
