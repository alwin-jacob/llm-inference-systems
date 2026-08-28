# Source register

| Source | Role | Evidence limit |
| --- | --- | --- |
| Pydantic models in `contracts.py` | Normative Stage 0 data contracts | Synthetic validation only |
| Generated files in `schemas/` | Machine-readable contract projection | Generated from Pydantic; no manual edits |
| `metrics.py` | Pure deterministic metric definitions | Synthetic fixture inputs only |
| `comparison.py` | Compatibility and delta rules | No cross-runtime/hardware comparison without explicit policy |
| Example JSON files | Workload/configuration validation fixtures | No runtime or performance evidence |
| Tests and `verify_stage0.py` | Adversarial and end-to-end local proof | `TEST_FIXTURE_ONLY` |
| `.github/workflows/ci.yml` | CI configuration | Configuration evidence only; no remote execution claim |

The percentile implementation follows the Hyndman-Fan Type 7 formula directly in source. No
external dataset, model output, historical result, or private source is incorporated.
