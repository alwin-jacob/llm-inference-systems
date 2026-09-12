# Repository agent instructions

* Preserve versioned contracts and schema compatibility. Add incompatible semantics under a new version rather than modifying historical contract bytes.
* Generate committed JSON Schemas from the Pydantic models; do not hand-edit generated schema files.
* Preserve deterministic canonicalization, raw-to-derived reconstruction, artifact integrity checks, and manifest-last persistence semantics.
* Keep client concurrency and server batching as independently evidenced quantities.
* Preserve exact request, stream, lifecycle, failure, cancellation, and telemetry evidence needed for reconstruction.
* Keep fixture-backed and runtime-backed evidence distinguishable through the existing typed contracts and artifact metadata.
* Do not represent protocol specifications, configuration, or planned execution state as observed runtime results.
* Maintain public-safety checks for credentials, private paths, sensitive environment data, and unsafe artifact content.
* Update tests and documentation when changing measurement semantics, schemas, artifact structure, or lifecycle behavior.
* Run the verification sequence documented in `README.md` before committing changes.
