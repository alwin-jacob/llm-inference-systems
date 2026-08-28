# Repository agent instructions

- Keep implementation and claims within the Stage 0 synthetic-fixture boundary.
- Never add a run, serve, benchmark, profile, deploy, model-download, GPU, or network path.
- Preserve strict schema versions, deterministic canonicalization, and failure records.
- Generate schemas from Pydantic models; never hand-edit committed schema files.
- Keep examples synthetic and put synthetic run artifacts only in tests.
- Run every check documented in `README.md` before a commit.
