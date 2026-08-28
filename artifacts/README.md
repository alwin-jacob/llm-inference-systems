# Artifact directory policy

No run artifacts are committed at Stage 0. Synthetic artifacts are constructed in memory by
tests and verification code so their numbers cannot be mistaken for measured benchmark output.

Any future artifact work requires a separately authorized stage and must preserve versioning,
failure retention, provenance, and evidence-scope rules.
