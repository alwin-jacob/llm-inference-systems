# Runtime environment lock

This directory contains the versioned Linux/CUDA environment specification used by the Stage 2 runtime protocol.

The machine-readable lock records the runtime package set, artifact identities, model snapshot identity, environment requirements, and verification inputs needed to construct a reproducible execution environment.

## Runtime packages

The specification pins the corresponding vLLM CUDA 12.9 artifact together with PyTorch-family package identities.

Torch and torchaudio artifact hashes are sourced from the official PyTorch package index.

The selected torchvision artifact hash remains unresolved because the package-index metadata does not expose the required SHA-256 value. The machine-readable lock records this incomplete state explicitly rather than representing the environment as fully resolved.

## Model snapshot

The runtime snapshot identity is pinned to:

```text
Qwen/Qwen2.5-0.5B-Instruct
revision: 7ae557604adf67be50417f59c2c2f167def9a775
```

The corresponding revision identity and required-file inventory are retained by the snapshot-manifest protocol.

## Environment verification

The execution specification records:

* exact package identities;
* Linux/CUDA environment requirements;
* runtime launch identity;
* model and tokenizer snapshot identity;
* package-version verification;
* resource requirements; and
* artifact-integrity inputs.

Package version inspection uses `importlib.metadata` so environment verification can inspect installed package metadata independently of runtime initialization.
