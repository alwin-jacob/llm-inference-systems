# Stage 2 Linux/CUDA execution lock

This directory is separate from the ordinary CPU-only development lock. Nothing here is installed
or executed by Stage 2A. The machine-readable specification pins a future Linux/CUDA environment,
but its status is `BLOCKED_BINARY_RETRIEVAL_AUTHORIZATION_REQUIRED` because the official PyTorch
simple-index metadata and a metadata-only HEAD response do not expose the SHA-256 of the selected
`torchvision` wheel. Stage 2A did not download that 9,290,444-byte binary to calculate it.

The vLLM CUDA 12.9 artifact uses the exact approved GitHub release URL and the
controller-authorized hash. The former `wheels.vllm.ai` location is not accepted as a substitute.
Torch and torchaudio hashes came from official index URL fragments. No resolver lock is represented
as frozen while the one unresolved artifact hash remains.

The future snapshot identity is the exact repository `Qwen/Qwen2.5-0.5B-Instruct` at revision
`7ae557604adf67be50417f59c2c2f167def9a775`, with its revision-specific source URL retained in the
machine-readable lock. Stage 2A did not retrieve the snapshot.

The pre-import version command in the lock uses `importlib.metadata` and does not import vLLM.
