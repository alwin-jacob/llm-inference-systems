# Stage 2 Linux/CUDA execution lock

This directory is separate from the ordinary CPU-only development lock. Nothing here is installed
or executed by Stage 2A. The machine-readable specification pins a future Linux/CUDA environment,
but its status is `BLOCKED_BINARY_RETRIEVAL_AUTHORIZATION_REQUIRED` because the official PyTorch
simple-index metadata and a metadata-only HEAD response do not expose the SHA-256 of the selected
`torchvision` wheel. Stage 2A did not download that 9,290,444-byte binary to calculate it.

The vLLM CUDA 12.9 artifact URL came from the official commit-specific metadata endpoint. Its hash
is the controller-authorized value. Torch and torchaudio hashes came from official index URL
fragments. No resolver lock is represented as frozen while the one unresolved artifact hash
remains.

The pre-import version command in the lock uses `importlib.metadata` and does not import vLLM.
