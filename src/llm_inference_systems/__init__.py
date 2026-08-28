"""Stage 0 measurement-contract foundation."""

from llm_inference_systems.contracts import (
    ARTIFACT_SCHEMA_VERSION,
    COMPARISON_CONTRACT_VERSION,
    MEASUREMENT_CONTRACT_VERSION,
)

__version__ = "0.1.0"

__all__ = [
    "ARTIFACT_SCHEMA_VERSION",
    "COMPARISON_CONTRACT_VERSION",
    "MEASUREMENT_CONTRACT_VERSION",
    "__version__",
]
