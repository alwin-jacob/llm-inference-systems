"""Versioned synthetic-fixture measurement and loopback harness contracts."""

from llm_inference_systems.contracts import (
    ARTIFACT_SCHEMA_VERSION,
    COMPARISON_CONTRACT_VERSION,
    MEASUREMENT_CONTRACT_VERSION,
)
from llm_inference_systems.stage1_contracts import STAGE1_MEASUREMENT_CONTRACT_VERSION

__version__ = "0.2.0"

__all__ = [
    "ARTIFACT_SCHEMA_VERSION",
    "COMPARISON_CONTRACT_VERSION",
    "MEASUREMENT_CONTRACT_VERSION",
    "STAGE1_MEASUREMENT_CONTRACT_VERSION",
    "__version__",
]
