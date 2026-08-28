"""Shared real-loopback Stage 1 fixture bundles for integration tests."""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from llm_inference_systems.artifact_io import ValidatedBundle
from llm_inference_systems.runner import run_fixture_to_directory
from tests.factories import ROOT


@pytest.fixture(scope="session")
def stage1_bundle_pair(
    tmp_path_factory: pytest.TempPathFactory,
) -> tuple[Path, ValidatedBundle, Path, ValidatedBundle]:
    root = tmp_path_factory.mktemp("stage1-bundles")

    async def execute() -> tuple[ValidatedBundle, ValidatedBundle]:
        workload_path = ROOT / "examples/workloads/streaming-fixture-v1.json"
        configuration_path = ROOT / "examples/configs/stage1-streaming-v1.json"
        fixture_path = ROOT / "examples/fixtures/streaming-fixture-v1.json"
        first = await run_fixture_to_directory(
            workload_path=workload_path,
            configuration_path=configuration_path,
            fixture_path=fixture_path,
            output_directory=root / "run-a",
        )
        second = await run_fixture_to_directory(
            workload_path=workload_path,
            configuration_path=configuration_path,
            fixture_path=fixture_path,
            output_directory=root / "run-b",
        )
        return first, second

    first, second = asyncio.run(execute())
    return root / "run-a", first, root / "run-b", second
