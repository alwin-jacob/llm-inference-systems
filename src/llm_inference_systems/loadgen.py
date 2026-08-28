"""Bounded deterministic closed-loop load generation for Stage 1."""

from __future__ import annotations

import asyncio

import httpx

from llm_inference_systems.stage1_contracts import (
    FixtureDefinition,
    Stage1RequestRecord,
    Stage1RunConfiguration,
    Stage1WorkloadDefinition,
    StreamEvidenceKind,
)
from llm_inference_systems.streaming import EvidenceCollector, execute_streaming_request


async def execute_closed_loop_workload(
    client: httpx.AsyncClient,
    workload: Stage1WorkloadDefinition,
    configuration: Stage1RunConfiguration,
    fixture: FixtureDefinition,
    collector: EvidenceCollector,
) -> tuple[Stage1RequestRecord, ...]:
    """Retain one warmup, then execute measured cases under an explicit semaphore."""

    fixtures = {case.case_id: case for case in fixture.cases}
    warmup_case = fixtures[workload.warmup_case.case_id]
    warmup_id = "warmup-001"
    await collector.record_stream_event(
        request_id=warmup_id,
        case_id=warmup_case.case_id,
        phase="WARMUP",
        kind=StreamEvidenceKind.CLIENT_REQUEST_STARTED,
    )
    warmup = await execute_streaming_request(
        client,
        warmup_case,
        configuration,
        phase="WARMUP",
        request_id=warmup_id,
        collector=collector,
    )
    await collector.record_stream_event(
        request_id=warmup_id,
        case_id=warmup_case.case_id,
        phase="WARMUP",
        kind=StreamEvidenceKind.CLIENT_REQUEST_ENDED,
    )

    semaphore = asyncio.Semaphore(configuration.load_shape.requested_client_concurrency)

    async def execute_measured(index: int) -> Stage1RequestRecord:
        workload_case = workload.measured_cases[index]
        fixture_case = fixtures[workload_case.case_id]
        request_id = f"measured-{index + 1:03d}"
        async with semaphore:
            await collector.record_stream_event(
                request_id=request_id,
                case_id=fixture_case.case_id,
                phase="MEASURED",
                kind=StreamEvidenceKind.CLIENT_REQUEST_STARTED,
            )
            result = await execute_streaming_request(
                client,
                fixture_case,
                configuration,
                phase="MEASURED",
                request_id=request_id,
                collector=collector,
            )
            await collector.record_stream_event(
                request_id=request_id,
                case_id=fixture_case.case_id,
                phase="MEASURED",
                kind=StreamEvidenceKind.CLIENT_REQUEST_ENDED,
            )
            return result

    tasks = [
        asyncio.create_task(execute_measured(index), name=f"stage1-measured-{index + 1:03d}")
        for index in range(len(workload.measured_cases))
    ]
    measured = await asyncio.gather(*tasks)
    return (warmup, *measured)
