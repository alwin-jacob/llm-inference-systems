"""Verify the CPU-fixture-only Stage 2A source boundary and frozen historical bytes."""

# ruff: noqa: E501 -- frozen SHA-256 inventory stays one auditable path/hash pair per line.

from __future__ import annotations

import ast
import hashlib
from pathlib import Path

from llm_inference_systems import __version__
from llm_inference_systems.canonical import canonical_json
from llm_inference_systems.schema_io import SCHEMA_MODELS, schema_sync_mismatches
from llm_inference_systems.stage2_contracts import (
    ExecutionLockStatus,
    Stage2CompletionRequest,
    Stage2ExecutionLock,
    Stage2RunConfiguration,
)

FROZEN_HASHES = {
    "artifacts/stage1-fixture/2026-08-27/README.md": "b2703214ac2be0be93dea56158229a467d4494d903ed49a4a8bab02329abf4e8",
    "artifacts/stage1-fixture/2026-08-27/comparison.json": "100113403dc2ed7af304c8de5c76754a922637262a39387dd8c60f2675365f16",
    "artifacts/stage1-fixture/2026-08-27/evidence-manifest.json": "fcf34ef410fe64e06bff9fdd2ef588e47a2f44947ce336fc2dd9f5128ee6fcf0",
    "artifacts/stage1-fixture/2026-08-27/run-a/manifest.json": "ed5746021818a05c205866c072be7b9d142d2470b6a19efb9e68d9dfcb65f52d",
    "artifacts/stage1-fixture/2026-08-27/run-a/requests.jsonl": "9300b91c7d43708b156590548235be4ccb29d52fff30bbeeb3404ff54ab354eb",
    "artifacts/stage1-fixture/2026-08-27/run-a/server-events.jsonl": "e8b26ac178c9250ecde1b7e6385904a87bb1ec6ad69b1ada0dfbc561c4bef842",
    "artifacts/stage1-fixture/2026-08-27/run-a/stream-events.jsonl": "ef2dc958d5536bcca8514b318afb838bfb9184899b06e38088b307eb80185499",
    "artifacts/stage1-fixture/2026-08-27/run-a/summary.json": "f44e2b1eab1b3b3324e60209f712690362d7e9ef266451093542297e0a259d63",
    "artifacts/stage1-fixture/2026-08-27/run-b/manifest.json": "513703289fed5402b1dc38992de09544dd95bdab1a1126b798d2536236c758b9",
    "artifacts/stage1-fixture/2026-08-27/run-b/requests.jsonl": "fa6fe1350a9ba7cbf1ac089f7e56c40dd789f0f2e2979a8f6783eecb00983bac",
    "artifacts/stage1-fixture/2026-08-27/run-b/server-events.jsonl": "8051347e0609e358db09a3f128b78ae53d2be5694716bd203e77ada9c160069d",
    "artifacts/stage1-fixture/2026-08-27/run-b/stream-events.jsonl": "17831a6e608fed05b4f17df5b01336e96e83ce6a4f8e2dda067ee2a4e5ab40cc",
    "artifacts/stage1-fixture/2026-08-27/run-b/summary.json": "d435cb1ce3731911ee1ba5a2b5c661f8a8b8265d60b2b70b4cfb5006a9ce8cac",
    "examples/configs/stage0-contract-v1.json": "a33947705bd4f9bfb81557e31047fd7688b27fe0bd1a9f18ed80f652b0d54d4d",
    "examples/configs/stage1-regression-policy-v1.json": "0d9ba900ca4957f12af6b98489c3813e5bec3d572d24151ecd280e30221ff849",
    "examples/configs/stage1-streaming-v1.json": "fc2f677fe2e6af65562ad563c82b92c55cb16f1dff3dae15b8a9627ddbaaf4ba",
    "examples/fixtures/streaming-fixture-v1.json": "de6eac7819a824e2ec268b08f1d8dd22773107acddd12f25dd9271937d02842b",
    "examples/workloads/deterministic-smoke-v1.json": "d0d019db9cecba1d979ba8a80091d1d8675f7a76b0c6f4ecb921ccb8d4907f57",
    "examples/workloads/streaming-fixture-v1.json": "2563e8cbe46d637c5f111de5aa11b362da21d6ecf2cda2e439d37a648225dba9",
    "schemas/comparison-policy-v0.1.0.schema.json": "057f26f26431f6ef4a38d70cae65cd17f62c4ddb9b369546388dd75e3d4079c6",
    "schemas/comparison-policy-v0.2.0.schema.json": "7d5580999774677a32c81f5eda249778376e98723d06411c8b99dedeb826f08f",
    "schemas/comparison-report-v0.1.0.schema.json": "539ae064052bdc0950a3f7fb9a63a3e4dfe6e7ae37adf0768ec7de26344bfcba",
    "schemas/comparison-report-v0.2.0.schema.json": "4e1160e9004cb8938fc630aeb55fa76a1ab9ec32e64c3f4d41fbf9ae5b8ceec9",
    "schemas/execution-manifest-v0.2.0.schema.json": "4b98878c7b58aa05eaecc94dc5c500b3e3b5f1ad18f5745fac3dca1a07b20a60",
    "schemas/fixture-definition-v0.2.0.schema.json": "971f3711577e78bfe186980724eaacb08f9f04ddd01b7cd2b1d462511f830738",
    "schemas/run-artifact-v0.1.0.schema.json": "4cce2174f5b6398c1b67f31c30154fd9201ad943c188a3a61f1bed923c74837b",
    "schemas/run-configuration-v0.1.0.schema.json": "85c28c2507067431dfefe11121f82ce4ad4deada972b2d4b86d2a82f0caccea1",
    "schemas/run-configuration-v0.2.0.schema.json": "f09a57427aec0cf0bb63c2221e54879ce001c73ec24c7d78f1f48e6e229f04c0",
    "schemas/workload-definition-v0.1.0.schema.json": "448ce2a53885b3289eb7505edbdaa75cb61fcd0dd85dc1af1312611b5ab4916a",
    "schemas/workload-definition-v0.2.0.schema.json": "605711239c7cac4b55bb68a86ac1fe065d762bfa38fd890716d7a1b5b99c9e46",
}
HISTORICAL_STAGE1_UV_LOCK_SHA256 = (
    "748fd114d05ea6e96c058f41b8a1ee0736d30339f100179e3ee7c47c7e6c59e6"
)
ORDINARY_UV_LOCK_SHA256 = "96419af34fa7338fcf99916d4db3a25f480d32c6c2ffcdfc7366b5138025036d"
VLLM_WHEEL_SOURCE_URL = (
    "https://github.com/vllm-project/vllm/releases/download/v0.28.0/"
    "vllm-0.28.0%2Bcu129-cp38-abi3-manylinux_2_28_x86_64.whl"
)
VLLM_WHEEL_SHA256 = "8ec943b66a0c6b4351d0778e99d7bacfca5788dd8eedd49425092bacb61c4397"
FORBIDDEN_IMPORT_ROOTS = frozenset(
    {
        "cuda",
        "cupy",
        "flashinfer",
        "huggingface_hub",
        "kaggle",
        "nvidia",
        "torch",
        "torchaudio",
        "torchvision",
        "transformers",
        "vllm",
    }
)
FORBIDDEN_LOCK_NAMES = frozenset(
    {
        "cupy",
        "flashinfer-python",
        "huggingface-hub",
        "kaggle",
        "nvidia-cuda-runtime",
        "torch",
        "torchaudio",
        "torchvision",
        "transformers",
        "vllm",
    }
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _verify_frozen_bytes(root: Path) -> None:
    for relative, expected in FROZEN_HASHES.items():
        path = root / relative
        if not path.is_file() or _sha256(path) != expected:
            raise AssertionError(f"frozen Stage 0/1 byte differs: {relative}")


def _verify_import_boundary(root: Path) -> None:
    candidates = [
        path
        for directory in ("src", "tests", "scripts")
        for path in sorted((root / directory).rglob("*.py"))
    ]
    for path in candidates:
        tree = ast.parse(path.read_bytes(), filename=path.as_posix())
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.Call)
                and (
                    (isinstance(node.func, ast.Name) and node.func.id == "__import__")
                    or (
                        isinstance(node.func, ast.Attribute)
                        and isinstance(node.func.value, ast.Name)
                        and node.func.value.id == "importlib"
                        and node.func.attr == "import_module"
                    )
                )
                and node.args
                and isinstance(node.args[0], ast.Constant)
            ):
                imported = node.args[0].value
                if (
                    isinstance(imported, str)
                    and imported.split(".", 1)[0] in FORBIDDEN_IMPORT_ROOTS
                ):
                    raise AssertionError(
                        f"forbidden dynamic ordinary import in {path.relative_to(root)}"
                    )
            names: tuple[str, ...]
            if isinstance(node, ast.Import):
                names = tuple(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module is not None:
                names = (node.module,)
            else:
                continue
            if any(name.split(".", 1)[0] in FORBIDDEN_IMPORT_ROOTS for name in names):
                raise AssertionError(f"forbidden ordinary import in {path.relative_to(root)}")


def _verify_ordinary_lock(root: Path) -> None:
    if _sha256(root / "uv.lock") != ORDINARY_UV_LOCK_SHA256:
        raise AssertionError("ordinary uv.lock bytes differ from the authorized Stage 2A lock")
    names: set[str] = set()
    for line in (root / "uv.lock").read_text().splitlines():
        if line.startswith("name = "):
            names.add(line.split('"', 2)[1].casefold())
    forbidden = names & FORBIDDEN_LOCK_NAMES
    if forbidden:
        raise AssertionError(f"forbidden ordinary lock dependencies: {sorted(forbidden)}")
    project_text = (root / "pyproject.toml").read_text()
    declared = _declared_forbidden_dependencies(project_text)
    if declared:
        raise AssertionError(f"forbidden ordinary declared dependencies: {sorted(declared)}")


def _declared_forbidden_dependencies(project_text: str) -> set[str]:
    folded = project_text.casefold()
    declared = {
        name for name in FORBIDDEN_LOCK_NAMES if f'"{name}' in folded or f"'{name}" in folded
    }
    return declared


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    if __version__ != "0.3.0":
        raise AssertionError("current package version differs from 0.3.0")
    Stage2RunConfiguration.model_validate_json(
        (root / "examples/configs/stage2a-protocol-fixture-v1.json").read_bytes()
    )
    Stage2CompletionRequest.model_validate_json(
        (root / "examples/fixtures/stage2a-completion-request-v1.json").read_bytes()
    )
    execution_lock = Stage2ExecutionLock.model_validate_json(
        (root / "execution-lock/stage2-execution-lock.json").read_bytes()
    )
    if (
        execution_lock.status
        is not ExecutionLockStatus.BLOCKED_BINARY_RETRIEVAL_AUTHORIZATION_REQUIRED
    ):
        raise AssertionError("execution-lock status differs")
    vllm_artifact = next(
        artifact for artifact in execution_lock.artifacts if artifact.package == "vllm"
    )
    if (
        vllm_artifact.source_url != VLLM_WHEEL_SOURCE_URL
        or vllm_artifact.sha256 != VLLM_WHEEL_SHA256
        or execution_lock.installed
        or execution_lock.executed
        or execution_lock.resolver_lock_claimed_complete
    ):
        raise AssertionError("Stage 2 execution lock differs from the exact blocked contract")
    if schema_sync_mismatches(root / "schemas"):
        raise AssertionError("Stage 2 generated schemas are not synchronized")
    _verify_frozen_bytes(root)
    _verify_import_boundary(root)
    _verify_ordinary_lock(root)
    print(
        canonical_json(
            {
                "execution_lock_status": execution_lock.status,
                "forbidden_ordinary_dependencies": False,
                "forbidden_runtime_imports": False,
                "historical_stage1_uv_lock_sha256": HISTORICAL_STAGE1_UV_LOCK_SHA256,
                "ordinary_uv_lock_sha256": _sha256(root / "uv.lock"),
                "package_version": __version__,
                "preserved_stage0_stage1_file_count": len(FROZEN_HASHES),
                "protocol_version": "0.3.0",
                "resolver_lock_claimed_complete": False,
                "schema_count": len(SCHEMA_MODELS),
                "stage2a_execution_scope": "TEST_FIXTURE_ONLY",
                "status": "verified",
                "vllm_wheel_sha256": vllm_artifact.sha256,
                "vllm_wheel_source_url": vllm_artifact.source_url,
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
