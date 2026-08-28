"""Manifest-last durable lifecycle for Stage 2 evidence bundles."""

from __future__ import annotations

import errno
import hashlib
import json
import os
import re
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath
from typing import Literal, Self

from pydantic import model_validator

from llm_inference_systems.canonical import canonical_json_bytes, sha256_identity
from llm_inference_systems.contracts import Identifier, NonNegativeInt, StrictModel
from llm_inference_systems.stage2_contracts import (
    BundleFileEntry,
    BundleState,
    Stage2BundleManifest,
    Stage2EvidenceBoundary,
)

MAX_STAGE2_FILE_BYTES = 16 * 1024 * 1024
ALLOWED_TEXT_SUFFIXES = frozenset({".json", ".jsonl", ".log", ".prom", ".txt"})
Reconstructor = Callable[[dict[str, bytes]], dict[str, bytes]]


class Stage2BundleError(ValueError):
    """Raised when a Stage 2 bundle is incomplete, unsafe, or irreconstructable."""


class BundleStateRecord(StrictModel):
    state: Literal[BundleState.INCOMPLETE, BundleState.INVALID]
    phase: Identifier
    reason: Identifier | None
    last_valid_boundary: Identifier | None
    sequence: NonNegativeInt

    @model_validator(mode="after")
    def validate_reason(self) -> Self:
        if self.state is BundleState.INVALID and self.reason is None:
            raise ValueError("invalid bundle state requires a reason")
        if self.state is BundleState.INCOMPLETE and self.reason is not None:
            raise ValueError("incomplete bundle state cannot have a terminal reason")
        return self


def _safe_relative(value: str) -> PurePosixPath:
    path = PurePosixPath(value)
    if value != path.as_posix() or path.is_absolute() or not path.parts or ".." in path.parts:
        raise Stage2BundleError("bundle path is unsafe")
    if any(part in {"", "."} for part in path.parts):
        raise Stage2BundleError("bundle path is not normalized")
    return path


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _default_sync_path(path: Path) -> None:
    flags = os.O_RDONLY
    descriptor = os.open(path, flags)
    try:
        os.fsync(descriptor)
    except OSError as error:
        if error.errno not in {errno.EINVAL, errno.ENOTSUP}:
            raise
    finally:
        os.close(descriptor)


def _check_tree(root: Path) -> tuple[Path, ...]:
    if root.is_symlink():
        raise Stage2BundleError("bundle root cannot be a symlink")
    paths = tuple(root.rglob("*"))
    if any(path.is_symlink() for path in paths):
        raise Stage2BundleError("bundle cannot contain symlinks")
    return paths


class Stage2BundleBuilder:
    """Create one inspectable staging bundle and atomically commit it once."""

    def __init__(
        self,
        parent: Path,
        bundle_name: str,
        *,
        repetition_index: int,
        boundary: Stage2EvidenceBoundary,
        source_commit: str,
        sync_path: Callable[[Path], None] = _default_sync_path,
        replace: Callable[[Path, Path], None] = os.replace,
    ) -> None:
        if re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,79}", bundle_name) is None:
            raise Stage2BundleError("bundle name is unsafe")
        self.parent = parent
        self.final_path = parent / bundle_name
        self.staging_path = parent / f".{bundle_name}.staging"
        self.repetition_index = repetition_index
        self.boundary = boundary
        self.source_commit = source_commit
        self._sync_path = sync_path
        self._replace = replace
        self._sequence = 0
        if self.final_path.exists() or self.staging_path.exists():
            raise Stage2BundleError("bundle target or staging directory already exists")
        parent.mkdir(parents=True, exist_ok=True)
        self.staging_path.mkdir()
        self._write_state(BundleState.INCOMPLETE, phase="CREATED", reason=None, last=None)
        self._sync_path(self.staging_path)
        self._sync_path(parent)

    @property
    def state_path(self) -> Path:
        return self.staging_path / "bundle-state.json"

    def _durable_write(self, path: Path, data: bytes) -> None:
        if len(data) > MAX_STAGE2_FILE_BYTES:
            raise Stage2BundleError("bundle file exceeds the Stage 2 size limit")
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
        try:
            with temporary.open("xb") as handle:
                handle.write(data)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, path)  # noqa: PTH105 - durability contract uses os.replace.
            self._sync_path(path.parent)
        finally:
            if temporary.exists():
                temporary.unlink()

    def _write_state(
        self,
        state: Literal[BundleState.INCOMPLETE, BundleState.INVALID],
        *,
        phase: str,
        reason: str | None,
        last: str | None,
    ) -> None:
        self._sequence += 1
        record = BundleStateRecord(
            state=state,
            phase=phase,
            reason=reason,
            last_valid_boundary=last,
            sequence=self._sequence,
        )
        self._durable_write(self.state_path, canonical_json_bytes(record) + b"\n")

    def _write_evidence(self, relative: str, data: bytes, *, prefix: str) -> None:
        self._require_incomplete()
        path = _safe_relative(relative)
        if path.parts[0] != prefix:
            raise Stage2BundleError(f"Stage 2 {prefix} evidence must remain beneath {prefix}/")
        if path.suffix not in ALLOWED_TEXT_SUFFIXES:
            raise Stage2BundleError("generated binary evidence is outside the Stage 2A policy")
        try:
            data.decode("utf-8")
        except UnicodeDecodeError as error:
            raise Stage2BundleError("Stage 2A durable evidence must be UTF-8 text") from error
        destination = self.staging_path.joinpath(*path.parts)
        if destination.exists():
            raise Stage2BundleError("bundle evidence file cannot be replaced")
        self._durable_write(destination, data)
        self._write_state(
            BundleState.INCOMPLETE,
            phase="EVIDENCE_RETAINED",
            reason=None,
            last=relative,
        )

    def write_raw(self, relative: str, data: bytes) -> None:
        self._write_evidence(relative, data, prefix="raw")

    def write_derived(self, relative: str, data: bytes) -> None:
        self._write_evidence(relative, data, prefix="derived")

    def invalidate(self, *, phase: str, reason: str, last_valid_boundary: str | None) -> None:
        if not self.staging_path.is_dir():
            raise Stage2BundleError("staging bundle no longer exists")
        self._require_incomplete()
        self._write_state(
            BundleState.INVALID,
            phase=phase,
            reason=reason,
            last=last_valid_boundary,
        )

    def _require_incomplete(self) -> BundleStateRecord:
        if not self.state_path.is_file():
            raise Stage2BundleError("bundle has no mutable incomplete state")
        state = BundleStateRecord.model_validate_json(self.state_path.read_bytes())
        if state.state is not BundleState.INCOMPLETE:
            raise Stage2BundleError("only an incomplete bundle can change state")
        return state

    def _inventory(self) -> tuple[BundleFileEntry, ...]:
        paths = _check_tree(self.staging_path)
        relative_files = sorted(
            path.relative_to(self.staging_path).as_posix()
            for path in paths
            if path.is_file() and path.name not in {"bundle-state.json", "evidence-manifest.json"}
        )
        return tuple(
            BundleFileEntry(
                path=relative,
                sha256=_sha256((self.staging_path / relative).read_bytes()),
                size=(self.staging_path / relative).stat().st_size,
            )
            for relative in relative_files
        )

    def _validate_reconstruction(self, reconstruct: Reconstructor) -> str:
        raw = {
            path.relative_to(self.staging_path).as_posix(): path.read_bytes()
            for path in _check_tree(self.staging_path)
            if path.is_file() and path.relative_to(self.staging_path).parts[0] == "raw"
        }
        expected_derived = reconstruct(raw)
        normalized: dict[str, bytes] = {}
        for relative, data in expected_derived.items():
            path = _safe_relative(relative)
            if path.parts[0] != "derived":
                raise Stage2BundleError("reconstruction output escaped derived/")
            normalized[path.as_posix()] = data
        actual = {
            path.relative_to(self.staging_path).as_posix(): path.read_bytes()
            for path in _check_tree(self.staging_path)
            if path.is_file() and path.relative_to(self.staging_path).parts[0] == "derived"
        }
        if normalized != actual:
            raise Stage2BundleError("derived evidence differs from exact raw reconstruction")
        return sha256_identity({path: _sha256(data) for path, data in sorted(actual.items())})

    def commit(
        self,
        reconstruct: Reconstructor,
        *,
        crash_before_manifest: bool = False,
    ) -> Stage2BundleManifest:
        self._require_incomplete()
        reconstruction_sha256 = self._validate_reconstruction(reconstruct)
        inventory = self._inventory()
        if not inventory or not any(entry.path.startswith("raw/") for entry in inventory):
            raise Stage2BundleError("committed bundles require retained raw evidence")
        if not any(entry.path.startswith("derived/") for entry in inventory):
            raise Stage2BundleError("committed bundles require reconstructed evidence")
        self.state_path.unlink()
        self._sync_path(self.staging_path)
        if crash_before_manifest:
            raise Stage2BundleError("simulated crash before evidence manifest")
        manifest = Stage2BundleManifest(
            schema_version="0.3.0",
            measurement_protocol_version="0.3.0",
            state=BundleState.COMMITTED,
            boundary=self.boundary,
            repetition_index=self.repetition_index,
            source_commit=self.source_commit,
            created_at_utc=datetime.now(UTC),
            files=inventory,
            reconstruction_sha256=reconstruction_sha256,
        )
        manifest_path = self.staging_path / "evidence-manifest.json"
        try:
            self._durable_write(manifest_path, canonical_json_bytes(manifest) + b"\n")
            self._sync_path(self.staging_path)
            self._replace(self.staging_path, self.final_path)
            self._sync_path(self.parent)
        except OSError as error:
            if self.staging_path.is_dir():
                if manifest_path.exists():
                    manifest_path.unlink()
                self._write_state(
                    BundleState.INVALID,
                    phase="COMMIT",
                    reason="DURABILITY_OPERATION_FAILED",
                    last="RECONSTRUCTION_VALIDATED",
                )
            raise Stage2BundleError("bundle fsync or atomic rename failed") from error
        return manifest


def inspect_bundle_state(directory: Path) -> BundleState:
    if not directory.is_dir() or directory.is_symlink():
        raise Stage2BundleError("bundle directory is missing or unsafe")
    manifest_path = directory / "evidence-manifest.json"
    if manifest_path.is_file():
        manifest = Stage2BundleManifest.model_validate_json(manifest_path.read_bytes())
        return manifest.state
    state_path = directory / "bundle-state.json"
    if not state_path.is_file():
        return BundleState.INCOMPLETE
    return BundleStateRecord.model_validate_json(state_path.read_bytes()).state


def validate_committed_bundle(
    directory: Path,
    reconstruct: Reconstructor,
) -> Stage2BundleManifest:
    if inspect_bundle_state(directory) is not BundleState.COMMITTED:
        raise Stage2BundleError("summaries may consume only committed bundles")
    paths = _check_tree(directory)
    manifest_path = directory / "evidence-manifest.json"
    manifest = Stage2BundleManifest.model_validate_json(manifest_path.read_bytes())
    actual = sorted(
        path.relative_to(directory).as_posix()
        for path in paths
        if path.is_file() and path.name != "evidence-manifest.json"
    )
    listed = [entry.path for entry in manifest.files]
    if actual != listed:
        raise Stage2BundleError("bundle file inventory is incomplete or unexpected")
    manifest_mtime = manifest_path.stat().st_mtime_ns
    for entry in manifest.files:
        path = directory / entry.path
        data = path.read_bytes()
        if len(data) != entry.size or _sha256(data) != entry.sha256:
            raise Stage2BundleError("bundle file size or hash differs")
        if path.stat().st_mtime_ns > manifest_mtime:
            raise Stage2BundleError("evidence manifest was written before an inventoried file")
    raw = {
        path.relative_to(directory).as_posix(): path.read_bytes()
        for path in paths
        if path.is_file() and path.relative_to(directory).parts[0] == "raw"
    }
    expected = reconstruct(raw)
    actual_derived = {
        path.relative_to(directory).as_posix(): path.read_bytes()
        for path in paths
        if path.is_file() and path.relative_to(directory).parts[0] == "derived"
    }
    if expected != actual_derived:
        raise Stage2BundleError("committed derived evidence does not reconstruct exactly")
    reconstruction_sha = sha256_identity(
        {path: _sha256(data) for path, data in sorted(actual_derived.items())}
    )
    if reconstruction_sha != manifest.reconstruction_sha256:
        raise Stage2BundleError("reconstruction identity differs")
    return manifest


def read_committed_summary(
    directory: Path,
    reconstruct: Reconstructor,
    *,
    relative_path: str = "derived/summary.json",
) -> object:
    validate_committed_bundle(directory, reconstruct)
    path = _safe_relative(relative_path)
    if path.parts[0] != "derived":
        raise Stage2BundleError("summary path must remain beneath derived/")
    try:
        return json.loads(directory.joinpath(*path.parts).read_bytes())
    except (OSError, json.JSONDecodeError) as error:
        raise Stage2BundleError("committed summary is unreadable") from error
