"""Manifest-last durable lifecycle for Stage 2 evidence bundles."""

from __future__ import annotations

import base64
import binascii
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
)

MAX_STAGE2_FILE_BYTES = 16 * 1024 * 1024
ALLOWED_TEXT_SUFFIXES = frozenset({".json", ".jsonl", ".log", ".prom", ".txt"})
Reconstructor = Callable[[dict[str, bytes]], dict[str, bytes]]

_SENSITIVE_EVIDENCE_PATTERNS = (
    re.compile(re.escape("-----BEGIN " + "PRIVATE KEY-----"), re.IGNORECASE),
    re.compile(r"\b(?:AKIA|ASIA)[0-9A-Z]{16}\b"),
    re.compile(r"\bgh[pousr]_[A-Za-z0-9]{30,}\b"),
    re.compile(r"\bsk-[A-Za-z0-9_-]{20,}\b"),
    re.compile(r"\bhf_[A-Za-z0-9]{20,}\b"),
    re.compile(r"(?im)(?:^|[,{])\s*[\"']?(?:proxy-)?authorization[\"']?\s*:\s*[\"']?\S+"),
    re.compile(r"(?im)(?:^|[,{])\s*[\"']?(?:cookie|set-cookie)[\"']?\s*:\s*[\"']?\S+"),
    re.compile(
        r"(?im)(?:^|[\"'])"
        r"(?:aws_secret_access_key|api_key|client_secret|access_token|"
        r"refresh_token|proxy_password|HF_TOKEN|HUGGING_FACE_HUB_TOKEN|"
        r"HUGGINGFACE_HUB_TOKEN|HUGGINGFACEHUB_API_TOKEN)"
        r"[\"']?\s*(?:=|:)\s*[\"']?\S+"
    ),
    re.compile(r"(?i)\bhttps?://[^\s/:@]+:[^\s/@]+@[^\s/]+"),
    re.compile(r"(?i)(?<![A-Za-z0-9._-])/(?:Users|home)/[A-Za-z0-9._-]+"),
    re.compile(r"(?i)(?:~|/[^\s\"']+)?/\.cache/(?:huggingface|torch|vllm)(?:/|\b)"),
    re.compile(r"(?i)\bGPU-[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}\b"),
    re.compile(r"(?i)\b(?:[A-Za-z0-9-]+\.)+[A-Za-z]{2,63}\b"),
    re.compile(r"(?i)\b(?:account|notebook)[_-]?id\s*(?:=|:)\s*[\"']?\S+"),
    re.compile(
        ("sam" + "sung") + r"[^\n]{0,80}(?:claim|ledger|control[-_ ]?plane)",
        re.IGNORECASE,
    ),
    re.compile(
        "|".join(
            re.escape(value)
            for value in (
                "medical" + "-record",
                "family" + "-record",
                "finance" + "-record",
                "loan" + "-application",
                "resume" + "_private",
                "immi" + "gration",
            )
        ),
        re.IGNORECASE,
    ),
)


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


def decoded_base64_evidence_texts(text: str) -> tuple[str, ...]:
    """Decode canonical JSON `*_base64` evidence fields for safety inspection."""

    documents: tuple[object, ...]
    try:
        documents = (json.loads(text),)
    except json.JSONDecodeError:
        documents = tuple(
            value for line in text.splitlines() if line.strip() for value in _parse_json_line(line)
        )
    decoded: list[str] = []

    def visit(item: object) -> None:
        if isinstance(item, dict):
            for key, child in item.items():
                if isinstance(key, str) and key.endswith("_base64") and isinstance(child, str):
                    try:
                        raw = base64.b64decode(child, validate=True)
                    except (binascii.Error, ValueError):
                        continue
                    if base64.b64encode(raw).decode("ascii") == child:
                        decoded.append(raw.decode("utf-8", errors="replace"))
                visit(child)
        elif isinstance(item, list):
            for child in item:
                visit(child)

    for document in documents:
        visit(document)
    return tuple(decoded)


def _parse_json_line(line: str) -> tuple[object, ...]:
    try:
        return (json.loads(line),)
    except json.JSONDecodeError:
        return ()


def _require_public_safe_evidence(text: str) -> None:
    scan_texts = (text, *decoded_base64_evidence_texts(text))
    if any(
        pattern.search(scan_text)
        for scan_text in scan_texts
        for pattern in _SENSITIVE_EVIDENCE_PATTERNS
    ):
        raise Stage2BundleError("durable Stage 2 evidence contains prohibited private material")


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


def _require_no_symlink_components(path: Path) -> None:
    current = Path(path.anchor) if path.is_absolute() else Path()
    parts = path.parts[1:] if path.is_absolute() else path.parts
    for part in parts:
        current /= part
        if current.is_symlink():
            raise Stage2BundleError("bundle parent path cannot contain symlinks")


class Stage2BundleBuilder:
    """Create one inspectable staging bundle and atomically commit it once."""

    def __init__(
        self,
        parent: Path,
        bundle_name: str,
        *,
        repetition_index: int,
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
        self.source_commit = source_commit
        self._sync_path = sync_path
        self._replace = replace
        self._sequence = 0
        _require_no_symlink_components(parent)
        if (
            self.final_path.exists()
            or self.final_path.is_symlink()
            or self.staging_path.exists()
            or self.staging_path.is_symlink()
        ):
            raise Stage2BundleError("bundle target or staging directory already exists")
        parent.mkdir(parents=True, exist_ok=True)
        _require_no_symlink_components(parent)
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

    def _prepare_evidence_destination(self, path: PurePosixPath) -> Path:
        current = self.staging_path
        for part in path.parts[:-1]:
            current /= part
            if current.is_symlink():
                raise Stage2BundleError("bundle evidence path cannot contain symlinks")
            if current.exists() and not current.is_dir():
                raise Stage2BundleError("bundle evidence parent is not a directory")
            if not current.exists():
                current.mkdir()
                self._sync_path(current.parent)
        destination = self.staging_path.joinpath(*path.parts)
        if destination.is_symlink():
            raise Stage2BundleError("bundle evidence destination cannot be a symlink")
        return destination

    def _write_state(
        self,
        state: Literal[BundleState.INCOMPLETE, BundleState.INVALID],
        *,
        phase: str,
        reason: str | None,
        last: str | None,
    ) -> None:
        self._write_state_at(
            self.staging_path,
            state,
            phase=phase,
            reason=reason,
            last=last,
        )

    def _write_state_at(
        self,
        directory: Path,
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
        self._durable_write(directory / "bundle-state.json", canonical_json_bytes(record) + b"\n")

    def _record_commit_failure(self) -> None:
        directory = self.final_path if self.final_path.is_dir() else self.staging_path
        if not directory.is_dir():
            raise Stage2BundleError("failed commit left no recoverable bundle directory")
        manifest_path = directory / "evidence-manifest.json"
        if manifest_path.exists():
            manifest_path.unlink()
        self._write_state_at(
            directory,
            BundleState.INVALID,
            phase="COMMIT",
            reason="DURABILITY_OPERATION_FAILED",
            last="RECONSTRUCTION_VALIDATED",
        )
        if directory == self.final_path:
            self._sync_path(self.parent)

    def _write_evidence(self, relative: str, data: bytes, *, prefix: str) -> None:
        self._require_incomplete()
        path = _safe_relative(relative)
        if path.parts[0] != prefix:
            raise Stage2BundleError(f"Stage 2 {prefix} evidence must remain beneath {prefix}/")
        if path.suffix not in ALLOWED_TEXT_SUFFIXES:
            raise Stage2BundleError("generated binary evidence is outside the Stage 2A policy")
        try:
            text = data.decode("utf-8")
        except UnicodeDecodeError as error:
            raise Stage2BundleError("Stage 2A durable evidence must be UTF-8 text") from error
        _require_public_safe_evidence(text)
        destination = self._prepare_evidence_destination(path)
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
        try:
            reconstruction_sha256 = self._validate_reconstruction(reconstruct)
            inventory = self._inventory()
            if not inventory or not any(entry.path.startswith("raw/") for entry in inventory):
                raise Stage2BundleError("committed bundles require retained raw evidence")
            if not any(entry.path.startswith("derived/") for entry in inventory):
                raise Stage2BundleError("committed bundles require reconstructed evidence")
        except Exception as error:
            self.invalidate(
                phase="COMMIT_VALIDATION",
                reason="RECONSTRUCTION_OR_INVENTORY_INVALID",
                last_valid_boundary="RAW_EVIDENCE_RETAINED",
            )
            if isinstance(error, Stage2BundleError):
                raise
            raise Stage2BundleError("bundle commit validation failed") from error
        if crash_before_manifest:
            raise Stage2BundleError("simulated crash before evidence manifest")
        manifest = Stage2BundleManifest(
            schema_version="0.3.0",
            measurement_protocol_version="0.3.0",
            state=BundleState.COMMITTED,
            repetition_index=self.repetition_index,
            source_commit=self.source_commit,
            created_at_utc=datetime.now(UTC),
            files=inventory,
            reconstruction_sha256=reconstruction_sha256,
        )
        manifest_bytes = canonical_json_bytes(manifest) + b"\n"
        try:
            self._sync_path(self.staging_path)
            self._replace(self.staging_path, self.final_path)
            self._sync_path(self.parent)
            (self.final_path / "bundle-state.json").unlink()
            self._sync_path(self.final_path)
            self._durable_write(self.final_path / "evidence-manifest.json", manifest_bytes)
        except OSError as error:
            try:
                self._record_commit_failure()
            except (OSError, Stage2BundleError) as recovery_error:
                raise Stage2BundleError(
                    "bundle commit failed and invalid-state durability could not be confirmed"
                ) from recovery_error
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
