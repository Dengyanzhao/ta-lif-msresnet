#!/usr/bin/env python3
"""Create or verify the independent, post-generation freeze manifest.

This command is deliberately read-only with respect to the protocol, author
sign-off record, and generated configuration matrix.  It writes exactly one
new file in create mode, and only after every validation has succeeded.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import os
import re
import subprocess
import sys
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
SRC_PATH = str(SRC_ROOT)
if SRC_PATH in sys.path:
    sys.path.remove(SRC_PATH)
sys.path.insert(0, SRC_PATH)

from talif_msresnet.config import (  # noqa: E402
    active_conditions_for_protocol,
    artifact_paths_for_protocol,
    canonical_json,
    confirmation_fields_for_protocol,
    generate_run_matrix,
    load_protocol,
    load_run_config,
    validate_run_mapping,
)


SCHEMA = "ta-lif-msresnet-freeze-manifest-v1"
SIGNED_STATUS = "Status: **SIGNED - AUTHOR APPROVALS COMPLETE; PROTOCOL FROZEN**"
V4_SIGNED_STATUS = (
    "Status: **AUTHORIZED - ACCOUNTABLE AUTHOR/USER APPROVAL RECORDED; "
    "PROTOCOL FROZEN**"
)
EXPECTED_SIGNERS = ("Yanzhao Deng", "Peng Yan", "Song Wang")
REQUIRED_SOURCE_PATHS = (
    "pyproject.toml",
    "src/talif_msresnet/config.py",
    "src/talif_msresnet/data.py",
    "src/talif_msresnet/freeze.py",
    "src/talif_msresnet/preflight.py",
    "src/talif_msresnet/train.py",
    "scripts/generate_run_configs.py",
    "scripts/create_freeze_manifest.py",
    "scripts/prepare_cifar10dvs.py",
    "scripts/run_matrix.py",
    "scripts/verify_cifar10dvs.py",
)
V3_REQUIRED_SOURCE_PATHS = (
    "src/talif_msresnet/pilot_v3.py",
    "src/talif_msresnet/statistics_v3.py",
    "scripts/analyze_v3_results.py",
    "scripts/evaluate_checkpoints.py",
    "scripts/pilot_health_gate_v3.py",
    "scripts/validate_v3_pilot.py",
)
V4_REQUIRED_SOURCE_PATHS = (
    "src/talif_msresnet/pilot_v4.py",
    "src/talif_msresnet/statistics_v3.py",
    "src/talif_msresnet/statistics_v4.py",
    "scripts/analyze_v3_results.py",
    "scripts/analyze_v4_results.py",
    "scripts/pilot_health_gate_v4.py",
    "scripts/validate_v4_pilot.py",
)
V5_REQUIRED_SOURCE_PATHS = (
    "src/talif_msresnet/pilot_v3.py",
    "src/talif_msresnet/pilot_v4.py",
    "src/talif_msresnet/pilot_v5.py",
    "src/talif_msresnet/statistics_v3.py",
    "src/talif_msresnet/statistics_v5.py",
    "scripts/analyze_v3_results.py",
    "scripts/analyze_v5_results.py",
    "scripts/calibrate_v5_health.py",
    "scripts/evaluate_checkpoints.py",
    "scripts/pilot_health_gate.py",
    "scripts/pilot_health_gate_v3.py",
    "scripts/pilot_health_gate_v5.py",
    "scripts/validate_v3_pilot.py",
    "scripts/validate_v5_pilot.py",
)
V6_REQUIRED_SOURCE_PATHS = (
    "src/talif_msresnet/benchmark.py",
    "src/talif_msresnet/benchmark_v6.py",
    "src/talif_msresnet/config_v6.py",
    "src/talif_msresnet/models.py",
    "src/talif_msresnet/neurons.py",
    "src/talif_msresnet/pathing.py",
    "src/talif_msresnet/pilot_v3.py",
    "src/talif_msresnet/pilot_v6.py",
    "src/talif_msresnet/utils.py",
    "src/talif_msresnet/v6_statistics.py",
    "scripts/analyze_v6_results.py",
    "scripts/archive_v6_evidence.py",
    "scripts/benchmark_checkpoint.py",
    "scripts/export_v6_validation_batch.py",
    "scripts/evaluate_checkpoints.py",
    "scripts/pilot_health_gate.py",
    "scripts/pilot_health_gate_v3.py",
    "scripts/pilot_health_gate_v6.py",
    "scripts/run_benchmarks.py",
    "scripts/run_v6_benchmarks.py",
    "scripts/validate_v3_pilot.py",
    "scripts/validate_v6_pilot.py",
)
V7_REQUIRED_SOURCE_PATHS = (
    "V7_MECHANISM_5090_RUNBOOK.md",
    "V7_STATISTICAL_ANALYSIS_AUDIT.md",
    "src/talif_msresnet/benchmark.py",
    "src/talif_msresnet/benchmark_v7.py",
    "src/talif_msresnet/config.py",
    "src/talif_msresnet/config_v6.py",
    "src/talif_msresnet/config_v7.py",
    "src/talif_msresnet/data.py",
    "src/talif_msresnet/diagnostics.py",
    "src/talif_msresnet/freeze.py",
    "src/talif_msresnet/models.py",
    "src/talif_msresnet/neurons.py",
    "src/talif_msresnet/ops.py",
    "src/talif_msresnet/pathing.py",
    "src/talif_msresnet/pilot_v3.py",
    "src/talif_msresnet/pilot_v4.py",
    "src/talif_msresnet/pilot_v5.py",
    "src/talif_msresnet/pilot_v6.py",
    "src/talif_msresnet/pilot_v7.py",
    "src/talif_msresnet/preflight.py",
    "src/talif_msresnet/train.py",
    "src/talif_msresnet/utils.py",
    "src/talif_msresnet/v7_statistics.py",
    "scripts/analyze_v7_results.py",
    "scripts/archive_v7_evidence.py",
    "scripts/create_freeze_manifest.py",
    "scripts/create_v7_instance_package.py",
    "scripts/evaluate_checkpoints.py",
    "scripts/export_v7_validation_batch.py",
    "scripts/generate_run_configs.py",
    "scripts/pilot_health_gate.py",
    "scripts/pilot_health_gate_v3.py",
    "scripts/pilot_health_gate_v4.py",
    "scripts/pilot_health_gate_v5.py",
    "scripts/pilot_health_gate_v6.py",
    "scripts/pilot_health_gate_v7.py",
    "scripts/preflight.py",
    "scripts/run_matrix.py",
    "scripts/run_v7_benchmarks.py",
    "scripts/validate_v3_pilot.py",
    "scripts/validate_v4_pilot.py",
    "scripts/validate_v5_pilot.py",
    "scripts/validate_v6_pilot.py",
    "scripts/validate_v7_pilot.py",
    "scripts/validate_v7_source_release.py",
    "scripts/v7_monitor_cn.py",
    "tests/test_v7_archive.py",
    "tests/test_v7_benchmark.py",
    "tests/test_v7_final_test_binding.py",
    "tests/test_v7_freeze_gates.py",
    "tests/test_v7_health_gate.py",
    "tests/test_v7_monitor_cn.py",
    "tests/test_v7_orchestration.py",
    "tests/test_v7_protocol.py",
    "tests/test_v7_provenance.py",
    "tests/test_v7_runtime.py",
    "tests/test_v7_source_release.py",
    "tests/test_v7_statistics.py",
    "tests/test_v7_training_orchestration.py",
    "tests/test_validate_v7_pilot.py",
)
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
COMMIT_RE = re.compile(r"^[0-9a-f]{40}$")
REMOTE_RE = re.compile(r"^(?:https?://|ssh://|git://|git@[^:]+:).+", re.IGNORECASE)
V5_RECOVERY_ORIGINAL_VALIDATION_SHA256 = (
    "5cc85680923b6eee1e454dcf4cf7667f4271dc531d5837a529e57bfcaccd7e86"
)
V5_RECOVERY_PILOT_EXECUTION_COMMIT = "6eeadd6389677347fe46ffa8d3bdec8c75455b44"
V5_RECOVERY_PROTOCOL_HASH = (
    "a5b2a664de5142438061f479a37f3b55401499104abb28ee3cf1d674ec1d4527"
)
V5_RECOVERY_TRAINING_ENVIRONMENT_SHA256 = (
    "f3b837c1554615bb97af91183ec450bf8f4bb43610ad47123aa1f161791d4fb6"
)
V5_RECOVERY_OUTPUT_NAME = "validation_path_recovery.json"
V5_RECOVERY_RELEASE_RECORD = "V5_VALIDATION_RECOVERY_RELEASE.json"
V5_RECOVERY_ALLOWED_CHANGED_PATHS = (
    "V5_TALIF_ONLY_RUNBOOK.md",
    "scripts/create_freeze_manifest.py",
    "scripts/validate_v3_pilot.py",
    "scripts/validate_v5_pilot.py",
    "src/talif_msresnet/freeze.py",
    "tests/test_v5_execution_gates.py",
    "tests/test_v5_release_flow.py",
    "tests/test_validate_v5_pilot.py",
)


class FreezeManifestError(RuntimeError):
    """Raised when an artifact cannot be represented as a valid freeze."""


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _stable_hash(value: Any) -> str:
    return _sha256_bytes(canonical_json(value).encode("utf-8"))


def _normalized_utf8(value: bytes, label: str) -> str:
    try:
        text = value.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise FreezeManifestError(f"{label} must be UTF-8 text") from exc
    return text.replace("\r\n", "\n").replace("\r", "\n")


def _inside(root: Path, value: str | Path, label: str, *, kind: str) -> Path:
    root = root.resolve()
    raw = Path(value)
    path = (raw if raw.is_absolute() else root / raw).resolve()
    try:
        path.relative_to(root)
    except ValueError as exc:
        raise FreezeManifestError(f"{label} must stay inside project root: {path}") from exc
    if kind == "file" and not path.is_file():
        raise FreezeManifestError(f"{label} is missing or is not a file: {path}")
    if kind == "directory" and not path.is_dir():
        raise FreezeManifestError(f"{label} is missing or is not a directory: {path}")
    if kind not in {"file", "directory", "output"}:
        raise AssertionError(f"Unsupported path kind: {kind}")
    return path


def _git(
    project_root: Path,
    arguments: Sequence[str],
    *,
    binary: bool = False,
    allow_failure: bool = False,
) -> bytes | str:
    completed = subprocess.run(
        ["git", "-C", str(project_root), *arguments],
        check=False,
        capture_output=True,
    )
    if completed.returncode != 0 and not allow_failure:
        detail = completed.stderr.decode("utf-8", errors="replace").strip()
        raise FreezeManifestError(
            f"Git command failed (git {' '.join(arguments)}): {detail or 'no details'}"
        )
    if binary:
        return completed.stdout
    return completed.stdout.decode("utf-8", errors="strict").strip()


def _git_repository(project_root: Path) -> Path:
    value = _git(project_root, ("rev-parse", "--show-toplevel"))
    assert isinstance(value, str)
    path = Path(value).resolve()
    try:
        project_root.resolve().relative_to(path)
    except ValueError as exc:
        raise FreezeManifestError("Project root is not inside the detected Git repository") from exc
    return path


def _resolve_commit(project_root: Path, requested: str, *, require_head: bool) -> str:
    requested = requested.strip().lower()
    if not COMMIT_RE.fullmatch(requested):
        raise FreezeManifestError("--freeze-commit must be an explicit 40-character commit ID")
    value = _git(project_root, ("rev-parse", "--verify", f"{requested}^{{commit}}"))
    assert isinstance(value, str)
    commit = value.lower()
    if commit != requested:
        raise FreezeManifestError("--freeze-commit did not resolve to the exact supplied commit")
    if require_head:
        head = _git(project_root, ("rev-parse", "HEAD"))
        assert isinstance(head, str)
        if head.lower() != commit:
            raise FreezeManifestError(
                "Creation requires HEAD to equal --freeze-commit; checkout the Phase A commit"
            )
    return commit


def _path_at_commit(repo_root: Path, commit: str, path: Path) -> bytes:
    try:
        relative = path.resolve().relative_to(repo_root).as_posix()
    except ValueError as exc:
        raise FreezeManifestError(f"Freeze input is outside the Git repository: {path}") from exc
    value = _git(repo_root, ("show", f"{commit}:{relative}"), binary=True)
    assert isinstance(value, bytes)
    return value


def _assert_committed_text(repo_root: Path, commit: str, path: Path, label: str) -> bytes:
    committed = _path_at_commit(repo_root, commit, path)
    current = path.read_bytes()
    if _normalized_utf8(committed, label) != _normalized_utf8(current, label):
        raise FreezeManifestError(f"{label} differs from the Phase A freeze commit: {path}")
    return committed


def _is_under(relative: str, directory: str) -> bool:
    directory = directory.rstrip("/")
    return relative == directory or relative.startswith(f"{directory}/")


def _assert_creation_worktree(
    repo_root: Path,
    project_root: Path,
    matrix_dir: Path,
    output_path: Path,
    *,
    allow_legacy_artifacts: bool = False,
    additional_artifact_dirs: Sequence[Path] = (),
) -> None:
    for arguments, label in (
        (("diff", "--quiet", "--no-ext-diff"), "unstaged tracked changes"),
        (("diff", "--cached", "--quiet", "--no-ext-diff"), "staged changes"),
    ):
        completed = subprocess.run(
            ["git", "-C", str(repo_root), *arguments],
            check=False,
            capture_output=True,
        )
        if completed.returncode == 1:
            raise FreezeManifestError(f"Git worktree has {label}; creation fails closed")
        if completed.returncode != 0:
            detail = completed.stderr.decode("utf-8", errors="replace").strip()
            raise FreezeManifestError(f"Cannot inspect Git {label}: {detail or 'no details'}")

    raw = _git(repo_root, ("ls-files", "--others", "--exclude-standard", "-z"), binary=True)
    assert isinstance(raw, bytes)
    project_relative = project_root.relative_to(repo_root).as_posix()
    project_prefix = "" if project_relative == "." else f"{project_relative}/"
    allowed_directories = {
        matrix_dir.relative_to(repo_root).as_posix(),
        *(
            f"{project_prefix}{name}"
            for name in ("data", "results", "checkpoints", "environment")
        ),
        *(path.resolve().relative_to(repo_root).as_posix() for path in additional_artifact_dirs),
    }
    allowed_file = output_path.relative_to(repo_root).as_posix()
    allowed_legacy_directories = (
        {f"{project_prefix}configs/generated"} if allow_legacy_artifacts else set()
    )
    allowed_legacy_files = (
        {f"{project_prefix}FREEZE_MANIFEST.json"} if allow_legacy_artifacts else set()
    )
    unexpected: list[str] = []
    for encoded in filter(None, raw.split(b"\0")):
        relative = encoded.decode("utf-8", errors="surrogateescape").replace("\\", "/")
        allowed = (
            relative == allowed_file
            or relative in allowed_legacy_files
            or any(
                _is_under(relative, root)
                for root in {*allowed_directories, *allowed_legacy_directories}
            )
        )
        if allowed:
            continue
        unexpected.append(relative)
    if unexpected:
        preview = ", ".join(unexpected[:5])
        suffix = " ..." if len(unexpected) > 5 else ""
        raise FreezeManifestError(
            f"Untracked files outside approved artifact directories: {preview}{suffix}"
        )


def _assert_runtime_source(
    repo_root: Path,
    project_root: Path,
    matrix_dir: Path,
    manifest_path: Path,
    freeze_commit: str,
    *,
    allow_legacy_artifacts: bool = False,
    additional_artifact_dirs: Sequence[Path] = (),
) -> None:
    """Require all executable source to still match the Phase A commit."""

    source_paths = (
        project_root / "pyproject.toml",
        project_root / "src",
        project_root / "scripts",
    )
    relative_sources = [path.relative_to(repo_root).as_posix() for path in source_paths]
    completed = subprocess.run(
        ["git", "-C", str(repo_root), "diff", "--quiet", freeze_commit, "--", *relative_sources],
        check=False,
        capture_output=True,
    )
    if completed.returncode == 1:
        raise FreezeManifestError(
            "Current executable source differs from the Phase A freeze commit"
        )
    if completed.returncode != 0:
        detail = completed.stderr.decode("utf-8", errors="replace").strip()
        raise FreezeManifestError(
            f"Cannot compare runtime source with the Phase A commit: {detail or 'no details'}"
        )

    raw = _git(repo_root, ("ls-files", "--others", "--exclude-standard", "-z"), binary=True)
    assert isinstance(raw, bytes)
    project_relative = project_root.relative_to(repo_root).as_posix()
    project_prefix = "" if project_relative == "." else f"{project_relative}/"
    allowed_directories = {
        matrix_dir.relative_to(repo_root).as_posix(),
        *(f"{project_prefix}{name}" for name in ("data", "results", "checkpoints", "environment")),
        *(path.resolve().relative_to(repo_root).as_posix() for path in additional_artifact_dirs),
    }
    allowed_file = manifest_path.relative_to(repo_root).as_posix()
    allowed_legacy_directories = (
        {f"{project_prefix}configs/generated"} if allow_legacy_artifacts else set()
    )
    allowed_legacy_files = (
        {f"{project_prefix}FREEZE_MANIFEST.json"} if allow_legacy_artifacts else set()
    )
    unexpected: list[str] = []
    for encoded in filter(None, raw.split(b"\0")):
        relative = encoded.decode("utf-8", errors="surrogateescape").replace("\\", "/")
        if "/__pycache__/" in f"/{relative}/" or relative.endswith(".pyc"):
            continue
        if (
            relative == allowed_file
            or relative in allowed_legacy_files
            or any(
                _is_under(relative, root)
                for root in {*allowed_directories, *allowed_legacy_directories}
            )
        ):
            continue
        unexpected.append(relative)
    if unexpected:
        preview = ", ".join(unexpected[:5])
        suffix = " ..." if len(unexpected) > 5 else ""
        raise FreezeManifestError(
            f"Untracked files could affect the frozen run: {preview}{suffix}"
        )


def _parse_aware_timestamp(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise FreezeManifestError(f"{label} must be a non-empty timezone-aware ISO-8601 string")
    normalized = value.strip()
    candidate = f"{normalized[:-1]}+00:00" if normalized.endswith("Z") else normalized
    try:
        parsed = datetime.fromisoformat(candidate)
    except ValueError as exc:
        raise FreezeManifestError(f"{label} must be a valid ISO-8601 timestamp") from exc
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise FreezeManifestError(f"{label} must include a UTC offset")
    return normalized


def _accountable_responsible_author(confirmed_by: Any, *, version: int) -> str:
    if not isinstance(confirmed_by, str) or not confirmed_by.strip():
        raise FreezeManifestError(
            "protocol_status.confirmed_by must identify the accountable author"
        )
    normalized = confirmed_by.strip()
    prohibited = [
        name for name in ("Peng Yan", "Song Wang")
        if name.casefold() in normalized.casefold()
    ]
    if prohibited:
        raise FreezeManifestError(
            f"Protocol v{version} must not claim unrecorded co-author approval in "
            "protocol_status.confirmed_by: " + ", ".join(prohibited)
        )
    responsible = normalized.split("(", 1)[0].strip()
    if not responsible:
        raise FreezeManifestError(
            "protocol_status.confirmed_by has no accountable-author identity"
        )
    return responsible


def _v4_responsible_author(confirmed_by: Any) -> str:
    """Backward-compatible wrapper retained for the v4 release tests."""

    return _accountable_responsible_author(confirmed_by, version=4)


def _validate_protocol_status(protocol: Mapping[str, Any]) -> dict[str, Any]:
    status = protocol.get("protocol_status")
    if not isinstance(status, Mapping):
        raise FreezeManifestError("protocol_status must be a mapping")
    if status.get("frozen") is not True:
        raise FreezeManifestError(
            "protocol_status.frozen is not true; this tool never changes it automatically"
        )
    confirmations = status.get("confirmations")
    if not isinstance(confirmations, Mapping):
        raise FreezeManifestError("protocol_status.confirmations must be a mapping")
    confirmation_fields = confirmation_fields_for_protocol(protocol)
    missing = [
        field for field in confirmation_fields if confirmations.get(field) is not True
    ]
    if missing:
        raise FreezeManifestError(f"Protocol confirmations are incomplete: {', '.join(missing)}")
    confirmed_by = status.get("confirmed_by")
    if not isinstance(confirmed_by, str) or not confirmed_by.strip():
        raise FreezeManifestError("protocol_status.confirmed_by must identify the human reviewers")
    version = int(protocol.get("protocol_version", 1))
    responsible_author: str | None = None
    if version in (4, 5, 6, 7):
        responsible_author = _accountable_responsible_author(
            confirmed_by, version=version
        )
    else:
        absent = [
            name
            for name in EXPECTED_SIGNERS
            if name.casefold() not in confirmed_by.casefold()
        ]
        if absent:
            raise FreezeManifestError(
                "protocol_status.confirmed_by does not name every expected signer: "
                + ", ".join(absent)
            )
    confirmed_at = _parse_aware_timestamp(status.get("confirmed_at"), "confirmed_at")
    validated = {"confirmed_by": confirmed_by.strip(), "confirmed_at": confirmed_at}
    if responsible_author is not None:
        validated["responsible_author"] = responsible_author
    return validated


def _validate_signoff(text: str, protocol: Mapping[str, Any]) -> None:
    version = int(protocol.get("protocol_version", 1))
    expected_status = V4_SIGNED_STATUS if version in (4, 5, 6, 7) else SIGNED_STATUS
    status_lines = re.findall(r"^Status:.*$", text, flags=re.MULTILINE)
    if status_lines != [expected_status]:
        raise FreezeManifestError(
            "Author record is not in the exact signed state documented in "
            "the protocol-bound preregistration sign-off"
        )
    if re.search(r"\[[^\]\r\n]*PENDING[^\]\r\n]*\]", text, flags=re.IGNORECASE):
        raise FreezeManifestError("Author record still contains a bracketed pending placeholder")
    for field in confirmation_fields_for_protocol(protocol):
        pattern = rf"^- \[[xX]\] `{re.escape(field)}`:"
        if re.search(pattern, text, flags=re.MULTILINE) is None:
            raise FreezeManifestError(f"Author checklist item is not checked: {field}")
    if version in (4, 5, 6, 7):
        status = protocol.get("protocol_status", {})
        confirmed_by = status.get("confirmed_by") if isinstance(status, Mapping) else None
        responsible = _accountable_responsible_author(
            confirmed_by, version=version
        )
        authorizer = (
            rf"^- Authorizer: {re.escape(responsible)}, accountable author and "
            rf"Codex task user\s*$"
        )
        if re.search(authorizer, text, flags=re.MULTILINE) is None:
            raise FreezeManifestError(
                f"Protocol v{version} sign-off does not identify the protocol-bound "
                "accountable author"
            )
        for name in ("Peng Yan", "Song Wang"):
            disclosure = rf"^- Independent approval from {re.escape(name)}: not asserted in this record\s*$"
            if re.search(disclosure, text, flags=re.MULTILINE) is None:
                raise FreezeManifestError(
                    f"Protocol v{version} sign-off must explicitly avoid asserting approval from {name}"
                )
            fabricated = (
                rf"^- {re.escape(name)}(?:, corresponding author)? - "
                rf"approval evidence/location:"
            )
            if re.search(fabricated, text, flags=re.MULTILINE | re.IGNORECASE):
                raise FreezeManifestError(
                    f"Protocol v{version} sign-off contains unrecorded approval evidence for {name}"
                )
        return
    for signer in EXPECTED_SIGNERS:
        pattern = (
            rf"^- {re.escape(signer)}(?:, corresponding author)? - "
            rf"approval evidence/location: `?([^`;\r\n]+)`?; date: "
            rf"`?(\d{{4}}-\d{{2}}-\d{{2}})`?\s*$"
        )
        match = re.search(pattern, text, flags=re.MULTILINE)
        if match is None or not match.group(1).strip():
            raise FreezeManifestError(f"Signed approval evidence/date is missing for {signer}")


def _read_json(path: Path, label: str) -> Mapping[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise FreezeManifestError(f"Cannot read {label}: {path}: {exc}") from exc
    if not isinstance(value, Mapping):
        raise FreezeManifestError(f"{label} must contain a JSON object")
    return value


def _validate_v3_pilot_acceptance(
    protocol: Mapping[str, Any],
    protocol_path: Path,
    project_root: Path,
) -> dict[str, Any]:
    """Re-run the read-only v3 pilot audit and bind its immutable PASS report."""

    acceptance = protocol.get("pilot_acceptance")
    if not isinstance(acceptance, Mapping):
        raise FreezeManifestError("Protocol v3 has no pilot_acceptance mapping")
    validation_value = acceptance.get("validation_output")
    if not isinstance(validation_value, str) or not validation_value.strip():
        raise FreezeManifestError("Protocol v3 has no pilot validation output path")
    validation_path = _inside(
        project_root,
        validation_value,
        "V3 pilot validation",
        kind="file",
    )
    stored = dict(_read_json(validation_path, "v3 pilot validation"))
    expected_top = {
        "schema_version": 1,
        "artifact_class": "NON_REPORTABLE_V3_TALIF_ONLY_PILOT_ACCEPTANCE",
        "reporting_eligibility": "FORBIDDEN_FROM_MANUSCRIPT_RESULTS",
        "confirmatory_analysis_eligibility": False,
        "status": "PASS",
        "pass": True,
        "decision": "ACCEPT_V3_TALIF_ONLY_120_EPOCH_PILOT",
        "exit_code": 0,
        "protocol_hash": _stable_hash(protocol),
        "acceptance_hash": _stable_hash(dict(acceptance)),
    }
    mismatches = [
        key for key, expected in expected_top.items() if stored.get(key) != expected
    ]
    datasets = stored.get("datasets")
    if not isinstance(datasets, Mapping) or set(datasets) != {"cifar100", "cifar10dvs"}:
        mismatches.append("datasets")
    elif any(
        not isinstance(datasets[name], Mapping)
        or datasets[name].get("status") != "PASS"
        or datasets[name].get("pass") is not True
        for name in ("cifar100", "cifar10dvs")
    ):
        mismatches.append("dataset PASS states")
    if stored.get("integrity_failures") != [] or stored.get("threshold_failures") != []:
        mismatches.append("failure records")
    if mismatches:
        raise FreezeManifestError(
            "V3 pilot validation is not an exact aggregate PASS: "
            + ", ".join(mismatches)
        )

    validator_path = project_root / "scripts" / "validate_v3_pilot.py"
    spec = importlib.util.spec_from_file_location(
        "talif_freeze_v3_pilot_validator", validator_path
    )
    if spec is None or spec.loader is None:
        raise FreezeManifestError("Cannot load the v3 pilot validator")
    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)
        current = module.validate_pilot(
            protocol_path=protocol_path,
            config_dir=project_root
            / artifact_paths_for_protocol(protocol)["pilot_matrix"],
            repository_root=project_root,
        )
    except Exception as exc:
        raise FreezeManifestError(
            f"Current v3 pilot artifacts fail revalidation: {type(exc).__name__}: {exc}"
        ) from exc
    stored_comparable = dict(stored)
    current_comparable = dict(current)
    stored_comparable.pop("validated_at", None)
    current_comparable.pop("validated_at", None)
    if stored_comparable != current_comparable:
        raise FreezeManifestError(
            "Stored v3 pilot validation differs from the current pilot artifacts"
        )
    return {
        "path": validation_path.relative_to(project_root).as_posix(),
        "sha256": _sha256_file(validation_path),
        "protocol_hash": stored["protocol_hash"],
        "acceptance_hash": stored["acceptance_hash"],
        "validated_at": stored.get("validated_at"),
        "datasets": {
            name: {
                "health_report_sha256": datasets[name].get("health_report_sha256"),
                "attempt_receipt_sha256": datasets[name].get(
                    "attempt_receipt_sha256"
                ),
                "pilot_plan_sha256": datasets[name].get("pilot_plan_sha256"),
                "environment_sha256": datasets[name].get("environment_sha256"),
            }
            for name in ("cifar100", "cifar10dvs")
        },
    }


def _validate_v4_pilot_acceptance(
    protocol: Mapping[str, Any],
    protocol_path: Path,
    project_root: Path,
) -> dict[str, Any]:
    """Re-run the v4 aggregate validator and bind the exact two-dataset PASS."""

    acceptance = protocol.get("pilot_acceptance")
    if not isinstance(acceptance, Mapping):
        raise FreezeManifestError("Protocol v4 has no pilot_acceptance mapping")
    validation_value = acceptance.get("validation_output")
    if not isinstance(validation_value, str) or not validation_value.strip():
        raise FreezeManifestError("Protocol v4 has no pilot validation output path")
    validation_path = _inside(
        project_root,
        validation_value,
        "V4 pilot validation",
        kind="file",
    )
    stored = dict(_read_json(validation_path, "v4 pilot validation"))

    validator_path = project_root / "scripts" / "validate_v4_pilot.py"
    if not validator_path.is_file():
        raise FreezeManifestError(f"V4 pilot validator is missing: {validator_path}")
    spec = importlib.util.spec_from_file_location(
        "talif_freeze_v4_pilot_validator", validator_path
    )
    if spec is None or spec.loader is None:
        raise FreezeManifestError("Cannot load the v4 pilot validator")
    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)
    except Exception as exc:
        raise FreezeManifestError(
            f"Cannot import the v4 pilot validator: {type(exc).__name__}: {exc}"
        ) from exc
    validate_pilot = getattr(module, "validate_pilot", None)
    if not callable(validate_pilot):
        raise FreezeManifestError("V4 pilot validator has no validate_pilot entry point")

    expected_top = {
        "schema_version": 1,
        "artifact_class": "NON_REPORTABLE_V4_TALIF_ONLY_PILOT_ACCEPTANCE",
        "reporting_eligibility": "FORBIDDEN_FROM_MANUSCRIPT_RESULTS",
        "confirmatory_analysis_eligibility": False,
        "status": "PASS",
        "pass": True,
        "decision": "ACCEPT_V4_TALIF_ONLY_120_EPOCH_PILOTS_RELEASE_FORMAL_FREEZE",
        "exit_code": 0,
        "protocol_hash": _stable_hash(protocol),
        "acceptance_hash": _stable_hash(dict(acceptance)),
    }
    mismatches = [
        key for key, expected in expected_top.items() if stored.get(key) != expected
    ]
    datasets = stored.get("datasets")
    dataset_names = ("cifar100", "cifar10dvs")
    if not isinstance(datasets, Mapping) or set(datasets) != set(dataset_names):
        mismatches.append("datasets")
    elif any(
        not isinstance(datasets[name], Mapping)
        or datasets[name].get("status") != "PASS"
        or datasets[name].get("pass") is not True
        for name in dataset_names
    ):
        mismatches.append("dataset PASS states")
    if stored.get("integrity_failures") != [] or stored.get("threshold_failures") != []:
        mismatches.append("failure records")
    if mismatches:
        raise FreezeManifestError(
            "V4 pilot validation is not an exact aggregate PASS: "
            + ", ".join(mismatches)
        )

    try:
        current = validate_pilot(
            protocol_path=protocol_path,
            config_dir=project_root
            / artifact_paths_for_protocol(protocol)["pilot_matrix"],
            repository_root=project_root,
        )
    except Exception as exc:
        raise FreezeManifestError(
            f"Current v4 pilot artifacts fail revalidation: {type(exc).__name__}: {exc}"
        ) from exc
    if not isinstance(current, Mapping):
        raise FreezeManifestError("V4 pilot validator returned a non-mapping result")
    stored_comparable = dict(stored)
    current_comparable = dict(current)
    stored_comparable.pop("validated_at", None)
    current_comparable.pop("validated_at", None)
    if stored_comparable != current_comparable:
        raise FreezeManifestError(
            "Stored v4 pilot validation differs from the current pilot artifacts"
        )

    assert isinstance(datasets, Mapping)
    dataset_evidence: dict[str, Any] = {}
    for name in dataset_names:
        evidence = datasets[name]
        assert isinstance(evidence, Mapping)
        dataset_evidence[name] = {
            "status": evidence.get("status"),
            "pass": evidence.get("pass"),
            **{
                str(key): value
                for key, value in evidence.items()
                if isinstance(key, str) and key.endswith("_sha256")
            },
        }
    return {
        "artifact_class": stored.get("artifact_class"),
        "status": "PASS",
        "pass": True,
        "path": validation_path.relative_to(project_root).as_posix(),
        "sha256": _sha256_file(validation_path),
        "protocol_hash": stored["protocol_hash"],
        "acceptance_hash": stored["acceptance_hash"],
        "validated_at": stored.get("validated_at"),
        "validator": {
            "path": validator_path.relative_to(project_root).as_posix(),
            "sha256": _sha256_file(validator_path),
        },
        "datasets": dataset_evidence,
    }


def _validate_v5_pilot_acceptance(
    protocol: Mapping[str, Any],
    protocol_path: Path,
    project_root: Path,
) -> dict[str, Any]:
    """Re-run the v5 validator and bind its exact two-dataset aggregate PASS."""

    acceptance = protocol.get("pilot_acceptance")
    if not isinstance(acceptance, Mapping):
        raise FreezeManifestError("Protocol v5 has no pilot_acceptance mapping")
    validation_value = acceptance.get("validation_output")
    if not isinstance(validation_value, str) or not validation_value.strip():
        raise FreezeManifestError("Protocol v5 has no pilot validation output path")
    validation_path = _inside(
        project_root,
        validation_value,
        "V5 pilot validation",
        kind="file",
    )
    canonical = dict(_read_json(validation_path, "v5 pilot validation"))

    validator_path = project_root / "scripts" / "validate_v5_pilot.py"
    if not validator_path.is_file():
        raise FreezeManifestError(f"V5 pilot validator is missing: {validator_path}")
    spec = importlib.util.spec_from_file_location(
        "talif_freeze_v5_pilot_validator", validator_path
    )
    if spec is None or spec.loader is None:
        raise FreezeManifestError("Cannot load the v5 pilot validator")
    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)
    except Exception as exc:
        raise FreezeManifestError(
            f"Cannot import the v5 pilot validator: {type(exc).__name__}: {exc}"
        ) from exc
    validate_pilot = getattr(module, "validate_pilot", None)
    if not callable(validate_pilot):
        raise FreezeManifestError("V5 pilot validator has no validate_pilot entry point")
    validate_recovery_release = getattr(module, "_recovery_release_binding", None)

    recovery_path = validation_path.with_name(V5_RECOVERY_OUTPUT_NAME)
    recovery: dict[str, Any] | None = None
    if canonical.get("status") == "INVALID" and recovery_path.is_file():
        recovery = dict(_read_json(recovery_path, "v5 pilot validation recovery"))
        recovery_mismatches = []
        if recovery.get("schema_version") != 1:
            recovery_mismatches.append("schema_version")
        if recovery.get("artifact_class") != "NON_REPORTABLE_V5_PILOT_VALIDATION_RECOVERY":
            recovery_mismatches.append("artifact_class")
        if recovery.get("status") != "PASS" or recovery.get("pass") is not True:
            recovery_mismatches.append("PASS state")
        if recovery.get("exit_code") != 0:
            recovery_mismatches.append("exit_code")
        if recovery.get("reporting_eligibility") != "FORBIDDEN_FROM_MANUSCRIPT_RESULTS":
            recovery_mismatches.append("reporting_eligibility")
        if recovery.get("confirmatory_analysis_eligibility") is not False:
            recovery_mismatches.append("confirmatory_analysis_eligibility")
        if recovery.get("decision") != (
            "RECOVER_V5_PILOT_PASS_AFTER_VALIDATOR_PATH_EQUIVALENCE_FIX"
        ):
            recovery_mismatches.append("decision")
        for key in ("pilot_execution_commit", "recovery_validator_commit"):
            if not isinstance(recovery.get(key), str) or COMMIT_RE.fullmatch(
                str(recovery.get(key))
            ) is None:
                recovery_mismatches.append(key)
        original = recovery.get("original_validation")
        if not isinstance(original, Mapping) or (
            original.get("path") != validation_path.relative_to(project_root).as_posix()
            or original.get("sha256") != _sha256_file(validation_path)
            or original.get("sha256") != V5_RECOVERY_ORIGINAL_VALIDATION_SHA256
            or original.get("status") != "INVALID"
        ):
            recovery_mismatches.append("original validation binding")
        validator = recovery.get("recovery_validator")
        if not isinstance(validator, Mapping) or (
            validator.get("path") != validator_path.relative_to(project_root).as_posix()
            or validator.get("sha256") != _sha256_file(validator_path)
        ):
            recovery_mismatches.append("recovery validator binding")
        expected_output = recovery_path.relative_to(project_root).as_posix()
        if recovery.get("output") != expected_output:
            recovery_mismatches.append("output")
        release_delta = recovery.get("release_delta")
        if not isinstance(release_delta, Mapping):
            recovery_mismatches.append("release_delta")
        else:
            expected_changes = [
                f"M\t{path}" for path in V5_RECOVERY_ALLOWED_CHANGED_PATHS
            ]
            if (
                release_delta.get("base_commit")
                != V5_RECOVERY_PILOT_EXECUTION_COMMIT
                or release_delta.get("recovery_commit")
                != recovery.get("recovery_validator_commit")
                or release_delta.get("allowed_changed_paths")
                != list(V5_RECOVERY_ALLOWED_CHANGED_PATHS)
                or release_delta.get("observed_changes") != expected_changes
                or release_delta.get("release_record")
                != V5_RECOVERY_RELEASE_RECORD
                or not isinstance(release_delta.get("release_record_sha256"), str)
                or SHA256_RE.fullmatch(release_delta["release_record_sha256"]) is None
                or release_delta.get("tracked_clean") is not True
            ):
                recovery_mismatches.append("release_delta")
        if not callable(validate_recovery_release):
            recovery_mismatches.append("recovery release validator")
        else:
            try:
                current_release_delta = validate_recovery_release(project_root)
            except Exception as exc:
                raise FreezeManifestError(
                    "Current v5 recovery release fails its independent seal check: "
                    f"{type(exc).__name__}: {exc}"
                ) from exc
            if not isinstance(release_delta, Mapping) or (
                dict(release_delta) != dict(current_release_delta)
            ):
                recovery_mismatches.append("independent release_delta revalidation")
        recovered = recovery.get("recovered_validation")
        if not isinstance(recovered, Mapping):
            recovery_mismatches.append("recovered validation")
            stored = {}
        else:
            stored = dict(recovered)
        if recovery.get("pilot_execution_commit") != V5_RECOVERY_PILOT_EXECUTION_COMMIT:
            recovery_mismatches.append("pilot_execution_commit")
        if recovery.get("protocol_hash") != V5_RECOVERY_PROTOCOL_HASH:
            recovery_mismatches.append("protocol_hash")
        if recovery.get("acceptance_hash") != canonical.get("acceptance_hash"):
            recovery_mismatches.append("acceptance_hash")
        if (
            isinstance(recovered, Mapping)
            and recovered.get("training_environment_sha256")
            != V5_RECOVERY_TRAINING_ENVIRONMENT_SHA256
        ):
            recovery_mismatches.append("training_environment_sha256")
        if recovery_mismatches:
            raise FreezeManifestError(
                "V5 pilot recovery binding is invalid: "
                + ", ".join(recovery_mismatches)
            )
    else:
        stored = canonical

    expected_top = {
        "schema_version": 1,
        "artifact_class": "NON_REPORTABLE_V5_TALIF_ONLY_PILOT_ACCEPTANCE",
        "reporting_eligibility": "FORBIDDEN_FROM_MANUSCRIPT_RESULTS",
        "confirmatory_analysis_eligibility": False,
        "status": "PASS",
        "pass": True,
        "decision": "ACCEPT_V5_TALIF_ONLY_120_EPOCH_PILOTS_RELEASE_FORMAL_FREEZE",
        "exit_code": 0,
        "protocol_hash": _stable_hash(protocol),
        "acceptance_hash": _stable_hash(dict(acceptance)),
    }
    mismatches = [
        key for key, expected in expected_top.items() if stored.get(key) != expected
    ]
    datasets = stored.get("datasets")
    dataset_names = ("cifar100", "cifar10dvs")
    if not isinstance(datasets, Mapping) or set(datasets) != set(dataset_names):
        mismatches.append("datasets")
    elif any(
        not isinstance(datasets[name], Mapping)
        or datasets[name].get("status") != "PASS"
        or datasets[name].get("pass") is not True
        for name in dataset_names
    ):
        mismatches.append("dataset PASS states")
    if stored.get("integrity_failures") != [] or stored.get("threshold_failures") != []:
        mismatches.append("failure records")
    training_environment_sha256 = stored.get("training_environment_sha256")
    if not isinstance(training_environment_sha256, str) or SHA256_RE.fullmatch(
        training_environment_sha256
    ) is None:
        mismatches.append("training_environment_sha256")
    elif isinstance(datasets, Mapping):
        dataset_environment_hashes = {
            evidence.get("environment_sha256")
            for evidence in datasets.values()
            if isinstance(evidence, Mapping)
        }
        if dataset_environment_hashes != {training_environment_sha256}:
            mismatches.append("dataset training environment bindings")
    if mismatches:
        raise FreezeManifestError(
            "V5 pilot validation is not an exact aggregate PASS: "
            + ", ".join(mismatches)
        )

    try:
        current = validate_pilot(
            protocol_path=protocol_path,
            config_dir=project_root
            / artifact_paths_for_protocol(protocol)["pilot_matrix"],
            repository_root=project_root,
            strict_health_validation=True,
            evidence_git_commit=(
                str(recovery["pilot_execution_commit"])
                if recovery is not None
                else None
            ),
            allow_equivalent_absolute_output_dir=recovery is not None,
        )
    except Exception as exc:
        raise FreezeManifestError(
            f"Current v5 pilot artifacts fail revalidation: {type(exc).__name__}: {exc}"
        ) from exc
    if not isinstance(current, Mapping):
        raise FreezeManifestError("V5 pilot validator returned a non-mapping result")
    stored_comparable = dict(stored)
    current_comparable = dict(current)
    stored_comparable.pop("validated_at", None)
    current_comparable.pop("validated_at", None)
    if stored_comparable != current_comparable:
        raise FreezeManifestError(
            "Stored v5 pilot validation differs from the current pilot artifacts"
        )

    assert isinstance(datasets, Mapping)
    dataset_evidence: dict[str, Any] = {}
    for name in dataset_names:
        evidence = datasets[name]
        assert isinstance(evidence, Mapping)
        dataset_evidence[name] = {
            "status": evidence.get("status"),
            "pass": evidence.get("pass"),
            "health_seed": evidence.get("health_seed"),
            "pilot_seed": evidence.get("pilot_seed"),
            **{
                str(key): value
                for key, value in evidence.items()
                if isinstance(key, str) and key.endswith("_sha256")
            },
        }
    binding = {
        "artifact_class": stored.get("artifact_class"),
        "status": "PASS",
        "pass": True,
        "path": (
            recovery_path if recovery is not None else validation_path
        ).relative_to(project_root).as_posix(),
        "sha256": _sha256_file(recovery_path if recovery is not None else validation_path),
        "protocol_hash": stored["protocol_hash"],
        "acceptance_hash": stored["acceptance_hash"],
        "validated_at": stored.get("validated_at"),
        "training_environment_sha256": training_environment_sha256,
        "validator": {
            "path": validator_path.relative_to(project_root).as_posix(),
            "sha256": _sha256_file(validator_path),
        },
        "datasets": dataset_evidence,
    }
    if recovery is not None:
        binding["recovery"] = {
            "artifact_class": recovery.get("artifact_class"),
            "path": recovery_path.relative_to(project_root).as_posix(),
            "sha256": _sha256_file(recovery_path),
            "pilot_execution_commit": recovery.get("pilot_execution_commit"),
            "recovery_validator_commit": recovery.get("recovery_validator_commit"),
            "original_validation": dict(recovery["original_validation"]),
        }
    return binding


def _validate_v6_pilot_acceptance(
    protocol: Mapping[str, Any],
    protocol_path: Path,
    project_root: Path,
) -> dict[str, Any]:
    """Re-run and bind the exact V6 six-condition CIFAR-100 pilot PASS."""

    acceptance = protocol.get("pilot_acceptance")
    if not isinstance(acceptance, Mapping):
        raise FreezeManifestError("Protocol v6 has no pilot_acceptance mapping")
    pilot_contract = acceptance.get("pilot")
    if not isinstance(pilot_contract, Mapping):
        raise FreezeManifestError("Protocol v6 has no pilot acceptance contract")
    validation_value = acceptance.get("validation_output")
    if not isinstance(validation_value, str) or not validation_value.strip():
        raise FreezeManifestError("Protocol v6 has no pilot validation output path")
    validation_path = _inside(
        project_root,
        validation_value,
        "V6 pilot validation",
        kind="file",
    )
    stored = dict(_read_json(validation_path, "v6 pilot validation"))

    validator_path = project_root / "scripts" / "validate_v6_pilot.py"
    if not validator_path.is_file():
        raise FreezeManifestError(f"V6 pilot validator is missing: {validator_path}")
    spec = importlib.util.spec_from_file_location(
        "talif_freeze_v6_pilot_validator", validator_path
    )
    if spec is None or spec.loader is None:
        raise FreezeManifestError("Cannot load the v6 pilot validator")
    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)
    except Exception as exc:
        raise FreezeManifestError(
            f"Cannot import the v6 pilot validator: {type(exc).__name__}: {exc}"
        ) from exc
    validate_pilot = getattr(module, "validate_pilot", None)
    if not callable(validate_pilot):
        raise FreezeManifestError("V6 pilot validator has no validate_pilot entry point")

    conditions = tuple(active_conditions_for_protocol(protocol))
    dataset_name = acceptance.get("dataset")
    health_seed = acceptance.get("health_seed")
    pilot_seed = acceptance.get("pilot_seed")
    run_handling = acceptance.get("run_handling")
    artifacts = artifact_paths_for_protocol(protocol)
    expected_top = {
        "schema_version": 1,
        "protocol_version": 6,
        "artifact_class": "NON_REPORTABLE_V6_MECHANISM_PILOT_ACCEPTANCE",
        "reporting_eligibility": "FORBIDDEN_FROM_MANUSCRIPT_RESULTS",
        "confirmatory_analysis_eligibility": False,
        "status": "PASS",
        "pass": True,
        "decision": "ACCEPT_V6_SIX_CONDITION_120_EPOCH_PILOT_RELEASE_FORMAL_FREEZE",
        "exit_code": 0,
        "protocol_hash": _stable_hash(protocol),
        "acceptance_hash": _stable_hash(dict(acceptance)),
        "required_epochs": pilot_contract.get("epochs"),
        "config_dir": artifacts.get("pilot_matrix"),
        "failure_action": (
            run_handling.get("failure_action")
            if isinstance(run_handling, Mapping)
            else None
        ),
    }
    mismatches = [
        key for key, expected in expected_top.items() if stored.get(key) != expected
    ]
    if dataset_name != "cifar100":
        mismatches.append("pilot_acceptance.dataset")
    if tuple(acceptance.get("conditions", ())) != conditions or len(conditions) != 6:
        mismatches.append("pilot_acceptance.conditions")
    if not isinstance(health_seed, int) or not isinstance(pilot_seed, int):
        mismatches.append("pilot seed bindings")
    for key in ("matrix_manifest_sha256", "run_manifest_csv_sha256"):
        if SHA256_RE.fullmatch(str(stored.get(key, ""))) is None:
            mismatches.append(key)

    datasets = stored.get("datasets")
    evidence: Mapping[str, Any] | None = None
    if not isinstance(datasets, Mapping) or set(datasets) != {"cifar100"}:
        mismatches.append("datasets")
    else:
        candidate = datasets.get("cifar100")
        if isinstance(candidate, Mapping):
            evidence = candidate
        else:
            mismatches.append("cifar100 dataset evidence")
    environment_sha256 = stored.get("training_environment_sha256")
    if SHA256_RE.fullmatch(str(environment_sha256 or "")) is None:
        mismatches.append("training_environment_sha256")

    if evidence is not None:
        expected_dataset = {
            "status": "PASS",
            "pass": True,
            "seed": pilot_seed,
            "health_seed": health_seed,
            "pilot_seed": pilot_seed,
            "results_root": acceptance.get("pilot_output_root"),
            "health_report_path": acceptance.get("health_output"),
            "attempt_receipt_path": acceptance.get("attempt_receipt"),
            "pilot_plan_path": acceptance.get("pilot_plan"),
            "environment_sha256": environment_sha256,
        }
        mismatches.extend(
            f"cifar100.{key}"
            for key, expected in expected_dataset.items()
            if evidence.get(key) != expected
        )
        for key in (
            "block_hash",
            "health_report_sha256",
            "attempt_receipt_sha256",
            "pilot_plan_sha256",
            "shared_weight_sha256",
            "split_manifest_sha256",
        ):
            if SHA256_RE.fullmatch(str(evidence.get(key, ""))) is None:
                mismatches.append(f"cifar100.{key}")
        runs = evidence.get("runs")
        if not isinstance(runs, Mapping) or tuple(runs) != conditions:
            mismatches.append("cifar100.runs")
        else:
            for condition in conditions:
                run = runs.get(condition)
                if (
                    not isinstance(run, Mapping)
                    or run.get("condition") != condition
                    or run.get("training_environment_sha256") != environment_sha256
                    or run.get("integrity_failures") != []
                    or run.get("threshold_failures") != []
                ):
                    mismatches.append(f"cifar100.runs.{condition}")
    if stored.get("integrity_failures") != [] or stored.get("threshold_failures") != []:
        mismatches.append("failure records")
    if mismatches:
        raise FreezeManifestError(
            "V6 pilot validation is not an exact six-condition PASS: "
            + ", ".join(dict.fromkeys(mismatches))
        )

    try:
        current = validate_pilot(
            protocol_path=protocol_path,
            config_dir=project_root / artifacts["pilot_matrix"],
            repository_root=project_root,
        )
    except Exception as exc:
        raise FreezeManifestError(
            f"Current v6 pilot artifacts fail revalidation: {type(exc).__name__}: {exc}"
        ) from exc
    if not isinstance(current, Mapping):
        raise FreezeManifestError("V6 pilot validator returned a non-mapping result")
    stored_comparable = dict(stored)
    current_comparable = dict(current)
    stored_comparable.pop("validated_at", None)
    current_comparable.pop("validated_at", None)
    if stored_comparable != current_comparable:
        raise FreezeManifestError(
            "Stored v6 pilot validation differs from the current pilot artifacts"
        )

    assert evidence is not None
    return {
        "artifact_class": stored["artifact_class"],
        "status": "PASS",
        "pass": True,
        "path": validation_path.relative_to(project_root).as_posix(),
        "sha256": _sha256_file(validation_path),
        "protocol_hash": stored["protocol_hash"],
        "acceptance_hash": stored["acceptance_hash"],
        "validated_at": stored.get("validated_at"),
        "training_environment_sha256": environment_sha256,
        "conditions": list(conditions),
        "validator": {
            "path": validator_path.relative_to(project_root).as_posix(),
            "sha256": _sha256_file(validator_path),
        },
        "datasets": {
            "cifar100": {
                "status": evidence["status"],
                "pass": evidence["pass"],
                "health_seed": evidence["health_seed"],
                "pilot_seed": evidence["pilot_seed"],
                "block_hash": evidence["block_hash"],
                "environment_sha256": evidence["environment_sha256"],
                "shared_weight_sha256": evidence["shared_weight_sha256"],
                "split_manifest_sha256": evidence["split_manifest_sha256"],
                "health_report_sha256": evidence["health_report_sha256"],
                "attempt_receipt_sha256": evidence["attempt_receipt_sha256"],
                "pilot_plan_sha256": evidence["pilot_plan_sha256"],
            }
        },
    }


def _validate_v7_pilot_acceptance(
    protocol: Mapping[str, Any],
    protocol_path: Path,
    project_root: Path,
) -> dict[str, Any]:
    """Re-run and bind the exact V7 six-condition CIFAR-100 pilot PASS."""

    acceptance = protocol.get("pilot_acceptance")
    if not isinstance(acceptance, Mapping):
        raise FreezeManifestError("Protocol v7 has no pilot_acceptance mapping")
    pilot_contract = acceptance.get("pilot")
    if not isinstance(pilot_contract, Mapping):
        raise FreezeManifestError("Protocol v7 has no pilot acceptance contract")
    validation_value = acceptance.get("validation_output")
    if not isinstance(validation_value, str) or not validation_value.strip():
        raise FreezeManifestError("Protocol v7 has no pilot validation output path")
    validation_path = _inside(
        project_root,
        validation_value,
        "V7 pilot validation",
        kind="file",
    )
    stored = dict(_read_json(validation_path, "v7 pilot validation"))

    validator_path = project_root / "scripts" / "validate_v7_pilot.py"
    if not validator_path.is_file():
        raise FreezeManifestError(f"V7 pilot validator is missing: {validator_path}")
    spec = importlib.util.spec_from_file_location(
        "talif_freeze_v7_pilot_validator", validator_path
    )
    if spec is None or spec.loader is None:
        raise FreezeManifestError("Cannot load the v7 pilot validator")
    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)
    except Exception as exc:
        raise FreezeManifestError(
            f"Cannot import the v7 pilot validator: {type(exc).__name__}: {exc}"
        ) from exc
    validate_pilot = getattr(module, "validate_pilot", None)
    if not callable(validate_pilot):
        raise FreezeManifestError("V7 pilot validator has no validate_pilot entry point")

    conditions = tuple(active_conditions_for_protocol(protocol))
    dataset_name = acceptance.get("dataset")
    health_seed = acceptance.get("health_seed")
    pilot_seed = acceptance.get("pilot_seed")
    run_handling = acceptance.get("run_handling")
    artifacts = artifact_paths_for_protocol(protocol)
    expected_top = {
        "schema_version": 1,
        "protocol_version": 7,
        "artifact_class": "NON_REPORTABLE_V7_MECHANISM_PILOT_ACCEPTANCE",
        "reporting_eligibility": "FORBIDDEN_FROM_MANUSCRIPT_RESULTS",
        "confirmatory_analysis_eligibility": False,
        "status": "PASS",
        "pass": True,
        "decision": "ACCEPT_V7_SIX_CONDITION_120_EPOCH_PILOT_RELEASE_FORMAL_FREEZE",
        "exit_code": 0,
        "protocol_path": protocol_path.relative_to(project_root).as_posix(),
        "protocol_file_sha256": _sha256_file(protocol_path),
        "protocol_hash": _stable_hash(protocol),
        "acceptance_hash": _stable_hash(dict(acceptance)),
        "required_epochs": pilot_contract.get("epochs"),
        "config_dir": artifacts.get("pilot_matrix"),
        "failure_action": (
            run_handling.get("failure_action")
            if isinstance(run_handling, Mapping)
            else None
        ),
    }
    mismatches = [
        key for key, expected in expected_top.items() if stored.get(key) != expected
    ]
    if dataset_name != "cifar100":
        mismatches.append("pilot_acceptance.dataset")
    if tuple(acceptance.get("conditions", ())) != conditions or conditions != (
        "M0",
        "M1",
        "M2",
        "M3",
        "M4",
        "PLIF",
    ):
        mismatches.append("pilot_acceptance.conditions")
    if not isinstance(health_seed, int) or not isinstance(pilot_seed, int):
        mismatches.append("pilot seed bindings")
    for key in ("matrix_manifest_sha256", "run_manifest_csv_sha256"):
        if SHA256_RE.fullmatch(str(stored.get(key, ""))) is None:
            mismatches.append(key)

    datasets = stored.get("datasets")
    evidence: Mapping[str, Any] | None = None
    if not isinstance(datasets, Mapping) or set(datasets) != {"cifar100"}:
        mismatches.append("datasets")
    else:
        candidate = datasets.get("cifar100")
        if isinstance(candidate, Mapping):
            evidence = candidate
        else:
            mismatches.append("cifar100 dataset evidence")
    environment_sha256 = stored.get("training_environment_sha256")
    if SHA256_RE.fullmatch(str(environment_sha256 or "")) is None:
        mismatches.append("training_environment_sha256")

    if evidence is not None:
        expected_dataset = {
            "status": "PASS",
            "pass": True,
            "seed": pilot_seed,
            "health_seed": health_seed,
            "pilot_seed": pilot_seed,
            "results_root": acceptance.get("pilot_output_root"),
            "health_report_path": acceptance.get("health_output"),
            "attempt_receipt_path": acceptance.get("attempt_receipt"),
            "pilot_plan_path": acceptance.get("pilot_plan"),
            "environment_sha256": environment_sha256,
        }
        mismatches.extend(
            f"cifar100.{key}"
            for key, expected in expected_dataset.items()
            if evidence.get(key) != expected
        )
        for key in (
            "block_hash",
            "health_report_sha256",
            "attempt_receipt_sha256",
            "pilot_plan_sha256",
            "shared_weight_sha256",
            "split_manifest_sha256",
        ):
            if SHA256_RE.fullmatch(str(evidence.get(key, ""))) is None:
                mismatches.append(f"cifar100.{key}")
        runs = evidence.get("runs")
        if not isinstance(runs, Mapping) or tuple(runs) != conditions:
            mismatches.append("cifar100.runs")
        else:
            for condition in conditions:
                run = runs.get(condition)
                expected_run_id = (
                    f"E9_cifar100_d20_t6_{condition}_s{pilot_seed}"
                )
                if (
                    not isinstance(run, Mapping)
                    or run.get("run_id") != expected_run_id
                    or run.get("condition") != condition
                    or run.get("training_environment_sha256") != environment_sha256
                    or run.get("integrity_failures") != []
                    or run.get("threshold_failures") != []
                ):
                    mismatches.append(f"cifar100.runs.{condition}")
    if stored.get("integrity_failures") != [] or stored.get("threshold_failures") != []:
        mismatches.append("failure records")
    if mismatches:
        raise FreezeManifestError(
            "V7 pilot validation is not an exact six-condition PASS: "
            + ", ".join(dict.fromkeys(mismatches))
        )

    try:
        current = validate_pilot(
            protocol_path=protocol_path,
            config_dir=project_root / artifacts["pilot_matrix"],
            repository_root=project_root,
        )
    except Exception as exc:
        raise FreezeManifestError(
            f"Current v7 pilot artifacts fail revalidation: {type(exc).__name__}: {exc}"
        ) from exc
    if not isinstance(current, Mapping):
        raise FreezeManifestError("V7 pilot validator returned a non-mapping result")
    stored_comparable = dict(stored)
    current_comparable = dict(current)
    stored_comparable.pop("validated_at", None)
    current_comparable.pop("validated_at", None)
    if stored_comparable != current_comparable:
        raise FreezeManifestError(
            "Stored v7 pilot validation differs from the current pilot artifacts"
        )

    assert evidence is not None
    return {
        "artifact_class": stored["artifact_class"],
        "status": "PASS",
        "pass": True,
        "path": validation_path.relative_to(project_root).as_posix(),
        "sha256": _sha256_file(validation_path),
        "protocol_file_sha256": stored["protocol_file_sha256"],
        "protocol_hash": stored["protocol_hash"],
        "acceptance_hash": stored["acceptance_hash"],
        "validated_at": stored.get("validated_at"),
        "training_environment_sha256": environment_sha256,
        "conditions": list(conditions),
        "pilot_matrix_manifest_sha256": stored["matrix_manifest_sha256"],
        "pilot_run_manifest_csv_sha256": stored["run_manifest_csv_sha256"],
        "validator": {
            "path": validator_path.relative_to(project_root).as_posix(),
            "sha256": _sha256_file(validator_path),
        },
        "datasets": {
            "cifar100": {
                "status": evidence["status"],
                "pass": evidence["pass"],
                "health_seed": evidence["health_seed"],
                "pilot_seed": evidence["pilot_seed"],
                "block_hash": evidence["block_hash"],
                "environment_sha256": evidence["environment_sha256"],
                "shared_weight_sha256": evidence["shared_weight_sha256"],
                "split_manifest_sha256": evidence["split_manifest_sha256"],
                "health_report_sha256": evidence["health_report_sha256"],
                "attempt_receipt_sha256": evidence["attempt_receipt_sha256"],
                "pilot_plan_sha256": evidence["pilot_plan_sha256"],
            }
        },
    }


def _manifest_row(resolved: Any, config_path: Path) -> dict[str, Any]:
    return {
        "run_id": resolved.runtime.run_id,
        "experiment": resolved.experiment,
        "dataset": resolved.data.dataset,
        "depth": resolved.model.depth,
        "time_steps": resolved.model.time_steps,
        "condition": resolved.model.condition,
        "topology": resolved.model.topology,
        "neuron": resolved.model.neuron,
        "seed": resolved.runtime.seed,
        "config_file": config_path.name,
        "config_hash": resolved.config_hash,
        "config_file_sha256": _sha256_file(config_path),
        "matrix_key": json.dumps(resolved.analysis["matrix_key"], separators=(",", ":")),
        "protocol_hash": resolved.analysis["protocol_hash"],
    }


def _csv_projection(row: Mapping[str, Any]) -> dict[str, str]:
    return {key: str(value) for key, value in row.items()}


def _validate_matrix(
    protocol: Mapping[str, Any],
    protocol_path: Path,
    matrix_dir: Path,
    project_root: Path,
) -> dict[str, Any]:
    matrix_manifest_path = matrix_dir / "matrix_manifest.json"
    run_manifest_path = matrix_dir / "run_manifest.csv"
    if not matrix_manifest_path.is_file() or not run_manifest_path.is_file():
        raise FreezeManifestError(
            "Generated matrix must contain matrix_manifest.json and run_manifest.csv"
        )
    matrix_manifest = _read_json(matrix_manifest_path, "matrix manifest")
    expected_runs = generate_run_matrix(protocol)
    protocol_hash = _stable_hash(protocol)
    matrix_hash = _stable_hash(expected_runs)
    expected_rows: list[dict[str, Any]] = []
    expected_names = {"matrix_manifest.json", "run_manifest.csv"}
    for raw in expected_runs:
        resolved = validate_run_mapping(raw, protocol)
        config_path = matrix_dir / f"{resolved.runtime.run_id}.yaml"
        if not config_path.is_file():
            raise FreezeManifestError(f"Generated config is missing: {config_path.name}")
        actual = load_run_config(config_path, protocol_path)
        if actual.as_dict() != resolved.as_dict():
            raise FreezeManifestError(
                f"Generated config differs from the fixed matrix: {config_path.name}"
            )
        expected_rows.append(_manifest_row(resolved, config_path))
        expected_names.add(config_path.name)

    actual_names = {path.name for path in matrix_dir.iterdir() if path.is_file()}
    nested = [path for path in matrix_dir.iterdir() if path.is_dir()]
    if actual_names != expected_names or nested:
        extra = sorted(actual_names - expected_names)
        missing = sorted(expected_names - actual_names)
        detail = f"extra={extra[:5]}, missing={missing[:5]}, nested={[p.name for p in nested[:5]]}"
        raise FreezeManifestError(
            f"Generated matrix directory is not an exact artifact set: {detail}"
        )

    try:
        with run_manifest_path.open("r", encoding="utf-8-sig", newline="") as handle:
            csv_rows = list(csv.DictReader(handle))
    except (OSError, UnicodeError, csv.Error) as exc:
        raise FreezeManifestError(f"Cannot read run_manifest.csv: {exc}") from exc
    expected_csv = [_csv_projection(row) for row in expected_rows]
    if csv_rows != expected_csv:
        raise FreezeManifestError(
            f"run_manifest.csv does not exactly describe the {len(expected_runs)} configs"
        )

    expected_protocol = protocol_path.relative_to(project_root).as_posix()
    expected_matrix_manifest = {
        "protocol": expected_protocol,
        "protocol_hash": protocol_hash,
        "run_count": len(expected_runs),
        "conditions": list(active_conditions_for_protocol(protocol)),
        "seeds": list(protocol["seeds"]),
        "matrix_hash": matrix_hash,
        "runs": expected_rows,
    }
    if dict(matrix_manifest) != expected_matrix_manifest:
        raise FreezeManifestError(
            "matrix_manifest.json does not exactly match the protocol-derived matrix"
        )

    config_bindings = [
        {"path": row["config_file"], "sha256": row["config_file_sha256"]}
        for row in expected_rows
    ]
    return {
        "directory": matrix_dir.relative_to(project_root).as_posix(),
        "matrix_hash": matrix_hash,
        "run_count": len(expected_runs),
        "matrix_manifest": {
            "path": matrix_manifest_path.name,
            "sha256": _sha256_file(matrix_manifest_path),
        },
        "run_manifest": {
            "path": run_manifest_path.name,
            "sha256": _sha256_file(run_manifest_path),
        },
        "config_count": len(config_bindings),
        "config_set_sha256": _stable_hash(config_bindings),
        **(
            {
                "conditions": list(active_conditions_for_protocol(protocol)),
                "seeds": list(protocol["seeds"]),
            }
            if int(protocol.get("protocol_version", 1)) == 7
            else {}
        ),
    }


def _repository_url(repo_root: Path, supplied: str | None) -> str:
    value = supplied
    if value is None:
        discovered = _git(repo_root, ("config", "--get", "remote.origin.url"), allow_failure=True)
        assert isinstance(discovered, str)
        value = discovered
    value = value.strip() if isinstance(value, str) else ""
    if not value or "pending" in value.casefold() or REMOTE_RE.fullmatch(value) is None:
        raise FreezeManifestError(
            "A non-placeholder remote repository URL is required via --repository-url or origin"
        )
    return value


def build_manifest(
    *,
    project_root: Path,
    protocol_path: Path,
    signoff_path: Path,
    matrix_dir: Path,
    output_path: Path,
    freeze_commit: str,
    repository_url: str | None,
    created_at: str,
    require_creation_state: bool,
) -> dict[str, Any]:
    """Validate every input and return a manifest without writing anything."""

    project_root = project_root.resolve()
    protocol_path = _inside(project_root, protocol_path, "Protocol", kind="file")
    signoff_path = _inside(project_root, signoff_path, "Author sign-off", kind="file")
    matrix_dir = _inside(project_root, matrix_dir, "Generated matrix", kind="directory")
    output_path = _inside(project_root, output_path, "Freeze manifest output", kind="output")
    if output_path == protocol_path or output_path == signoff_path:
        raise FreezeManifestError("Output cannot replace a freeze input")
    if output_path.is_relative_to(matrix_dir):
        raise FreezeManifestError("Output must be outside the generated matrix directory")

    protocol = load_protocol(protocol_path)
    version = int(protocol.get("protocol_version", 1))
    is_v3 = version == 3
    is_v4 = version == 4
    is_v5 = version == 5
    is_v6 = version == 6
    is_v7 = version == 7
    is_isolated_protocol = version in (3, 4, 5, 6, 7)
    allow_legacy_artifacts = version in (3, 4, 5)
    artifacts = artifact_paths_for_protocol(protocol)
    if is_isolated_protocol:
        expected_paths = {
            "protocol": (project_root / artifacts["protocol"]).resolve(),
            "signoff": (project_root / artifacts["signoff"]).resolve(),
            "matrix": (project_root / artifacts["formal_matrix"]).resolve(),
            "freeze manifest": (project_root / artifacts["freeze_manifest"]).resolve(),
        }
        observed_paths = {
            "protocol": protocol_path,
            "signoff": signoff_path,
            "matrix": matrix_dir,
            "freeze manifest": output_path,
        }
        mismatches = [
            f"{label}: {observed_paths[label]} != {expected}"
            for label, expected in expected_paths.items()
            if observed_paths[label] != expected
        ]
        if mismatches:
            raise FreezeManifestError(
                f"Protocol v{version} freeze inputs must use their isolated artifact paths: "
                + "; ".join(mismatches)
            )
    confirmation = _validate_protocol_status(protocol)
    signoff_text = signoff_path.read_text(encoding="utf-8")
    _validate_signoff(signoff_text, protocol)
    created_at = _parse_aware_timestamp(created_at, "created_at")
    if is_v3:
        pilot_validation = _validate_v3_pilot_acceptance(
            protocol, protocol_path, project_root
        )
    elif is_v4:
        pilot_validation = _validate_v4_pilot_acceptance(
            protocol, protocol_path, project_root
        )
    elif is_v5:
        pilot_validation = _validate_v5_pilot_acceptance(
            protocol, protocol_path, project_root
        )
    elif is_v6:
        pilot_validation = _validate_v6_pilot_acceptance(
            protocol, protocol_path, project_root
        )
    elif is_v7:
        pilot_validation = _validate_v7_pilot_acceptance(
            protocol, protocol_path, project_root
        )
    else:
        pilot_validation = None

    repo_root = _git_repository(project_root)
    commit = _resolve_commit(project_root, freeze_commit, require_head=require_creation_state)
    if is_v5 and isinstance(pilot_validation, Mapping):
        recovery = pilot_validation.get("recovery")
        if isinstance(recovery, Mapping) and recovery.get(
            "recovery_validator_commit"
        ) != commit:
            raise FreezeManifestError(
                "V5 recovery validator commit must equal the formal freeze commit"
            )
    committed_protocol = _assert_committed_text(
        repo_root, commit, protocol_path, "Protocol at freeze commit"
    )
    committed_signoff = _assert_committed_text(
        repo_root, commit, signoff_path, "Author sign-off at freeze commit"
    )
    if is_v3:
        required_sources = (*REQUIRED_SOURCE_PATHS, *V3_REQUIRED_SOURCE_PATHS)
    elif is_v4:
        required_sources = (*REQUIRED_SOURCE_PATHS, *V4_REQUIRED_SOURCE_PATHS)
    elif is_v5:
        required_sources = (*REQUIRED_SOURCE_PATHS, *V5_REQUIRED_SOURCE_PATHS)
    elif is_v6:
        required_sources = (*REQUIRED_SOURCE_PATHS, *V6_REQUIRED_SOURCE_PATHS)
    elif is_v7:
        required_sources = (*REQUIRED_SOURCE_PATHS, *V7_REQUIRED_SOURCE_PATHS)
    else:
        required_sources = REQUIRED_SOURCE_PATHS
    committed_gate_sources: dict[str, bytes] = {}
    for relative in required_sources:
        source_path = project_root / relative
        if not source_path.is_file():
            raise FreezeManifestError(f"Required experiment source is missing: {relative}")
        committed_source = _assert_committed_text(
            repo_root, commit, source_path, f"Required source {relative}"
        )
        if (is_v4 and relative in V4_REQUIRED_SOURCE_PATHS) or (
            is_v5 and relative in V5_REQUIRED_SOURCE_PATHS
        ) or (
            is_v6 and relative in V6_REQUIRED_SOURCE_PATHS
        ) or (
            is_v7 and relative in V7_REQUIRED_SOURCE_PATHS
        ):
            committed_gate_sources[relative] = committed_source
    if require_creation_state:
        additional_artifact_dirs = (
            (project_root / artifacts["pilot_matrix"],)
            if is_isolated_protocol
            else ()
        )
        _assert_creation_worktree(
            repo_root,
            project_root,
            matrix_dir,
            output_path,
            allow_legacy_artifacts=allow_legacy_artifacts,
            additional_artifact_dirs=additional_artifact_dirs,
        )
    remote = _repository_url(repo_root, repository_url)
    tree = _git(repo_root, ("rev-parse", f"{commit}^{{tree}}"))
    commit_time = _git(repo_root, ("show", "-s", "--format=%cI", commit))
    assert isinstance(tree, str) and isinstance(commit_time, str)

    matrix = _validate_matrix(protocol, protocol_path, matrix_dir, project_root)
    if is_v7 and (
        matrix.get("run_count") != 48
        or matrix.get("config_count") != 48
        or matrix.get("conditions") != ["M0", "M1", "M2", "M3", "M4", "PLIF"]
        or matrix.get("seeds") != list(protocol["seeds"])
        or len(protocol["seeds"]) != 8
    ):
        raise FreezeManifestError(
            "Protocol v7 formal matrix must bind exactly 8 seeds x 6 ordered conditions"
        )
    project_in_repo = project_root.relative_to(repo_root)

    def project_relative(path: Path) -> str:
        return path.relative_to(project_root).as_posix()

    manifest = {
        "schema": SCHEMA,
        "created_at": created_at,
        "repository": {
            "url": remote,
            "freeze_commit": commit,
            "freeze_tree": tree,
            "freeze_commit_time": commit_time,
            "project_path": project_in_repo.as_posix() or ".",
        },
        "protocol": {
            "path": project_relative(protocol_path),
            "file_sha256": _sha256_bytes(committed_protocol),
            "canonical_sha256": _stable_hash(protocol),
            "version": protocol["protocol_version"],
            **confirmation,
        },
        "signed_record": {
            "path": project_relative(signoff_path),
            "file_sha256": _sha256_bytes(committed_signoff),
        },
        "generated_matrix": matrix,
    }
    if pilot_validation is not None:
        manifest["pilot_validation"] = pilot_validation
    if is_v4 or is_v5 or is_v6 or is_v7:
        gate_source_paths = (
            V4_REQUIRED_SOURCE_PATHS
            if is_v4
            else V5_REQUIRED_SOURCE_PATHS
            if is_v5
            else V6_REQUIRED_SOURCE_PATHS
            if is_v6
            else V7_REQUIRED_SOURCE_PATHS
        )
        manifest["gate_sources"] = {
            relative: {
                "path": relative,
                "file_sha256": _sha256_bytes(committed_gate_sources[relative]),
            }
            for relative in gate_source_paths
        }
    return manifest


def _atomic_create_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lock_path = Path(f"{path}.lock")
    try:
        lock_fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError as exc:
        raise FreezeManifestError(f"Manifest creation lock already exists: {lock_path}") from exc
    temporary: Path | None = None
    try:
        if path.exists():
            raise FreezeManifestError(
                f"Refusing to overwrite existing manifest: {path}; use --verify instead"
            )
        fd, name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
        temporary = Path(name)
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as handle:
            json.dump(value, handle, indent=2, sort_keys=True, ensure_ascii=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        if path.exists():
            raise FreezeManifestError(
                f"Manifest appeared during creation; refusing overwrite: {path}"
            )
        os.replace(temporary, path)
        temporary = None
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
        os.close(lock_fd)
        lock_path.unlink(missing_ok=True)


def create_manifest(
    *,
    project_root: Path,
    protocol_path: Path,
    signoff_path: Path,
    matrix_dir: Path,
    output_path: Path,
    freeze_commit: str,
    repository_url: str | None,
    created_at: str | None = None,
) -> dict[str, Any]:
    output_path = _inside(project_root, output_path, "Freeze manifest output", kind="output")
    if output_path.exists():
        raise FreezeManifestError(
            f"Refusing to overwrite existing manifest: {output_path}; use --verify instead"
        )
    timestamp = created_at or datetime.now().astimezone().isoformat(timespec="seconds")
    manifest = build_manifest(
        project_root=project_root,
        protocol_path=protocol_path,
        signoff_path=signoff_path,
        matrix_dir=matrix_dir,
        output_path=output_path,
        freeze_commit=freeze_commit,
        repository_url=repository_url,
        created_at=timestamp,
        require_creation_state=True,
    )
    _atomic_create_json(output_path, manifest)
    return manifest


def verify_manifest(*, project_root: Path, manifest_path: Path) -> dict[str, Any]:
    manifest_path = _inside(project_root, manifest_path, "Freeze manifest", kind="file")
    stored = dict(_read_json(manifest_path, "freeze manifest"))
    if stored.get("schema") != SCHEMA:
        raise FreezeManifestError(f"Unsupported freeze manifest schema: {stored.get('schema')!r}")
    repository = stored.get("repository")
    protocol = stored.get("protocol")
    signed_record = stored.get("signed_record")
    generated = stored.get("generated_matrix")
    required_objects = (repository, protocol, signed_record, generated)
    if not all(isinstance(value, Mapping) for value in required_objects):
        raise FreezeManifestError("Freeze manifest is missing a required object")
    assert isinstance(repository, Mapping)
    assert isinstance(protocol, Mapping)
    assert isinstance(signed_record, Mapping)
    assert isinstance(generated, Mapping)
    matrix_dir = _inside(
        project_root,
        Path(str(generated.get("directory", ""))),
        "Generated matrix",
        kind="directory",
    )
    repo_root = _git_repository(project_root)
    freeze_commit = _resolve_commit(
        project_root,
        str(repository.get("freeze_commit", "")),
        require_head=False,
    )
    bound_protocol_path = _inside(
        project_root,
        Path(str(protocol.get("path", ""))),
        "Protocol",
        kind="file",
    )
    bound_protocol = load_protocol(bound_protocol_path)
    bound_version = int(bound_protocol.get("protocol_version", 1))
    is_isolated_protocol = bound_version in (3, 4, 5, 6, 7)
    allow_legacy_artifacts = bound_version in (3, 4, 5)
    additional_artifact_dirs = (
        (
            project_root
            / artifact_paths_for_protocol(bound_protocol)["pilot_matrix"],
        )
        if is_isolated_protocol
        else ()
    )
    _assert_runtime_source(
        repo_root,
        project_root.resolve(),
        matrix_dir,
        manifest_path,
        freeze_commit,
        allow_legacy_artifacts=allow_legacy_artifacts,
        additional_artifact_dirs=additional_artifact_dirs,
    )
    created_at = _parse_aware_timestamp(stored.get("created_at"), "created_at")
    expected = build_manifest(
        project_root=project_root,
        protocol_path=Path(str(protocol.get("path", ""))),
        signoff_path=Path(str(signed_record.get("path", ""))),
        matrix_dir=Path(str(generated.get("directory", ""))),
        output_path=manifest_path,
        freeze_commit=str(repository.get("freeze_commit", "")),
        repository_url=str(repository.get("url", "")),
        created_at=created_at,
        require_creation_state=False,
    )
    if stored != expected:
        raise FreezeManifestError(
            "Freeze manifest content differs from the current bound artifacts"
        )
    return stored


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--protocol", type=Path, default=Path("configs/protocol.yaml"))
    parser.add_argument("--signoff", type=Path)
    parser.add_argument("--matrix-dir", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--freeze-commit",
        help="Required in create mode: explicit 40-character Phase A commit ID",
    )
    parser.add_argument(
        "--repository-url",
        help="Remote URL to bind; defaults to remote.origin.url",
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Verify the existing --output without writing anything",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    project_root = args.project_root.resolve()
    try:
        protocol_path = args.protocol if args.protocol.is_absolute() else project_root / args.protocol
        protocol = load_protocol(protocol_path)
        artifacts = artifact_paths_for_protocol(protocol)
        signoff_path = args.signoff or Path(artifacts["signoff"])
        matrix_dir = args.matrix_dir or Path(artifacts["formal_matrix"])
        output_path = args.output or Path(artifacts["freeze_manifest"])
        if args.verify:
            verify_manifest(project_root=project_root, manifest_path=output_path)
            output = _inside(project_root, output_path, "Freeze manifest", kind="file")
            print(f"PASS: verified {output}")
            print(f"Freeze manifest SHA-256: {_sha256_file(output)}")
            return 0
        if args.freeze_commit is None:
            raise FreezeManifestError("--freeze-commit is required in create mode")
        create_manifest(
            project_root=project_root,
            protocol_path=protocol_path,
            signoff_path=signoff_path,
            matrix_dir=matrix_dir,
            output_path=output_path,
            freeze_commit=args.freeze_commit,
            repository_url=args.repository_url,
        )
        output = _inside(project_root, output_path, "Freeze manifest", kind="file")
        print(f"Created {output}")
        print(f"Freeze manifest SHA-256: {_sha256_file(output)}")
        print("Protocol, confirmations, and author approvals were not modified.")
        return 0
    except (FreezeManifestError, OSError, ValueError) as exc:
        print(f"ERROR: freeze manifest not created or verified: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
