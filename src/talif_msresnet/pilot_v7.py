"""Fail-closed evidence contracts for the isolated V7 mechanism pilot."""

from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as distribution_version
from pathlib import Path
from typing import Any

from .config import RunConfig, canonical_json, load_run_config, validate_run_mapping
from .config_v7 import (
    V7_ACTIVE_CONDITIONS,
    V7_ARTIFACT_PATHS,
    V7_CIFAR100_PROVENANCE_CONTRACT,
    V7_FORMAL_SEEDS,
    V7_HEALTH_SEED,
    V7_PILOT_ACCEPTANCE,
    V7_PILOT_SEED,
    generate_v7_pilot_matrix,
    validate_v7_protocol,
)
from .pathing import artifact_path_reference
from .pilot_v3 import repository_git_identity
from .utils import sha256_file, stable_hash, utc_now

HEALTH_SCHEMA_VERSION = 1
HEALTH_ARTIFACT_CLASS = "NON_REPORTABLE_V7_MECHANISM_IMPLEMENTATION_HEALTH_GATE"
ATTEMPT_SCHEMA_VERSION = 1
ATTEMPT_ARTIFACT_CLASS = "NON_REPORTABLE_V7_MECHANISM_HEALTH_ATTEMPT"
ATTEMPT_STATUS = "ATTEMPT_CLAIMED_SEED_CONSUMED_NO_RETRY"
REPORTING_ELIGIBILITY = "FORBIDDEN_FROM_MANUSCRIPT_RESULTS"
HEALTH_SEED_DISPOSITION = "CONSUMED_NONREPORTING_V7_HEALTH_NEVER_RETRY"
PILOT_SEED_DISPOSITION = "RESERVED_UNCONSUMED_NONREPORTING_V7_PILOT"
PILOT_PLAN_SEED_DISPOSITION = "CONSUMED_BY_THIS_NONREPORTABLE_V7_PILOT_BLOCK"
V7_HEALTH_RUNTIME_SOURCE_PATHS: tuple[str, ...] = (
    "scripts/pilot_health_gate_v7.py",
    "scripts/pilot_health_gate.py",
    "scripts/pilot_health_gate_v3.py",
    "scripts/validate_v7_pilot.py",
    "scripts/generate_run_configs.py",
    "scripts/run_matrix.py",
    "src/talif_msresnet/config.py",
    "src/talif_msresnet/config_v6.py",
    "src/talif_msresnet/config_v7.py",
    "src/talif_msresnet/data.py",
    "src/talif_msresnet/freeze.py",
    "src/talif_msresnet/models.py",
    "src/talif_msresnet/neurons.py",
    "src/talif_msresnet/pathing.py",
    "src/talif_msresnet/pilot_v3.py",
    "src/talif_msresnet/pilot_v4.py",
    "src/talif_msresnet/pilot_v5.py",
    "src/talif_msresnet/pilot_v6.py",
    "src/talif_msresnet/pilot_v7.py",
    "src/talif_msresnet/preflight.py",
    "src/talif_msresnet/train.py",
    "src/talif_msresnet/utils.py",
)
V7_HEALTH_RECOVERY_SCHEMA = (
    "ta-lif-msresnet-v7-health-compatibility-recovery-release-v1"
)
V7_HEALTH_RECOVERY_RELEASE_RECORD = (
    "V7_HEALTH_COMPATIBILITY_RECOVERY_RELEASE.json"
)
V7_HEALTH_RECOVERY_BASE_COMMIT = "de2c3bf295624e9d83caf67a078b830d51208861"
V7_HEALTH_RECOVERY_HEALTH_SHA256 = (
    "2419a7241dba1b95ee96757c362aee00e50bdeb9a201b90db5e287a0fbc53c56"
)
V7_HEALTH_RECOVERY_HEALTH_PATH = (
    "results/pilot/v7_mechanism/health_cifar100_s1068798027.json"
)
V7_HEALTH_RECOVERY_ATTEMPT_SHA256 = (
    "26f02b0df23b99354d90a76f4df63f78a949cf0a7216541b511d075c067c8281"
)
V7_HEALTH_RECOVERY_ATTEMPT_PATH = (
    "results/pilot/v7_mechanism/health_cifar100_s1068798027.attempt.json"
)
V7_HEALTH_RECOVERY_PROTOCOL_HASH = (
    "ce8240e28147a87ee76a3953f8097613ecc33d683339ad66d9e73a126104265c"
)
V7_HEALTH_RECOVERY_SPLIT_SOURCE_FINGERPRINT = "cifar100:50000"
V7_HEALTH_RECOVERY_CORRECTED_FIELD = "source.split_source_fingerprint"
V7_HEALTH_RECOVERY_ALLOWED_CHANGED_PATHS: tuple[str, ...] = (
    "V7_MECHANISM_5090_RUNBOOK.md",
    "scripts/create_freeze_manifest.py",
    "scripts/pilot_health_gate_v7.py",
    "scripts/validate_v7_pilot.py",
    "src/talif_msresnet/freeze.py",
    "src/talif_msresnet/pilot_v7.py",
    "tests/test_v7_freeze_gates.py",
    "tests/test_v7_health_gate.py",
    "tests/test_validate_v7_pilot.py",
)
SIGNED_STATUS = (
    "Status: **AUTHORIZED - ACCOUNTABLE AUTHOR/USER APPROVAL RECORDED; "
    "PROTOCOL FROZEN**"
)
_HEX = frozenset("0123456789abcdef")


class PilotV7Error(RuntimeError):
    """Raised when V7 health or pilot evidence violates the frozen contract."""


@dataclass(frozen=True)
class PilotBlock:
    dataset: str
    health_seed: int
    pilot_seed: int
    conditions: tuple[str, ...]
    health_output: Path
    attempt_receipt: Path
    pilot_output_root: Path
    pilot_plan: Path
    formal_output_root: Path
    protocol_hash: str
    acceptance_hash: str
    block_hash: str

    @property
    def seed(self) -> int:
        return self.pilot_seed


def _inside(root: Path, value: Any, label: str) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise PilotV7Error(f"{label} must be a non-empty repository-relative path")
    path = (root / value).resolve()
    try:
        path.relative_to(root.resolve())
    except ValueError as exc:
        raise PilotV7Error(f"{label} escapes the repository: {path}") from exc
    return path


def _is_sha256(value: Any) -> bool:
    return isinstance(value, str) and len(value) == 64 and set(value) <= _HEX


def validate_v7_runtime_environment(environment_contract: Mapping[str, Any]) -> str:
    """Check V7's explicitly frozen torchvision distribution version.

    The legacy health gate already checks CUDA, GPU, and PyTorch.  V7 keeps
    torchvision separate because its dataset implementation is part of the
    CIFAR-100 input pipeline and should not drift with a compatible-looking
    PyTorch installation.
    """

    expected = environment_contract.get("torchvision_version")
    if not isinstance(expected, str) or not expected:
        raise PilotV7Error("V7 environment contract has no frozen torchvision version")
    try:
        observed = distribution_version("torchvision")
    except PackageNotFoundError as exc:
        raise PilotV7Error("V7 environment has no installed torchvision distribution") from exc
    if observed != expected:
        raise PilotV7Error(
            f"torchvision mismatch: {observed!r} != {expected!r}"
        )
    return observed


def _read_json(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise PilotV7Error(f"Cannot read {label}: {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise PilotV7Error(f"{label} must contain a JSON object")
    return value


def _git_output(repository_root: Path, *arguments: str) -> str:
    completed = subprocess.run(
        ["git", *arguments],
        cwd=repository_root,
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        detail = completed.stderr.strip() or completed.stdout.strip()
        raise PilotV7Error(
            "Cannot inspect the V7 health recovery release "
            f"(git {' '.join(arguments)}): {detail or 'no details'}"
        )
    return completed.stdout.strip()


def _git_bytes(repository_root: Path, *arguments: str) -> bytes:
    completed = subprocess.run(
        ["git", *arguments],
        cwd=repository_root,
        check=False,
        capture_output=True,
    )
    if completed.returncode != 0:
        detail = completed.stderr.decode("utf-8", errors="replace").strip()
        raise PilotV7Error(
            "Cannot inspect the V7 health recovery release "
            f"(git {' '.join(arguments)}): {detail or 'no details'}"
        )
    return completed.stdout


def _require_full_commit(value: Any, label: str) -> str:
    if not isinstance(value, str) or len(value) != 40 or not set(value) <= _HEX:
        raise PilotV7Error(f"{label} is not a full lowercase Git commit hash")
    return value


def require_v7_author_freeze(
    protocol: Mapping[str, Any], *, repository_root: str | Path
) -> dict[str, Any]:
    """Bind V7 health/pilot execution to the accountable-author record."""

    validated = validate_v7_protocol(protocol)
    root = Path(repository_root).resolve()
    status = validated["protocol_status"]
    confirmed_by = str(status["confirmed_by"]).strip()
    for unrecorded in ("Peng Yan", "Song Wang"):
        if unrecorded.casefold() in confirmed_by.casefold():
            raise PilotV7Error(f"V7 must not assert unrecorded approval from {unrecorded}")
    responsible = confirmed_by.split("(", 1)[0].strip()
    signoff = _inside(root, V7_ARTIFACT_PATHS["signoff"], "V7 sign-off")
    if not signoff.is_file():
        raise PilotV7Error(f"V7 sign-off is missing: {signoff}")
    text = signoff.read_text(encoding="utf-8")
    status_lines = re.findall(r"^Status:.*$", text, flags=re.MULTILINE)
    if status_lines != [SIGNED_STATUS]:
        raise PilotV7Error("V7 sign-off does not contain its exact authorized status")
    authorizer = (
        rf"^- Authorizer: {re.escape(responsible)}, accountable author and "
        rf"Codex task user\s*$"
    )
    if re.search(authorizer, text, flags=re.MULTILINE) is None:
        raise PilotV7Error("V7 sign-off does not identify the accountable author")
    for field in status["confirmations"]:
        if re.search(rf"^- \[[xX]\] `{re.escape(field)}`:", text, re.MULTILINE) is None:
            raise PilotV7Error(f"V7 sign-off checklist is missing {field}")
    return {
        "responsible_author": responsible,
        "confirmed_at": status["confirmed_at"],
        "signoff": artifact_path_reference(signoff, root),
        "signoff_sha256": sha256_file(signoff),
    }


def resolve_pilot_block(
    protocol: Mapping[str, Any],
    dataset: str | None = None,
    *,
    repository_root: str | Path,
) -> PilotBlock:
    """Resolve the one exact CIFAR-100 V7 health/pilot block."""

    if dataset is not None and str(dataset) != "cifar100":
        raise PilotV7Error("V7 core health/pilot supports only cifar100")
    validated = validate_v7_protocol(protocol)
    root = Path(repository_root).resolve()
    acceptance = validated["pilot_acceptance"]
    conditions = tuple(str(value) for value in acceptance["conditions"])
    if conditions != V7_ACTIVE_CONDITIONS:
        raise PilotV7Error("V7 pilot condition order differs from the frozen six-condition order")
    health_seed = int(acceptance["health_seed"])
    pilot_seed = int(acceptance["pilot_seed"])
    if health_seed != V7_HEALTH_SEED or pilot_seed != V7_PILOT_SEED:
        raise PilotV7Error("V7 health/pilot seed binding differs from the seed ledger")
    if {health_seed, pilot_seed} & set(V7_FORMAL_SEEDS):
        raise PilotV7Error("V7 health or pilot seed overlaps the formal seed set")
    protocol_hash = stable_hash(validated)
    acceptance_hash = stable_hash(acceptance)
    payload = {
        "dataset": "cifar100",
        "health_seed": health_seed,
        "pilot_seed": pilot_seed,
        "conditions": list(conditions),
        "protocol_hash": protocol_hash,
        "acceptance_hash": acceptance_hash,
    }
    block = PilotBlock(
        dataset="cifar100",
        health_seed=health_seed,
        pilot_seed=pilot_seed,
        conditions=conditions,
        health_output=_inside(root, acceptance["health_output"], "V7 health output"),
        attempt_receipt=_inside(root, acceptance["attempt_receipt"], "V7 attempt receipt"),
        pilot_output_root=_inside(
            root, acceptance["pilot_output_root"], "V7 pilot output root"
        ),
        pilot_plan=_inside(root, acceptance["pilot_plan"], "V7 pilot plan"),
        formal_output_root=_inside(
            root, V7_ARTIFACT_PATHS["formal_results"], "V7 formal output root"
        ),
        protocol_hash=protocol_hash,
        acceptance_hash=acceptance_hash,
        block_hash=stable_hash(payload),
    )
    if block.pilot_output_root == block.formal_output_root:
        raise PilotV7Error("V7 pilot and formal roots must remain separate")
    return block


def expected_pilot_configs(protocol: Mapping[str, Any]) -> tuple[RunConfig, ...]:
    validated = validate_v7_protocol(protocol)
    configs = tuple(
        validate_run_mapping(raw, validated) for raw in generate_v7_pilot_matrix(validated)
    )
    if tuple(config.model.condition for config in configs) != V7_ACTIVE_CONDITIONS:
        raise PilotV7Error("Generated V7 pilot matrix has the wrong condition order")
    return configs


def attempt_receipt_payload(
    *,
    block: PilotBlock,
    protocol_path: Path,
    config_paths: Sequence[Path],
    configs: Sequence[RunConfig],
    git_identity: Mapping[str, Any],
    repository_root: Path,
    runtime_sources: Mapping[str, str],
) -> dict[str, Any]:
    if tuple(config.model.condition for config in configs) != block.conditions:
        raise PilotV7Error("Health receipt configs do not cover the ordered V7 conditions")
    source = health_source_binding(
        block=block,
        protocol_path=protocol_path,
        config_paths=config_paths,
        configs=configs,
        repository_root=repository_root,
        runtime_sources=runtime_sources,
    )
    return {
        "schema_version": ATTEMPT_SCHEMA_VERSION,
        "artifact_class": ATTEMPT_ARTIFACT_CLASS,
        "status": ATTEMPT_STATUS,
        "reporting_eligibility": REPORTING_ELIGIBILITY,
        "claimed_at": utc_now(),
        "dataset": block.dataset,
        "health_seed": block.health_seed,
        "pilot_seed": block.pilot_seed,
        "conditions": list(block.conditions),
        "protocol_hash": block.protocol_hash,
        "acceptance_hash": block.acceptance_hash,
        "block_hash": block.block_hash,
        "git_commit": git_identity.get("git_commit"),
        "tracked_clean": git_identity.get("tracked_clean"),
        "health_seed_disposition": HEALTH_SEED_DISPOSITION,
        "pilot_seed_disposition": PILOT_SEED_DISPOSITION,
        "source": source,
    }


def health_source_binding(
    *,
    block: PilotBlock,
    protocol_path: Path,
    config_paths: Sequence[Path],
    configs: Sequence[RunConfig],
    repository_root: Path,
    runtime_sources: Mapping[str, str],
) -> dict[str, Any]:
    """Build the exact committed source binding shared by all V7 health stages."""

    if tuple(config.model.condition for config in configs) != block.conditions:
        raise PilotV7Error("V7 health source configs have the wrong condition order")
    if len(config_paths) != len(configs):
        raise PilotV7Error("V7 health source config paths and configs differ in length")
    if not runtime_sources or any(
        not isinstance(path, str) or not _is_sha256(digest)
        for path, digest in runtime_sources.items()
    ):
        raise PilotV7Error("V7 health runtime source hashes are incomplete")
    return {
        "protocol": artifact_path_reference(protocol_path, repository_root),
        "protocol_file_sha256": sha256_file(protocol_path),
        "configs": [
            {
                "condition": config.model.condition,
                "path": artifact_path_reference(path, repository_root),
                "file_sha256": sha256_file(path),
                "config_hash": config.config_hash,
            }
            for path, config in zip(config_paths, configs)
        ],
        "runtime_sources_sha256": dict(runtime_sources),
    }


def current_health_source_binding(
    *,
    protocol: Mapping[str, Any],
    protocol_path: Path,
    block: PilotBlock,
    repository_root: Path,
) -> dict[str, Any]:
    """Rebuild the canonical V7 health source binding from current files."""

    root = repository_root.resolve()
    expected = expected_pilot_configs(protocol)
    config_dir = _inside(
        root, V7_ARTIFACT_PATHS["pilot_matrix"], "V7 pilot matrix"
    )
    config_paths: list[Path] = []
    configs: list[RunConfig] = []
    for frozen in expected:
        path = config_dir / f"{frozen.runtime.run_id}.yaml"
        if not path.is_file():
            raise PilotV7Error(f"V7 pilot config is missing: {path}")
        try:
            observed = load_run_config(path, protocol_path)
        except (OSError, ValueError) as exc:
            raise PilotV7Error(f"Cannot validate V7 pilot config {path}: {exc}") from exc
        if observed.config_hash != frozen.config_hash:
            raise PilotV7Error(
                f"V7 pilot config differs from the frozen matrix: {observed.runtime.run_id}"
            )
        config_paths.append(path)
        configs.append(observed)

    runtime_sources: dict[str, str] = {}
    for relative in V7_HEALTH_RUNTIME_SOURCE_PATHS:
        path = _inside(root, relative, "V7 health runtime source")
        if not path.is_file():
            raise PilotV7Error(f"Required V7 health runtime source is missing: {path}")
        runtime_sources[relative] = sha256_file(path)
    return health_source_binding(
        block=block,
        protocol_path=protocol_path,
        config_paths=tuple(config_paths),
        configs=tuple(configs),
        repository_root=root,
        runtime_sources=runtime_sources,
    )


def json_file_payload_sha256(value: Mapping[str, Any]) -> str:
    """Hash the exact UTF-8 bytes written by :func:`exclusive_create_json`."""

    text = json.dumps(value, indent=2, sort_keys=True, ensure_ascii=True) + "\n"
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def exclusive_create_json(path: str | Path, value: Mapping[str, Any]) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    flags = os.O_CREAT | os.O_EXCL | os.O_WRONLY
    descriptor = os.open(target, flags, 0o600)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            json.dump(value, handle, indent=2, sort_keys=True, ensure_ascii=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
    except BaseException:
        target.unlink(missing_ok=True)
        raise
    return target


def validate_health_recovery_release(
    repository_root: str | Path,
    *,
    health_path: str | Path | None = None,
    attempt_receipt_path: str | Path | None = None,
    report: Mapping[str, Any] | None = None,
    receipt: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Validate the one sealed compatibility fix for the consumed V7 health seed.

    The health result remains evidence from its original commit.  This validator
    permits only one exact implementation commit followed by one record-only
    seal commit, and independently binds the immutable health and receipt bytes.
    """

    root = Path(repository_root).resolve()
    recovery_commit = _require_full_commit(
        _git_output(root, "rev-parse", "HEAD"), "V7 recovery release commit"
    )
    record_path = root / V7_HEALTH_RECOVERY_RELEASE_RECORD
    record = _read_json(record_path, "V7 health recovery release record")
    implementation_commit = _require_full_commit(
        record.get("implementation_commit"), "V7 recovery implementation commit"
    )

    seal_parents = _git_output(
        root, "rev-list", "--parents", "-n", "1", recovery_commit
    ).split()
    if seal_parents != [recovery_commit, implementation_commit]:
        raise PilotV7Error(
            "V7 recovery release must be the one-parent seal of its "
            "implementation commit"
        )
    implementation_parents = _git_output(
        root, "rev-list", "--parents", "-n", "1", implementation_commit
    ).split()
    if implementation_parents != [
        implementation_commit,
        V7_HEALTH_RECOVERY_BASE_COMMIT,
    ]:
        raise PilotV7Error(
            "V7 recovery implementation must be the direct child of the "
            "consumed-health commit"
        )
    if _git_output(root, "status", "--porcelain", "--untracked-files=no"):
        raise PilotV7Error("V7 health recovery requires a clean tracked Git worktree")

    implementation_changes = tuple(
        line.strip()
        for line in _git_output(
            root,
            "diff",
            "--name-status",
            "--find-renames",
            V7_HEALTH_RECOVERY_BASE_COMMIT,
            implementation_commit,
        ).splitlines()
        if line.strip()
    )
    expected_changes = tuple(
        f"M\t{path}" for path in V7_HEALTH_RECOVERY_ALLOWED_CHANGED_PATHS
    )
    if implementation_changes != expected_changes:
        raise PilotV7Error(
            "V7 health recovery implementation delta is not the exact "
            "modification-only allowlist: "
            f"observed={list(implementation_changes)} "
            f"expected={list(expected_changes)}"
        )
    seal_changes = tuple(
        line.strip()
        for line in _git_output(
            root,
            "diff",
            "--name-status",
            "--find-renames",
            implementation_commit,
            recovery_commit,
        ).splitlines()
        if line.strip()
    )
    if seal_changes != (f"A\t{V7_HEALTH_RECOVERY_RELEASE_RECORD}",):
        raise PilotV7Error(
            "V7 health recovery seal must add only the release record"
        )

    implementation_tree = _git_output(
        root, "rev-parse", f"{implementation_commit}^{{tree}}"
    )
    expected_record = {
        "schema": V7_HEALTH_RECOVERY_SCHEMA,
        "base_health_commit": V7_HEALTH_RECOVERY_BASE_COMMIT,
        "implementation_commit": implementation_commit,
        "implementation_tree": implementation_tree,
        "health_report_path": V7_HEALTH_RECOVERY_HEALTH_PATH,
        "health_report_sha256": V7_HEALTH_RECOVERY_HEALTH_SHA256,
        "attempt_receipt_path": V7_HEALTH_RECOVERY_ATTEMPT_PATH,
        "attempt_receipt_sha256": V7_HEALTH_RECOVERY_ATTEMPT_SHA256,
        "protocol_hash": V7_HEALTH_RECOVERY_PROTOCOL_HASH,
        "corrected_field": V7_HEALTH_RECOVERY_CORRECTED_FIELD,
        "split_source_fingerprint": V7_HEALTH_RECOVERY_SPLIT_SOURCE_FINGERPRINT,
        "allowed_changed_paths": list(V7_HEALTH_RECOVERY_ALLOWED_CHANGED_PATHS),
    }
    mismatches = [
        key for key, expected in expected_record.items() if record.get(key) != expected
    ]
    exact_record_keys = set(expected_record) | {
        "implementation_file_sha256",
        "record_sha256",
    }
    if set(record) != exact_record_keys:
        mismatches.append("record_keys")
    implementation_file_sha256 = record.get("implementation_file_sha256")
    if not isinstance(implementation_file_sha256, Mapping) or set(
        implementation_file_sha256
    ) != set(V7_HEALTH_RECOVERY_ALLOWED_CHANGED_PATHS):
        mismatches.append("implementation_file_sha256")
        implementation_file_sha256 = {}
    for relative in V7_HEALTH_RECOVERY_ALLOWED_CHANGED_PATHS:
        committed_sha256 = hashlib.sha256(
            _git_bytes(root, "show", f"{implementation_commit}:{relative}")
        ).hexdigest()
        if implementation_file_sha256.get(relative) != committed_sha256:
            mismatches.append(f"implementation_file_sha256.{relative}")
    record_without_hash = dict(record)
    record_sha256 = record_without_hash.pop("record_sha256", None)
    if record_sha256 != stable_hash(record_without_hash):
        mismatches.append("record_sha256")
    if mismatches:
        raise PilotV7Error(
            "V7 health recovery record is not the sealed reviewed "
            "implementation: " + ", ".join(mismatches)
        )

    evidence_supplied = any(
        value is not None
        for value in (health_path, attempt_receipt_path, report, receipt)
    )
    if evidence_supplied:
        if health_path is None or attempt_receipt_path is None:
            raise PilotV7Error(
                "V7 health recovery evidence requires both canonical artifact paths"
            )
        health_file = Path(health_path).resolve()
        receipt_file = Path(attempt_receipt_path).resolve()
        if not health_file.is_file() or not receipt_file.is_file():
            raise PilotV7Error("V7 health recovery evidence files are missing")
        try:
            health_relative = health_file.relative_to(root).as_posix()
            receipt_relative = receipt_file.relative_to(root).as_posix()
        except ValueError as exc:
            raise PilotV7Error(
                "V7 health recovery evidence paths escape the repository"
            ) from exc
        if health_relative != V7_HEALTH_RECOVERY_HEALTH_PATH:
            raise PilotV7Error("V7 health recovery report path is not authorized")
        if receipt_relative != V7_HEALTH_RECOVERY_ATTEMPT_PATH:
            raise PilotV7Error("V7 health recovery receipt path is not authorized")
        if sha256_file(health_file) != V7_HEALTH_RECOVERY_HEALTH_SHA256:
            raise PilotV7Error("V7 health recovery report SHA-256 is not authorized")
        if sha256_file(receipt_file) != V7_HEALTH_RECOVERY_ATTEMPT_SHA256:
            raise PilotV7Error("V7 health recovery receipt SHA-256 is not authorized")
        stored_report = _read_json(health_file, "V7 recovered health report")
        stored_receipt = _read_json(receipt_file, "V7 recovered health receipt")
        if report is not None and dict(report) != stored_report:
            raise PilotV7Error("In-memory V7 health report differs from its sealed bytes")
        if receipt is not None and dict(receipt) != stored_receipt:
            raise PilotV7Error("In-memory V7 health receipt differs from its sealed bytes")
        source = stored_report.get("source")
        evidence_mismatches: list[str] = []
        if stored_report.get("status") != "PASS" or stored_report.get("pass") is not True:
            evidence_mismatches.append("health PASS state")
        if stored_report.get("git_commit") != V7_HEALTH_RECOVERY_BASE_COMMIT:
            evidence_mismatches.append("health git_commit")
        if stored_receipt.get("git_commit") != V7_HEALTH_RECOVERY_BASE_COMMIT:
            evidence_mismatches.append("receipt git_commit")
        if stored_report.get("protocol_hash") != V7_HEALTH_RECOVERY_PROTOCOL_HASH:
            evidence_mismatches.append("health protocol_hash")
        if stored_receipt.get("protocol_hash") != V7_HEALTH_RECOVERY_PROTOCOL_HASH:
            evidence_mismatches.append("receipt protocol_hash")
        if not isinstance(source, Mapping) or source.get(
            "split_source_fingerprint"
        ) != V7_HEALTH_RECOVERY_SPLIT_SOURCE_FINGERPRINT:
            evidence_mismatches.append("health split_source_fingerprint")
        if stored_report.get(
            "attempt_receipt_sha256"
        ) != V7_HEALTH_RECOVERY_ATTEMPT_SHA256:
            evidence_mismatches.append("health attempt_receipt_sha256")
        if evidence_mismatches:
            raise PilotV7Error(
                "V7 health recovery evidence does not match the consumed PASS: "
                + ", ".join(evidence_mismatches)
            )

    return {
        "schema": V7_HEALTH_RECOVERY_SCHEMA,
        "base_health_commit": V7_HEALTH_RECOVERY_BASE_COMMIT,
        "implementation_commit": implementation_commit,
        "implementation_tree": implementation_tree,
        "recovery_commit": recovery_commit,
        "health_report_path": V7_HEALTH_RECOVERY_HEALTH_PATH,
        "health_report_sha256": V7_HEALTH_RECOVERY_HEALTH_SHA256,
        "attempt_receipt_path": V7_HEALTH_RECOVERY_ATTEMPT_PATH,
        "attempt_receipt_sha256": V7_HEALTH_RECOVERY_ATTEMPT_SHA256,
        "protocol_hash": V7_HEALTH_RECOVERY_PROTOCOL_HASH,
        "corrected_field": V7_HEALTH_RECOVERY_CORRECTED_FIELD,
        "split_source_fingerprint": V7_HEALTH_RECOVERY_SPLIT_SOURCE_FINGERPRINT,
        "allowed_changed_paths": list(V7_HEALTH_RECOVERY_ALLOWED_CHANGED_PATHS),
        "observed_changes": list(implementation_changes),
        "implementation_file_sha256": dict(implementation_file_sha256),
        "release_record": V7_HEALTH_RECOVERY_RELEASE_RECORD,
        "record_sha256": str(record["record_sha256"]),
        "release_record_sha256": sha256_file(record_path),
        "tracked_clean": True,
    }


def _validate_recovered_current_source(
    *,
    repository_root: Path,
    receipt_source: Mapping[str, Any],
    current_source: Mapping[str, Any],
    recovery: Mapping[str, Any],
) -> None:
    for key in ("protocol", "protocol_file_sha256", "configs"):
        if current_source.get(key) != receipt_source.get(key):
            raise PilotV7Error(
                f"V7 recovered health source changed a frozen {key} binding"
            )
    old_runtime = receipt_source.get("runtime_sources_sha256")
    current_runtime = current_source.get("runtime_sources_sha256")
    implementation_sha256 = recovery.get("implementation_file_sha256")
    if not isinstance(old_runtime, Mapping) or not isinstance(
        current_runtime, Mapping
    ) or not isinstance(implementation_sha256, Mapping):
        raise PilotV7Error("V7 recovered health runtime source binding is malformed")
    if set(old_runtime) != set(V7_HEALTH_RUNTIME_SOURCE_PATHS) or set(
        current_runtime
    ) != set(V7_HEALTH_RUNTIME_SOURCE_PATHS):
        raise PilotV7Error("V7 recovered health runtime source set changed")
    for relative in V7_HEALTH_RUNTIME_SOURCE_PATHS:
        base_sha256 = hashlib.sha256(
            _git_bytes(
                repository_root,
                "show",
                f"{V7_HEALTH_RECOVERY_BASE_COMMIT}:{relative}",
            )
        ).hexdigest()
        if old_runtime.get(relative) != base_sha256:
            raise PilotV7Error(
                f"V7 consumed health source is not bound to its base commit: {relative}"
            )
        expected = (
            implementation_sha256[relative]
            if relative in implementation_sha256
            else old_runtime[relative]
        )
        if current_runtime.get(relative) != expected:
            raise PilotV7Error(
                f"V7 recovered runtime source is not sealed: {relative}"
            )


def validate_attempt_receipt(
    receipt: Mapping[str, Any], *, block: PilotBlock
) -> dict[str, Any]:
    expected = {
        "schema_version": ATTEMPT_SCHEMA_VERSION,
        "artifact_class": ATTEMPT_ARTIFACT_CLASS,
        "status": ATTEMPT_STATUS,
        "reporting_eligibility": REPORTING_ELIGIBILITY,
        "dataset": block.dataset,
        "health_seed": block.health_seed,
        "pilot_seed": block.pilot_seed,
        "conditions": list(block.conditions),
        "protocol_hash": block.protocol_hash,
        "acceptance_hash": block.acceptance_hash,
        "block_hash": block.block_hash,
        "health_seed_disposition": HEALTH_SEED_DISPOSITION,
        "pilot_seed_disposition": PILOT_SEED_DISPOSITION,
    }
    mismatches = [key for key, value in expected.items() if receipt.get(key) != value]
    if receipt.get("tracked_clean") is not True:
        mismatches.append("tracked_clean")
    if not isinstance(receipt.get("claimed_at"), str):
        mismatches.append("claimed_at")
    commit = receipt.get("git_commit")
    if not isinstance(commit, str) or len(commit) != 40 or not set(commit) <= _HEX:
        mismatches.append("git_commit")
    source = receipt.get("source")
    if not isinstance(source, Mapping):
        mismatches.append("source")
    else:
        if not isinstance(source.get("protocol"), str) or not _is_sha256(
            source.get("protocol_file_sha256")
        ):
            mismatches.append("source.protocol")
        configs = source.get("configs")
        if not isinstance(configs, list) or len(configs) != len(block.conditions):
            mismatches.append("source.configs")
        else:
            for condition, item in zip(block.conditions, configs):
                if not isinstance(item, Mapping) or (
                    item.get("condition") != condition
                    or not isinstance(item.get("path"), str)
                    or not _is_sha256(item.get("file_sha256"))
                    or not _is_sha256(item.get("config_hash"))
                ):
                    mismatches.append(f"source.configs.{condition}")
        runtime_sources = source.get("runtime_sources_sha256")
        if not isinstance(runtime_sources, Mapping) or set(
            runtime_sources
        ) != set(V7_HEALTH_RUNTIME_SOURCE_PATHS) or any(
            not isinstance(path, str) or not _is_sha256(digest)
            for path, digest in runtime_sources.items()
        ):
            mismatches.append("source.runtime_sources_sha256")
    if mismatches:
        raise PilotV7Error("Invalid V7 health attempt receipt: " + ", ".join(mismatches))
    return dict(receipt)


def validate_health_report_payload(
    report: Mapping[str, Any],
    *,
    block: PilotBlock,
    receipt: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate a canonical V7 health PASS without rerunning the seed."""

    validate_attempt_receipt(receipt, block=block)
    expected = {
        "schema_version": HEALTH_SCHEMA_VERSION,
        "artifact_class": HEALTH_ARTIFACT_CLASS,
        "reporting_eligibility": REPORTING_ELIGIBILITY,
        "confirmatory_analysis_eligibility": False,
        "status": "PASS",
        "pass": True,
        "dataset": block.dataset,
        "health_seed": block.health_seed,
        "pilot_seed": block.pilot_seed,
        "conditions": list(block.conditions),
        "protocol_hash": block.protocol_hash,
        "acceptance_hash": block.acceptance_hash,
        "block_hash": block.block_hash,
        "health_seed_disposition": HEALTH_SEED_DISPOSITION,
        "pilot_seed_disposition": PILOT_SEED_DISPOSITION,
    }
    mismatches = [key for key, value in expected.items() if report.get(key) != value]
    if report.get("integrity_anomalies") != []:
        mismatches.append("integrity_anomalies")
    if report.get("failures") != []:
        mismatches.append("failures")
    if not _is_sha256(report.get("attempt_receipt_sha256")):
        mismatches.append("attempt_receipt_sha256")
    if report.get("git_commit") != receipt.get("git_commit"):
        mismatches.append("git_commit")
    if report.get("tracked_clean") is not True:
        mismatches.append("tracked_clean")
    if report.get("attempt_receipt") != V7_PILOT_ACCEPTANCE["attempt_receipt"]:
        mismatches.append("attempt_receipt")
    environment = report.get("environment")
    if not isinstance(environment, Mapping) or not _is_sha256(
        environment.get("training_environment_sha256")
    ):
        mismatches.append("training_environment_sha256")
    elif (
        environment.get("contract") != V7_PILOT_ACCEPTANCE["environment"]
        or not isinstance(environment.get("training_environment_identity"), Mapping)
        or not isinstance(environment.get("gpu_idle_precheck"), Mapping)
        or not isinstance(
            environment.get("gpu_idle_precheck", {}).get("device_uuid"), str
        )
    ):
        mismatches.append("environment")
    source = report.get("source")
    preclaim = report.get("preclaim")
    receipt_source = receipt.get("source")
    preclaim_source = preclaim.get("source") if isinstance(preclaim, Mapping) else None
    if not isinstance(source, Mapping):
        mismatches.append("source")
    elif not isinstance(receipt_source, Mapping):
        mismatches.append("receipt.source")
    else:
        for key in (
            "protocol",
            "protocol_file_sha256",
            "configs",
            "runtime_sources_sha256",
        ):
            if source.get(key) != receipt_source.get(key):
                mismatches.append(f"source.{key}")
            if not isinstance(preclaim_source, Mapping) or (
                preclaim_source.get(key) != receipt_source.get(key)
            ):
                mismatches.append(f"preclaim.source.{key}")
    expected_provenance = {
        "cifar100_source_provenance": V7_CIFAR100_PROVENANCE_CONTRACT[
            "source_provenance_path"
        ],
        "cifar100_source_provenance_sha256": V7_CIFAR100_PROVENANCE_CONTRACT[
            "source_provenance_sha256"
        ],
        "cifar100_archive": V7_CIFAR100_PROVENANCE_CONTRACT["archive_path"],
        "cifar100_archive_sha256": V7_CIFAR100_PROVENANCE_CONTRACT["archive_sha256"],
        "cifar100_train_pickle": V7_CIFAR100_PROVENANCE_CONTRACT["train_pickle_path"],
        "cifar100_train_pickle_sha256": V7_CIFAR100_PROVENANCE_CONTRACT[
            "train_pickle_sha256"
        ],
        "cifar100_test_pickle": V7_CIFAR100_PROVENANCE_CONTRACT["test_pickle_path"],
        "cifar100_test_pickle_sha256": V7_CIFAR100_PROVENANCE_CONTRACT[
            "test_pickle_sha256"
        ],
        "cifar100_meta_pickle": V7_CIFAR100_PROVENANCE_CONTRACT["meta_pickle_path"],
        "cifar100_meta_pickle_sha256": V7_CIFAR100_PROVENANCE_CONTRACT[
            "meta_pickle_sha256"
        ],
        "cifar100_split_manifest": V7_CIFAR100_PROVENANCE_CONTRACT[
            "split_manifest_path"
        ],
        "cifar100_split_manifest_sha256": V7_CIFAR100_PROVENANCE_CONTRACT[
            "split_manifest_sha256"
        ],
    }
    report_provenance = source.get("cifar100_provenance") if isinstance(source, Mapping) else None
    preclaim_provenance = (
        preclaim.get("cifar100_provenance") if isinstance(preclaim, Mapping) else None
    )
    if report_provenance != expected_provenance:
        mismatches.append("source.cifar100_provenance")
    if preclaim_provenance != expected_provenance:
        mismatches.append("preclaim.cifar100_provenance")
    execution_batch_sha256 = (
        source.get("fixed_batch_sha256") if isinstance(source, Mapping) else None
    )
    execution_split_sha256 = (
        source.get("split_manifest_sha256") if isinstance(source, Mapping) else None
    )
    if not _is_sha256(execution_batch_sha256):
        mismatches.append("source.fixed_batch_sha256")
    if not _is_sha256(execution_split_sha256):
        mismatches.append("source.split_manifest_sha256")
    if isinstance(source, Mapping):
        required_source = {
            "reference_config_hash": lambda value: _is_sha256(value),
            "reference_config_seed": lambda value: value == block.pilot_seed,
            "health_seed": lambda value: value == block.health_seed,
            "pilot_seed": lambda value: value == block.pilot_seed,
            "fixed_batch_shape": lambda value: isinstance(value, list) and bool(value),
            "fixed_batch_size": lambda value: isinstance(value, int) and value > 0,
            "split_source_fingerprint": lambda value: (
                value == V7_HEALTH_RECOVERY_SPLIT_SOURCE_FINGERPRINT
            ),
        }
        for key, predicate in required_source.items():
            if not predicate(source.get(key)):
                mismatches.append(f"source.{key}")
    if not isinstance(preclaim, Mapping) or preclaim.get("pass") is not True:
        mismatches.append("preclaim.pass")
    else:
        if preclaim.get("fixed_batch_sha256") != execution_batch_sha256:
            mismatches.append("preclaim.fixed_batch_sha256")
        if preclaim.get("fixed_batch_reconstruction_sha256") != execution_batch_sha256:
            mismatches.append("preclaim.fixed_batch_reconstruction_sha256")
        if preclaim.get("fixed_batch_reconstruction_count") != 2:
            mismatches.append("preclaim.fixed_batch_reconstruction_count")
        if preclaim.get("split_manifest_sha256") != execution_split_sha256:
            mismatches.append("preclaim.split_manifest_sha256")
        if isinstance(environment, Mapping) and preclaim.get(
            "training_environment_sha256"
        ) != environment.get("training_environment_sha256"):
            mismatches.append("preclaim.training_environment_sha256")
        if isinstance(environment, Mapping) and preclaim.get(
            "training_environment_identity"
        ) != environment.get("training_environment_identity"):
            mismatches.append("preclaim.training_environment_identity")
        preclaim_idle = preclaim.get("gpu_idle_precheck")
        report_idle = environment.get("gpu_idle_precheck") if isinstance(environment, Mapping) else None
        if not isinstance(preclaim_idle, Mapping) or not isinstance(report_idle, Mapping) or (
            preclaim_idle.get("device_uuid") != report_idle.get("device_uuid")
        ):
            mismatches.append("preclaim.gpu_idle_precheck")
    checks = report.get("checks")
    by_condition = checks.get("by_condition") if isinstance(checks, Mapping) else None
    health_contract = V7_PILOT_ACCEPTANCE["health"]
    expected_adaptive_updates = health_contract[
        "expected_adaptive_update_by_condition"
    ]
    if not isinstance(checks, Mapping) or checks.get("contract") != health_contract:
        mismatches.append("checks.contract")
    if isinstance(checks, Mapping) and checks.get(
        "expected_adaptive_update_by_condition"
    ) != expected_adaptive_updates:
        mismatches.append("checks.expected_adaptive_update_by_condition")
    if not isinstance(by_condition, Mapping) or tuple(by_condition) != block.conditions:
        mismatches.append("checks.by_condition")
    else:
        for condition in block.conditions:
            item = by_condition.get(condition)
            if not isinstance(item, Mapping) or (
                item.get("status") != "PASS" or item.get("pass") is not True
            ):
                mismatches.append(f"checks.by_condition.{condition}.status")
                continue
            if item.get("finite") is not True or item.get("resume_exact") is not True:
                mismatches.append(f"checks.by_condition.{condition}.numerics_resume")
            expects_adaptive = expected_adaptive_updates[condition]
            states = item.get("adaptive_enabled_by_epoch")
            expected_states = [False] * 5 + [expects_adaptive]
            if states != expected_states:
                mismatches.append(f"checks.by_condition.{condition}.adaptive_schedule")
            if item.get(
                "expected_adaptive_parameter_update_nonzero"
            ) is not expects_adaptive:
                mismatches.append(
                    f"checks.by_condition.{condition}.adaptive_expectation"
                )
            names = item.get("adaptive_parameter_names")
            if expects_adaptive:
                if not isinstance(names, list) or not names or item.get(
                    "adaptive_parameter_update_nonzero"
                ) is not True:
                    mismatches.append(f"checks.by_condition.{condition}.adaptive_update")
            elif names != [] or item.get("adaptive_parameter_update_nonzero") is not False:
                mismatches.append(
                    f"checks.by_condition.{condition}.unexpected_adaptive_group"
                )
            for detail in ("fixed_batch", "formal_schedule_boundary", "checkpoint_resume"):
                evidence = item.get(detail)
                if not isinstance(evidence, Mapping) or evidence.get("pass") is not True:
                    mismatches.append(f"checks.by_condition.{condition}.{detail}")
    if not isinstance(checks, Mapping) or checks.get("initial_forward_equivalent") is not True:
        mismatches.append("checks.initial_forward_equivalent")
    if not isinstance(checks, Mapping) or checks.get("shared_initialization_equal") is not True:
        mismatches.append("checks.shared_initialization_equal")
    initial = checks.get("initial_evidence") if isinstance(checks, Mapping) else None
    if not isinstance(initial, Mapping) or (
        initial.get("pass") is not True
        or initial.get("initial_forward_equivalent") is not True
        or initial.get("shared_initialization_equal") is not True
    ):
        mismatches.append("checks.initial_evidence")
    else:
        for key in ("output_sha256_by_condition", "shared_weight_sha256_by_condition"):
            values = initial.get(key)
            if not isinstance(values, Mapping) or tuple(values) != block.conditions or any(
                not _is_sha256(values.get(condition)) for condition in block.conditions
            ):
                mismatches.append(f"checks.initial_evidence.{key}")
    if mismatches:
        raise PilotV7Error("V7 health report is not an exact PASS: " + ", ".join(mismatches))
    return dict(report)


def validate_health_report(
    health_path: str | Path,
    protocol_path: str | Path,
    dataset: str | None = None,
    *,
    repository_root: str | Path,
    require_current_tracked_clean: bool = False,
) -> dict[str, Any]:
    from .config import load_protocol

    root = Path(repository_root).resolve()
    protocol_file = Path(protocol_path).resolve()
    protocol = load_protocol(protocol_file)
    block = resolve_pilot_block(protocol, dataset, repository_root=root)
    path = Path(health_path).resolve()
    if path != block.health_output or not path.is_file():
        raise PilotV7Error("V7 health report path is missing or noncanonical")
    if not block.attempt_receipt.is_file():
        raise PilotV7Error("V7 health attempt receipt is missing")
    receipt = _read_json(block.attempt_receipt, "V7 health attempt receipt")
    report = validate_health_report_payload(
        _read_json(path, "V7 health report"), block=block, receipt=receipt
    )
    if report.get("attempt_receipt_sha256") != sha256_file(block.attempt_receipt):
        raise PilotV7Error("V7 health report does not bind the attempt receipt")
    current_source = current_health_source_binding(
        protocol=protocol,
        protocol_path=protocol_file,
        block=block,
        repository_root=root,
    )
    direct_source_match = receipt.get("source") == current_source
    direct_identity_match = True
    if require_current_tracked_clean:
        identity = repository_git_identity(root)
        direct_identity_match = identity.get("tracked_clean") is True and identity.get(
            "git_commit"
        ) == report.get("git_commit")
    if direct_source_match and direct_identity_match:
        return report

    try:
        recovery = validate_health_recovery_release(
            root,
            health_path=path,
            attempt_receipt_path=block.attempt_receipt,
            report=report,
            receipt=receipt,
        )
    except PilotV7Error as exc:
        raise PilotV7Error(
            "V7 health attempt source binding differs from current frozen files "
            f"and no exact sealed recovery applies: {exc}"
        ) from exc
    receipt_source = receipt.get("source")
    if not isinstance(receipt_source, Mapping):
        raise PilotV7Error("V7 recovered health receipt has no source binding")
    _validate_recovered_current_source(
        repository_root=root,
        receipt_source=receipt_source,
        current_source=current_source,
        recovery=recovery,
    )
    recovered_report = dict(report)
    recovered_report["compatibility_recovery"] = recovery
    return recovered_report


def expected_pilot_plan_payload(
    *,
    block: PilotBlock,
    protocol_path: Path,
    config_dir: Path,
    manifest_rows: Sequence[Mapping[str, Any]],
    health_report: Mapping[str, Any],
    repository_root: Path,
) -> dict[str, Any]:
    configs = expected_pilot_configs(validate_v7_protocol_from_path(protocol_path))
    rows = {str(row.get("run_id")): row for row in manifest_rows}
    planned: list[dict[str, Any]] = []
    for config in configs:
        run_id = config.runtime.run_id
        row = rows.get(run_id)
        path = config_dir / f"{run_id}.yaml"
        if row is None or not path.is_file():
            raise PilotV7Error(f"V7 pilot matrix is missing {run_id}")
        planned.append(
            {
                "condition": config.model.condition,
                "run_id": run_id,
                "config_file": artifact_path_reference(path, repository_root),
                "config_file_sha256": sha256_file(path),
                "config_hash": config.config_hash,
                "seed": block.pilot_seed,
            }
        )
    environment = health_report.get("environment")
    if not isinstance(environment, Mapping):
        raise PilotV7Error("V7 health PASS has no environment binding")
    compatibility_recovery = health_report.get("compatibility_recovery")
    git_commit = health_report.get("git_commit")
    recovery_fields: dict[str, Any] = {}
    if compatibility_recovery is not None:
        if not isinstance(compatibility_recovery, Mapping):
            raise PilotV7Error("V7 health compatibility recovery binding is malformed")
        verified_recovery = validate_health_recovery_release(
            repository_root,
            health_path=block.health_output,
            attempt_receipt_path=block.attempt_receipt,
        )
        if dict(compatibility_recovery) != verified_recovery:
            raise PilotV7Error("V7 health compatibility recovery binding changed")
        git_commit = verified_recovery["recovery_commit"]
        recovery_fields = {
            "health_execution_git_commit": health_report.get("git_commit"),
            "health_compatibility_recovery": verified_recovery,
        }
    return {
        "version": 7,
        "artifact_class": "NON_REPORTABLE_V7_MECHANISM_PILOT_PLAN",
        "non_reportable": True,
        "reporting_eligibility": REPORTING_ELIGIBILITY,
        "protocol_path": artifact_path_reference(protocol_path, repository_root),
        "protocol_hash": block.protocol_hash,
        "acceptance_hash": block.acceptance_hash,
        "block_hash": block.block_hash,
        "git_commit": git_commit,
        **recovery_fields,
        "health_report": artifact_path_reference(block.health_output, repository_root),
        "health_report_sha256": sha256_file(block.health_output),
        "attempt_receipt": artifact_path_reference(
            block.attempt_receipt, repository_root
        ),
        "attempt_receipt_sha256": sha256_file(block.attempt_receipt),
        "training_environment_sha256": environment.get("training_environment_sha256"),
        "dataset": block.dataset,
        "seed": block.pilot_seed,
        "health_seed": block.health_seed,
        "pilot_seed": block.pilot_seed,
        "health_seed_disposition": HEALTH_SEED_DISPOSITION,
        "pilot_seed_disposition": PILOT_PLAN_SEED_DISPOSITION,
        "conditions": list(block.conditions),
        "pilot_output_root": artifact_path_reference(
            block.pilot_output_root, repository_root
        ),
        "formal_output_root": artifact_path_reference(
            block.formal_output_root, repository_root
        ),
        "runs": planned,
    }


def validate_v7_protocol_from_path(path: str | Path) -> dict[str, Any]:
    """Load lazily to avoid a config/pilot import cycle in callers."""

    from .config import load_protocol

    return validate_v7_protocol(load_protocol(path))


def canonical_health_payload_hash(value: Mapping[str, Any]) -> str:
    return stable_hash(json.loads(canonical_json(value)))
