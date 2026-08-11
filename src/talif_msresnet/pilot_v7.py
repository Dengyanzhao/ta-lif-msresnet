"""Fail-closed evidence contracts for the isolated V7 mechanism pilot."""

from __future__ import annotations

import json
import hashlib
import os
import re
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
            "split_source_fingerprint": lambda value: _is_sha256(value),
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
    current_source = current_health_source_binding(
        protocol=protocol,
        protocol_path=protocol_file,
        block=block,
        repository_root=root,
    )
    if receipt.get("source") != current_source:
        raise PilotV7Error(
            "V7 health attempt source binding differs from current frozen files"
        )
    report = validate_health_report_payload(
        _read_json(path, "V7 health report"), block=block, receipt=receipt
    )
    if report.get("attempt_receipt_sha256") != sha256_file(block.attempt_receipt):
        raise PilotV7Error("V7 health report does not bind the attempt receipt")
    if require_current_tracked_clean:
        identity = repository_git_identity(root)
        if identity.get("tracked_clean") is not True or identity.get("git_commit") != report.get(
            "git_commit"
        ):
            raise PilotV7Error("Current Git identity differs from the V7 health PASS")
    return report


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
    return {
        "version": 7,
        "artifact_class": "NON_REPORTABLE_V7_MECHANISM_PILOT_PLAN",
        "non_reportable": True,
        "reporting_eligibility": REPORTING_ELIGIBILITY,
        "protocol_path": artifact_path_reference(protocol_path, repository_root),
        "protocol_hash": block.protocol_hash,
        "acceptance_hash": block.acceptance_hash,
        "block_hash": block.block_hash,
        "git_commit": health_report.get("git_commit"),
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
