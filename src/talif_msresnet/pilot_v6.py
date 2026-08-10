"""Fail-closed evidence contracts for the isolated V6 mechanism pilot."""

from __future__ import annotations

import json
import os
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as distribution_version
from pathlib import Path
from typing import Any

from .config import RunConfig, canonical_json, validate_run_mapping
from .config_v6 import (
    V6_ACTIVE_CONDITIONS,
    V6_ARTIFACT_PATHS,
    V6_FORMAL_SEEDS,
    V6_HEALTH_SEED,
    V6_PILOT_SEED,
    generate_v6_pilot_matrix,
    validate_v6_protocol,
)
from .pathing import artifact_path_reference
from .pilot_v3 import repository_git_identity
from .utils import sha256_file, stable_hash, utc_now

HEALTH_SCHEMA_VERSION = 1
HEALTH_ARTIFACT_CLASS = "NON_REPORTABLE_V6_MECHANISM_IMPLEMENTATION_HEALTH_GATE"
ATTEMPT_SCHEMA_VERSION = 1
ATTEMPT_ARTIFACT_CLASS = "NON_REPORTABLE_V6_MECHANISM_HEALTH_ATTEMPT"
ATTEMPT_STATUS = "ATTEMPT_CLAIMED_SEED_CONSUMED_NO_RETRY"
REPORTING_ELIGIBILITY = "FORBIDDEN_FROM_MANUSCRIPT_RESULTS"
HEALTH_SEED_DISPOSITION = "CONSUMED_NONREPORTING_V6_HEALTH_NEVER_RETRY"
PILOT_SEED_DISPOSITION = "RESERVED_UNCONSUMED_NONREPORTING_V6_PILOT"
PILOT_PLAN_SEED_DISPOSITION = "CONSUMED_BY_THIS_NONREPORTABLE_V6_PILOT_BLOCK"
SIGNED_STATUS = (
    "Status: **AUTHORIZED - ACCOUNTABLE AUTHOR/USER APPROVAL RECORDED; "
    "PROTOCOL FROZEN**"
)
_HEX = frozenset("0123456789abcdef")


class PilotV6Error(RuntimeError):
    """Raised when V6 health or pilot evidence violates the frozen contract."""


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
        raise PilotV6Error(f"{label} must be a non-empty repository-relative path")
    path = (root / value).resolve()
    try:
        path.relative_to(root.resolve())
    except ValueError as exc:
        raise PilotV6Error(f"{label} escapes the repository: {path}") from exc
    return path


def _is_sha256(value: Any) -> bool:
    return isinstance(value, str) and len(value) == 64 and set(value) <= _HEX


def validate_v6_runtime_environment(environment_contract: Mapping[str, Any]) -> str:
    """Check V6's explicitly frozen torchvision distribution version.

    The legacy health gate already checks CUDA, GPU, and PyTorch.  V6 keeps
    torchvision separate because its dataset implementation is part of the
    CIFAR-100 input pipeline and should not drift with a compatible-looking
    PyTorch installation.
    """

    expected = environment_contract.get("torchvision_version")
    if not isinstance(expected, str) or not expected:
        raise PilotV6Error("V6 environment contract has no frozen torchvision version")
    try:
        observed = distribution_version("torchvision")
    except PackageNotFoundError as exc:
        raise PilotV6Error("V6 environment has no installed torchvision distribution") from exc
    if observed != expected:
        raise PilotV6Error(
            f"torchvision mismatch: {observed!r} != {expected!r}"
        )
    return observed


def _read_json(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise PilotV6Error(f"Cannot read {label}: {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise PilotV6Error(f"{label} must contain a JSON object")
    return value


def require_v6_author_freeze(
    protocol: Mapping[str, Any], *, repository_root: str | Path
) -> dict[str, Any]:
    """Bind V6 health/pilot execution to the accountable-author record."""

    validated = validate_v6_protocol(protocol)
    root = Path(repository_root).resolve()
    status = validated["protocol_status"]
    confirmed_by = str(status["confirmed_by"]).strip()
    for unrecorded in ("Peng Yan", "Song Wang"):
        if unrecorded.casefold() in confirmed_by.casefold():
            raise PilotV6Error(f"V6 must not assert unrecorded approval from {unrecorded}")
    responsible = confirmed_by.split("(", 1)[0].strip()
    signoff = _inside(root, V6_ARTIFACT_PATHS["signoff"], "V6 sign-off")
    if not signoff.is_file():
        raise PilotV6Error(f"V6 sign-off is missing: {signoff}")
    text = signoff.read_text(encoding="utf-8")
    status_lines = re.findall(r"^Status:.*$", text, flags=re.MULTILINE)
    if status_lines != [SIGNED_STATUS]:
        raise PilotV6Error("V6 sign-off does not contain its exact authorized status")
    authorizer = (
        rf"^- Authorizer: {re.escape(responsible)}, accountable author and "
        rf"Codex task user\s*$"
    )
    if re.search(authorizer, text, flags=re.MULTILINE) is None:
        raise PilotV6Error("V6 sign-off does not identify the accountable author")
    for field in status["confirmations"]:
        if re.search(rf"^- \[[xX]\] `{re.escape(field)}`:", text, re.MULTILINE) is None:
            raise PilotV6Error(f"V6 sign-off checklist is missing {field}")
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
    """Resolve the one exact CIFAR-100 V6 health/pilot block."""

    if dataset is not None and str(dataset) != "cifar100":
        raise PilotV6Error("V6 core health/pilot supports only cifar100")
    validated = validate_v6_protocol(protocol)
    root = Path(repository_root).resolve()
    acceptance = validated["pilot_acceptance"]
    conditions = tuple(str(value) for value in acceptance["conditions"])
    if conditions != V6_ACTIVE_CONDITIONS:
        raise PilotV6Error("V6 pilot condition order differs from the frozen six-condition order")
    health_seed = int(acceptance["health_seed"])
    pilot_seed = int(acceptance["pilot_seed"])
    if health_seed != V6_HEALTH_SEED or pilot_seed != V6_PILOT_SEED:
        raise PilotV6Error("V6 health/pilot seed binding differs from the seed ledger")
    if {health_seed, pilot_seed} & set(V6_FORMAL_SEEDS):
        raise PilotV6Error("V6 health or pilot seed overlaps the formal seed set")
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
        health_output=_inside(root, acceptance["health_output"], "V6 health output"),
        attempt_receipt=_inside(root, acceptance["attempt_receipt"], "V6 attempt receipt"),
        pilot_output_root=_inside(
            root, acceptance["pilot_output_root"], "V6 pilot output root"
        ),
        pilot_plan=_inside(root, acceptance["pilot_plan"], "V6 pilot plan"),
        formal_output_root=_inside(
            root, V6_ARTIFACT_PATHS["formal_results"], "V6 formal output root"
        ),
        protocol_hash=protocol_hash,
        acceptance_hash=acceptance_hash,
        block_hash=stable_hash(payload),
    )
    if block.pilot_output_root == block.formal_output_root:
        raise PilotV6Error("V6 pilot and formal roots must remain separate")
    return block


def expected_pilot_configs(protocol: Mapping[str, Any]) -> tuple[RunConfig, ...]:
    validated = validate_v6_protocol(protocol)
    configs = tuple(
        validate_run_mapping(raw, validated) for raw in generate_v6_pilot_matrix(validated)
    )
    if tuple(config.model.condition for config in configs) != V6_ACTIVE_CONDITIONS:
        raise PilotV6Error("Generated V6 pilot matrix has the wrong condition order")
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
        raise PilotV6Error("Health receipt configs do not cover the ordered V6 conditions")
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
        "source": {
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
        },
    }


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
    if mismatches:
        raise PilotV6Error("Invalid V6 health attempt receipt: " + ", ".join(mismatches))
    return dict(receipt)


def validate_health_report_payload(
    report: Mapping[str, Any],
    *,
    block: PilotBlock,
    receipt: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate a canonical V6 health PASS without rerunning the seed."""

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
    if report.get("attempt_receipt_sha256") is None:
        mismatches.append("attempt_receipt_sha256")
    environment = report.get("environment")
    if not isinstance(environment, Mapping) or not _is_sha256(
        environment.get("training_environment_sha256")
    ):
        mismatches.append("training_environment_sha256")
    checks = report.get("checks")
    by_condition = checks.get("by_condition") if isinstance(checks, Mapping) else None
    if not isinstance(by_condition, Mapping) or tuple(by_condition) != block.conditions:
        mismatches.append("checks.by_condition")
    else:
        adaptive = {"M2", "M3", "M4", "PLIF"}
        for condition in block.conditions:
            item = by_condition.get(condition)
            if not isinstance(item, Mapping) or item.get("status") != "PASS":
                mismatches.append(f"checks.by_condition.{condition}.status")
                continue
            if item.get("finite") is not True or item.get("resume_exact") is not True:
                mismatches.append(f"checks.by_condition.{condition}.numerics_resume")
            states = item.get("adaptive_enabled_by_epoch")
            expected_states = [False] * 5 + [condition in adaptive]
            if states != expected_states:
                mismatches.append(f"checks.by_condition.{condition}.adaptive_schedule")
            names = item.get("adaptive_parameter_names")
            if condition in adaptive:
                if not isinstance(names, list) or not names or item.get(
                    "adaptive_parameter_update_nonzero"
                ) is not True:
                    mismatches.append(f"checks.by_condition.{condition}.adaptive_update")
            elif names != []:
                mismatches.append(f"checks.by_condition.{condition}.unexpected_adaptive_group")
    if not isinstance(checks, Mapping) or checks.get("initial_forward_equivalent") is not True:
        mismatches.append("checks.initial_forward_equivalent")
    if mismatches:
        raise PilotV6Error("V6 health report is not an exact PASS: " + ", ".join(mismatches))
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
        raise PilotV6Error("V6 health report path is missing or noncanonical")
    if not block.attempt_receipt.is_file():
        raise PilotV6Error("V6 health attempt receipt is missing")
    receipt = _read_json(block.attempt_receipt, "V6 health attempt receipt")
    report = validate_health_report_payload(
        _read_json(path, "V6 health report"), block=block, receipt=receipt
    )
    if report.get("attempt_receipt_sha256") != sha256_file(block.attempt_receipt):
        raise PilotV6Error("V6 health report does not bind the attempt receipt")
    if require_current_tracked_clean:
        identity = repository_git_identity(root)
        if identity.get("tracked_clean") is not True or identity.get("git_commit") != report.get(
            "git_commit"
        ):
            raise PilotV6Error("Current Git identity differs from the V6 health PASS")
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
    configs = expected_pilot_configs(validate_v6_protocol_from_path(protocol_path))
    rows = {str(row.get("run_id")): row for row in manifest_rows}
    planned: list[dict[str, Any]] = []
    for config in configs:
        run_id = config.runtime.run_id
        row = rows.get(run_id)
        path = config_dir / f"{run_id}.yaml"
        if row is None or not path.is_file():
            raise PilotV6Error(f"V6 pilot matrix is missing {run_id}")
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
        raise PilotV6Error("V6 health PASS has no environment binding")
    return {
        "version": 6,
        "artifact_class": "NON_REPORTABLE_V6_MECHANISM_PILOT_PLAN",
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


def validate_v6_protocol_from_path(path: str | Path) -> dict[str, Any]:
    """Load lazily to avoid a config/pilot import cycle in callers."""

    from .config import load_protocol

    return validate_v6_protocol(load_protocol(path))


def canonical_health_payload_hash(value: Mapping[str, Any]) -> str:
    return stable_hash(json.loads(canonical_json(value)))
