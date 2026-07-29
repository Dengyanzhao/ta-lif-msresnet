"""Fail-closed contracts for the threshold-free TA-LIF-only v5 health gate.

Protocol v5 separates the one-shot implementation-health seed from the
non-reportable pilot seed.  This module contains no GPU execution.  It binds
the frozen development evidence, validates the exclusive attempt receipt and
health report, and constructs the exact dataset-specific pilot plan.
"""

from __future__ import annotations

import json
import math
import os
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml

from .config import (
    RunConfig,
    active_conditions_for_protocol,
    artifact_paths_for_protocol,
    confirmation_fields_for_protocol,
    generate_v5_pilot_matrix,
    validate_run_mapping,
)
from .pathing import artifact_path_reference
from .pilot_v3 import repository_git_identity
from .utils import sha256_file, stable_hash, utc_now

V5_PILOT_CONDITIONS = ("C1", "C2")
V5_PILOT_DATASETS = ("cifar100", "cifar10dvs")
HEALTH_SCHEMA_VERSION = 5
HEALTH_ARTIFACT_CLASS = "NON_REPORTING_V5_IMPLEMENTATION_HEALTH_GATE"
ATTEMPT_SCHEMA_VERSION = 1
ATTEMPT_ARTIFACT_CLASS = "NON_REPORTING_V5_HEALTH_ATTEMPT"
ATTEMPT_STATUS = "ATTEMPT_CLAIMED_SEED_CONSUMED_NO_RETRY"
REPORTING_ELIGIBILITY = "FORBIDDEN_FROM_MANUSCRIPT_RESULTS"
SIGNED_STATUS = (
    "Status: **AUTHORIZED - ACCOUNTABLE AUTHOR/USER APPROVAL RECORDED; PROTOCOL FROZEN**"
)
PROHIBITED_UNRECORDED_APPROVERS = ("Peng Yan", "Song Wang")
HEALTH_SEED_DISPOSITION = "CONSUMED_NONREPORTING_V5_HEALTH_NEVER_RETRY"
PILOT_SEED_DISPOSITION = "RESERVED_UNCONSUMED_NONREPORTING_V5_PILOT"
DEVELOPMENT_SEED_DISPOSITION = (
    "DEVELOPMENT_ONLY_EXCLUDE_FROM_V5_HEALTH_PILOT_AND_FORMAL"
)
HEALTH_DECISION_BASIS = "deterministic_mechanism_and_numerical_integrity_only"
LEARNING_METRICS_ROLE = "descriptive_only_no_pass_fail_threshold"
TIMING_HEALTH_ROLE = "BOUND_FOR_PILOT_ONLY_NOT_EVALUATED_BY_HEALTH"
SCHEDULE_HEALTH_ROLE = "BOUND_FOR_PILOT_ONLY_NOT_EVALUATED_BY_HEALTH"


class PilotV5Error(RuntimeError):
    """Raised when v5 evidence cannot satisfy its frozen contract."""


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
    development_probe_hash: str
    development_probe_binding: Mapping[str, Any]
    block_hash: str

    @property
    def seed(self) -> int:
        """Return the pilot seed for legacy pilot-plan consumers."""

        return self.pilot_seed


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise PilotV5Error(f"{label} must be a mapping")
    return value


def _positive_int(value: Any, label: str, *, allow_zero: bool = False) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise PilotV5Error(f"{label} must be an integer")
    minimum = 0 if allow_zero else 1
    if value < minimum:
        raise PilotV5Error(f"{label} must be >= {minimum}")
    return int(value)


def _inside_repository(value: Any, repository_root: Path, label: str) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise PilotV5Error(f"{label} must be a non-empty repository-relative path")
    path = Path(value)
    if path.is_absolute() or ".." in path.parts:
        raise PilotV5Error(f"{label} must be repository-relative without '..'")
    resolved = (repository_root / path).resolve()
    try:
        resolved.relative_to(repository_root.resolve())
    except ValueError as exc:
        raise PilotV5Error(f"{label} escapes the repository") from exc
    return resolved


def _is_within(path: Path, root: Path) -> bool:
    resolved = path.resolve()
    resolved_root = root.resolve()
    return resolved == resolved_root or resolved_root in resolved.parents


def _is_sha256(value: Any) -> bool:
    return bool(
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _is_git_commit(value: Any) -> bool:
    return bool(
        isinstance(value, str)
        and len(value) == 40
        and all(character in "0123456789abcdef" for character in value)
    )


def _responsible_author(value: Any) -> str:
    if not isinstance(value, str) or not value.strip():
        raise PilotV5Error("protocol_status.confirmed_by must identify the authorizer")
    normalized = value.strip()
    prohibited = [
        name
        for name in PROHIBITED_UNRECORDED_APPROVERS
        if name.casefold() in normalized.casefold()
    ]
    if prohibited:
        raise PilotV5Error(
            "v5 confirmed_by claims unrecorded co-author approval: "
            + ", ".join(prohibited)
        )
    responsible = normalized.split("(", 1)[0].strip()
    if not responsible:
        raise PilotV5Error("protocol_status.confirmed_by has no author identity")
    return responsible


def require_v5_author_freeze(
    protocol: Mapping[str, Any],
    *,
    repository_root: str | Path,
) -> dict[str, Any]:
    """Require the exact accountable-author authorization before v5 GPU work."""

    if int(protocol.get("protocol_version", 0)) != 5:
        raise PilotV5Error("Author-freeze validation requires protocol_version 5")
    status = _mapping(protocol.get("protocol_status"), "protocol.protocol_status")
    if status.get("frozen") is not True:
        raise PilotV5Error("v5 GPU work requires protocol_status.frozen=true")
    confirmations = _mapping(
        status.get("confirmations"), "protocol.protocol_status.confirmations"
    )
    missing = [
        field
        for field in confirmation_fields_for_protocol(protocol)
        if confirmations.get(field) is not True
    ]
    if missing:
        raise PilotV5Error("v5 author confirmations are incomplete: " + ", ".join(missing))

    confirmed_by = str(status.get("confirmed_by") or "").strip()
    responsible = _responsible_author(confirmed_by)
    confirmed_at = str(status.get("confirmed_at") or "").strip()
    candidate = f"{confirmed_at[:-1]}+00:00" if confirmed_at.endswith("Z") else confirmed_at
    try:
        timestamp = datetime.fromisoformat(candidate)
    except ValueError as exc:
        raise PilotV5Error("protocol_status.confirmed_at is not valid ISO-8601") from exc
    if timestamp.tzinfo is None or timestamp.utcoffset() is None:
        raise PilotV5Error("protocol_status.confirmed_at must include a UTC offset")

    root = Path(repository_root).resolve()
    signoff = _inside_repository(
        artifact_paths_for_protocol(protocol)["signoff"], root, "v5 sign-off"
    )
    if not signoff.is_file():
        raise PilotV5Error(f"The v5 authorization record is missing: {signoff}")
    text = signoff.read_text(encoding="utf-8")
    if re.findall(r"^Status:.*$", text, flags=re.MULTILINE) != [SIGNED_STATUS]:
        raise PilotV5Error("The v5 authorization record is not in its exact frozen state")
    if re.search(r"\[[^\]\r\n]*PENDING[^\]\r\n]*\]", text, flags=re.IGNORECASE):
        raise PilotV5Error("The v5 authorization record contains a pending placeholder")
    for field in confirmation_fields_for_protocol(protocol):
        if re.search(
            rf"^- \[[xX]\] `{re.escape(field)}`:", text, flags=re.MULTILINE
        ) is None:
            raise PilotV5Error(f"The v5 authorization checklist is not checked: {field}")
    authorizer = (
        rf"^- Authorizer: {re.escape(responsible)}, accountable author and "
        rf"Codex task user\s*$"
    )
    if re.search(authorizer, text, flags=re.MULTILINE) is None:
        raise PilotV5Error("The v5 authorization record does not identify the authorizer")
    for name in PROHIBITED_UNRECORDED_APPROVERS:
        disclosure = (
            rf"^- Independent approval from {re.escape(name)}: "
            rf"not asserted in this record\s*$"
        )
        if re.search(disclosure, text, flags=re.MULTILINE) is None:
            raise PilotV5Error(f"The v5 record lacks the non-approval disclosure for {name}")
    return {
        "confirmed_by": confirmed_by,
        "responsible_author": responsible,
        "confirmed_at": confirmed_at,
        "signoff": artifact_path_reference(signoff, root),
        "signoff_sha256": sha256_file(signoff),
    }


def _development_probe_binding(
    acceptance: Mapping[str, Any], repository_root: Path
) -> dict[str, Any]:
    contract = _mapping(
        acceptance.get("development_probe"), "pilot_acceptance.development_probe"
    )
    reports = _mapping(contract.get("datasets"), "development_probe.datasets")
    if tuple(reports) != V5_PILOT_DATASETS:
        raise PilotV5Error(
            "development_probe.datasets must be exactly ordered cifar100,cifar10dvs"
        )
    if not _is_git_commit(contract.get("expected_git_commit")):
        raise PilotV5Error("development_probe.expected_git_commit is invalid")
    for key in (
        "plan_file_sha256",
        "plan_hash",
        "development_entrypoint_sha256",
        "training_source_sha256",
        "v4_termination_record_sha256",
        "v4_attempt_receipt_sha256",
        "v4_failure_report_sha256",
    ):
        if not _is_sha256(contract.get(key)):
            raise PilotV5Error(f"development_probe.{key} is invalid")
    plan_path = _inside_repository(
        contract.get("plan"), repository_root, "development_probe.plan"
    )
    report_binding: dict[str, Any] = {}
    used_seeds: set[int] = set()
    for dataset in V5_PILOT_DATASETS:
        item = _mapping(reports.get(dataset), f"development_probe.datasets.{dataset}")
        path = _inside_repository(
            item.get("path"), repository_root, f"development_probe.{dataset}.path"
        )
        digest = item.get("sha256")
        if not _is_sha256(digest):
            raise PilotV5Error(f"development_probe.{dataset}.sha256 is invalid")
        fixed_seed = _positive_int(
            item.get("fixed_batch_seed"),
            f"development_probe.{dataset}.fixed_batch_seed",
        )
        formal_seed = _positive_int(
            item.get("formal_schedule_seed"),
            f"development_probe.{dataset}.formal_schedule_seed",
        )
        if fixed_seed == formal_seed or {fixed_seed, formal_seed} & used_seeds:
            raise PilotV5Error("development probe seed identities are not disjoint")
        used_seeds.update((fixed_seed, formal_seed))
        report_binding[dataset] = {
            "path": artifact_path_reference(path, repository_root),
            "sha256": digest,
            "fixed_batch_seed": fixed_seed,
            "formal_schedule_seed": formal_seed,
        }
    return {
        "artifact_class": contract.get("artifact_class"),
        "reporting_eligibility": contract.get("reporting_eligibility"),
        "confirmatory_analysis_eligibility": contract.get(
            "confirmatory_analysis_eligibility"
        ),
        "expected_status": contract.get("expected_status"),
        "expected_git_commit": contract.get("expected_git_commit"),
        "expected_tracked_clean": contract.get("expected_tracked_clean"),
        "plan": artifact_path_reference(plan_path, repository_root),
        "plan_file_sha256": contract.get("plan_file_sha256"),
        "plan_hash": contract.get("plan_hash"),
        "development_entrypoint_sha256": contract.get(
            "development_entrypoint_sha256"
        ),
        "training_source_sha256": contract.get("training_source_sha256"),
        "v4_termination_record_sha256": contract.get(
            "v4_termination_record_sha256"
        ),
        "v4_attempt_receipt_sha256": contract.get("v4_attempt_receipt_sha256"),
        "v4_failure_report_sha256": contract.get("v4_failure_report_sha256"),
        "reports": report_binding,
    }


def resolve_pilot_block(
    protocol: Mapping[str, Any],
    dataset: str,
    *,
    repository_root: str | Path,
) -> PilotBlock:
    """Resolve one exact v5 block with separate health and pilot seed roles."""

    root = Path(repository_root).resolve()
    if int(protocol.get("protocol_version", 0)) != 5:
        raise PilotV5Error("The v5 pilot layer requires protocol_version 5")
    conditions = tuple(active_conditions_for_protocol(protocol))
    if conditions != V5_PILOT_CONDITIONS:
        raise PilotV5Error("v5 active conditions must be exactly ordered C1,C2")
    acceptance = _mapping(protocol.get("pilot_acceptance"), "protocol.pilot_acceptance")
    if tuple(acceptance.get("conditions", ())) != conditions:
        raise PilotV5Error("pilot_acceptance.conditions must be exactly ordered C1,C2")
    datasets = _mapping(acceptance.get("datasets"), "protocol.pilot_acceptance.datasets")
    if tuple(datasets) != V5_PILOT_DATASETS:
        raise PilotV5Error(
            "pilot_acceptance.datasets must be exactly ordered cifar100,cifar10dvs"
        )
    normalized = str(dataset).lower().replace("-", "")
    if normalized not in V5_PILOT_DATASETS:
        raise PilotV5Error(f"Unsupported v5 pilot dataset: {dataset!r}")
    profile = _mapping(datasets.get(normalized), f"pilot_acceptance.datasets.{normalized}")
    health_seed = _positive_int(profile.get("health_seed"), f"{normalized}.health_seed")
    pilot_seed = _positive_int(profile.get("pilot_seed"), f"{normalized}.pilot_seed")
    if health_seed == pilot_seed:
        raise PilotV5Error("v5 health and pilot seeds must be distinct")

    health_output = _inside_repository(
        profile.get("health_output"), root, f"{normalized}.health_output"
    )
    attempt_receipt = _inside_repository(
        profile.get("attempt_receipt"), root, f"{normalized}.attempt_receipt"
    )
    pilot_output_root = _inside_repository(
        profile.get("pilot_output_root"), root, f"{normalized}.pilot_output_root"
    )
    pilot_plan = _inside_repository(
        profile.get("pilot_plan"), root, f"{normalized}.pilot_plan"
    )
    formal_output_root = _inside_repository(protocol.get("output_root"), root, "output_root")
    if len({health_output, attempt_receipt, pilot_output_root, pilot_plan}) != 4:
        raise PilotV5Error(f"{normalized} v5 artifact paths must be distinct")
    if _is_within(pilot_output_root, formal_output_root) or _is_within(
        formal_output_root, pilot_output_root
    ):
        raise PilotV5Error("v5 pilot and formal output roots must be disjoint")
    if _is_within(health_output, pilot_output_root) or _is_within(
        attempt_receipt, pilot_output_root
    ):
        raise PilotV5Error("v5 health evidence must be outside the pilot output root")

    implementation = _mapping(
        acceptance.get("implementation_health"), "pilot_acceptance.implementation_health"
    )
    if implementation.get("decision_basis") != HEALTH_DECISION_BASIS:
        raise PilotV5Error("v5 implementation-health decision basis differs")
    if implementation.get("thresholds_evaluated") != []:
        raise PilotV5Error("v5 implementation health must evaluate no thresholds")
    if implementation.get("learning_metrics_role") != LEARNING_METRICS_ROLE:
        raise PilotV5Error("v5 learning metrics must remain descriptive")
    fixed = _mapping(implementation.get("fixed_batch"), "implementation_health.fixed_batch")
    formal = _mapping(
        implementation.get("formal_schedule"), "implementation_health.formal_schedule"
    )
    checkpoint = _mapping(
        implementation.get("checkpoint_resume"), "implementation_health.checkpoint_resume"
    )
    _positive_int(fixed.get("batch_size"), "implementation_health.fixed_batch.batch_size")
    steps = [
        _positive_int(value, "fixed_batch.observation_steps[]", allow_zero=True)
        for value in fixed.get("observation_steps", ())
    ]
    if not steps or steps != sorted(set(steps)) or steps[0] != 0 or steps[-1] == 0:
        raise PilotV5Error("fixed_batch observation steps must increase from zero")
    if fixed.get("minimum_ta_routed_gradient_coverage_observation") != 0.0:
        raise PilotV5Error("v5 TA health must not use a calibrated coverage threshold")
    epochs = _positive_int(formal.get("epochs"), "formal_schedule.epochs")
    _positive_int(
        formal.get("train_batches_per_epoch"), "formal_schedule.train_batches_per_epoch"
    )
    _positive_int(
        formal.get("validation_batches_per_epoch"),
        "formal_schedule.validation_batches_per_epoch",
    )
    activation = _positive_int(
        formal.get("required_activation_epoch_zero_based"),
        "formal_schedule.required_activation_epoch_zero_based",
        allow_zero=True,
    )
    if list(formal.get("required_pre_activation_epochs", ())) != list(range(activation)):
        raise PilotV5Error("formal_schedule pre-activation epochs are incomplete")
    if list(formal.get("required_post_activation_epochs", ())) != list(
        range(activation, epochs)
    ):
        raise PilotV5Error("formal_schedule must cross and cover the TA activation epoch")
    if checkpoint.get("required") is not True:
        raise PilotV5Error("v5 exact checkpoint-resume validation is required")

    development_binding = _development_probe_binding(acceptance, root)
    role_seeds = {
        health_seed,
        pilot_seed,
        *(
            int(value)
            for item in development_binding["reports"].values()
            for value in (item["fixed_batch_seed"], item["formal_schedule_seed"])
        ),
    }
    if len(role_seeds) != 6:
        raise PilotV5Error("v5 health, pilot, and development seed roles overlap")

    protocol_hash = stable_hash(protocol)
    acceptance_hash = stable_hash(dict(acceptance))
    development_hash = stable_hash(development_binding)
    block_binding = {
        "dataset": normalized,
        "profile": dict(profile),
        "conditions": list(conditions),
        "evidence": {"development_probe": development_binding},
        "implementation_health": {
            "fixed_batch": dict(fixed),
            "formal_schedule": dict(formal),
            "checkpoint_resume": dict(checkpoint),
        },
        "environment": acceptance.get("environment"),
        "timing": acceptance.get("timing"),
        "schedule": acceptance.get("schedule"),
    }
    return PilotBlock(
        dataset=normalized,
        health_seed=health_seed,
        pilot_seed=pilot_seed,
        conditions=conditions,
        health_output=health_output,
        attempt_receipt=attempt_receipt,
        pilot_output_root=pilot_output_root,
        pilot_plan=pilot_plan,
        formal_output_root=formal_output_root,
        protocol_hash=protocol_hash,
        acceptance_hash=acceptance_hash,
        development_probe_hash=development_hash,
        development_probe_binding=development_binding,
        block_hash=stable_hash(block_binding),
    )


def expected_pilot_configs(
    protocol: Mapping[str, Any], dataset: str
) -> tuple[RunConfig, RunConfig]:
    rows = [
        row
        for row in generate_v5_pilot_matrix(protocol)
        if row.get("data", {}).get("dataset") == dataset
    ]
    configs = tuple(validate_run_mapping(row, protocol) for row in rows)
    if len(configs) != 2 or tuple(item.model.condition for item in configs) != ("C1", "C2"):
        raise PilotV5Error(f"{dataset} does not generate one ordered C1/C2 pilot block")
    profile = _mapping(
        _mapping(protocol["pilot_acceptance"], "pilot_acceptance")["datasets"],
        "pilot_acceptance.datasets",
    )[dataset]
    pilot_seed = int(_mapping(profile, f"datasets.{dataset}")["pilot_seed"])
    if any(config.runtime.seed != pilot_seed for config in configs):
        raise PilotV5Error("v5 pilot configs must use the pilot seed, never the health seed")
    return configs  # type: ignore[return-value]


def _read_json(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise PilotV5Error(f"Cannot read {label} {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise PilotV5Error(f"{label} must be a JSON object")
    return value


def validate_development_probe_report_payload(
    report: Mapping[str, Any],
    *,
    dataset: str,
    binding: Mapping[str, Any],
) -> None:
    """Validate the frozen semantic identity of one development-only report."""

    reports = _mapping(binding.get("reports"), "development probe reports")
    dataset_binding = _mapping(reports.get(dataset), f"development probe {dataset}")
    failures: list[str] = []

    def require(result: bool, message: str) -> None:
        if not result:
            failures.append(message)

    expected = {
        "artifact_class": binding.get("artifact_class"),
        "reporting_eligibility": binding.get("reporting_eligibility"),
        "confirmatory_analysis_eligibility": binding.get(
            "confirmatory_analysis_eligibility"
        ),
        "status": binding.get("expected_status"),
        "pass": None,
        "thresholds_evaluated": [],
        "dataset": dataset,
        "conditions": list(V5_PILOT_CONDITIONS),
        "plan": binding.get("plan"),
        "plan_file_sha256": binding.get("plan_file_sha256"),
        "plan_hash": binding.get("plan_hash"),
        "git_commit": binding.get("expected_git_commit"),
        "tracked_clean": binding.get("expected_tracked_clean"),
        "integrity_anomalies": [],
    }
    for key, expected_value in expected.items():
        require(report.get(key) == expected_value, f"development report {dataset} {key} mismatch")
    require(report.get("fatal_error") in (None, ""), f"development report {dataset} is fatal")

    seeds = report.get("development_seeds")
    require(isinstance(seeds, Mapping), f"development report {dataset} seeds are missing")
    if isinstance(seeds, Mapping):
        require(
            seeds.get("fixed_batch") == dataset_binding.get("fixed_batch_seed"),
            f"development report {dataset} fixed-batch seed mismatch",
        )
        require(
            seeds.get("formal_schedule") == dataset_binding.get("formal_schedule_seed"),
            f"development report {dataset} formal-schedule seed mismatch",
        )
        require(
            seeds.get("disposition") == DEVELOPMENT_SEED_DISPOSITION,
            f"development report {dataset} seed disposition mismatch",
        )

    source = report.get("source")
    require(isinstance(source, Mapping), f"development report {dataset} source is missing")
    if isinstance(source, Mapping):
        for key in (
            "development_entrypoint_sha256",
            "training_source_sha256",
            "v4_termination_record_sha256",
        ):
            require(
                source.get(key) == binding.get(key),
                f"development report {dataset} source {key} mismatch",
            )
    v4_evidence = report.get("v4_failure_evidence")
    require(
        isinstance(v4_evidence, Mapping),
        f"development report {dataset} v4 failure evidence is missing",
    )
    if isinstance(v4_evidence, Mapping):
        require(
            v4_evidence.get("receipt_sha256")
            == binding.get("v4_attempt_receipt_sha256"),
            f"development report {dataset} v4 receipt mismatch",
        )
        require(
            v4_evidence.get("report_sha256") == binding.get("v4_failure_report_sha256"),
            f"development report {dataset} v4 failure report mismatch",
        )

    for key in ("fixed_batch_probe_by_condition", "formal_schedule_probe_by_condition"):
        results = report.get(key)
        require(isinstance(results, Mapping), f"development report {dataset} {key} missing")
        if isinstance(results, Mapping):
            require(
                tuple(results) == V5_PILOT_CONDITIONS,
                f"development report {dataset} {key} condition order mismatch",
            )
            for condition in V5_PILOT_CONDITIONS:
                item = results.get(condition)
                require(
                    isinstance(item, Mapping) and item.get("integrity_anomalies") == [],
                    f"development report {dataset} {key}/{condition} has anomalies",
                )
    if failures:
        raise PilotV5Error(
            "V5 development-probe evidence rejected:\n- " + "\n- ".join(failures)
        )


def validate_development_probe_evidence(
    protocol: Mapping[str, Any],
    *,
    repository_root: str | Path,
) -> dict[str, Any]:
    """Validate both exact development JSON files, their hashes, and semantics."""

    root = Path(repository_root).resolve()
    acceptance = _mapping(protocol.get("pilot_acceptance"), "protocol.pilot_acceptance")
    binding = _development_probe_binding(acceptance, root)
    plan_path = _inside_repository(binding["plan"], root, "development probe plan")
    if not plan_path.is_file() or sha256_file(plan_path) != binding["plan_file_sha256"]:
        raise PilotV5Error("The exact v5 development-probe plan is missing or changed")
    try:
        plan = yaml.safe_load(plan_path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise PilotV5Error(f"Cannot read the v5 development-probe plan: {exc}") from exc
    if not isinstance(plan, Mapping) or stable_hash(plan) != binding["plan_hash"]:
        raise PilotV5Error("The v5 development-probe plan hash differs")

    for dataset in V5_PILOT_DATASETS:
        item = _mapping(binding["reports"][dataset], f"development report {dataset}")
        path = _inside_repository(item["path"], root, f"development report {dataset}")
        if not path.is_file() or sha256_file(path) != item["sha256"]:
            raise PilotV5Error(
                f"The exact {dataset} development-probe JSON is missing or changed"
            )
        validate_development_probe_report_payload(
            _read_json(path, f"{dataset} development report"),
            dataset=dataset,
            binding=binding,
        )
    return binding


def attempt_receipt_payload(
    *,
    block: PilotBlock,
    protocol_path: str | Path,
    config_path: str | Path,
    config: RunConfig,
    git_identity: Mapping[str, Any],
    repository_root: str | Path,
    runtime_sources: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Build the durable claim made before any health-seed GPU operation."""

    if config.data.dataset != block.dataset or config.model.condition != "C1":
        raise PilotV5Error("The attempt receipt requires the dataset's C1 pilot config")
    if config.runtime.seed != block.pilot_seed:
        raise PilotV5Error("The reference config must carry the pilot seed, not the health seed")
    root = Path(repository_root).resolve()
    protocol_file = Path(protocol_path).resolve()
    config_file = Path(config_path).resolve()
    return {
        "schema_version": ATTEMPT_SCHEMA_VERSION,
        "artifact_class": ATTEMPT_ARTIFACT_CLASS,
        "status": ATTEMPT_STATUS,
        "reporting_eligibility": REPORTING_ELIGIBILITY,
        "confirmatory_analysis_eligibility": False,
        "dataset": block.dataset,
        "seed": block.health_seed,
        "health_seed": block.health_seed,
        "pilot_seed": block.pilot_seed,
        "conditions": list(block.conditions),
        "protocol_hash": block.protocol_hash,
        "acceptance_hash": block.acceptance_hash,
        "development_probe_hash": block.development_probe_hash,
        "block_hash": block.block_hash,
        "git_commit": git_identity.get("git_commit"),
        "tracked_clean": git_identity.get("tracked_clean"),
        "protocol": artifact_path_reference(protocol_file, root),
        "protocol_file_sha256": sha256_file(protocol_file),
        "reference_config": artifact_path_reference(config_file, root),
        "reference_config_sha256": sha256_file(config_file),
        "reference_config_hash": config.config_hash,
        "reference_config_seed": config.runtime.seed,
        "evidence": {"development_probe": dict(block.development_probe_binding)},
        "runtime_sources_sha256": dict(runtime_sources or {}),
        "claimed_at": utc_now(),
        "seed_disposition": HEALTH_SEED_DISPOSITION,
        "health_seed_disposition": HEALTH_SEED_DISPOSITION,
        "pilot_seed_disposition": PILOT_SEED_DISPOSITION,
    }


def exclusive_create_json(path: str | Path, value: Mapping[str, Any]) -> Path:
    """Create a durable, standards-compliant JSON artifact without overwrite."""

    destination = Path(path).resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = (
        json.dumps(
            dict(value),
            indent=2,
            sort_keys=True,
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")
    descriptor = os.open(destination, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "wb", closefd=True) as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    return destination


def validate_attempt_receipt(
    path: str | Path,
    *,
    block: PilotBlock,
    protocol_path: str | Path,
    repository_root: str | Path,
    reference_config: RunConfig | None = None,
) -> dict[str, Any]:
    receipt_path = Path(path).resolve()
    if receipt_path != block.attempt_receipt or not receipt_path.is_file():
        raise PilotV5Error("The canonical dataset-specific v5 attempt receipt is missing")
    receipt = _read_json(receipt_path, "v5 health attempt receipt")
    expected = {
        "schema_version": ATTEMPT_SCHEMA_VERSION,
        "artifact_class": ATTEMPT_ARTIFACT_CLASS,
        "status": ATTEMPT_STATUS,
        "reporting_eligibility": REPORTING_ELIGIBILITY,
        "confirmatory_analysis_eligibility": False,
        "dataset": block.dataset,
        "seed": block.health_seed,
        "health_seed": block.health_seed,
        "pilot_seed": block.pilot_seed,
        "conditions": list(block.conditions),
        "protocol_hash": block.protocol_hash,
        "acceptance_hash": block.acceptance_hash,
        "development_probe_hash": block.development_probe_hash,
        "block_hash": block.block_hash,
        "protocol_file_sha256": sha256_file(protocol_path),
        "evidence": {"development_probe": dict(block.development_probe_binding)},
        "seed_disposition": HEALTH_SEED_DISPOSITION,
        "health_seed_disposition": HEALTH_SEED_DISPOSITION,
        "pilot_seed_disposition": PILOT_SEED_DISPOSITION,
    }
    failures = [
        f"attempt receipt {key} mismatch"
        for key, expected_value in expected.items()
        if receipt.get(key) != expected_value
    ]
    if receipt.get("tracked_clean") is not True:
        failures.append("attempt receipt was not claimed from a clean tracked tree")
    if not _is_git_commit(receipt.get("git_commit")):
        failures.append("attempt receipt git_commit is invalid")
    root = Path(repository_root).resolve()
    if receipt.get("protocol") != artifact_path_reference(protocol_path, root):
        failures.append("attempt receipt protocol path mismatch")
    try:
        claimed_at = str(receipt.get("claimed_at") or "")
        parsed = datetime.fromisoformat(
            f"{claimed_at[:-1]}+00:00" if claimed_at.endswith("Z") else claimed_at
        )
        if parsed.tzinfo is None or parsed.utcoffset() is None:
            raise ValueError
    except ValueError:
        failures.append("attempt receipt claimed_at is not timezone-aware ISO-8601")
    if reference_config is not None:
        if reference_config.runtime.seed != block.pilot_seed:
            failures.append("reference config does not use the v5 pilot seed")
        if receipt.get("reference_config_hash") != reference_config.config_hash:
            failures.append("attempt receipt reference config hash mismatch")
        if receipt.get("reference_config_seed") != block.pilot_seed:
            failures.append("attempt receipt reference config seed mismatch")
    runtime_sources = receipt.get("runtime_sources_sha256")
    if not isinstance(runtime_sources, Mapping) or not runtime_sources:
        failures.append("attempt receipt runtime source hashes are missing")
    elif any(not isinstance(key, str) or not _is_sha256(value) for key, value in runtime_sources.items()):
        failures.append("attempt receipt runtime source hashes are invalid")
    if failures:
        raise PilotV5Error("V5 health attempt receipt rejected:\n- " + "\n- ".join(failures))
    return receipt


def validate_fixed_batch_shape(
    dataset: str,
    shape: Sequence[Any],
    *,
    batch_size: int,
    time_steps: int,
    in_channels: int,
) -> list[int]:
    if not isinstance(shape, Sequence) or isinstance(shape, (str, bytes)):
        raise PilotV5Error("fixed batch shape must be a sequence")
    if any(isinstance(value, bool) or not isinstance(value, int) or value <= 0 for value in shape):
        raise PilotV5Error("fixed batch shape must contain positive integers")
    values = [int(value) for value in shape]
    if not values or values[0] != batch_size:
        raise PilotV5Error("fixed batch shape must bind the exact executed batch slice")
    if dataset == "cifar100":
        if len(values) != 4 or values[1] != in_channels:
            raise PilotV5Error("CIFAR-100 fixed input must be [B,C,H,W]")
    elif dataset == "cifar10dvs":
        if len(values) != 5 or values[1] != time_steps or values[2] != in_channels:
            raise PilotV5Error("CIFAR10-DVS fixed input must be [B,T,C,H,W]")
    else:
        raise PilotV5Error(f"Unsupported fixed-batch dataset: {dataset}")
    return values


def _finite(value: Any, *, positive: bool = False) -> bool:
    valid = (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    )
    return bool(valid and (not positive or float(value) > 0.0))


def nonfinite_paths(value: Any, path: str = "root") -> list[str]:
    """Return paths to all explicitly nonfinite numeric observations."""

    if isinstance(value, float):
        return [] if math.isfinite(value) else [path]
    if isinstance(value, Mapping):
        paths: list[str] = []
        for key, item in value.items():
            paths.extend(nonfinite_paths(item, f"{path}.{key}"))
        return paths
    if isinstance(value, (list, tuple)):
        paths = []
        for index, item in enumerate(value):
            paths.extend(nonfinite_paths(item, f"{path}[{index}]"))
        return paths
    return []


def json_safe(value: Any) -> Any:
    """Replace nonfinite floats with JSON null after they have been recorded."""

    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, Mapping):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [json_safe(item) for item in value]
    if isinstance(value, list):
        return [json_safe(item) for item in value]
    return value


def _append_recorded_anomalies(
    failures: list[str], prefix: str, item: Mapping[str, Any]
) -> None:
    recorded = item.get("integrity_anomalies")
    if recorded != []:
        if isinstance(recorded, list) and recorded:
            failures.extend(f"{prefix}: {message}" for message in recorded)
        else:
            failures.append(f"{prefix}: integrity_anomalies is missing or malformed")


def _fixed_batch_anomalies(
    item: Any,
    *,
    condition: str,
    health_seed: int,
    settings: Mapping[str, Any],
    checkpoint_settings: Mapping[str, Any],
) -> list[str]:
    prefix = f"fixed_batch/{condition}"
    if not isinstance(item, Mapping):
        return [f"{prefix}: report is missing"]
    failures: list[str] = []
    _append_recorded_anomalies(failures, prefix, item)
    if item.get("condition") != condition or item.get("seed") != health_seed:
        failures.append(f"{prefix}: condition or health seed mismatch")
    if item.get("optimizer_regime") != settings.get("optimizer_regime"):
        failures.append(f"{prefix}: optimizer regime mismatch")
    if item.get("learning_metrics_role") != LEARNING_METRICS_ROLE:
        failures.append(f"{prefix}: learning metrics are not descriptive-only")
    expected_steps = list(settings.get("observation_steps", ()))
    observations = item.get("observations")
    if not isinstance(observations, list) or [
        row.get("step") for row in observations if isinstance(row, Mapping)
    ] != expected_steps:
        failures.append(f"{prefix}: observation steps are incomplete")
    history = item.get("training_history")
    maximum_step = int(expected_steps[-1]) if expected_steps else 0
    if not isinstance(history, list) or [
        row.get("step") for row in history if isinstance(row, Mapping)
    ] != list(range(1, maximum_step + 1)):
        failures.append(f"{prefix}: training history is incomplete")
    if isinstance(observations, list) and observations:
        final = observations[-1]
        delta = final.get("parameter_delta") if isinstance(final, Mapping) else None
        if not isinstance(delta, Mapping) or not _finite(delta.get("l2"), positive=True):
            failures.append(f"{prefix}: no finite positive parameter update was observed")
        elif int(delta.get("updated_parameter_tensors", 0)) <= 0:
            failures.append(f"{prefix}: no parameter tensor was updated")

    shared = item.get("shared_conv_and_fc_gradients")
    if not isinstance(shared, Mapping) or shared.get("pass") is not True:
        failures.append(f"{prefix}: shared convolution/classifier gradients failed")
    elif (
        int(shared.get("required_parameter_count", 0)) <= 0
        or shared.get("passed_parameter_count") != shared.get("required_parameter_count")
    ):
        failures.append(f"{prefix}: shared gradient coverage is incomplete")

    ta = item.get("ta_gradients")
    if condition == "C2":
        required_ta = (
            isinstance(ta, Mapping)
            and ta.get("pass") is True
            and ta.get("every_ta_parameter_tensor_has_nonzero_routed_coverage") is True
            and ta.get("all_ta_gradients_present_and_finite_on_every_probe") is True
            and ta.get("no_nonzero_gradient_in_unrouted_slots") is True
            and int(ta.get("routed_parameter_slots", 0)) > 0
            and int(ta.get("covered_parameter_slots", 0)) > 0
        )
        if not required_ta:
            failures.append(f"{prefix}: C2 TA routing or gradients failed")
    elif not isinstance(ta, Mapping) or ta.get("status") != "not_applicable_lif_condition":
        failures.append(f"{prefix}: C1 TA status is invalid")

    checkpoint = item.get("checkpoint_boundary")
    if not isinstance(checkpoint, Mapping) or checkpoint.get("pass") is not True:
        failures.append(f"{prefix}: checkpoint exact-resume comparison failed")
    elif (
        checkpoint.get("comparison") != "bitwise exact next training step"
        or checkpoint.get("resume_loader") != checkpoint_settings.get("resume_loader")
        or checkpoint.get("checkpoint_name") != checkpoint_settings.get("checkpoint_name")
        or checkpoint.get("resume_epoch_match") is not True
        or checkpoint.get("mismatch_count") != 0
        or checkpoint.get("mismatch_paths") != []
    ):
        failures.append(f"{prefix}: checkpoint exact-resume evidence is incomplete")
    elif condition == "C2" and not (
        checkpoint.get("ta_activation_boundary_crossed_on_resume") is True
        and checkpoint.get("ta_enabled_at_checkpoint") is False
        and checkpoint.get("ta_enabled_on_next_step") is True
        and checkpoint.get("checkpoint_epoch_zero_based") == 4
        and checkpoint.get("expected_start_epoch_zero_based") == 5
        and checkpoint.get("resumed_start_epoch_zero_based") == 5
        and checkpoint.get("ta_activation_epoch_zero_based") == 5
    ):
        failures.append(f"{prefix}: exact resume did not cross the epoch-4/5 TA boundary")
    return failures


def _formal_schedule_anomalies(
    item: Any,
    *,
    condition: str,
    health_seed: int,
    settings: Mapping[str, Any],
    ta_lr_scale: float,
) -> list[str]:
    prefix = f"formal_schedule/{condition}"
    if not isinstance(item, Mapping):
        return [f"{prefix}: report is missing"]
    failures: list[str] = []
    _append_recorded_anomalies(failures, prefix, item)
    epochs = int(settings["epochs"])
    train_batches = int(settings["train_batches_per_epoch"])
    validation_batches = int(settings["validation_batches_per_epoch"])
    activation = int(settings["required_activation_epoch_zero_based"])
    if item.get("condition") != condition or item.get("seed") != health_seed:
        failures.append(f"{prefix}: condition or health seed mismatch")
    for key, expected in (
        ("epochs", epochs),
        ("train_batches_per_epoch", train_batches),
        ("validation_batches_per_epoch", validation_batches),
        ("ta_activation_epoch_zero_based", activation),
    ):
        if item.get(key) != expected:
            failures.append(f"{prefix}: {key} mismatch")
    if item.get("learning_metrics_role") != LEARNING_METRICS_ROLE:
        failures.append(f"{prefix}: learning metrics are not descriptive-only")
    rows = item.get("epoch_metrics")
    if not isinstance(rows, list) or [
        row.get("epoch") for row in rows if isinstance(row, Mapping)
    ] != list(range(epochs)):
        failures.append(f"{prefix}: epoch history is incomplete")
        return failures

    expected_states = [condition == "C2" and epoch >= activation for epoch in range(epochs)]
    if [row.get("ta_enabled") for row in rows] != expected_states:
        failures.append(f"{prefix}: TA activation state sequence mismatch")
    for epoch, row in enumerate(rows):
        if not isinstance(row, Mapping):
            failures.append(f"{prefix}: epoch {epoch} is malformed")
            continue
        if row.get("train_batches") != train_batches:
            failures.append(f"{prefix}: epoch {epoch} train-batch count mismatch")
        if row.get("validation_batches") != validation_batches:
            failures.append(f"{prefix}: epoch {epoch} validation-batch count mismatch")
        if row.get("absolute_lr_schedule_match") is not True:
            failures.append(f"{prefix}: epoch {epoch} absolute LR schedule mismatch")
        if not _finite(row.get("expected_base_lr"), positive=True):
            failures.append(f"{prefix}: epoch {epoch} expected base LR is invalid")

    if condition == "C2":
        pre = rows[:activation]
        post = rows[activation:]
        if any(
            not isinstance(row.get("ta_parameter_delta"), Mapping)
            or float(row["ta_parameter_delta"].get("l2", math.nan)) != 0.0
            for row in pre
        ):
            failures.append(f"{prefix}: TA parameters changed before activation")
        if any(int(row.get("ta_optimizer_state_count_after", -1)) != 0 for row in pre):
            failures.append(f"{prefix}: TA optimizer state existed before activation")
        if not post:
            failures.append(f"{prefix}: no post-activation epoch was executed")
        else:
            final_delta = post[-1].get("ta_parameter_delta")
            if not isinstance(final_delta, Mapping) or not _finite(
                final_delta.get("l2"), positive=True
            ):
                failures.append(f"{prefix}: TA parameters did not update after activation")
            first = post[0]
            gradient = first.get("ta_gradient_observation")
            if not isinstance(gradient, Mapping) or not (
                int(gradient.get("parameter_tensor_count", 0)) > 0
                and gradient.get("gradient_present_count")
                == gradient.get("parameter_tensor_count")
                and gradient.get("finite_gradient_count")
                == gradient.get("parameter_tensor_count")
                and int(gradient.get("nonzero_gradient_count", 0)) > 0
            ):
                failures.append(f"{prefix}: TA gradients failed at activation")
            ratio = first.get("ta_lr_over_base_lr")
            if not _finite(ratio, positive=True) or not math.isclose(
                float(ratio), ta_lr_scale, rel_tol=0.0, abs_tol=1e-12
            ):
                failures.append(f"{prefix}: TA/base LR ratio mismatch at activation")
    return failures


def implementation_health_anomalies(
    fixed_by_condition: Mapping[str, Any],
    formal_by_condition: Mapping[str, Any],
    *,
    block: PilotBlock,
    protocol: Mapping[str, Any],
    reference_config: RunConfig,
) -> list[str]:
    """Recompute all and only the frozen v5 implementation-health anomalies."""

    acceptance = _mapping(protocol.get("pilot_acceptance"), "protocol.pilot_acceptance")
    implementation = _mapping(
        acceptance.get("implementation_health"), "implementation_health"
    )
    fixed = _mapping(implementation.get("fixed_batch"), "implementation_health.fixed_batch")
    formal = _mapping(
        implementation.get("formal_schedule"), "implementation_health.formal_schedule"
    )
    checkpoint = _mapping(
        implementation.get("checkpoint_resume"), "implementation_health.checkpoint_resume"
    )
    anomalies: list[str] = []
    if tuple(fixed_by_condition) != block.conditions:
        anomalies.append("fixed_batch: condition order differs from C1,C2")
    if tuple(formal_by_condition) != block.conditions:
        anomalies.append("formal_schedule: condition order differs from C1,C2")
    for condition in block.conditions:
        anomalies.extend(
            _fixed_batch_anomalies(
                fixed_by_condition.get(condition),
                condition=condition,
                health_seed=block.health_seed,
                settings=fixed,
                checkpoint_settings=checkpoint,
            )
        )
        anomalies.extend(
            _formal_schedule_anomalies(
                formal_by_condition.get(condition),
                condition=condition,
                health_seed=block.health_seed,
                settings=formal,
                ta_lr_scale=float(reference_config.optimizer.ta_lr_scale),
            )
        )
    for path in nonfinite_paths(
        {"fixed_batch": fixed_by_condition, "formal_schedule": formal_by_condition}
    ):
        anomalies.append(f"nonfinite observation at {path}")
    return list(dict.fromkeys(anomalies))


def validate_health_report_payload(
    report: Mapping[str, Any],
    *,
    block: PilotBlock,
    protocol: Mapping[str, Any],
    reference_config: RunConfig,
    receipt: Mapping[str, Any],
    receipt_sha256: str,
) -> None:
    """Validate a v5 PASS without applying any learning-performance cutoff."""

    acceptance = _mapping(protocol.get("pilot_acceptance"), "protocol.pilot_acceptance")
    implementation = _mapping(
        acceptance.get("implementation_health"), "pilot_acceptance.implementation_health"
    )
    fixed_contract = _mapping(
        implementation.get("fixed_batch"), "implementation_health.fixed_batch"
    )
    formal_contract = _mapping(
        implementation.get("formal_schedule"), "implementation_health.formal_schedule"
    )
    checkpoint_contract = _mapping(
        implementation.get("checkpoint_resume"), "implementation_health.checkpoint_resume"
    )
    failures: list[str] = []

    def require(result: bool, message: str) -> None:
        if not result:
            failures.append(message)

    expected_top = {
        "schema_version": HEALTH_SCHEMA_VERSION,
        "artifact_class": HEALTH_ARTIFACT_CLASS,
        "reporting_eligibility": REPORTING_ELIGIBILITY,
        "confirmatory_analysis_eligibility": False,
        "status": "PASS",
        "pass": True,
        "thresholds_evaluated": [],
        "decision_basis": HEALTH_DECISION_BASIS,
        "learning_metrics_role": LEARNING_METRICS_ROLE,
        "dataset": block.dataset,
        "seed": block.health_seed,
        "health_seed": block.health_seed,
        "pilot_seed": block.pilot_seed,
        "conditions": list(block.conditions),
        "protocol_hash": block.protocol_hash,
        "acceptance_hash": block.acceptance_hash,
        "development_probe_hash": block.development_probe_hash,
        "block_hash": block.block_hash,
        "git_commit": receipt.get("git_commit"),
        "attempt_receipt_sha256": receipt_sha256,
        "health_seed_disposition": HEALTH_SEED_DISPOSITION,
        "pilot_seed_disposition": PILOT_SEED_DISPOSITION,
        "integrity_anomalies": [],
        "failures": [],
    }
    for key, expected_value in expected_top.items():
        require(report.get(key) == expected_value, f"health report {key} mismatch")
    require(report.get("tracked_clean") is True, "health report tracked_clean must be true")
    require(report.get("fatal_error") in (None, ""), "health report contains a fatal error")

    evidence = report.get("evidence")
    require(isinstance(evidence, Mapping), "health evidence is missing")
    if isinstance(evidence, Mapping):
        require(
            evidence.get("development_probe") == block.development_probe_binding,
            "development-probe evidence binding mismatch",
        )

    source = report.get("source")
    require(isinstance(source, Mapping), "health source evidence is missing")
    if isinstance(source, Mapping):
        expected_source = {
            "dataset": block.dataset,
            "health_seed": block.health_seed,
            "pilot_seed": block.pilot_seed,
            "reference_config_hash": reference_config.config_hash,
            "reference_config_seed": block.pilot_seed,
            "fixed_batch_size": int(fixed_contract["batch_size"]),
        }
        for key, expected_value in expected_source.items():
            require(source.get(key) == expected_value, f"health source {key} mismatch")
        for key in ("split_manifest_sha256", "fixed_batch_sha256"):
            require(_is_sha256(source.get(key)), f"health source {key} is invalid")
        try:
            validate_fixed_batch_shape(
                block.dataset,
                source.get("fixed_batch_shape", ()),
                batch_size=int(fixed_contract["batch_size"]),
                time_steps=reference_config.model.time_steps,
                in_channels=reference_config.model.in_channels,
            )
        except PilotV5Error as exc:
            failures.append(str(exc))
        if block.dataset == "cifar10dvs":
            require(
                _is_sha256(source.get("dvs_index_sha256"))
                and source.get("dvs_index_sha256")
                == source.get("split_source_fingerprint"),
                "DVS index/source identity is invalid",
            )
        runtime_sources = source.get("runtime_sources_sha256")
        require(
            isinstance(runtime_sources, Mapping)
            and runtime_sources == receipt.get("runtime_sources_sha256"),
            "runtime source hash binding mismatch",
        )

    environment_binding = _mapping(
        acceptance.get("environment"), "pilot_acceptance.environment"
    )
    environment = report.get("environment")
    require(isinstance(environment, Mapping), "environment evidence is missing")
    if isinstance(environment, Mapping):
        require(
            environment.get("contract") == dict(environment_binding),
            "environment contract binding mismatch",
        )
        hardware = environment.get("hardware")
        require(isinstance(hardware, Mapping), "hardware evidence is missing")
        if isinstance(hardware, Mapping):
            require(
                str(environment_binding["expected_gpu_substring"]).lower()
                in str(hardware.get("name", "")).lower(),
                "GPU binding mismatch",
            )
        require(
            environment.get("pytorch") == environment_binding.get("pytorch_version"),
            "PyTorch version mismatch",
        )
        require(
            str(environment.get("cuda_version"))
            == str(environment_binding.get("cuda_runtime")),
            "CUDA runtime mismatch",
        )
        for key, expected_value in (
            ("precision", "float32"),
            ("amp", False),
            ("torch_deterministic", True),
            ("torch_deterministic_warn_only", False),
            ("cudnn_benchmark", False),
            ("cudnn_deterministic", True),
        ):
            require(environment.get(key) == expected_value, f"environment {key} mismatch")
        idle = environment.get("gpu_idle_precheck")
        require(
            isinstance(idle, Mapping) and idle.get("pass") is True,
            "GPU idle precheck did not pass",
        )

    timing = report.get("timing")
    require(isinstance(timing, Mapping), "timing binding is missing")
    if isinstance(timing, Mapping):
        require(
            timing.get("contract") == acceptance.get("timing"),
            "timing contract binding mismatch",
        )
        require(timing.get("health_role") == TIMING_HEALTH_ROLE, "timing health role mismatch")
        require(timing.get("evaluated") is False, "timing must not determine v5 health")
    schedule = report.get("schedule")
    require(isinstance(schedule, Mapping), "schedule binding is missing")
    if isinstance(schedule, Mapping):
        require(
            schedule.get("contract") == acceptance.get("schedule"),
            "schedule contract binding mismatch",
        )
        require(
            schedule.get("health_role") == SCHEDULE_HEALTH_ROLE,
            "schedule health role mismatch",
        )
        require(schedule.get("evaluated") is False, "pilot schedule must not determine health")

    observed = report.get("implementation_health")
    require(isinstance(observed, Mapping), "implementation-health evidence is missing")
    if isinstance(observed, Mapping):
        expected_implementation = {
            "decision_basis": HEALTH_DECISION_BASIS,
            "thresholds_evaluated": [],
            "learning_metrics_role": LEARNING_METRICS_ROLE,
            "integrity_anomalies": [],
        }
        for key, expected_value in expected_implementation.items():
            require(
                observed.get(key) == expected_value,
                f"implementation health {key} mismatch",
            )
        fixed = observed.get("fixed_batch")
        formal = observed.get("formal_schedule")
        checkpoint = observed.get("checkpoint_resume")
        require(isinstance(fixed, Mapping), "fixed-batch health evidence is missing")
        require(isinstance(formal, Mapping), "formal-schedule health evidence is missing")
        require(isinstance(checkpoint, Mapping), "checkpoint-resume evidence is missing")
        fixed_results: Mapping[str, Any] = {}
        formal_results: Mapping[str, Any] = {}
        if isinstance(fixed, Mapping):
            require(fixed.get("contract") == dict(fixed_contract), "fixed-batch contract mismatch")
            candidate = fixed.get("by_condition")
            require(isinstance(candidate, Mapping), "fixed-batch condition evidence is missing")
            if isinstance(candidate, Mapping):
                fixed_results = candidate
        if isinstance(formal, Mapping):
            require(
                formal.get("contract") == dict(formal_contract),
                "formal-schedule contract mismatch",
            )
            candidate = formal.get("by_condition")
            require(isinstance(candidate, Mapping), "formal condition evidence is missing")
            if isinstance(candidate, Mapping):
                formal_results = candidate
        if isinstance(checkpoint, Mapping):
            require(
                checkpoint.get("contract") == dict(checkpoint_contract),
                "checkpoint-resume contract mismatch",
            )
            expected_checkpoints = {
                condition: fixed_results.get(condition, {}).get("checkpoint_boundary")
                if isinstance(fixed_results.get(condition), Mapping)
                else None
                for condition in block.conditions
            }
            require(
                checkpoint.get("by_condition") == expected_checkpoints,
                "checkpoint-resume condition binding mismatch",
            )
        if fixed_results and formal_results:
            recomputed = implementation_health_anomalies(
                fixed_results,
                formal_results,
                block=block,
                protocol=protocol,
                reference_config=reference_config,
            )
            require(recomputed == [], "implementation anomalies recompute as: " + "; ".join(recomputed))

    if failures:
        raise PilotV5Error("V5 pilot health report rejected:\n- " + "\n- ".join(failures))


def validate_health_report(
    report_path: str | Path,
    protocol_path: str | Path,
    dataset: str,
    *,
    repository_root: str | Path,
    expected_git_commit: str | None = None,
    require_current_tracked_clean: bool = True,
) -> dict[str, Any]:
    from .config import load_protocol

    root = Path(repository_root).resolve()
    protocol_file = Path(protocol_path).resolve()
    protocol = load_protocol(protocol_file)
    block = resolve_pilot_block(protocol, dataset, repository_root=root)
    path = Path(report_path).resolve()
    if path != block.health_output or not path.is_file():
        raise PilotV5Error("The canonical dataset-specific v5 health report is missing")
    validate_development_probe_evidence(protocol, repository_root=root)
    reference = expected_pilot_configs(protocol, block.dataset)[0]
    receipt = validate_attempt_receipt(
        block.attempt_receipt,
        block=block,
        protocol_path=protocol_file,
        repository_root=root,
        reference_config=reference,
    )
    report = _read_json(path, "v5 health report")
    validate_health_report_payload(
        report,
        block=block,
        protocol=protocol,
        reference_config=reference,
        receipt=receipt,
        receipt_sha256=sha256_file(block.attempt_receipt),
    )
    if expected_git_commit is None or require_current_tracked_clean:
        identity = repository_git_identity(root)
        if expected_git_commit is None:
            expected_git_commit = str(identity["git_commit"])
        if require_current_tracked_clean and identity["tracked_clean"] is not True:
            raise PilotV5Error("Current tracked worktree is not clean")
    if report.get("git_commit") != expected_git_commit:
        raise PilotV5Error("Health report Git commit differs from the required commit")
    return report


def expected_pilot_plan_payload(
    *,
    block: PilotBlock,
    protocol_path: str | Path,
    config_dir: str | Path,
    manifest_rows: Sequence[Mapping[str, Any]],
    health_report: Mapping[str, Any],
    repository_root: str | Path,
) -> dict[str, Any]:
    root = Path(repository_root).resolve()
    config_root = Path(config_dir).resolve()
    selected = [row for row in manifest_rows if str(row.get("dataset")) == block.dataset]
    if len(selected) != 2 or [str(row.get("condition")) for row in selected] != ["C1", "C2"]:
        raise PilotV5Error("A v5 pilot plan requires one ordered C1/C2 dataset block")
    if any(int(row.get("seed", -1)) != block.pilot_seed for row in selected):
        raise PilotV5Error("A v5 pilot plan may contain only the dataset pilot seed")
    runs: list[dict[str, Any]] = []
    for row in selected:
        config_path = (config_root / str(row.get("config_file", ""))).resolve()
        if config_path.parent != config_root or not config_path.is_file():
            raise PilotV5Error(f"Pilot config is missing or escapes its directory: {config_path}")
        runs.append(
            {
                "condition": str(row["condition"]),
                "run_id": str(row["run_id"]),
                "config_file": artifact_path_reference(config_path, root),
                "config_file_sha256": sha256_file(config_path),
                "config_hash": str(row["config_hash"]),
                "seed": block.pilot_seed,
            }
        )
    return {
        "version": 5,
        "artifact_class": "NON_REPORTABLE_V5_PILOT_PLAN",
        "non_reportable": True,
        "dataset": block.dataset,
        "seed": block.pilot_seed,
        "health_seed": block.health_seed,
        "pilot_seed": block.pilot_seed,
        "conditions": list(block.conditions),
        "protocol_path": artifact_path_reference(protocol_path, root),
        "protocol_hash": block.protocol_hash,
        "acceptance_hash": block.acceptance_hash,
        "development_probe_hash": block.development_probe_hash,
        "block_hash": block.block_hash,
        "git_commit": health_report.get("git_commit"),
        "health_report": artifact_path_reference(block.health_output, root),
        "health_report_sha256": sha256_file(block.health_output),
        "attempt_receipt": artifact_path_reference(block.attempt_receipt, root),
        "attempt_receipt_sha256": sha256_file(block.attempt_receipt),
        "pilot_output_root": artifact_path_reference(block.pilot_output_root, root),
        "formal_output_root": artifact_path_reference(block.formal_output_root, root),
        "evidence": {"development_probe": dict(block.development_probe_binding)},
        "thresholds_evaluated_by_health": [],
        "health_seed_disposition": HEALTH_SEED_DISPOSITION,
        "pilot_seed_disposition": "CONSUMED_BY_THIS_NONREPORTABLE_V5_PILOT_BLOCK",
        "runs": runs,
    }
