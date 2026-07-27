from __future__ import annotations

import hashlib
import csv
import json
import sys
from pathlib import Path

import pytest
import torch
import yaml

pytest.importorskip("torch")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

V2R1_PROTOCOL = ROOT / "configs" / "protocol_v2_pilot.yaml"
V2R1_CONFIG_DIR_NAME = "v2_pilot_generated"
V2R2_PROTOCOL = ROOT / "configs" / "protocol_v2r2_seed88_of80_e120.yaml"
V2R2_CONFIG_DIR_NAME = "v2r2_seed88_of80_e120_generated"

import generate_run_configs  # noqa: E402
import validate_v2_pilot as validator  # noqa: E402
from talif_msresnet.config import load_protocol, load_run_config  # noqa: E402


def _environment_identity() -> tuple[str, str]:
    identity = {
        "device": "cuda:0",
        "hardware": {
            "name": "NVIDIA GeForce RTX 5090",
            "compute_capability": "12.0",
            "total_memory_bytes": 33679736832,
            "multiprocessor_count": 170,
        },
        "software": {
            "platform": "Linux-test",
            "python": "3.11.12",
            "pytorch": "2.9.1+cu128",
            "numpy": "2.0.0",
            "cuda_version": "12.8",
            "cudnn_version": 91002,
        },
        "precision": "float32",
        "determinism": {
            "requested": True,
            "torch_algorithms_enabled": True,
            "torch_warn_only": False,
            "cudnn_benchmark": False,
            "cudnn_deterministic": True,
            "cublas_workspace_config": ":4096:8",
        },
    }
    encoded = json.dumps(identity, sort_keys=True, separators=(",", ":"))
    digest = hashlib.sha256(encoded.encode("utf-8")).hexdigest()
    return encoded, digest


def _pilot_tree(
    tmp_path: Path,
    *,
    protocol_path: Path = V2R1_PROTOCOL,
    config_dir_name: str = V2R1_CONFIG_DIR_NAME,
    best_accuracy_by_condition: dict[str, float] | None = None,
) -> tuple[Path, Path, Path]:
    protocol = load_protocol(protocol_path)
    acceptance = protocol["pilot_acceptance"]
    pilot_seed = int(acceptance["seed"])
    config_dir = tmp_path / "configs" / config_dir_name
    generate_run_configs.generate(protocol_path, config_dir)
    manifest = json.loads((config_dir / "matrix_manifest.json").read_text(encoding="utf-8"))
    results_root = tmp_path / protocol["pilot_acceptance"]["pilot_output_root"]
    results_root.mkdir(parents=True)
    environment, environment_hash = _environment_identity()
    identity = json.loads(environment)
    health_hardware = {"device": identity["device"], **identity["hardware"]}
    shared_hash = "a" * 64
    split_hash = "b" * 64
    consolidated: list[dict[str, object]] = []
    matrix_events: list[str] = []
    health_path = tmp_path / acceptance["health_output"]
    health_path.parent.mkdir(parents=True, exist_ok=True)
    health_report = {
        "status": "PASS",
        "pass": True,
        "protocol_hash": manifest["protocol_hash"],
        "acceptance_hash": validator.stable_hash(acceptance),
        "git_commit": "c" * 40,
        "tracked_clean": True,
        "reporting_eligibility": "FORBIDDEN_FROM_MANUSCRIPT_RESULTS",
        "confirmatory_analysis_eligibility": False,
        "source": {
            "split_manifest_sha256": split_hash,
            "fixed_batch_sha256": "d" * 64,
            "fixed_batch_shape": [64, 3, 32, 32],
        },
        "environment": {
            "hardware": health_hardware,
            "platform": identity["software"]["platform"],
            "python": identity["software"]["python"],
            "pytorch": identity["software"]["pytorch"],
            "numpy": identity["software"]["numpy"],
            "cuda_version": identity["software"]["cuda_version"],
            "gpu_idle_precheck": {"device_uuid": "GPU-test"},
        },
    }
    health_path.write_text(json.dumps(health_report), encoding="utf-8")
    plan_path = tmp_path / acceptance["pilot_plan"]
    plan_path.parent.mkdir(parents=True, exist_ok=True)
    plan_runs = [
        {
            "condition": row["condition"],
            "run_id": row["run_id"],
            "config_file": validator.artifact_path_reference(
                config_dir / row["config_file"], tmp_path
            ),
            "config_file_sha256": row["config_file_sha256"],
            "config_hash": row["config_hash"],
        }
        for row in manifest["runs"]
    ]
    plan = {
        "version": 2,
        "non_reportable": True,
        "protocol_path": validator.artifact_path_reference(protocol_path, tmp_path),
        "protocol_hash": manifest["protocol_hash"],
        "acceptance_hash": validator.stable_hash(acceptance),
        "git_commit": health_report["git_commit"],
        "health_report": validator.artifact_path_reference(health_path, tmp_path),
        "health_report_sha256": validator.sha256_file(health_path),
        "pilot_output_root": validator.artifact_path_reference(results_root, tmp_path),
        "formal_output_root": validator.artifact_path_reference(
            tmp_path / protocol["output_root"], tmp_path
        ),
        "runs": plan_runs,
    }
    plan_path.write_text(json.dumps(plan), encoding="utf-8")
    orchestration_evidence = {
        "pilot_plan": validator.artifact_path_reference(plan_path, tmp_path),
        "pilot_plan_sha256": validator.sha256_file(plan_path),
        "plan_version": 2,
        "protocol_hash": manifest["protocol_hash"],
        "acceptance_hash": validator.stable_hash(acceptance),
        "git_commit": health_report["git_commit"],
        "health_report": validator.artifact_path_reference(health_path, tmp_path),
        "health_report_sha256": validator.sha256_file(health_path),
    }
    launch_context = {
        "pass": True,
        "device_uuid": "GPU-test",
        "hardware": health_hardware,
        "split_manifest_sha256": split_hash,
        "fixed_batch_sha256": "d" * 64,
        "fixed_batch_shape": [64, 3, 32, 32],
    }
    matrix_events.append(
        json.dumps({"event": "pilot_context_validated", "context": launch_context})
    )

    for row in manifest["runs"]:
        condition = row["condition"]
        run_id = row["run_id"]
        config = load_run_config(config_dir / row["config_file"], protocol_path)
        run_dir = results_root / run_id
        run_dir.mkdir()
        resolved = config.as_dict()
        resolved["runtime"]["output_dir"] = str(results_root)
        target_accuracy = (best_accuracy_by_condition or {}).get(condition, 0.65)
        best_loss = 1.0 - 0.001 * 119
        metrics = {
            "run_id": run_id,
            "condition": condition,
            "seed": pilot_seed,
            "dataset": "cifar100",
            "config_hash": row["config_hash"],
            "protocol_hash": manifest["protocol_hash"],
            "status": "complete",
            "failed": 0,
            "best_epoch": 120,
            "best_val_accuracy": target_accuracy,
            "best_val_loss": best_loss,
            "converged": int(target_accuracy >= 0.60),
            "convergence_epoch": 1 if target_accuracy >= 0.60 else "",
            "training_environment_identity": environment,
            "training_environment_sha256": environment_hash,
            "shared_weight_sha256": shared_hash,
            "split_manifest_sha256": split_hash,
        }
        consolidated.append(metrics)
        run_manifest = {
            "run_id": run_id,
            "status": "complete",
            "config_hash": row["config_hash"],
            "shared_weight_sha256": shared_hash,
            "split_manifest_sha256": split_hash,
            "ta_activation_epoch_zero_based": 6,
            "ta_parameter_names": [] if condition in {"C1", "C3"} else ["n.center"],
            "environment": {"training_environment_sha256": environment_hash},
            "orchestrator_evidence": orchestration_evidence,
        }
        events = [
            json.dumps(
                {
                    "event": "run_started",
                    "orchestrator_evidence": orchestration_evidence,
                }
            )
        ]
        history = []
        for epoch in range(120):
            ta_enabled = condition in {"C2", "C4"} and epoch >= 6
            seconds = 2.0 if ta_enabled else 1.0
            val_loss = 1.0 - 0.001 * epoch
            train = {"loss": 2.0 - 0.001 * epoch, "accuracy": 0.5, "seconds": seconds}
            val = {"loss": val_loss, "accuracy": target_accuracy}
            best_val = {"loss": val_loss, "accuracy": target_accuracy, "epoch": epoch}
            history.append(
                {
                    **train,
                    "epoch": epoch,
                    "val_accuracy": target_accuracy,
                    "val_loss": val_loss,
                }
            )
            events.append(
                json.dumps(
                    {
                        "event": "epoch_completed",
                        "epoch": epoch,
                        "ta_enabled": ta_enabled,
                        "train": train,
                        "val": val,
                        "best_val": best_val,
                    }
                )
            )
        (run_dir / "seed_metrics.json").write_text(json.dumps(metrics), encoding="utf-8")
        (run_dir / "run_manifest.json").write_text(json.dumps(run_manifest), encoding="utf-8")
        (run_dir / "resolved_config.json").write_text(json.dumps(resolved), encoding="utf-8")
        (run_dir / "events.jsonl").write_text("\n".join(events) + "\n", encoding="utf-8")
        checkpoint = {
            "model_state": {"weight": torch.tensor([1.0])},
            "optimizer_state": {"state": {}, "param_groups": []},
            "scheduler_state": {"last_epoch": 120},
            "epoch": 119,
            "best_val": {
                "accuracy": target_accuracy,
                "loss": best_loss,
                "epoch": 119,
            },
            "config": resolved,
            "config_hash": row["config_hash"],
            "execution_hash": config.execution_hash,
            "train_history": history,
            "training_environment_sha256": environment_hash,
        }
        torch.save(checkpoint, run_dir / "best.pt")
        torch.save(checkpoint, run_dir / "last.pt")
        (run_dir / "attempt_001.stdout.log").write_text("done\n", encoding="utf-8")
        (run_dir / "attempt_001.stderr.log").write_text("", encoding="utf-8")
        matrix_events.append(
            json.dumps(
                {
                    "event": "run_context_validated",
                    "run_id": run_id,
                    "context": launch_context,
                }
            )
        )
        matrix_events.append(
            json.dumps(
                {
                    "event": "run_finished",
                    "run_id": run_id,
                    "returncode": 0,
                    "action": "fresh",
                    "attempt": 1,
                }
            )
        )
    with (results_root / "seed_metrics.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        fields = list(consolidated[0])
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(consolidated)
    (results_root / "matrix_events.jsonl").write_text(
        "\n".join(matrix_events) + "\n", encoding="utf-8"
    )
    return protocol_path, config_dir, results_root


def test_complete_120_epoch_pilot_passes(tmp_path: Path) -> None:
    protocol, configs, results = _pilot_tree(tmp_path)

    report = validator.validate_pilot(
        protocol_path=protocol,
        config_dir=configs,
        results_root=results,
        repository_root=tmp_path,
        strict_health_validation=False,
    )

    assert report["status"] == "PASS"
    assert report["decision"] == "ACCEPT_120_EPOCH_SCHEDULE_FOR_FORMAL_V2_DESIGN"
    assert not report["integrity_failures"]
    assert not report["threshold_failures"]
    assert report["runs"]["C2"]["timing"]["ratio"] == 2.0
    assert report["runs"]["C1"]["timing"]["status"] == "NOT_APPLICABLE_LIF"


def test_v2r2_seed88_pilot_passes_with_its_bound_artifact_paths(
    tmp_path: Path,
) -> None:
    protocol_path, configs, results = _pilot_tree(
        tmp_path,
        protocol_path=V2R2_PROTOCOL,
        config_dir_name=V2R2_CONFIG_DIR_NAME,
    )
    protocol = load_protocol(protocol_path)
    acceptance = protocol["pilot_acceptance"]

    report = validator.validate_pilot(
        protocol_path=protocol_path,
        config_dir=configs,
        results_root=results,
        repository_root=tmp_path,
        strict_health_validation=False,
    )

    assert acceptance["seed"] == 88
    assert acceptance["overfit"]["steps"] == 80
    assert configs.name == V2R2_CONFIG_DIR_NAME
    assert report["status"] == "PASS"
    assert report["protocol_path"] == str(V2R2_PROTOCOL.resolve())
    assert report["results_root"] == str((tmp_path / acceptance["pilot_output_root"]).resolve())
    assert report["health_report_path"] == str((tmp_path / acceptance["health_output"]).resolve())
    assert report["pilot_plan_path"] == str((tmp_path / acceptance["pilot_plan"]).resolve())
    assert {condition: run["run_id"] for condition, run in report["runs"].items()} == {
        condition: f"E1_cifar100_d20_t6_{condition}_s88" for condition in ("C1", "C2", "C3", "C4")
    }


def test_v2r2_validator_rejects_seed77_output_root(tmp_path: Path) -> None:
    _old_protocol, _old_configs, old_results = _pilot_tree(tmp_path)
    new_protocol, new_configs, _new_results = _pilot_tree(
        tmp_path,
        protocol_path=V2R2_PROTOCOL,
        config_dir_name=V2R2_CONFIG_DIR_NAME,
    )

    with pytest.raises(validator.PilotValidationError, match="Results root differs"):
        validator.validate_pilot(
            protocol_path=new_protocol,
            config_dir=new_configs,
            results_root=old_results,
            repository_root=tmp_path,
            strict_health_validation=False,
        )


@pytest.mark.parametrize(
    ("acceptance_path_key", "failure_fragment"),
    (
        ("health_output", "pilot health report protocol_hash"),
        ("pilot_plan", "pilot plan protocol_path differs"),
    ),
)
def test_v2r2_validator_rejects_seed77_bound_evidence(
    tmp_path: Path,
    acceptance_path_key: str,
    failure_fragment: str,
) -> None:
    old_protocol, _old_configs, _old_results = _pilot_tree(tmp_path)
    new_protocol, new_configs, new_results = _pilot_tree(
        tmp_path,
        protocol_path=V2R2_PROTOCOL,
        config_dir_name=V2R2_CONFIG_DIR_NAME,
    )
    old_acceptance = load_protocol(old_protocol)["pilot_acceptance"]
    new_acceptance = load_protocol(new_protocol)["pilot_acceptance"]
    old_artifact = tmp_path / old_acceptance[acceptance_path_key]
    new_artifact = tmp_path / new_acceptance[acceptance_path_key]
    new_artifact.write_bytes(old_artifact.read_bytes())

    report = validator.validate_pilot(
        protocol_path=new_protocol,
        config_dir=new_configs,
        results_root=new_results,
        repository_root=tmp_path,
        strict_health_validation=False,
    )

    assert report["status"] == "INVALID"
    assert any(failure_fragment in failure for failure in report["integrity_failures"])


def test_accuracy_failure_requires_fresh_160_epoch_protocol(tmp_path: Path) -> None:
    protocol, configs, results = _pilot_tree(
        tmp_path, best_accuracy_by_condition={"C4": 0.59}
    )

    report = validator.validate_pilot(
        protocol_path=protocol,
        config_dir=configs,
        results_root=results,
        repository_root=tmp_path,
        strict_health_validation=False,
    )

    assert report["status"] == "FAIL"
    assert report["decision"] == "REQUIRE_FRESH_160_EPOCH_PILOT"
    assert report["fallback"]["resume_from_120_epoch_pilot"] is False


def test_slow_ta_epoch_ratio_requires_fresh_160_epoch_protocol(tmp_path: Path) -> None:
    protocol, configs, results = _pilot_tree(tmp_path)
    target = results / "E1_cifar100_d20_t6_C2_s77" / "events.jsonl"
    events = [json.loads(line) for line in target.read_text(encoding="utf-8").splitlines()]
    for event in events:
        if event.get("ta_enabled"):
            event["train"]["seconds"] = 3.1
    target.write_text("\n".join(json.dumps(item) for item in events) + "\n", encoding="utf-8")
    for checkpoint_name in ("best.pt", "last.pt"):
        checkpoint_path = target.parent / checkpoint_name
        checkpoint = torch.load(checkpoint_path, weights_only=False)
        for row in checkpoint["train_history"]:
            if row["epoch"] >= 6:
                row["seconds"] = 3.1
        torch.save(checkpoint, checkpoint_path)

    report = validator.validate_pilot(
        protocol_path=protocol,
        config_dir=configs,
        results_root=results,
        repository_root=tmp_path,
        strict_health_validation=False,
    )

    assert report["status"] == "FAIL"
    assert any("ratio" in item for item in report["threshold_failures"])


def test_mixed_training_environment_invalidates_pilot(tmp_path: Path) -> None:
    protocol, configs, results = _pilot_tree(tmp_path)
    target = results / "E1_cifar100_d20_t6_C3_s77" / "seed_metrics.json"
    metrics = json.loads(target.read_text(encoding="utf-8"))
    identity = json.loads(metrics["training_environment_identity"])
    identity["software"]["platform"] = "Linux-other"
    encoded = json.dumps(identity, sort_keys=True, separators=(",", ":"))
    metrics["training_environment_identity"] = encoded
    metrics["training_environment_sha256"] = hashlib.sha256(encoded.encode("utf-8")).hexdigest()
    target.write_text(json.dumps(metrics), encoding="utf-8")

    report = validator.validate_pilot(
        protocol_path=protocol,
        config_dir=configs,
        results_root=results,
        repository_root=tmp_path,
        strict_health_validation=False,
    )

    assert report["status"] == "INVALID"
    assert report["decision"] == "BLOCK_FORMAL_V2_AND_INVESTIGATE_PILOT_INTEGRITY"
    assert any("not identical" in item for item in report["integrity_failures"])


def test_protocol_valid_but_noncanonical_generated_yaml_is_invalid(tmp_path: Path) -> None:
    protocol, configs, results = _pilot_tree(tmp_path)
    manifest_path = configs / "matrix_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    target = configs / manifest["runs"][0]["config_file"]
    config = yaml.safe_load(target.read_text(encoding="utf-8"))
    config["runtime"]["log_every"] = 99
    target.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    manifest["runs"][0]["config_file_sha256"] = validator.sha256_file(target)
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    report = validator.validate_pilot(
        protocol_path=protocol,
        config_dir=configs,
        results_root=results,
        repository_root=tmp_path,
        strict_health_validation=False,
    )

    assert report["status"] == "INVALID"
    assert any("in-memory protocol matrix" in item for item in report["integrity_failures"])


def test_terminal_summary_cannot_override_epoch_evidence(tmp_path: Path) -> None:
    protocol, configs, results = _pilot_tree(tmp_path)
    target = results / "E1_cifar100_d20_t6_C4_s77" / "seed_metrics.json"
    metrics = json.loads(target.read_text(encoding="utf-8"))
    metrics["best_val_accuracy"] = 0.99
    target.write_text(json.dumps(metrics), encoding="utf-8")

    report = validator.validate_pilot(
        protocol_path=protocol,
        config_dir=configs,
        results_root=results,
        repository_root=tmp_path,
        strict_health_validation=False,
    )

    assert report["status"] == "INVALID"
    assert any("best_val_accuracy" in item for item in report["integrity_failures"])


def test_tampered_checkpoint_is_invalid(tmp_path: Path) -> None:
    protocol, configs, results = _pilot_tree(tmp_path)
    target = results / "E1_cifar100_d20_t6_C3_s77" / "last.pt"
    checkpoint = torch.load(target, weights_only=False)
    checkpoint["epoch"] = 118
    torch.save(checkpoint, target)

    report = validator.validate_pilot(
        protocol_path=protocol,
        config_dir=configs,
        results_root=results,
        repository_root=tmp_path,
        strict_health_validation=False,
    )

    assert report["status"] == "INVALID"
    assert any("last.pt epoch" in item for item in report["integrity_failures"])


def test_same_environment_resume_after_failure_can_pass(tmp_path: Path) -> None:
    protocol, configs, results = _pilot_tree(tmp_path)
    run_id = "E1_cifar100_d20_t6_C2_s77"
    run_dir = results / run_id
    manifest = json.loads((run_dir / "run_manifest.json").read_text(encoding="utf-8"))
    environment_hash = manifest["environment"]["training_environment_sha256"]
    manifest["resume_history"] = [
        {
            "resumed_at": "2026-07-27T00:00:00Z",
            "checkpoint": "last.pt",
            "training_environment_sha256": environment_hash,
        }
    ]
    (run_dir / "run_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    config_hash = manifest["config_hash"]
    (run_dir / "failure.json").write_text(
        json.dumps({"run_id": run_id, "config_hash": config_hash, "epoch": 50}),
        encoding="utf-8",
    )
    (run_dir / "attempt_002.stdout.log").write_text("resumed\n", encoding="utf-8")
    (run_dir / "attempt_002.stderr.log").write_text("", encoding="utf-8")
    trainer_events_path = run_dir / "events.jsonl"
    trainer_events = [
        json.loads(line)
        for line in trainer_events_path.read_text(encoding="utf-8").splitlines()
    ]
    epoch_51_index = next(
        index
        for index, event in enumerate(trainer_events)
        if event.get("event") == "epoch_completed" and event.get("epoch") == 51
    )
    abandoned_epoch = dict(trainer_events[epoch_51_index])
    trainer_events[epoch_51_index:epoch_51_index] = [
        abandoned_epoch,
        {
            "event": "run_started",
            "orchestrator_evidence": manifest["orchestrator_evidence"],
        },
        {"event": "resumed", "checkpoint": str(run_dir / "last.pt"), "start_epoch": 51},
    ]
    trainer_events_path.write_text(
        "\n".join(json.dumps(item) for item in trainer_events) + "\n",
        encoding="utf-8",
    )
    path = results / "matrix_events.jsonl"
    events = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    success_index = next(
        index
        for index, event in enumerate(events)
        if event.get("run_id") == run_id and event.get("event") == "run_finished"
    )
    events[success_index]["action"] = "resume"
    events[success_index]["attempt"] = 2
    events.insert(
        success_index,
        {
            "event": "run_failed",
            "run_id": run_id,
            "returncode": 1,
            "action": "fresh",
            "attempt": 1,
        },
    )
    path.write_text("\n".join(json.dumps(item) for item in events) + "\n", encoding="utf-8")

    report = validator.validate_pilot(
        protocol_path=protocol,
        config_dir=configs,
        results_root=results,
        repository_root=tmp_path,
        strict_health_validation=False,
    )

    assert report["status"] == "PASS"
    assert report["runs"]["C2"]["historical_failure"]["epoch"] == 50


def test_unresolved_historical_failure_is_invalid(tmp_path: Path) -> None:
    protocol, configs, results = _pilot_tree(tmp_path)
    run_id = "E1_cifar100_d20_t6_C2_s77"
    run_dir = results / run_id
    manifest = json.loads((run_dir / "run_manifest.json").read_text(encoding="utf-8"))
    (run_dir / "failure.json").write_text(
        json.dumps({"run_id": run_id, "config_hash": manifest["config_hash"]}),
        encoding="utf-8",
    )
    path = results / "matrix_events.jsonl"
    with path.open("a", encoding="utf-8") as handle:
        handle.write(
            json.dumps(
                {
                    "event": "run_failed",
                    "run_id": run_id,
                    "returncode": 1,
                    "action": "resume",
                    "attempt": 2,
                }
            )
            + "\n"
        )

    report = validator.validate_pilot(
        protocol_path=protocol,
        config_dir=configs,
        results_root=results,
        repository_root=tmp_path,
        strict_health_validation=False,
    )

    assert report["status"] == "INVALID"
    assert any("unresolved matrix failure" in item for item in report["integrity_failures"])
