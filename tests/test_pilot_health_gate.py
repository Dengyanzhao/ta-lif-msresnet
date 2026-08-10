"""CPU tests for the non-reporting CUDA pilot health gate."""

from __future__ import annotations

import dataclasses
import json
import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import pilot_health_gate as gate  # noqa: E402
from talif_msresnet.config import (  # noqa: E402
    DataConfig,
    ModelConfig,
    OptimizerConfig,
    RunConfig,
    RuntimeConfig,
)
from talif_msresnet.models import build_model  # noqa: E402


def _tiny_config(condition: str = "C1") -> RunConfig:
    topology, neuron = gate.CONDITIONS[condition]
    return RunConfig(
        protocol_version=1,
        experiment="NONREPORTING_TEST",
        data=DataConfig(dataset="cifar100", num_classes=4, in_channels=3),
        model=ModelConfig(
            condition=condition,
            topology=topology,
            neuron=neuron,
            depth=20,
            time_steps=2,
            num_classes=4,
            in_channels=3,
            base_channels=2,
            neuron_cfg={
                "tau": 0.5,
                "threshold": 1.0,
                "width": 1.0,
                "delta_min": 0.05,
            },
        ),
        optimizer=OptimizerConfig(
            epochs=2,
            batch_size=2,
            lr=0.1,
            milestones=(100,),
        ),
        runtime=RuntimeConfig(
            seed=77,
            device="cpu",
            output_dir="results/nonreporting-tests",
            run_id="NONREPORTING_TEST",
            log_every=100,
            deterministic=False,
            dry_run=False,
            limit_batches=1,
        ),
    )


def test_output_path_rejects_formal_tree_and_existing_file(tmp_path: Path) -> None:
    formal = tmp_path / "results" / "runs"
    with pytest.raises(gate.HealthGateError, match="formal root"):
        gate.validate_output_path(
            formal / "health.json",
            formal_roots=[formal],
            repository_root=tmp_path,
        )

    existing = tmp_path / "results" / "nonreporting" / "health.json"
    existing.parent.mkdir(parents=True)
    existing.write_text("{}", encoding="utf-8")
    with pytest.raises(FileExistsError, match="overwrite"):
        gate.validate_output_path(
            existing,
            formal_roots=[formal],
            repository_root=tmp_path,
        )


def test_output_path_allows_new_nonreporting_destination(tmp_path: Path) -> None:
    output = tmp_path / "results" / "nonreporting" / "health.json"
    resolved = gate.validate_output_path(
        output,
        formal_roots=[tmp_path / "results" / "runs"],
        repository_root=tmp_path,
    )
    assert resolved == output.resolve()


def test_current_launch_context_rechecks_device_environment_and_data(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    protocol = gate.load_protocol(ROOT / "configs" / "protocol_v2_pilot.yaml")
    monkeypatch.setenv("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    gate.seed_everything(77, deterministic=True)
    hardware = {
        "device": "cuda:0",
        "name": "NVIDIA GeForce RTX 5090",
        "compute_capability": "12.0",
        "total_memory_bytes": 32,
        "multiprocessor_count": 170,
    }
    idle = {"pass": True, "device_uuid": "GPU-current"}
    inputs = torch.arange(64 * 3 * 2 * 2, dtype=torch.float32).reshape(64, 3, 2, 2)
    targets = torch.arange(64, dtype=torch.long)
    batch_hash = gate.tensor_batch_sha256(inputs, targets)
    current = {
        **gate.environment_manifest(),
        "hardware": hardware,
        "precision": "float32",
        "amp": False,
        "cublas_workspace_config": ":4096:8",
        "torch_deterministic": True,
        "torch_deterministic_warn_only": False,
        "cudnn_benchmark": False,
        "cudnn_deterministic": True,
        "gpu_idle_precheck": idle,
    }
    report = {
        "environment": current,
        "source": {
            "split_manifest_sha256": "a" * 64,
            "fixed_batch_sha256": batch_hash,
            "fixed_batch_shape": list(inputs.shape),
        },
    }
    monkeypatch.setattr(gate, "require_target_cuda", lambda *_args, **_kwargs: hardware)
    monkeypatch.setattr(gate, "validate_runtime_environment", lambda *_args: None)
    monkeypatch.setattr(gate, "gpu_idle_precheck", lambda *_args, **_kwargs: idle)
    monkeypatch.setattr(
        gate,
        "load_fixed_real_batch",
        lambda *_args, **_kwargs: (
            _tiny_config(),
            inputs,
            targets,
            {"manifest_sha256": "a" * 64},
        ),
    )

    evidence = gate.validate_current_runtime_against_health_report(
        report,
        protocol=protocol,
        reference_config=_tiny_config(),
        device="cuda:0",
    )
    assert evidence["pass"] is True
    assert evidence["device_uuid"] == "GPU-current"
    assert evidence["fixed_batch_sha256"] == batch_hash

    report["source"]["fixed_batch_sha256"] = "b" * 64
    with pytest.raises(gate.HealthGateError, match="fixed CIFAR-100 batch"):
        gate.validate_current_runtime_against_health_report(
            report,
            protocol=protocol,
            reference_config=_tiny_config(),
            device="cuda:0",
        )


def test_v2_protocol_binds_health_thresholds_and_canonical_output() -> None:
    output = ROOT / "results" / "pilot" / "v2_seed77_e120_health.json"
    args = gate.build_parser().parse_args(["--output", str(output)])
    config = gate.load_run_config(args.config, args.protocol)
    protocol = gate.load_protocol(args.protocol)

    gate.validate_protocol_health_binding(args, config, protocol, output)

    args.max_ta_step_ratio = 3.01
    with pytest.raises(gate.HealthGateError, match="differ from protocol"):
        gate.validate_protocol_health_binding(args, config, protocol, output)

    args.max_ta_step_ratio = 3.0
    tampered = dataclasses.replace(
        config,
        model=dataclasses.replace(config.model, base_channels=2),
    )
    with pytest.raises(gate.HealthGateError, match="protocol-generated"):
        gate.validate_protocol_health_binding(args, tampered, protocol, output)

    args.expected_gpu_substring = ""
    with pytest.raises(gate.HealthGateError, match="expected-gpu-substring"):
        gate.validate_protocol_health_binding(args, config, protocol, output)


def test_v2r2_protocol_requires_explicit_seed_and_overfit_step_bindings() -> None:
    protocol_path = ROOT / "configs" / "protocol_v2r2_seed88_of80_e120.yaml"
    protocol = gate.load_protocol(protocol_path)
    acceptance = protocol["pilot_acceptance"]
    assert acceptance["seed"] == 88
    assert acceptance["overfit"]["steps"] == 80
    assert acceptance["health_output"] == (
        "results/pilot/v2r2_seed88_of80_e120_health.json"
    )
    assert acceptance["validation_output"] == (
        "results/pilot/v2r2_seed88_of80_e120_validation.json"
    )
    assert acceptance["pilot_output_root"] == (
        "results/pilot/v2r2_seed88_of80_e120"
    )
    assert acceptance["pilot_plan"] == (
        "environment/unfrozen_pilot_plan_v2r2_seed88_of80_e120.json"
    )
    expected_runs = [
        run
        for run in gate.generate_run_matrix(protocol)
        if run["condition"] == "C1" and run["seed"] == 88
    ]
    assert len(expected_runs) == 1
    config = gate.validate_run_mapping(expected_runs[0], protocol)
    output = ROOT / acceptance["health_output"]
    config_path = (
        ROOT
        / "configs"
        / "v2r2_seed88_of80_e120_generated"
        / f"{config.runtime.run_id}.yaml"
    )

    explicit_args = gate.build_parser().parse_args(
        [
            "--config",
            str(config_path),
            "--protocol",
            str(protocol_path),
            "--output",
            str(output),
            "--seed",
            "88",
            "--overfit-steps",
            "80",
        ]
    )
    assert explicit_args.config == config_path
    gate.validate_protocol_health_binding(
        explicit_args, config, protocol, output
    )

    default_args = gate.build_parser().parse_args(
        [
            "--protocol",
            str(protocol_path),
            "--output",
            str(output),
        ]
    )
    assert (default_args.seed, default_args.overfit_steps) == (77, 40)
    with pytest.raises(gate.HealthGateError, match="differ from protocol"):
        gate.validate_protocol_health_binding(
            default_args, config, protocol, output
        )


def test_read_only_health_report_validator_accepts_only_bound_pass(tmp_path: Path) -> None:
    protocol_path = ROOT / "configs" / "protocol_v2_pilot.yaml"
    protocol = gate.load_protocol(protocol_path)
    acceptance = protocol["pilot_acceptance"]
    report_path = tmp_path / acceptance["health_output"]
    report_path.parent.mkdir(parents=True)
    source_config = tmp_path / "configs" / "v2_pilot_generated" / ("E1_cifar100_d20_t6_C1_s77.yaml")
    source_config.parent.mkdir(parents=True)
    source_config.write_bytes(
        (ROOT / "configs" / "v2_pilot_generated" / source_config.name).read_bytes()
    )
    commit = "a" * 40
    report = {
        "schema_version": 2,
        "artifact_class": gate.ARTIFACT_CLASS,
        "reporting_eligibility": "FORBIDDEN_FROM_MANUSCRIPT_RESULTS",
        "confirmatory_analysis_eligibility": False,
        "status": "PASS",
        "pass": True,
        "protocol_hash": gate.stable_hash(protocol),
        "acceptance_hash": gate.stable_hash(acceptance),
        "git_commit": commit,
        "tracked_clean": True,
        "seed": acceptance["seed"],
        "conditions": acceptance["conditions"],
        "thresholds": {
            "overfit_steps": acceptance["overfit"]["steps"],
            "minimum_overfit_accuracy": acceptance["overfit"]["minimum_accuracy"],
            "maximum_overfit_loss_fraction": acceptance["overfit"]["maximum_loss_fraction"],
            "minimum_ta_routed_gradient_coverage": acceptance["overfit"][
                "minimum_ta_routed_gradient_coverage"
            ],
            "maximum_ta_enabled_over_frozen_step_ratio": acceptance["timing"][
                "maximum_ta_enabled_over_frozen_ratio"
            ],
        },
        "source": {
            "config": str(source_config),
            "config_sha256": gate.sha256_file(source_config),
            "protocol": str(protocol_path.resolve()),
            "dataset": acceptance["dataset"],
            "overfit_batch_size": acceptance["overfit"]["batch_size"],
            "timing_batch_size": acceptance["timing"]["batch_size"],
            "protocol_sha256": gate.sha256_file(protocol_path),
            "split_manifest_sha256": "0" * 64,
            "fixed_batch_sha256": "1" * 64,
            "fixed_batch_shape": [64, 3, 32, 32],
        },
        "environment": {
            "hardware": {"name": "NVIDIA GeForce RTX 5090"},
            "pytorch": acceptance["environment"]["pytorch_version"],
            "cuda_version": acceptance["environment"]["cuda_runtime"],
            "precision": "float32",
            "amp": False,
            "torch_deterministic": True,
            "torch_deterministic_warn_only": False,
            "cudnn_benchmark": False,
            "cudnn_deterministic": True,
            "gpu_idle_precheck": {
                "pass": True,
                "source": "nvidia-smi",
                "device_selector": "GPU-test",
                "device_uuid": "GPU-test",
                "cuda_visible_index": 0,
                "measurement_point": "immediately_before_cuda_timing",
                "settle_seconds": 1.0,
                "samples_percent": [0, 1, 0],
                "maximum_observed_percent": 1,
                "maximum_allowed_percent": 10,
            },
        },
        "functional_health_by_condition": {
            condition: {
                "pass": True,
                "overfit": {
                    "pass": True,
                    "steps": acceptance["overfit"]["steps"],
                    "loss_fraction": 0.5,
                    "final_accuracy": 0.75,
                    "train_loss_history": [1.0] * acceptance["overfit"]["steps"],
                },
                "shared_conv_and_fc_gradients": {
                    "pass": True,
                    "required_parameter_count": 1,
                    "passed_parameter_count": 1,
                    "parameters": {
                        "stem_conv.weight": {
                            "pass": True,
                            "finite": True,
                            "nonzero_count": 1,
                        }
                    },
                },
                "ta_gradients": (
                    {
                        "pass": True,
                        "aggregate_routed_gradient_coverage": 1.0,
                        "covered_parameter_slots": 1,
                        "every_ta_parameter_tensor_has_nonzero_routed_coverage": True,
                        "all_ta_gradients_present_and_finite_on_every_probe": True,
                        "no_nonzero_gradient_in_unrouted_slots": True,
                    }
                    if condition in {"C2", "C4"}
                    else {"pass": True, "status": "not_applicable_lif_condition"}
                ),
                "checkpoint_next_step": {
                    "pass": True,
                    "resume_loader": "talif_msresnet.train._load_resume",
                    "checkpoint_name": "last.pt",
                    "checkpoint_epoch_zero_based": 0,
                    "expected_start_epoch_zero_based": 1,
                    "ta_activation_epoch_zero_based": 1,
                    "ta_enabled_at_checkpoint": False,
                    "ta_enabled_on_next_step": condition in {"C2", "C4"},
                    "ta_activation_boundary_crossed_on_resume": condition in {"C2", "C4"},
                    "resume_epoch_match": True,
                    "scheduler_stepped_before_last_checkpoint": True,
                    "precheckpoint_epoch_count": 1,
                    "mismatch_count": 0,
                },
            }
            for condition in gate.CONDITIONS_IN_ORDER
        },
        "cuda_train_step_timing": {
            "pass": True,
            "conditions": {
                "C1": {
                    "lif_standard": {
                        "cuda_step_median_ms": 1.0,
                        "peak_allocated_bytes": 1,
                    }
                },
                "C2": {
                    "rounds": [
                        {"order": ["ta_frozen", "ta_enabled"]},
                        {"order": ["ta_enabled", "ta_frozen"]},
                    ],
                    "aggregate": {
                        "ta_frozen_peak_allocated_bytes": 1,
                        "ta_enabled_peak_allocated_bytes": 1,
                    },
                },
                "C3": {
                    "lif_standard": {
                        "cuda_step_median_ms": 1.0,
                        "peak_allocated_bytes": 1,
                    }
                },
                "C4": {
                    "rounds": [
                        {"order": ["ta_frozen", "ta_enabled"]},
                        {"order": ["ta_enabled", "ta_frozen"]},
                    ],
                    "aggregate": {
                        "ta_frozen_peak_allocated_bytes": 1,
                        "ta_enabled_peak_allocated_bytes": 1,
                    },
                },
            },
            "ta_enabled_over_frozen_ratios": {
                condition: {
                    "pass": True,
                    "ta_enabled_over_frozen": 1.5,
                    "round_ratios": [1.4, 1.5],
                }
                for condition in ("C2", "C4")
            },
        },
        "failures": [],
    }
    report_path.write_text(json.dumps(report), encoding="utf-8")

    accepted = gate.validate_pilot_health_report(
        report_path,
        protocol_path,
        repository_root=tmp_path,
        expected_git_commit=commit,
        require_current_tracked_clean=False,
    )
    assert accepted["pass"] is True

    report["acceptance_hash"] = "0" * 64
    report_path.write_text(json.dumps(report), encoding="utf-8")
    with pytest.raises(gate.HealthGateError, match="acceptance_hash"):
        gate.validate_pilot_health_report(
            report_path,
            protocol_path,
            repository_root=tmp_path,
            expected_git_commit=commit,
            require_current_tracked_clean=False,
        )

    report["acceptance_hash"] = gate.stable_hash(acceptance)
    idle = report["environment"]["gpu_idle_precheck"]
    idle["samples_percent"] = [100, 100, 100]
    idle["maximum_observed_percent"] = 0
    report_path.write_text(json.dumps(report), encoding="utf-8")
    with pytest.raises(gate.HealthGateError, match="GPU idle maximum"):
        gate.validate_pilot_health_report(
            report_path,
            protocol_path,
            repository_root=tmp_path,
            expected_git_commit=commit,
            require_current_tracked_clean=False,
        )

    idle["samples_percent"] = [0, 1, 0]
    idle["maximum_observed_percent"] = 1
    checkpoint = report["functional_health_by_condition"]["C2"][
        "checkpoint_next_step"
    ]
    checkpoint["ta_activation_boundary_crossed_on_resume"] = False
    report_path.write_text(json.dumps(report), encoding="utf-8")
    with pytest.raises(gate.HealthGateError, match="frozen-to-enabled"):
        gate.validate_pilot_health_report(
            report_path,
            protocol_path,
            repository_root=tmp_path,
            expected_git_commit=commit,
            require_current_tracked_clean=False,
        )

    checkpoint["ta_activation_boundary_crossed_on_resume"] = True
    report["functional_health_by_condition"]["C2"]["overfit"] = {"pass": True}
    report_path.write_text(json.dumps(report), encoding="utf-8")
    with pytest.raises(gate.HealthGateError, match="C2 overfit"):
        gate.validate_pilot_health_report(
            report_path,
            protocol_path,
            repository_root=tmp_path,
            expected_git_commit=commit,
            require_current_tracked_clean=False,
        )


def test_shared_gradient_gate_requires_every_conv_and_fc_gradient() -> None:
    model = build_model(dataclasses.asdict(_tiny_config().model))
    logits = model(torch.randn(2, 3, 8, 8))
    torch.nn.functional.cross_entropy(logits, torch.tensor([1, 3])).backward()

    report = gate.inspect_shared_gradients(model)

    assert report["pass"]
    assert report["required_parameter_count"] == 23
    assert report["passed_parameter_count"] == 23

    model.stem_conv.weight.grad = None
    failed = gate.inspect_shared_gradients(model)
    assert not failed["pass"]
    assert not failed["parameters"]["stem_conv.weight"]["pass"]


def test_ta_summary_exempts_unrouted_history_slots_but_requires_coverage() -> None:
    routes = {"neuron": {0, 1}}
    accumulator = {
        "neuron.center": {
            "bank_size": 4,
            "finite_on_every_probe": True,
            "gradient_present_on_every_probe": True,
            "nonzero_slots": {0, 1},
            "probe_count": 2,
        },
        "neuron.raw_width": {
            "bank_size": 4,
            "finite_on_every_probe": True,
            "gradient_present_on_every_probe": True,
            "nonzero_slots": {0, 1},
            "probe_count": 2,
        },
    }

    report = gate.summarize_ta_gradients(routes, accumulator, minimum_coverage=1.0)

    assert report["pass"]
    assert report["aggregate_routed_gradient_coverage"] == 1.0
    assert report["parameters"]["neuron.center"]["unrouted_slots_exempt"] == [2, 3]

    accumulator["neuron.raw_width"]["nonzero_slots"] = set()
    failed = gate.summarize_ta_gradients(routes, accumulator, minimum_coverage=0.5)
    assert not failed["pass"]
    assert not failed["every_ta_parameter_tensor_has_nonzero_routed_coverage"]


def test_actual_ta_step_collects_finite_routed_parameter_gradients() -> None:
    torch.manual_seed(0)
    config = _tiny_config("C2")
    device = torch.device("cpu")
    model, optimizer, _scheduler, _ta = gate._build_training_objects(
        config,
        device,
        ta_enabled=True,
    )
    routes: dict[str, set[int]] = {}
    accumulator = gate._new_ta_gradient_accumulator(model)
    batch = (torch.randn(2, 3, 8, 8), torch.tensor([1, 3]))

    gate._step_with_ta_tracking(
        model,
        optimizer,
        config,
        batch,
        device,
        0,
        routes,
        accumulator,
    )
    report = gate.summarize_ta_gradients(routes, accumulator, minimum_coverage=0.0)

    assert report["routed_parameter_slots"] > 0
    assert report["covered_parameter_slots"] > 0
    assert report["all_ta_gradients_present_and_finite_on_every_probe"]
    assert report["no_nonzero_gradient_in_unrouted_slots"]


@pytest.mark.parametrize("condition", ["C1", "C2"])
def test_checkpoint_reload_reproduces_exact_next_cpu_step(
    tmp_path: Path,
    condition: str,
) -> None:
    config = _tiny_config(condition)
    if condition == "C2":
        config = dataclasses.replace(
            config,
            optimizer=dataclasses.replace(
                config.optimizer,
                epochs=120,
                milestones=(2, 5, 7),
                ta_start_fraction=0.05,
            ),
        )
    device = torch.device("cpu")
    batch = (torch.randn(2, 3, 8, 8), torch.tensor([1, 3]))
    report = gate.checkpoint_next_step_check(
        config=config,
        batch=batch,
        device=device,
        checkpoint_path=tmp_path / condition / "last.pt",
    )

    assert report["pass"]
    assert report["mismatch_count"] == 0
    assert report["checkpoint_name"] == "last.pt"
    assert report["resume_loader"] == "talif_msresnet.train._load_resume"
    assert report["scheduler_stepped_before_last_checkpoint"] is True
    if condition == "C2":
        assert report["checkpoint_epoch_zero_based"] == 5
        assert report["expected_start_epoch_zero_based"] == 6
        assert report["precheckpoint_epoch_count"] == 6
        assert report["ta_enabled_at_checkpoint"] is False
        assert report["ta_enabled_on_next_step"] is True
        assert report["ta_activation_boundary_crossed_on_resume"] is True


def test_cuda_timing_uses_reversed_rounds_and_conservative_ratio(
    monkeypatch,
    tmp_path: Path,
) -> None:
    calls: list[tuple[str, bool]] = []
    samples = {
        ("C1", False): [1.0],
        ("C2", False): [2.0, 1.0],
        ("C2", True): [4.0, 3.0],
        ("C3", False): [1.0],
        ("C4", False): [4.0, 2.0],
        ("C4", True): [6.0, 5.0],
    }

    def fake_timed_mode(
        config: RunConfig,
        *,
        device: torch.device,
        batch: tuple[torch.Tensor, torch.Tensor],
        ta_enabled: bool,
        warmup: int,
        iterations: int,
    ) -> dict[str, object]:
        del device, batch, warmup, iterations
        key = (config.model.condition, ta_enabled)
        calls.append(key)
        value = samples[key].pop(0)
        return {
            "cuda_step_ms": [value],
            "cuda_step_median_ms": value,
            "peak_allocated_bytes": int(value * 100),
        }

    monkeypatch.setattr(gate, "_timed_mode", fake_timed_mode)
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: None)
    report = gate.run_cuda_timing_gate(
        _tiny_config(),
        seed=77,
        device=torch.device("cpu"),
        output_parent=tmp_path,
        batch=(torch.randn(2, 3, 8, 8), torch.tensor([1, 3])),
        warmup=2,
        iterations=5,
        maximum_ratio=3.0,
    )

    assert calls == [
        ("C1", False),
        ("C2", False),
        ("C2", True),
        ("C2", True),
        ("C2", False),
        ("C3", False),
        ("C4", False),
        ("C4", True),
        ("C4", True),
        ("C4", False),
    ]
    assert report["conditions"]["C2"]["rounds"][0]["order"] == [
        "ta_frozen",
        "ta_enabled",
    ]
    assert report["conditions"]["C2"]["rounds"][1]["order"] == [
        "ta_enabled",
        "ta_frozen",
    ]
    assert report["ta_enabled_over_frozen_ratios"]["C2"]["round_ratios"] == [
        2.0,
        3.0,
    ]
    assert report["ta_enabled_over_frozen_ratios"]["C2"]["ta_enabled_over_frozen"] == 3.0
    assert report["ta_enabled_over_frozen_ratios"]["C2"]["pass"] is True


def test_run_gate_samples_gpu_immediately_before_cuda_timing(
    monkeypatch,
    tmp_path: Path,
) -> None:
    config = dataclasses.replace(
        _tiny_config(),
        runtime=dataclasses.replace(_tiny_config().runtime, deterministic=True),
    )
    args = gate.build_parser().parse_args(["--output", str(tmp_path / "health.json")])
    inputs = torch.zeros(8, 3, 8, 8)
    targets = torch.zeros(8, dtype=torch.long)
    events: list[str] = []
    idle = {
        "pass": True,
        "source": "nvidia-smi",
        "device_selector": "GPU-test",
        "device_uuid": "GPU-test",
        "cuda_visible_index": 0,
        "measurement_point": "immediately_before_cuda_timing",
        "settle_seconds": 1.0,
        "samples_percent": [0, 0, 0],
        "maximum_observed_percent": 0,
        "maximum_allowed_percent": 10,
    }

    monkeypatch.setattr(
        gate,
        "require_target_cuda",
        lambda _device, _expected: {"name": "NVIDIA GeForce RTX 5090"},
    )
    monkeypatch.setattr(
        gate,
        "repository_git_identity",
        lambda _root: {"git_commit": "a" * 40, "tracked_clean": True},
    )
    monkeypatch.setattr(gate, "validate_runtime_environment", lambda *_args: None)
    monkeypatch.setattr(gate, "seed_everything", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(gate, "environment_manifest", lambda: {})
    monkeypatch.setattr(
        gate,
        "load_fixed_real_batch",
        lambda _config, **_kwargs: (
            config,
            inputs,
            targets,
            {"manifest_sha256": "f" * 64},
        ),
    )

    def fake_condition_health(_config: RunConfig, *, condition: str, **_kwargs):
        events.append(f"functional-{condition}")
        return {"pass": True}

    monkeypatch.setattr(gate, "run_condition_health", fake_condition_health)
    monkeypatch.setattr(gate.gc, "collect", lambda: 0)
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: None)
    monkeypatch.setattr(torch.Tensor, "to", lambda tensor, *_args, **_kwargs: tensor)
    monkeypatch.setattr(gate, "gpu_idle_precheck", lambda _device: events.append("idle") or idle)
    monkeypatch.setattr(
        gate,
        "run_cuda_timing_gate",
        lambda *_args, **_kwargs: events.append("timing")
        or {"pass": True, "ta_enabled_over_frozen_ratios": {}},
    )

    report = gate.run_gate(args, config, tmp_path / "health.json")

    assert events == [
        "functional-C1",
        "functional-C2",
        "functional-C3",
        "functional-C4",
        "idle",
        "timing",
    ]
    assert report["environment"]["gpu_idle_precheck"] == idle


def test_cuda_requirement_fails_closed_on_cpu() -> None:
    with pytest.raises(gate.HealthGateError, match="requires a real CUDA"):
        gate.require_target_cuda(torch.device("cpu"), "RTX 5090")


def test_gpu_idle_precheck_rejects_busy_samples(monkeypatch) -> None:
    monkeypatch.setattr(torch.cuda, "synchronize", lambda _device: None)
    monkeypatch.setattr(gate.time, "sleep", lambda _seconds: None)
    properties = type("Properties", (), {"uuid": "test-uuid"})()
    monkeypatch.setattr(torch.cuda, "get_device_properties", lambda _index: properties)
    values = iter([0, 5, 11])
    commands: list[list[str]] = []

    def fake_run(command: list[str], **_kwargs):
        commands.append(command)
        return gate.subprocess.CompletedProcess(
            command,
            returncode=0,
            stdout=f"{next(values)}\n",
            stderr="",
        )

    monkeypatch.setattr(gate.subprocess, "run", fake_run)

    with pytest.raises(gate.HealthGateError, match="not idle enough"):
        gate.gpu_idle_precheck(torch.device("cuda:0"))

    assert len(commands) == 3
    assert all("--id=GPU-test-uuid" in command for command in commands)


def test_gpu_idle_precheck_fails_closed_on_malformed_nvidia_smi(monkeypatch) -> None:
    monkeypatch.setattr(torch.cuda, "synchronize", lambda _device: None)
    monkeypatch.setattr(gate.time, "sleep", lambda _seconds: None)
    properties = type("Properties", (), {"uuid": "test-uuid"})()
    monkeypatch.setattr(torch.cuda, "get_device_properties", lambda _index: properties)
    monkeypatch.setattr(
        gate.subprocess,
        "run",
        lambda command, **_kwargs: gate.subprocess.CompletedProcess(
            command,
            returncode=0,
            stdout="N/A\n",
            stderr="",
        ),
    )

    with pytest.raises(gate.HealthGateError, match="Cannot verify target GPU"):
        gate.gpu_idle_precheck(torch.device("cuda:0"), samples=1)


def test_gpu_idle_precheck_requires_a_target_uuid(monkeypatch) -> None:
    properties = type("Properties", (), {"uuid": ""})()
    monkeypatch.setattr(torch.cuda, "get_device_properties", lambda _index: properties)

    with pytest.raises(gate.HealthGateError, match="has no UUID"):
        gate.gpu_idle_precheck(torch.device("cuda:0"), samples=1)


def test_comparison_treats_matching_nan_as_equal_and_detects_tensor_change() -> None:
    assert gate._comparison_mismatches({"x": float("nan")}, {"x": float("nan")}) == []
    mismatches = gate._comparison_mismatches(
        {"x": torch.tensor([1.0])},
        {"x": torch.tensor([2.0])},
    )
    assert mismatches == ["root.x"]
