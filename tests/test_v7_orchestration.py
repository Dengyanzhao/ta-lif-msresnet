from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

pytest.importorskip("torch")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import run_matrix as matrix_runner

from talif_msresnet.config_v7 import (
    V7_ACTIVE_CONDITIONS,
    V7_FORMAL_SEEDS,
    V7_PILOT_SEED,
)


def _pilot_rows(*, seed: int = V7_PILOT_SEED) -> list[dict[str, str]]:
    return [
        {
            "experiment": "E9",
            "dataset": "cifar100",
            "depth": "20",
            "time_steps": "6",
            "seed": str(seed),
            "condition": condition,
            "run_id": f"E9_cifar100_d20_t6_{condition}_s{seed}",
            "config_file": f"E9_cifar100_d20_t6_{condition}_s{seed}.yaml",
        }
        for condition in V7_ACTIVE_CONDITIONS
    ]


def _v7_block(tmp_path: Path) -> SimpleNamespace:
    health = tmp_path / "health.json"
    health.write_text("{}\n", encoding="utf-8")
    return SimpleNamespace(
        dataset="cifar100",
        conditions=V7_ACTIVE_CONDITIONS,
        health_seed=1_068_798_027,
        pilot_seed=V7_PILOT_SEED,
        pilot_output_root=tmp_path / "pilot",
        pilot_plan=tmp_path / "pilot-plan.json",
        health_output=health,
    )


def _patch_v7_contract_dependencies(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> SimpleNamespace:
    block = _v7_block(tmp_path)
    monkeypatch.setattr(matrix_runner, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(
        matrix_runner,
        "resolve_v7_pilot_block",
        lambda *_args, **_kwargs: block,
    )
    monkeypatch.setattr(
        matrix_runner,
        "validate_v7_health_report",
        lambda *_args, **_kwargs: {"status": "PASS", "pass": True},
    )
    return block


def test_v7_pilot_launch_requires_exact_ordered_six_condition_block(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    block = _patch_v7_contract_dependencies(monkeypatch, tmp_path)

    resolved, output_root, plan, report = (
        matrix_runner._validate_v7_pilot_launch_contract(
            protocol={"protocol_version": 7},
            protocol_path=tmp_path / "protocol.yaml",
            selected_rows=_pilot_rows(),
            dataset="cifar100",
            output_root=None,
            pilot_plan=None,
        )
    )

    assert resolved is block
    assert output_root == block.pilot_output_root
    assert plan == block.pilot_plan
    assert report["pass"] is True


@pytest.mark.parametrize(
    ("mutator", "message"),
    [
        (lambda rows: rows[:-1], "exactly one complete six-condition block"),
        (
            lambda rows: [rows[1], rows[0], *rows[2:]],
            "launch order must be exactly",
        ),
        (
            lambda rows: [dict(row, seed=str(V7_PILOT_SEED + 1)) for row in rows],
            "differs from the pilot-seed binding",
        ),
        (
            lambda rows: [dict(rows[0], dataset="cifar10"), *rows[1:]],
            "one dataset/depth/T/seed block",
        ),
        (
            lambda rows: [dict(rows[0], condition="LIF"), *rows[1:]],
            "launch order must be exactly",
        ),
    ],
)
def test_v7_pilot_launch_fails_closed_for_tampered_rows(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mutator,
    message: str,
) -> None:
    _patch_v7_contract_dependencies(monkeypatch, tmp_path)

    with pytest.raises(ValueError, match=message):
        matrix_runner._validate_v7_pilot_launch_contract(
            protocol={"protocol_version": 7},
            protocol_path=tmp_path / "protocol.yaml",
            selected_rows=mutator(_pilot_rows()),
            dataset="cifar100",
            output_root=None,
            pilot_plan=None,
        )


def test_v7_pilot_launch_rejects_wrong_dataset_and_path_overrides(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_v7_contract_dependencies(monkeypatch, tmp_path)
    common = {
        "protocol": {"protocol_version": 7},
        "protocol_path": tmp_path / "protocol.yaml",
        "selected_rows": _pilot_rows(),
        "output_root": None,
        "pilot_plan": None,
    }

    with pytest.raises(ValueError, match="Protocol V7 pilot dataset must be 'cifar100'"):
        matrix_runner._validate_v7_pilot_launch_contract(
            **common,
            dataset="cifar10",
        )
    with pytest.raises(ValueError, match="cannot override the V7 pilot output root"):
        matrix_runner._validate_v7_pilot_launch_contract(
            **{**common, "output_root": tmp_path / "other-output"},
            dataset="cifar100",
        )
    with pytest.raises(ValueError, match="cannot override the V7 pilot plan"):
        matrix_runner._validate_v7_pilot_launch_contract(
            **{**common, "pilot_plan": tmp_path / "other-plan.json"},
            dataset="cifar100",
        )


@pytest.mark.parametrize(
    ("selection", "message"),
    [
        ({"condition": "M0"}, "condition, limit, and experiment filters are forbidden"),
        ({"limit": 1}, "condition, limit, and experiment filters are forbidden"),
        ({"experiment": "E9"}, "condition, limit, and experiment filters are forbidden"),
    ],
)
def test_v7_pilot_rejects_condition_limit_and_experiment_filters(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    selection: dict[str, object],
    message: str,
) -> None:
    config_dir = tmp_path / "configs" / "v7-pilot"
    config_dir.mkdir(parents=True)
    protocol = {
        "protocol_version": 7,
        "study_stage": "health_pilot_formal",
        "artifact_paths": {"pilot_matrix": "configs/v7-pilot"},
    }
    monkeypatch.setattr(matrix_runner, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(matrix_runner, "load_protocol", lambda _path: protocol)
    monkeypatch.setattr(matrix_runner, "_read_manifest", lambda _path: _pilot_rows())
    monkeypatch.setattr(matrix_runner, "require_v7_author_freeze", lambda *_a, **_k: {})
    monkeypatch.setattr(
        matrix_runner,
        "resolve_v7_pilot_block",
        lambda *_a, **_k: pytest.fail("forbidden selection reached block resolution"),
    )

    with pytest.raises(ValueError, match=message):
        matrix_runner.run_matrix(
            config_dir=config_dir,
            protocol_path=tmp_path / "configs" / "protocol-v7.yaml",
            allow_unfrozen_pilot=True,
            dataset="cifar100",
            device="cuda:0",
            **selection,
        )


def _formal_rows() -> list[dict[str, str]]:
    return [
        {
            "experiment": "E9",
            "dataset": "cifar100",
            "depth": "20",
            "time_steps": "6",
            "seed": str(seed),
            "condition": condition,
            "run_id": f"E9_cifar100_d20_t6_{condition}_s{seed}",
            "config_file": f"E9_cifar100_d20_t6_{condition}_s{seed}.yaml",
        }
        for seed in V7_FORMAL_SEEDS
        for condition in V7_ACTIVE_CONDITIONS
    ]


def test_v7_formal_matrix_has_exact_48_frozen_run_identities() -> None:
    protocol = matrix_runner.load_protocol(ROOT / "configs" / "protocol_v7_mechanism.yaml")
    expected = [
        f"E9_cifar100_d20_t6_{condition}_s{seed}"
        for seed in V7_FORMAL_SEEDS
        for condition in V7_ACTIVE_CONDITIONS
    ]
    generated = matrix_runner.generate_run_matrix(protocol)
    observed = [matrix_runner.validate_run_mapping(raw, protocol).runtime.run_id for raw in generated]

    assert len(observed) == 48
    assert observed == expected


def test_v7_formal_rejects_same_size_matrix_with_tampered_run_identity(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    protocol = matrix_runner.load_protocol(ROOT / "configs" / "protocol_v7_mechanism.yaml")
    formal_matrix = tmp_path / protocol["artifact_paths"]["formal_matrix"]
    formal_matrix.mkdir(parents=True)
    rows = _formal_rows()
    rows[0] = dict(rows[0], run_id=rows[1]["run_id"])
    (formal_matrix / "matrix_manifest.json").write_text(
        json.dumps(
            {
                "protocol_hash": "p" * 64,
                "runs": [{"run_id": row["run_id"]} for row in rows],
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(matrix_runner, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(matrix_runner, "load_protocol", lambda _path: protocol)
    monkeypatch.setattr(matrix_runner, "_read_manifest", lambda _path: rows)
    monkeypatch.setattr(
        matrix_runner,
        "check_protocol",
        lambda *_args, **_kwargs: SimpleNamespace(
            ok=True,
            errors=[],
            protocol_hash="p" * 64,
        ),
    )
    monkeypatch.setattr(matrix_runner, "verify_formal_freeze", lambda **_kwargs: {})
    monkeypatch.setattr(
        matrix_runner.subprocess,
        "run",
        lambda *_a, **_k: pytest.fail("tampered formal matrix reached subprocess"),
    )

    with pytest.raises(RuntimeError, match="generated run manifest differs from the protocol"):
        matrix_runner.run_matrix(
            config_dir=formal_matrix,
            protocol_path=tmp_path / protocol["artifact_paths"]["protocol"],
            device="cuda:0",
            stop_on_error=True,
        )
    assert not list(tmp_path.rglob("attempt_*.stdout.log"))


@pytest.mark.parametrize(
    "selection",
    [
        {"condition": "M0"},
        {"limit": 1},
        {"dataset": "cifar100"},
        {"experiment": "E9"},
        {"config_path": "inside-formal-matrix"},
    ],
)
def test_v7_formal_rejects_every_partial_selection(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    selection: dict[str, object],
) -> None:
    formal_matrix = tmp_path / "configs" / "v7-formal"
    formal_matrix.mkdir(parents=True)
    formal_results = tmp_path / "results" / "formal-v7"
    protocol_path = tmp_path / "configs" / "protocol-v7.yaml"
    protocol = {
        "protocol_version": 7,
        "study_stage": "health_pilot_formal",
        "active_conditions": list(V7_ACTIVE_CONDITIONS),
        "matrix": {"primary": [{"seeds": list(V7_FORMAL_SEEDS)}]},
        "artifact_paths": {
            "formal_matrix": str(formal_matrix.relative_to(tmp_path)),
            "pilot_matrix": "configs/v7-pilot",
            "formal_results": str(formal_results.relative_to(tmp_path)),
            "freeze_manifest": "FREEZE_MANIFEST_V7.json",
        },
    }
    monkeypatch.setattr(matrix_runner, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(matrix_runner, "load_protocol", lambda _path: protocol)
    monkeypatch.setattr(matrix_runner, "_read_manifest", lambda _path: _formal_rows())
    monkeypatch.setattr(
        matrix_runner,
        "check_protocol",
        lambda *_args, **_kwargs: SimpleNamespace(
            ok=True,
            errors=[],
            protocol_hash="p" * 64,
        ),
    )
    monkeypatch.setattr(
        matrix_runner,
        "verify_formal_freeze",
        lambda **_kwargs: pytest.fail("partial selection reached the freeze gate"),
    )
    kwargs: dict[str, object] = {
        "config_dir": formal_matrix,
        "protocol_path": protocol_path,
    }
    kwargs.update(selection)
    if kwargs.get("config_path") == "inside-formal-matrix":
        kwargs["config_path"] = formal_matrix / "one-run.yaml"

    with pytest.raises(ValueError, match="complete frozen 48-run matrix"):
        matrix_runner.run_matrix(**kwargs)


def test_v7_pilot_author_freeze_failure_precedes_health_and_runtime(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config_dir = tmp_path / "configs" / "v7-pilot"
    config_dir.mkdir(parents=True)
    protocol_path = tmp_path / "configs" / "protocol-v7.yaml"
    protocol = {
        "protocol_version": 7,
        "study_stage": "health_pilot_formal",
        "artifact_paths": {"pilot_matrix": "configs/v7-pilot"},
    }
    monkeypatch.setattr(matrix_runner, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(matrix_runner, "load_protocol", lambda _path: protocol)
    monkeypatch.setattr(matrix_runner, "_read_manifest", lambda _path: _pilot_rows())
    monkeypatch.setattr(
        matrix_runner,
        "require_v7_author_freeze",
        lambda *_a, **_k: (_ for _ in ()).throw(
            matrix_runner.PilotV7Error("author freeze missing")
        ),
    )
    monkeypatch.setattr(
        matrix_runner,
        "validate_v7_health_report",
        lambda *_a, **_k: pytest.fail("health validation must not run without author freeze"),
    )
    monkeypatch.setattr(
        matrix_runner,
        "validate_v7_current_runtime",
        lambda *_a, **_k: pytest.fail("runtime validation must not run without author freeze"),
    )

    with pytest.raises(ValueError, match="requires author freeze"):
        matrix_runner.run_matrix(
            config_dir=config_dir,
            protocol_path=protocol_path,
            allow_unfrozen_pilot=True,
            dataset="cifar100",
            device="cuda:0",
        )
    assert not list(tmp_path.rglob("attempt_*.stdout.log"))


def test_v7_pilot_rejects_v6_health_evidence_before_any_run(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    block = _patch_v7_contract_dependencies(monkeypatch, tmp_path)
    monkeypatch.setattr(
        matrix_runner,
        "validate_v7_health_report",
        lambda *_a, **_k: (_ for _ in ()).throw(
            matrix_runner.PilotV7Error("V6 health evidence cannot authorize V7")
        ),
    )

    with pytest.raises(RuntimeError, match="health-report gate blocked execution"):
        matrix_runner._validate_v7_pilot_launch_contract(
            protocol={"protocol_version": 7},
            protocol_path=tmp_path / "protocol.yaml",
            selected_rows=_pilot_rows(),
            dataset=block.dataset,
            output_root=None,
            pilot_plan=None,
        )


def test_v7_consumed_run_identity_allows_only_checkpoint_resume_or_terminal_skip(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "attempt_001.stdout.log").write_text("failed\n", encoding="utf-8")

    with pytest.raises(RuntimeError, match="forbids a fresh retry"):
        matrix_runner._continuation_action(
            run_dir,
            "run",
            "config-hash",
            resume_matrix=True,
            forbid_fresh_after_attempt=True,
        )

    checkpoint = run_dir / "last.pt"
    checkpoint.write_bytes(b"checkpoint")
    assert matrix_runner._continuation_action(
        run_dir,
        "run",
        "config-hash",
        resume_matrix=True,
        forbid_fresh_after_attempt=True,
    ) == ("resume", checkpoint.resolve())

    checkpoint.unlink()
    (run_dir / "seed_metrics.json").write_text(
        json.dumps(
            {
                "run_id": "run",
                "config_hash": "config-hash",
                "status": "complete",
                "failed": 0,
            }
        ),
        encoding="utf-8",
    )
    (run_dir / "best.pt").write_bytes(b"best")
    assert matrix_runner._continuation_action(
        run_dir,
        "run",
        "config-hash",
        resume_matrix=True,
        forbid_fresh_after_attempt=True,
    ) == ("skip", None)


def test_v7_formal_freeze_failure_blocks_before_matrix_or_subprocess(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    formal_matrix = tmp_path / "configs" / "v7-formal"
    formal_matrix.mkdir(parents=True)
    protocol_path = tmp_path / "configs" / "protocol-v7.yaml"
    protocol = {
        "protocol_version": 7,
        "study_stage": "health_pilot_formal",
        "artifact_paths": {
            "formal_matrix": "configs/v7-formal",
            "formal_results": "results/formal-v7",
            "pilot_matrix": "configs/v7-pilot",
            "freeze_manifest": "FREEZE_MANIFEST_V7.json",
        },
        "matrix": {"primary": [{"seeds": list(V7_FORMAL_SEEDS)}]},
    }
    monkeypatch.setattr(matrix_runner, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(matrix_runner, "load_protocol", lambda _path: protocol)
    monkeypatch.setattr(matrix_runner, "_read_manifest", lambda _path: _formal_rows())
    monkeypatch.setattr(
        matrix_runner,
        "check_protocol",
        lambda *_args, **_kwargs: SimpleNamespace(
            ok=True,
            errors=[],
            protocol_hash="p" * 64,
        ),
    )
    monkeypatch.setattr(
        matrix_runner,
        "verify_formal_freeze",
        lambda **_kwargs: (_ for _ in ()).throw(
            matrix_runner.FreezeGateError("V7 pilot validation is missing or modified")
        ),
    )
    monkeypatch.setattr(
        matrix_runner.subprocess,
        "run",
        lambda *_a, **_k: pytest.fail("formal freeze failure reached subprocess"),
    )

    with pytest.raises(RuntimeError, match="Formal freeze-manifest gate blocked execution"):
        matrix_runner.run_matrix(
            config_dir=formal_matrix,
            protocol_path=protocol_path,
            device="cuda:0",
            stop_on_error=True,
        )
    assert not list(tmp_path.rglob("attempt_*.stdout.log"))


def test_v7_pilot_revalidates_runtime_before_matrix_and_every_run(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config_dir = tmp_path / "configs" / "v7-pilot"
    config_dir.mkdir(parents=True)
    rows = _pilot_rows()
    rows_by_file: dict[str, dict[str, str]] = {}
    for row in rows:
        path = config_dir / row["config_file"]
        path.write_text(f"condition: {row['condition']}\n", encoding="utf-8")
        rows_by_file[path.name] = row
    (config_dir / "matrix_manifest.json").write_text(
        json.dumps({"protocol_hash": "p" * 64}),
        encoding="utf-8",
    )
    protocol_path = tmp_path / "configs" / "protocol-v7.yaml"
    protocol_path.write_text("protocol_version: 7\n", encoding="utf-8")
    block = _v7_block(tmp_path)
    protocol = {
        "protocol_version": 7,
        "study_stage": "health_pilot_formal",
        "artifact_paths": {
            "pilot_matrix": str(config_dir.relative_to(tmp_path)),
            "formal_matrix": "configs/v7-formal",
            "formal_results": "results/formal-v7",
        },
    }
    reference_config = SimpleNamespace(
        data=SimpleNamespace(dataset="cifar100"),
        runtime=SimpleNamespace(seed=V7_PILOT_SEED),
    )
    calls: list[str] = []
    commands: list[list[str]] = []

    monkeypatch.setattr(matrix_runner, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(matrix_runner, "load_protocol", lambda _path: protocol)
    monkeypatch.setattr(matrix_runner, "_read_manifest", lambda _path: rows)
    monkeypatch.setattr(matrix_runner, "require_v7_author_freeze", lambda *_a, **_k: calls.append("author") or {})
    monkeypatch.setattr(
        matrix_runner,
        "resolve_v7_pilot_block",
        lambda *_args, **_kwargs: block,
    )
    monkeypatch.setattr(
        matrix_runner,
        "validate_v7_health_report",
        lambda *_args, **_kwargs: calls.append("health") or {"status": "PASS", "pass": True},
    )
    monkeypatch.setattr(
        matrix_runner,
        "check_protocol",
        lambda *_args, **_kwargs: SimpleNamespace(
            ok=True,
            errors=[],
            protocol_hash="p" * 64,
        ),
    )
    monkeypatch.setattr(matrix_runner, "_validate_v2_generated_matrix", lambda **_k: None)
    monkeypatch.setattr(matrix_runner, "generate_v7_pilot_matrix", lambda _p: [])
    monkeypatch.setattr(
        matrix_runner,
        "expected_v7_pilot_configs",
        lambda _protocol: (reference_config,),
    )

    def validate_runtime(*_args, **_kwargs) -> dict[str, object]:
        calls.append("runtime")
        return {"pass": True, "call": calls.count("runtime")}

    monkeypatch.setattr(matrix_runner, "validate_v7_current_runtime", validate_runtime)
    plan_path = tmp_path / "pilot-plan.json"
    monkeypatch.setattr(
        matrix_runner,
        "_prepare_v7_pilot_plan",
        lambda **_kwargs: plan_path,
    )

    def load_config(path: Path, _protocol: Path) -> SimpleNamespace:
        row = rows_by_file[Path(path).name]
        return SimpleNamespace(
            runtime=SimpleNamespace(
                run_id=row["run_id"],
                deterministic=True,
            ),
            config_hash=f"hash-{row['condition']}",
        )

    monkeypatch.setattr(matrix_runner, "load_run_config", load_config)

    def run_command(command: list[str], **_kwargs) -> SimpleNamespace:
        commands.append(command)
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(matrix_runner.subprocess, "run", run_command)

    exit_code = matrix_runner.run_matrix(
        config_dir=config_dir,
        protocol_path=protocol_path,
        allow_unfrozen_pilot=True,
        dataset="cifar100",
        device="cuda:0",
        stop_on_error=True,
    )

    assert exit_code == 0
    assert calls == ["author", "health", "runtime"] + ["runtime"] * len(V7_ACTIVE_CONDITIONS)
    assert len(commands) == len(V7_ACTIVE_CONDITIONS)
    assert all("--orchestrator-pilot-plan" in command for command in commands)
