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

from talif_msresnet.config_v6 import (
    V6_ACTIVE_CONDITIONS,
    V6_FORMAL_SEEDS,
    V6_PILOT_SEED,
)


def _pilot_rows(*, seed: int = V6_PILOT_SEED) -> list[dict[str, str]]:
    return [
        {
            "experiment": "E6",
            "dataset": "cifar100",
            "depth": "20",
            "time_steps": "6",
            "seed": str(seed),
            "condition": condition,
            "run_id": f"E6_cifar100_d20_t6_{condition}_s{seed}",
            "config_file": f"{condition}.yaml",
        }
        for condition in V6_ACTIVE_CONDITIONS
    ]


def _v6_block(tmp_path: Path) -> SimpleNamespace:
    health = tmp_path / "health.json"
    health.write_text("{}\n", encoding="utf-8")
    return SimpleNamespace(
        dataset="cifar100",
        conditions=V6_ACTIVE_CONDITIONS,
        health_seed=1_707_261_715,
        pilot_seed=V6_PILOT_SEED,
        pilot_output_root=tmp_path / "pilot",
        pilot_plan=tmp_path / "pilot-plan.json",
        health_output=health,
    )


def _patch_v6_launch_dependencies(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> SimpleNamespace:
    block = _v6_block(tmp_path)
    monkeypatch.setattr(matrix_runner, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(
        matrix_runner,
        "resolve_v6_pilot_block",
        lambda *_args, **_kwargs: block,
    )
    monkeypatch.setattr(
        matrix_runner,
        "validate_v6_health_report",
        lambda *_args, **_kwargs: {"status": "PASS", "pass": True},
    )
    return block


def test_v6_pilot_launch_requires_exact_ordered_six_condition_block(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    block = _patch_v6_launch_dependencies(monkeypatch, tmp_path)

    resolved, output_root, plan, report = (
        matrix_runner._validate_v6_pilot_launch_contract(
            protocol={"protocol_version": 6},
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
            lambda rows: [dict(row, seed=str(V6_PILOT_SEED + 1)) for row in rows],
            "differs from the pilot-seed binding",
        ),
        (
            lambda rows: [dict(rows[0], dataset="cifar10"), *rows[1:]],
            "one dataset/depth/T/seed block",
        ),
    ],
)
def test_v6_pilot_launch_fails_closed_for_tampered_rows(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mutator,
    message: str,
) -> None:
    _patch_v6_launch_dependencies(monkeypatch, tmp_path)

    with pytest.raises(ValueError, match=message):
        matrix_runner._validate_v6_pilot_launch_contract(
            protocol={"protocol_version": 6},
            protocol_path=tmp_path / "protocol.yaml",
            selected_rows=mutator(_pilot_rows()),
            dataset="cifar100",
            output_root=None,
            pilot_plan=None,
        )


def test_v6_pilot_launch_rejects_wrong_dataset_and_path_overrides(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_v6_launch_dependencies(monkeypatch, tmp_path)
    common = {
        "protocol": {"protocol_version": 6},
        "protocol_path": tmp_path / "protocol.yaml",
        "selected_rows": _pilot_rows(),
        "output_root": None,
        "pilot_plan": None,
    }

    with pytest.raises(ValueError, match="dataset must be 'cifar100'"):
        matrix_runner._validate_v6_pilot_launch_contract(
            **common,
            dataset="cifar10",
        )
    with pytest.raises(ValueError, match="cannot override the V6 pilot output root"):
        matrix_runner._validate_v6_pilot_launch_contract(
            **{**common, "output_root": tmp_path / "other-output"},
            dataset="cifar100",
        )
    with pytest.raises(ValueError, match="cannot override the V6 pilot plan"):
        matrix_runner._validate_v6_pilot_launch_contract(
            **{**common, "pilot_plan": tmp_path / "other-plan.json"},
            dataset="cifar100",
        )


def _formal_rows() -> list[dict[str, str]]:
    return [
        {
            "experiment": "E6",
            "dataset": "cifar100",
            "depth": "20",
            "time_steps": "6",
            "seed": str(seed),
            "condition": condition,
            "run_id": f"E6_cifar100_d20_t6_{condition}_s{seed}",
            "config_file": f"E6_cifar100_d20_t6_{condition}_s{seed}.yaml",
        }
        for seed in V6_FORMAL_SEEDS
        for condition in V6_ACTIVE_CONDITIONS
    ]


@pytest.mark.parametrize(
    "selection",
    [
        {"condition": "M0"},
        {"limit": 1},
        {"dataset": "cifar100"},
        {"experiment": "E6"},
        {"config_path": "inside-formal-matrix"},
    ],
)
def test_v6_formal_rejects_every_partial_selection(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    selection: dict[str, object],
) -> None:
    formal_matrix = tmp_path / "configs" / "v6-formal"
    formal_results = tmp_path / "results" / "formal-v6"
    protocol_path = tmp_path / "configs" / "protocol-v6.yaml"
    protocol = {
        "protocol_version": 6,
        "study_stage": "health_pilot_formal",
        "active_conditions": list(V6_ACTIVE_CONDITIONS),
        "matrix": {"primary": [{"seeds": list(V6_FORMAL_SEEDS)}]},
        "artifact_paths": {
            "formal_matrix": str(formal_matrix.relative_to(tmp_path)),
            "pilot_matrix": "configs/v6-pilot",
            "formal_results": str(formal_results.relative_to(tmp_path)),
            "freeze_manifest": "FREEZE_MANIFEST_V6.json",
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


def test_v6_consumed_run_identity_allows_only_checkpoint_resume_or_skip(
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


def test_v6_pilot_revalidates_runtime_before_matrix_and_every_run(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config_dir = tmp_path / "configs" / "v6-pilot"
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
    protocol_path = tmp_path / "configs" / "protocol-v6.yaml"
    protocol_path.write_text("protocol_version: 6\n", encoding="utf-8")
    block = _v6_block(tmp_path)
    protocol = {
        "protocol_version": 6,
        "study_stage": "health_pilot_formal",
        "artifact_paths": {
            "pilot_matrix": str(config_dir.relative_to(tmp_path)),
            "formal_matrix": "configs/v6-formal",
            "formal_results": "results/formal-v6",
        },
    }
    reference_config = SimpleNamespace(
        data=SimpleNamespace(dataset="cifar100"),
        runtime=SimpleNamespace(seed=V6_PILOT_SEED),
    )
    runtime_calls: list[str] = []
    commands: list[list[str]] = []

    monkeypatch.setattr(matrix_runner, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(matrix_runner, "load_protocol", lambda _path: protocol)
    monkeypatch.setattr(matrix_runner, "_read_manifest", lambda _path: rows)
    monkeypatch.setattr(matrix_runner, "require_v6_author_freeze", lambda *_a, **_k: {})
    monkeypatch.setattr(
        matrix_runner,
        "resolve_v6_pilot_block",
        lambda *_args, **_kwargs: block,
    )
    monkeypatch.setattr(
        matrix_runner,
        "validate_v6_health_report",
        lambda *_args, **_kwargs: {"status": "PASS", "pass": True},
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
    monkeypatch.setattr(matrix_runner, "generate_v6_pilot_matrix", lambda _p: [])
    monkeypatch.setattr(
        matrix_runner,
        "expected_v6_pilot_configs",
        lambda _protocol: (reference_config,),
    )

    def validate_runtime(*_args, **_kwargs) -> dict[str, object]:
        runtime_calls.append(str(_kwargs["device"]))
        return {"pass": True, "call": len(runtime_calls)}

    monkeypatch.setattr(matrix_runner, "validate_v6_current_runtime", validate_runtime)
    plan_path = tmp_path / "pilot-plan.json"
    monkeypatch.setattr(
        matrix_runner,
        "_prepare_v6_pilot_plan",
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
    assert runtime_calls == ["cuda:0"] * 7
    assert len(commands) == len(V6_ACTIVE_CONDITIONS)
    assert all("--orchestrator-pilot-plan" in command for command in commands)
