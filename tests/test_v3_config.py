from __future__ import annotations

import copy
import json
import sys
from datetime import datetime
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from generate_run_configs import generate, main as generate_main  # noqa: E402
from talif_msresnet.config import (  # noqa: E402
    ConfigError,
    V3_ACTIVE_CONDITIONS,
    V3_ANALYSIS_CONTRACT,
    V3_ARTIFACT_PATHS,
    V3_FORMAL_SEEDS,
    V3_PILOT_ACCEPTANCE,
    V3_PROTOCOL_CONFIRMATION_FIELDS,
    V3_RETIRED_SEEDS,
    active_conditions_for_protocol,
    generate_run_matrix,
    generate_v3_pilot_matrix,
    load_protocol,
    validate_protocol,
    validate_run_mapping,
)


PROTOCOL_PATH = ROOT / "configs" / "protocol_v3_talif_only.yaml"


@pytest.fixture(scope="module")
def protocol():
    return load_protocol(PROTOCOL_PATH)


def test_v3_protocol_is_signed_and_isolated(protocol) -> None:
    assert protocol["protocol_version"] == 3
    assert protocol["study_stage"] == "pilot_and_formal"
    assert active_conditions_for_protocol(protocol) == V3_ACTIVE_CONDITIONS
    assert tuple(protocol["seeds"]) == V3_FORMAL_SEEDS
    assert set(protocol["seeds"]).isdisjoint(V3_RETIRED_SEEDS)
    assert protocol["artifact_paths"] == V3_ARTIFACT_PATHS
    assert protocol["pilot_acceptance"] == V3_PILOT_ACCEPTANCE
    assert protocol["analysis"] == V3_ANALYSIS_CONTRACT

    status = protocol["protocol_status"]
    assert status["frozen"] is True
    assert status["confirmed_by"] == "Yanzhao Deng; Peng Yan; Song Wang"
    assert status["confirmed_at"] == "2026-07-28T19:18:37+08:00"
    confirmed_at = datetime.fromisoformat(status["confirmed_at"])
    assert confirmed_at.tzinfo is not None
    assert confirmed_at.utcoffset() is not None
    assert set(status["confirmations"]) == set(V3_PROTOCOL_CONFIRMATION_FIELDS)
    assert all(status["confirmations"].values())

    signoff = (ROOT / V3_ARTIFACT_PATHS["signoff"]).read_text(encoding="utf-8")
    assert "Status: **SIGNED - AUTHOR APPROVALS COMPLETE; PROTOCOL FROZEN**" in signoff
    assert signoff.lower().count("- [x]") == len(V3_PROTOCOL_CONFIRMATION_FIELDS)
    assert "PENDING" not in signoff


def test_active_documents_match_v3_failure_and_gated_v4_authorization() -> None:
    documents = {
        "readme": ROOT / "README.md",
        "runbook": ROOT / "V3_TALIF_ONLY_RUNBOOK.md",
        "design": ROOT / "V3_TALIF_ONLY_PROTOCOL_DRAFT.md",
        "v4_draft": ROOT / "V4_TALIF_ONLY_PROTOCOL_DRAFT.md",
        "v4_runbook": ROOT / "V4_TALIF_ONLY_RUNBOOK.md",
        "v4_signoff": ROOT / "PREREGISTRATION_SIGNOFF_V4_TALIF_ONLY.md",
        "closure": ROOT / "MS_RESNET_ROUTE_CLOSURE.md",
        "signoff": ROOT / V3_ARTIFACT_PATHS["signoff"],
    }
    normalized = {
        name: " ".join(path.read_text(encoding="utf-8").split()).casefold()
        for name, path in documents.items()
    }

    assert "v3 cifar-100 pilot failed" in normalized["readme"]
    assert "protocol v4 is now frozen" in normalized["readme"]
    assert "formal v4 remains" in normalized["readme"]
    assert "blocked until both dataset pilots" in normalized["readme"]
    assert "v3 pilot failed" in normalized["runbook"]
    assert "formal permanently" in normalized["runbook"]
    assert "blocked" in normalized["runbook"]
    assert "phase a: author/source freeze (completed)" in normalized["runbook"]
    assert "author-approved and phase a-frozen" in normalized["design"]
    assert "verification status: unverified" in normalized["v4_draft"]
    assert (
        "draft - author confirmation required; execution not authorized"
        in normalized["v4_draft"]
    )
    assert "this is the only authorized execution order" in normalized["v4_runbook"]
    assert "formal training is authorized only" in normalized["v4_runbook"]
    assert "accountable author/user approval recorded" in normalized["v4_signoff"]
    assert "signed and phase a-frozen" in normalized["closure"]
    assert "phase b's isolated matrix generation, health gates" in normalized["signoff"]

    combined = " ".join(normalized.values())
    stale_claims = (
        "unfrozen c1-versus-c2 ta-lif-only v3 design",
        "checked-in v3 protocol is currently unsigned and unfrozen",
        "verification status: unverified and unfrozen",
        "executable but unsigned `configs/protocol_v3_talif_only.yaml`",
        "prespecified analysis contract pending author sign-off",
        "required author sign-off before execution",
        "before the yaml can be frozen",
        "study draft only. it is not frozen",
        "phase b and all runtime gates remain mandatory before any gpu health gate",
    )
    assert not [claim for claim in stale_claims if claim in combined]

    runbook = normalized["runbook"]
    assert runbook.index("run exactly one dataset-specific v3 health gate") < runbook.index(
        "create and verify `freeze_manifest_v3_talif_only.json`"
    )
    assert runbook.index(
        "create and verify `freeze_manifest_v3_talif_only.json`"
    ) < runbook.index("formal execution and analysis")


def test_v3_formal_matrix_is_twenty_complete_c1_c2_pairs(protocol) -> None:
    runs = generate_run_matrix(protocol)

    assert len(runs) == 20
    assert {run["condition"] for run in runs} == {"C1", "C2"}
    assert {run["seed"] for run in runs} == set(V3_FORMAL_SEEDS)
    assert {run["runtime"]["output_dir"] for run in runs} == {
        V3_ARTIFACT_PATHS["formal_results"]
    }
    assert len({run["run_id"] for run in runs}) == 20
    assert all(run["final_test"] is False for run in runs)

    groups = {(run["data"]["dataset"], run["seed"]) for run in runs}
    assert len(groups) == 10
    for dataset, seed in groups:
        block = [
            run
            for run in runs
            if run["data"]["dataset"] == dataset and run["seed"] == seed
        ]
        assert [run["condition"] for run in block] == ["C1", "C2"]


def test_v3_pilot_matrix_is_two_isolated_dataset_blocks(protocol) -> None:
    runs = generate_v3_pilot_matrix(protocol)

    assert len(runs) == 4
    assert [run["condition"] for run in runs] == ["C1", "C2", "C1", "C2"]
    expected_seeds = {
        dataset: values["seed"]
        for dataset, values in V3_PILOT_ACCEPTANCE["datasets"].items()
    }
    assert {
        run["data"]["dataset"]: run["seed"] for run in runs
    } == expected_seeds
    for run in runs:
        dataset = run["data"]["dataset"]
        assert run["runtime"]["output_dir"] == V3_PILOT_ACCEPTANCE["datasets"][
            dataset
        ]["pilot_output_root"]
        assert run["seed"] not in V3_FORMAL_SEEDS
        assert run["seed"] not in V3_RETIRED_SEEDS


@pytest.mark.parametrize(
    ("mutation", "message"),
    (
        (
            lambda value: value.update({"active_conditions": ["C1", "C2", "C3"]}),
            "active_conditions",
        ),
        (
            lambda value: value.update({"output_root": "results/runs"}),
            "output_root",
        ),
        (
            lambda value: value["artifact_paths"].update(
                {"formal_matrix": "configs/generated"}
            ),
            "artifact_paths",
        ),
        (
            lambda value: value["analysis"]["primary_accuracy_test"].update(
                {"sidedness": "two_sided"}
            ),
            "analysis",
        ),
        (
            lambda value: value["protocol_status"]["confirmations"].pop(
                "model_selection_and_one_time_test_access"
            ),
            "confirmations",
        ),
    ),
)
def test_v3_frozen_identity_cannot_drift(protocol, mutation, message) -> None:
    changed = copy.deepcopy(protocol)
    mutation(changed)
    with pytest.raises(ConfigError, match=message):
        validate_protocol(changed)


def test_v3_run_parser_rejects_inactive_c3_and_c4(protocol) -> None:
    baseline = generate_run_matrix(protocol)[0]
    for condition in ("C3", "C4"):
        changed = copy.deepcopy(baseline)
        changed["condition"] = condition
        changed["model"]["condition"] = condition
        with pytest.raises(ConfigError, match="inactive for protocol v3"):
            validate_run_mapping(changed, protocol)


def test_v3_formal_and_pilot_manifests_are_separate_and_reproducible(
    protocol, tmp_path: Path
) -> None:
    formal_output = tmp_path / "formal"
    pilot_output = tmp_path / "pilot"

    formal_runs = generate(PROTOCOL_PATH, formal_output)
    pilot_runs = generate(PROTOCOL_PATH, pilot_output, stage="pilot")
    formal_manifest = json.loads(
        (formal_output / "matrix_manifest.json").read_text(encoding="utf-8")
    )
    pilot_manifest = json.loads(
        (pilot_output / "matrix_manifest.json").read_text(encoding="utf-8")
    )

    assert len(formal_runs) == formal_manifest["run_count"] == 20
    assert formal_manifest["conditions"] == ["C1", "C2"]
    assert formal_manifest["seeds"] == list(V3_FORMAL_SEEDS)
    assert len(list(formal_output.glob("*.yaml"))) == 20
    assert len(pilot_runs) == pilot_manifest["run_count"] == 4
    assert pilot_manifest["conditions"] == ["C1", "C2"]
    assert pilot_manifest["seeds"] == [
        item["seed"] for item in V3_PILOT_ACCEPTANCE["datasets"].values()
    ]
    assert len(list(pilot_output.glob("*.yaml"))) == 4
    assert {run["run_id"] for run in formal_runs}.isdisjoint(
        run["run_id"] for run in pilot_runs
    )

    formal_csv = (formal_output / "run_manifest.csv").read_bytes()
    pilot_csv = (pilot_output / "run_manifest.csv").read_bytes()
    generate(PROTOCOL_PATH, formal_output)
    generate(PROTOCOL_PATH, pilot_output, stage="pilot")
    assert (formal_output / "run_manifest.csv").read_bytes() == formal_csv
    assert (pilot_output / "run_manifest.csv").read_bytes() == pilot_csv


def test_v3_cli_refuses_a_legacy_or_arbitrary_matrix_path(tmp_path: Path) -> None:
    with pytest.raises(SystemExit, match="isolated matrix path"):
        generate_main(
            [
                "--protocol",
                str(PROTOCOL_PATH),
                "--output",
                str(tmp_path / "wrong"),
            ]
        )


def test_legacy_matrix_sizes_and_conditions_remain_unchanged() -> None:
    v1 = load_protocol(ROOT / "configs" / "protocol.yaml")
    v2 = load_protocol(ROOT / "configs" / "protocol_v2_pilot.yaml")
    v2r2 = load_protocol(ROOT / "configs" / "protocol_v2r2_seed88_of80_e120.yaml")

    assert len(generate_run_matrix(v1)) == 40
    assert len(generate_run_matrix(v2)) == 4
    assert len(generate_run_matrix(v2r2)) == 4
    assert active_conditions_for_protocol(v1) == ("C1", "C2", "C3", "C4")
    assert active_conditions_for_protocol(v2) == ("C1", "C2", "C3", "C4")
