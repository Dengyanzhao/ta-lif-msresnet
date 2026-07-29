from __future__ import annotations

import sys
from copy import deepcopy
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from analyze_v5_results import (
    V5_ANALYSIS_RESULTS_REFERENCE,
    V5_FORMAL_RESULTS_REFERENCE,
    V5_PROTOCOL_REFERENCE,
    isolated_analysis_paths,
    validate_v5_protocol_contract,
)

from talif_msresnet.config import load_protocol


def test_v5_cli_contract_matches_frozen_protocol_and_fails_closed() -> None:
    protocol = load_protocol(ROOT / V5_PROTOCOL_REFERENCE)
    validate_v5_protocol_contract(protocol)

    changed = deepcopy(protocol)
    changed["protocol_version"] = 4
    with pytest.raises(ValueError, match="protocol_version"):
        validate_v5_protocol_contract(changed)

    changed = deepcopy(protocol)
    changed["analysis"]["bootstrap"]["resamples"] = 9999
    with pytest.raises(ValueError, match="analysis.bootstrap"):
        validate_v5_protocol_contract(changed)


def test_v5_cli_paths_are_isolated_from_pilot_and_legacy_results() -> None:
    protocol = load_protocol(ROOT / V5_PROTOCOL_REFERENCE)
    root = Path("C:/isolated-v5-analysis-test-root")
    protocol_path = root / V5_PROTOCOL_REFERENCE

    input_path, output_path = isolated_analysis_paths(
        protocol_path,
        protocol,
        None,
        None,
        repository_root=root,
    )
    assert input_path == (root / V5_FORMAL_RESULTS_REFERENCE).resolve()
    assert output_path == (root / V5_ANALYSIS_RESULTS_REFERENCE).resolve()

    with pytest.raises(ValueError, match="isolated formal-results root"):
        isolated_analysis_paths(
            protocol_path,
            protocol,
            root / "results" / "pilot" / "v5_talif_only",
            None,
            repository_root=root,
        )
    with pytest.raises(ValueError, match="isolated analysis-results root"):
        isolated_analysis_paths(
            protocol_path,
            protocol,
            None,
            root / "results" / "analysis" / "v4_talif_only",
            repository_root=root,
        )


def test_v5_cli_rejects_legacy_protocol_path() -> None:
    protocol = load_protocol(ROOT / V5_PROTOCOL_REFERENCE)
    root = Path("C:/isolated-v5-analysis-test-root")
    with pytest.raises(ValueError, match="requires protocol"):
        isolated_analysis_paths(
            root / "configs" / "protocol.yaml",
            protocol,
            None,
            None,
            repository_root=root,
        )
