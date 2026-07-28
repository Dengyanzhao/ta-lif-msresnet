from __future__ import annotations

import csv
import importlib.util
import json
import shutil
import subprocess
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest
import yaml


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from talif_msresnet.config import PROTOCOL_CONFIRMATION_FIELDS  # noqa: E402


def _load_tool() -> ModuleType:
    path = ROOT / "scripts" / "create_freeze_manifest.py"
    spec = importlib.util.spec_from_file_location("create_freeze_manifest", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


tool = _load_tool()


def _run_git(root: Path, *arguments: str) -> str:
    completed = subprocess.run(
        ["git", "-C", str(root), *arguments],
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    return completed.stdout.strip()


def _signed_record(*, signed: bool = True) -> str:
    status = (
        tool.SIGNED_STATUS
        if signed
        else "Status: **DRAFT - NOT SIGNED, NOT FROZEN**"
    )
    checklist = "\n".join(
        f"- [x] `{field}`: reviewed." for field in PROTOCOL_CONFIRMATION_FIELDS
    )
    song_approval = (
        "- Song Wang, corresponding author - approval evidence/location: "
        "`institutional-record-3`; date: `2026-07-21`"
    )
    return f"""# Prespecified protocol sign-off record

{status}

## Author review checklist

{checklist}

## Signatures

- Yanzhao Deng - approval evidence/location: `institutional-record-1`; date: `2026-07-21`
- Peng Yan - approval evidence/location: `institutional-record-2`; date: `2026-07-21`
{song_approval}
"""


def _protocol(*, frozen: bool = True) -> dict[str, Any]:
    value = yaml.safe_load((ROOT / "configs" / "protocol.yaml").read_text(encoding="utf-8"))
    status = value["protocol_status"]
    status["frozen"] = frozen
    status["confirmed_by"] = "Yanzhao Deng; Peng Yan; Song Wang"
    status["confirmed_at"] = "2026-07-21T18:30:00+08:00"
    status["confirmations"] = {
        field: frozen for field in PROTOCOL_CONFIRMATION_FIELDS
    }
    return value


def _copy_required_source(project: Path) -> None:
    for relative in tool.REQUIRED_SOURCE_PATHS:
        source = ROOT / relative
        target = project / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)


def _write_phase_a(project: Path, *, frozen: bool = True, signed: bool = True) -> str:
    (project / "configs").mkdir(parents=True, exist_ok=True)
    protocol_path = project / "configs" / "protocol.yaml"
    protocol_path.write_text(
        yaml.safe_dump(_protocol(frozen=frozen), sort_keys=False, allow_unicode=False),
        encoding="utf-8",
        newline="\n",
    )
    (project / "PREREGISTRATION_SIGNOFF.md").write_text(
        _signed_record(signed=signed), encoding="utf-8", newline="\n"
    )
    _copy_required_source(project)
    _run_git(project, "init", "--quiet")
    _run_git(project, "config", "user.email", "freeze-test@example.invalid")
    _run_git(project, "config", "user.name", "Freeze Test")
    _run_git(project, "add", ".")
    _run_git(project, "commit", "--quiet", "-m", "Phase A freeze")
    return _run_git(project, "rev-parse", "HEAD")


def _write_yaml(path: Path, value: dict[str, Any]) -> None:
    path.write_text(
        yaml.safe_dump(value, sort_keys=False, allow_unicode=False),
        encoding="utf-8",
        newline="\n",
    )


def _generate_matrix(project: Path) -> Path:
    protocol_path = project / "configs" / "protocol.yaml"
    protocol = tool.load_protocol(protocol_path)
    raw_runs = tool.generate_run_matrix(protocol)
    matrix_dir = project / "configs" / "generated"
    matrix_dir.mkdir(parents=True)
    rows: list[dict[str, Any]] = []
    for raw in raw_runs:
        resolved = tool.validate_run_mapping(raw, protocol)
        config_path = matrix_dir / f"{resolved.runtime.run_id}.yaml"
        _write_yaml(config_path, resolved.as_dict())
        rows.append(tool._manifest_row(resolved, config_path))
    with (matrix_dir / "run_manifest.csv").open(
        "w", encoding="utf-8-sig", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    matrix_manifest = {
        "protocol": protocol_path.relative_to(project).as_posix(),
        "protocol_hash": tool._stable_hash(protocol),
        "run_count": len(raw_runs),
        "conditions": list(tool.active_conditions_for_protocol(protocol)),
        "seeds": protocol["seeds"],
        "matrix_hash": tool._stable_hash(raw_runs),
        "runs": rows,
    }
    (matrix_dir / "matrix_manifest.json").write_text(
        json.dumps(matrix_manifest, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    return matrix_dir


def _complete_project(tmp_path: Path) -> tuple[Path, str, Path]:
    project = tmp_path / "project"
    project.mkdir()
    commit = _write_phase_a(project)
    matrix_dir = _generate_matrix(project)
    return project, commit, matrix_dir


def _create(project: Path, commit: str, matrix_dir: Path) -> dict[str, Any]:
    return tool.create_manifest(
        project_root=project,
        protocol_path=Path("configs/protocol.yaml"),
        signoff_path=Path("PREREGISTRATION_SIGNOFF.md"),
        matrix_dir=matrix_dir,
        output_path=Path("FREEZE_MANIFEST.json"),
        freeze_commit=commit,
        repository_url="https://example.invalid/ta-lif-msresnet.git",
        created_at="2026-07-21T19:00:00+08:00",
    )


def test_create_binds_two_stages_without_modifying_signed_inputs(tmp_path: Path) -> None:
    project, commit, matrix_dir = _complete_project(tmp_path)
    protocol_path = project / "configs" / "protocol.yaml"
    signoff_path = project / "PREREGISTRATION_SIGNOFF.md"
    before = (protocol_path.read_bytes(), signoff_path.read_bytes())

    manifest = _create(project, commit, matrix_dir)

    assert (protocol_path.read_bytes(), signoff_path.read_bytes()) == before
    assert manifest["repository"]["freeze_commit"] == commit
    assert manifest["protocol"]["canonical_sha256"] == tool._stable_hash(
        tool.load_protocol(protocol_path)
    )
    assert manifest["generated_matrix"]["run_count"] == 40
    matrix_manifest = json.loads(
        (matrix_dir / "matrix_manifest.json").read_text(encoding="utf-8")
    )
    assert matrix_manifest["protocol"] == "configs/protocol.yaml"
    assert "manifest_sha256" not in manifest
    assert tool.verify_manifest(
        project_root=project, manifest_path=Path("FREEZE_MANIFEST.json")
    ) == manifest


@pytest.mark.parametrize(
    ("frozen", "signed", "message"),
    [
        (False, True, "never changes it automatically"),
        (True, False, "not in the exact signed state"),
    ],
)
def test_incomplete_phase_a_fails_without_output(
    tmp_path: Path, frozen: bool, signed: bool, message: str
) -> None:
    project = tmp_path / "project"
    project.mkdir()
    commit = _write_phase_a(project, frozen=frozen, signed=signed)
    matrix_dir = project / "configs" / "generated"
    matrix_dir.mkdir()

    with pytest.raises(tool.FreezeManifestError, match=message):
        _create(project, commit, matrix_dir)

    assert not (project / "FREEZE_MANIFEST.json").exists()


def test_uncommitted_signed_record_fails_without_output(tmp_path: Path) -> None:
    project, commit, matrix_dir = _complete_project(tmp_path)
    signoff = project / "PREREGISTRATION_SIGNOFF.md"
    signoff.write_text(
        signoff.read_text(encoding="utf-8") + "\nPost-commit edit.\n",
        encoding="utf-8",
        newline="\n",
    )

    with pytest.raises(tool.FreezeManifestError, match="differs from the Phase A"):
        _create(project, commit, matrix_dir)

    assert not (project / "FREEZE_MANIFEST.json").exists()


def test_matrix_tampering_fails_without_output(tmp_path: Path) -> None:
    project, commit, matrix_dir = _complete_project(tmp_path)
    config_path = next(matrix_dir.glob("*.yaml"))
    config_path.write_text(
        config_path.read_text(encoding="utf-8") + "# changed after generation\n",
        encoding="utf-8",
        newline="\n",
    )

    with pytest.raises(tool.FreezeManifestError, match="manifest"):
        _create(project, commit, matrix_dir)

    assert not (project / "FREEZE_MANIFEST.json").exists()


def test_existing_manifest_is_not_overwritten(tmp_path: Path) -> None:
    project, commit, matrix_dir = _complete_project(tmp_path)
    _create(project, commit, matrix_dir)
    output = project / "FREEZE_MANIFEST.json"
    before = output.read_bytes()

    with pytest.raises(tool.FreezeManifestError, match="Refusing to overwrite"):
        _create(project, commit, matrix_dir)

    assert output.read_bytes() == before


def test_verify_rejects_runtime_source_changed_after_freeze(tmp_path: Path) -> None:
    project, _commit, matrix_dir = _complete_project(tmp_path)
    _create(project, _commit, matrix_dir)
    source = project / "src" / "talif_msresnet" / "config.py"
    source.write_text(source.read_text(encoding="utf-8") + "\n# post-freeze edit\n", encoding="utf-8")

    with pytest.raises(tool.FreezeManifestError, match="executable source differs"):
        tool.verify_manifest(project_root=project, manifest_path=Path("FREEZE_MANIFEST.json"))
