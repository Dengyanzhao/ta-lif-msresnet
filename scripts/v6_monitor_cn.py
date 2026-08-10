#!/usr/bin/env python3
"""只读显示 V6 实验进度、单 run 预计完成时间和整体预计完成时间。"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import subprocess
import sys
import time
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) in sys.path:
    sys.path.remove(str(SRC_ROOT))
sys.path.insert(0, str(SRC_ROOT))

from talif_msresnet.config import generate_run_matrix, load_protocol
from talif_msresnet.config_v6 import (
    V6_ACTIVE_CONDITIONS,
    generate_v6_pilot_matrix,
    validate_v6_protocol,
)


@dataclass(frozen=True)
class RunProgress:
    run_id: str
    condition: str
    index: int
    total: int
    state: str
    epoch_completed: int
    epoch_total: int
    elapsed_seconds: float | None
    expected_total_seconds: float | None
    note: str


@dataclass(frozen=True)
class MonitorSnapshot:
    stage: str
    results_root: Path
    total_runs: int
    runs: tuple[RunProgress, ...]
    overall_remaining_seconds: float | None
    overall_finish_time: datetime | None
    gpu_summary: str | None


class V6MonitorError(RuntimeError):
    """Raised when the frozen V6 monitoring inputs are malformed."""


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--protocol",
        type=Path,
        default=PROJECT_ROOT / "configs" / "protocol_v6_mechanism.yaml",
    )
    parser.add_argument(
        "--stage",
        choices=("pilot", "formal"),
        default="formal",
        help="pilot 监控 6 个 non-reportable run；formal 监控 48 个正式 run。",
    )
    parser.add_argument(
        "--watch-seconds",
        type=float,
        default=0.0,
        help="大于零时循环刷新；例如 60。",
    )
    parser.add_argument("--json", action="store_true", help="输出机器可读快照。")
    parser.add_argument("--no-gpu", action="store_true", help="不查询 nvidia-smi。")
    return parser


def _read_json(path: Path) -> dict[str, Any] | None:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) else None


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    records: list[dict[str, Any]] = []
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeError):
        return records
    for line in lines:
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            records.append(value)
    return records


def _parse_time(value: Any) -> datetime | None:
    if not isinstance(value, str) or not value.strip():
        return None
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=timezone.utc)
    return parsed


def _finite(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    number = float(value)
    return number if math.isfinite(number) and number >= 0 else None


def _expected_runs(protocol: Mapping[str, Any], stage: str) -> list[dict[str, Any]]:
    raw = generate_v6_pilot_matrix(protocol) if stage == "pilot" else generate_run_matrix(protocol)
    expected_conditions = V6_ACTIVE_CONDITIONS
    result: list[dict[str, Any]] = []
    for item in raw:
        run_id = item.get("run_id")
        condition = item.get("model", {}).get("condition")
        if not isinstance(run_id, str) or condition not in expected_conditions:
            raise V6MonitorError("V6 运行矩阵与冻结条件定义不一致")
        result.append({"run_id": run_id, "condition": str(condition)})
    expected_count = 6 if stage == "pilot" else 48
    if len(result) != expected_count:
        raise V6MonitorError(f"V6 {stage} 矩阵数量错误：{len(result)} != {expected_count}")
    return result


def _results_root(protocol: Mapping[str, Any], stage: str) -> Path:
    artifacts = protocol["artifact_paths"]
    if stage == "formal":
        return (PROJECT_ROOT / str(artifacts["formal_results"])).resolve()
    acceptance = protocol["pilot_acceptance"]
    return (PROJECT_ROOT / str(acceptance["pilot_output_root"])).resolve()


def _matrix_events(root: Path) -> dict[str, dict[str, Any]]:
    states: dict[str, dict[str, Any]] = {}
    for event in _read_jsonl(root / "matrix_events.jsonl"):
        run_id = event.get("run_id")
        if not isinstance(run_id, str):
            continue
        current = states.setdefault(run_id, {})
        kind = event.get("event")
        if kind == "run_started":
            current["started"] = event
        elif kind in {"run_finished", "run_failed", "run_blocked", "run_skipped_complete"}:
            current["terminal"] = event
    return states


def _epoch_progress(run_dir: Path) -> tuple[int, float | None, datetime | None]:
    epochs = [
        event
        for event in _read_jsonl(run_dir / "events.jsonl")
        if event.get("event") == "epoch_completed"
    ]
    if not epochs:
        return 0, None, None
    timestamps = [_parse_time(event.get("timestamp")) for event in epochs]
    timestamps = [item for item in timestamps if item is not None]
    start = timestamps[0] if timestamps else None
    latest = timestamps[-1] if timestamps else None
    seconds: float | None = None
    if start is not None and latest is not None and len(epochs) > 1:
        seconds = max(0.0, (latest - start).total_seconds())
    elif len(epochs) == 1:
        train = epochs[0].get("train")
        if isinstance(train, Mapping):
            seconds = _finite(train.get("seconds"))
    return len(epochs), seconds, latest


def _completed_duration(run_dir: Path, matrix_state: Mapping[str, Any]) -> float | None:
    terminal = matrix_state.get("terminal")
    if isinstance(terminal, Mapping):
        duration = _finite(terminal.get("duration_s"))
        if duration is not None:
            return duration
    metrics = _read_json(run_dir / "seed_metrics.json")
    if metrics and metrics.get("status") == "complete":
        duration = _finite(metrics.get("duration_s"))
        if duration is not None:
            return duration
        started = _parse_time(metrics.get("started_at"))
        finished = _parse_time(metrics.get("finished_at"))
        if started is not None and finished is not None:
            return max(0.0, (finished - started).total_seconds())
    return None


def _state_for_run(
    *,
    run_id: str,
    condition: str,
    index: int,
    total: int,
    root: Path,
    matrix_state: Mapping[str, Any],
    epoch_total: int,
    now: datetime,
) -> RunProgress:
    run_dir = root / run_id
    metrics = _read_json(run_dir / "seed_metrics.json")
    failure = (run_dir / "failure.json").is_file()
    terminal = matrix_state.get("terminal")
    terminal_event = terminal.get("event") if isinstance(terminal, Mapping) else None
    duration = _completed_duration(run_dir, matrix_state)
    epochs, epoch_elapsed, _latest_epoch = _epoch_progress(run_dir)
    if metrics and metrics.get("status") == "complete" and not failure:
        return RunProgress(
            run_id,
            condition,
            index,
            total,
            "completed",
            epoch_total,
            epoch_total,
            duration,
            duration,
            "完成",
        )
    if failure or terminal_event in {"run_failed", "run_blocked"}:
        message = terminal.get("message") if isinstance(terminal, Mapping) else None
        return RunProgress(
            run_id,
            condition,
            index,
            total,
            "failed",
            epochs,
            epoch_total,
            duration,
            None,
            str(message or "失败或被门禁阻止"),
        )
    started = matrix_state.get("started")
    if isinstance(started, Mapping):
        start_unix = _finite(started.get("timestamp"))
        elapsed = max(0.0, now.timestamp() - start_unix) if start_unix is not None else None
        expected: float | None = None
        if epochs > 0 and elapsed is not None:
            expected = elapsed * epoch_total / epochs
        elif epoch_elapsed is not None and epochs > 0:
            expected = epoch_elapsed * epoch_total / max(1, epochs - 1)
        return RunProgress(
            run_id,
            condition,
            index,
            total,
            "running",
            epochs,
            epoch_total,
            elapsed,
            expected,
            "按已完成 epoch 外推" if expected is not None else "等待首个 epoch 完成后估算",
        )
    if terminal_event == "run_skipped_complete":
        return RunProgress(
            run_id,
            condition,
            index,
            total,
            "completed",
            epoch_total,
            epoch_total,
            duration,
            duration,
            "已跳过的完成 run",
        )
    return RunProgress(
        run_id, condition, index, total, "pending", 0, epoch_total, None, None, "等待启动"
    )


def _condition_medians(runs: Iterable[RunProgress]) -> tuple[dict[str, float], float | None]:
    grouped: dict[str, list[float]] = {}
    all_durations: list[float] = []
    for run in runs:
        if run.state != "completed" or run.expected_total_seconds is None:
            continue
        grouped.setdefault(run.condition, []).append(run.expected_total_seconds)
        all_durations.append(run.expected_total_seconds)
    medians = {condition: statistics.median(values) for condition, values in grouped.items()}
    return medians, statistics.median(all_durations) if all_durations else None


def _gpu_summary() -> str | None:
    try:
        output = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=name,utilization.gpu,memory.used,memory.total,temperature.gpu",
                "--format=csv,noheader,nounits",
            ],
            text=True,
            stderr=subprocess.DEVNULL,
            timeout=10,
        ).strip()
    except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired):
        return None
    return output or None


def collect_snapshot(
    protocol: Mapping[str, Any],
    *,
    stage: str,
    now: datetime | None = None,
    include_gpu: bool = True,
) -> MonitorSnapshot:
    now = now or datetime.now().astimezone()
    rows = _expected_runs(protocol, stage)
    root = _results_root(protocol, stage)
    events = _matrix_events(root)
    epoch_total = (
        int(protocol["pilot_acceptance"]["pilot"]["epochs"])
        if stage == "pilot"
        else int(protocol["optimizer"]["epochs"])
    )
    runs = tuple(
        _state_for_run(
            run_id=row["run_id"],
            condition=row["condition"],
            index=index,
            total=len(rows),
            root=root,
            matrix_state=events.get(row["run_id"], {}),
            epoch_total=epoch_total,
            now=now,
        )
        for index, row in enumerate(rows, start=1)
    )
    condition_medians, global_median = _condition_medians(runs)
    remaining: float | None = 0.0
    for run in runs:
        if run.state == "completed":
            continue
        if run.state == "failed":
            remaining = None
            break
        expected = (
            run.expected_total_seconds or condition_medians.get(run.condition) or global_median
        )
        if expected is None:
            remaining = None
            break
        elapsed = run.elapsed_seconds if run.state == "running" and run.elapsed_seconds else 0.0
        remaining += max(0.0, expected - elapsed)
    finish = (
        now.fromtimestamp(now.timestamp() + remaining, tz=now.tzinfo)
        if remaining is not None
        else None
    )
    return MonitorSnapshot(
        stage=stage,
        results_root=root,
        total_runs=len(rows),
        runs=runs,
        overall_remaining_seconds=remaining,
        overall_finish_time=finish,
        gpu_summary=None if not include_gpu else _gpu_summary(),
    )


def _format_duration(seconds: float | None) -> str:
    if seconds is None:
        return "待估算"
    seconds = max(0, round(seconds))
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours}小时{minutes:02d}分{seconds:02d}秒"


def _format_time(value: datetime | None) -> str:
    return (
        value.astimezone().strftime("%Y-%m-%d %H:%M:%S %z")
        if value
        else "待首个有效 epoch/run 后估算"
    )


def _as_json(snapshot: MonitorSnapshot) -> dict[str, Any]:
    return {
        "stage": snapshot.stage,
        "results_root": str(snapshot.results_root),
        "total_runs": snapshot.total_runs,
        "completed": sum(run.state == "completed" for run in snapshot.runs),
        "running": sum(run.state == "running" for run in snapshot.runs),
        "failed": sum(run.state == "failed" for run in snapshot.runs),
        "pending": sum(run.state == "pending" for run in snapshot.runs),
        "overall_remaining_seconds": snapshot.overall_remaining_seconds,
        "overall_finish_time": snapshot.overall_finish_time.isoformat()
        if snapshot.overall_finish_time
        else None,
        "gpu_summary": snapshot.gpu_summary,
        "runs": [run.__dict__ for run in snapshot.runs],
    }


def print_snapshot(snapshot: MonitorSnapshot) -> None:
    completed = sum(run.state == "completed" for run in snapshot.runs)
    running = sum(run.state == "running" for run in snapshot.runs)
    failed = sum(run.state == "failed" for run in snapshot.runs)
    pending = sum(run.state == "pending" for run in snapshot.runs)
    print(
        f"V6 中文监控 | 阶段: {snapshot.stage} | 当前时间: {datetime.now().astimezone():%Y-%m-%d %H:%M:%S %z}"
    )
    print(f"结果目录: {snapshot.results_root}")
    print(
        f"总进度: 完成 {completed}/{snapshot.total_runs}，运行中 {running}，等待 {pending}，失败/阻断 {failed}"
    )
    print(f"剩余预计: {_format_duration(snapshot.overall_remaining_seconds)}")
    print(f"预计全部完成: {_format_time(snapshot.overall_finish_time)}")
    if snapshot.gpu_summary:
        print(f"GPU: {snapshot.gpu_summary}")
    print("-" * 108)
    print(f"{'序号':<5}{'条件':<8}{'状态':<12}{'Epoch':<12}{'本 run 剩余':<18}{'Run ID'}")
    for run in snapshot.runs:
        remaining = None
        if run.expected_total_seconds is not None:
            remaining = max(0.0, run.expected_total_seconds - (run.elapsed_seconds or 0.0))
        epoch = f"{run.epoch_completed}/{run.epoch_total}"
        print(
            f"{run.index:<5}{run.condition:<8}{run.state:<12}{epoch:<12}"
            f"{_format_duration(remaining):<18}{run.run_id}"
        )
        if run.state in {"failed", "running"}:
            print(f"     说明: {run.note}")
    if failed:
        print("检测到失败/阻断。不要自动重跑；先保留日志，并按 runbook 的恢复规则处理。")


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.watch_seconds < 0:
        raise SystemExit("--watch-seconds 必须大于或等于 0")
    protocol = validate_v6_protocol(load_protocol(args.protocol.resolve()))
    while True:
        snapshot = collect_snapshot(protocol, stage=args.stage, include_gpu=not args.no_gpu)
        if args.json:
            print(json.dumps(_as_json(snapshot), ensure_ascii=False, sort_keys=True))
        else:
            if args.watch_seconds > 0 and sys.stdout.isatty():
                print("\033[2J\033[H", end="")
            print_snapshot(snapshot)
        if args.watch_seconds <= 0:
            return 0
        try:
            time.sleep(args.watch_seconds)
        except KeyboardInterrupt:
            print("\n已停止监控。实验进程未被修改。")
            return 0


if __name__ == "__main__":
    raise SystemExit(main())
