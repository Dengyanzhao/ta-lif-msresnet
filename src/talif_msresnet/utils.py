"""Small reproducibility, logging, and checkpoint helpers."""

from __future__ import annotations

import csv
import hashlib
import json
import os
import platform
import random
import socket
import tempfile
import time
from contextlib import contextmanager
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version as distribution_version
from pathlib import Path
from typing import Any, Dict, Iterator, Mapping, Sequence

import numpy as np
import torch


DETERMINISTIC_CUBLAS_WORKSPACE_CONFIG = ":4096:8"
SUPPORTED_CUBLAS_WORKSPACE_CONFIGS = frozenset({":4096:8", ":16:8"})


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def seed_everything(seed: int, deterministic: bool = True) -> None:
    """Seed Python, NumPy, and PyTorch without hiding determinism failures."""

    if deterministic:
        # CuBLAS reads this before its first workspace-backed operation. Set it
        # before any CUDA seeding or tensor work in this process.
        workspace_config = os.environ.setdefault(
            "CUBLAS_WORKSPACE_CONFIG", DETERMINISTIC_CUBLAS_WORKSPACE_CONFIG
        )
        if workspace_config not in SUPPORTED_CUBLAS_WORKSPACE_CONFIGS:
            raise RuntimeError(
                "Unsupported CUBLAS_WORKSPACE_CONFIG for deterministic execution: "
                f"{workspace_config!r}; use :4096:8 or :16:8"
            )
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        try:
            torch.use_deterministic_algorithms(True)
        except (AttributeError, TypeError):  # older PyTorch
            pass


def worker_seed_fn(worker_id: int) -> None:
    """DataLoader worker initializer derived from PyTorch's worker seed."""

    seed = torch.initial_seed() % (2**32)
    np.random.seed(seed)
    random.seed(seed)


def json_safe(value: Any) -> Any:
    """Convert tensors and NumPy values to JSON-safe structures."""

    if isinstance(value, torch.Tensor):
        if value.numel() == 1:
            return value.detach().cpu().item()
        return value.detach().cpu().tolist()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [json_safe(v) for v in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return repr(value)


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: str | Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while True:
            block = handle.read(chunk_size)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def stable_hash(value: Any) -> str:
    payload = json.dumps(json_safe(value), sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return sha256_bytes(payload.encode("utf-8"))


def _atomic_replace(path: Path, writer: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent))
    os.close(fd)
    temp_path = Path(temporary)
    try:
        writer(temp_path)
        os.replace(temp_path, path)
    finally:
        temp_path.unlink(missing_ok=True)


def atomic_write_json(path: str | Path, value: Any) -> None:
    path = Path(path)

    def writer(temp_path: Path) -> None:
        with temp_path.open("w", encoding="utf-8", newline="\n") as handle:
            json.dump(json_safe(value), handle, indent=2, sort_keys=True, ensure_ascii=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())

    _atomic_replace(path, writer)


def append_jsonl(path: str | Path, value: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(json_safe(value), sort_keys=True, ensure_ascii=False) + "\n"
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(line)
        handle.flush()
        os.fsync(handle.fileno())


@contextmanager
def file_lock(target: str | Path, timeout: float = 30.0) -> Iterator[None]:
    """Portable lock-file guard for short manifest/CSV appends."""

    lock_path = Path(f"{target}.lock")
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    deadline = time.monotonic() + timeout
    fd: int | None = None
    while fd is None:
        try:
            fd = os.open(str(lock_path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            os.write(fd, f"{os.getpid()} {utc_now()}\n".encode("ascii"))
        except FileExistsError:
            # Clear only clearly stale locks; a live writer holds the file for
            # milliseconds, so five minutes is deliberately conservative.
            try:
                age = time.time() - lock_path.stat().st_mtime
                if age > 300:
                    lock_path.unlink(missing_ok=True)
                    continue
            except FileNotFoundError:
                continue
            if time.monotonic() >= deadline:
                raise TimeoutError(f"Timed out waiting for {lock_path}")
            time.sleep(0.05)
    try:
        yield
    finally:
        if fd is not None:
            os.close(fd)
        lock_path.unlink(missing_ok=True)


def append_csv_row(path: str | Path, row: Mapping[str, Any], fieldnames: Sequence[str]) -> None:
    """Append a stable-schema CSV row, protected against concurrent runs."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    unknown = set(row) - set(fieldnames)
    if unknown:
        raise ValueError(f"CSV row has unknown fields: {sorted(unknown)}")
    with file_lock(path):
        exists = path.exists() and path.stat().st_size > 0
        if exists:
            with path.open("r", encoding="utf-8-sig", newline="") as handle:
                header = next(csv.reader(handle), [])
            if list(header) != list(fieldnames):
                raise ValueError(f"Existing CSV header differs from expected schema: {path}")
        with path.open("a", encoding="utf-8-sig", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="raise")
            if not exists:
                writer.writeheader()
            writer.writerow({key: json_safe(row.get(key, "")) for key in fieldnames})
            handle.flush()
            os.fsync(handle.fileno())


def upsert_csv_row(
    path: str | Path,
    row: Mapping[str, Any],
    fieldnames: Sequence[str],
    *,
    key_field: str,
    require_existing: bool = False,
) -> None:
    """Atomically insert or replace one uniquely keyed row.

    Duplicate existing keys are rejected because silently choosing one would
    corrupt the seed-block audit trail.
    """

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if key_field not in fieldnames or key_field not in row:
        raise ValueError(f"Missing CSV key field {key_field!r}")
    unknown = set(row) - set(fieldnames)
    if unknown:
        raise ValueError(f"CSV row has unknown fields: {sorted(unknown)}")
    key = str(row[key_field])
    with file_lock(path):
        existing: list[dict[str, Any]] = []
        if path.exists() and path.stat().st_size > 0:
            with path.open("r", encoding="utf-8-sig", newline="") as handle:
                reader = csv.DictReader(handle)
                if list(reader.fieldnames or []) != list(fieldnames):
                    raise ValueError(f"Existing CSV header differs from expected schema: {path}")
                existing = list(reader)
        matches = [index for index, item in enumerate(existing) if str(item.get(key_field, "")) == key]
        if len(matches) > 1:
            raise ValueError(f"Existing CSV contains duplicate {key_field}={key!r}: {path}")
        if require_existing and not matches:
            raise ValueError(f"No existing CSV row for {key_field}={key!r}: {path}")
        normalized = {name: json_safe(row.get(name, "")) for name in fieldnames}
        if matches:
            existing[matches[0]] = normalized
        else:
            existing.append(normalized)

        def writer(temp_path: Path) -> None:
            with temp_path.open("w", encoding="utf-8-sig", newline="") as handle:
                csv_writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="raise")
                csv_writer.writeheader()
                csv_writer.writerows(existing)
                handle.flush()
                os.fsync(handle.fileno())

        _atomic_replace(path, writer)


class JSONLLogger:
    """Context-free event logger used by each run."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)

    def log(self, event: str, **values: Any) -> None:
        append_jsonl(self.path, {"timestamp": utc_now(), "event": event, **values})


def environment_manifest() -> Dict[str, Any]:
    cuda_device = None
    if torch.cuda.is_available():
        cuda_device = torch.cuda.get_device_name(torch.cuda.current_device())
    try:
        torchvision_version: str | None = distribution_version("torchvision")
    except PackageNotFoundError:
        torchvision_version = None
    return {
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "pytorch": torch.__version__,
        "torchvision": torchvision_version,
        "numpy": np.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda,
        "cuda_device": cuda_device,
        "pid": os.getpid(),
    }


def capture_rng_state() -> Dict[str, Any]:
    state: Dict[str, Any] = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def restore_rng_state(state: Mapping[str, Any]) -> None:
    if "python" in state:
        random.setstate(state["python"])
    if "numpy" in state:
        np.random.set_state(state["numpy"])
    if "torch" in state:
        torch.set_rng_state(state["torch"])
    if "cuda" in state and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda"])


def atomic_torch_save(path: str | Path, value: Any) -> None:
    path = Path(path)

    def writer(temp_path: Path) -> None:
        torch.save(value, temp_path)

    _atomic_replace(path, writer)


def save_checkpoint(path: str | Path, payload: Mapping[str, Any]) -> None:
    required = {
        "model_state", "optimizer_state", "scheduler_state", "epoch",
        "best_val", "config", "config_hash",
    }
    missing = required - set(payload)
    if missing:
        raise ValueError(f"Checkpoint missing required keys: {sorted(missing)}")
    atomic_torch_save(path, dict(payload))


def load_checkpoint(path: str | Path, map_location: str | torch.device = "cpu") -> Dict[str, Any]:
    try:
        value = torch.load(Path(path), map_location=map_location, weights_only=False)
    except TypeError:  # PyTorch before weights_only
        value = torch.load(Path(path), map_location=map_location)
    if not isinstance(value, dict):
        raise ValueError(f"Checkpoint is not a mapping: {path}")
    return value


class AverageMeter:
    def __init__(self) -> None:
        self.total = 0.0
        self.count = 0

    def update(self, value: float, count: int = 1) -> None:
        self.total += float(value) * int(count)
        self.count += int(count)

    @property
    def average(self) -> float:
        return self.total / self.count if self.count else float("nan")


def resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"CUDA device requested but CUDA is unavailable: {requested}")
    return device
