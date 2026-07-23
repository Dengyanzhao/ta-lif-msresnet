# Hengyuan Cloud RTX 5090 smoke runbook

## Material Passport

- Origin Skill: academic-research-suite / experiment-agent
- Origin Mode: plan
- Origin Date: 2026-07-22
- Verification Status: VERIFIED against the current local repository interfaces
- Version Label: hengyuan_5090_smoke_v2

## 1. Scope

This runbook performs eight engineering-only dry runs:

- CIFAR-100/depth-20/T=6: C1, C2, C3, C4, seed 11;
- CIFAR10-DVS/depth-20/T=10: C1, C2, C3, C4, seed 11.

Each run keeps the protocol batch size of 64 and executes one epoch containing
at most two training batches and two validation batches. These outputs are not
paper results, do not count toward the five formal seeds, and must never be
copied into `results/runs`.

Do not add `--allow-unfrozen-pilot`. That flag starts a different, full
240-epoch non-reportable pilot and is incompatible with `--dry-run`.

## 2. Locations used in this runbook

There are three separate locations:

1. Local Windows project:
   `D:\论文解构\王\ta-lif-msresnet`
2. Hengyuan Cloud upload/data area: shown by the Hengyuan file manager. The
   examples below use `/hy-tmp`; replace it if the mounted data disk differs.
3. Hengyuan Cloud Linux project root:
   `/hy-tmp/ta-lif-msresnet`

All PowerShell commands in Section 4 run on the local Windows computer. All
Bash commands from Section 6 onward run in the Hengyuan Cloud terminal.

## 3. What the upload must contain

Required source and protocol content:

```text
ta-lif-msresnet/
  pyproject.toml
  README.md
  HENGYUAN_5090_SMOKE_RUNBOOK.md
  .gitignore
  CITATION.cff.template
  IMPLEMENTATION_DECISIONS.md
  LICENSE_CHOICE_REQUIRED.md
  PREREGISTRATION_SIGNOFF.md
  THIRD_PARTY_NOTICES.md
  configs/
    protocol.yaml
    generated/                 # 40 YAML + 2 matrix manifests
  environment/
  examples/
  scripts/
  src/
  tests/
  checkpoints/README.md
  results/README.md
  results/run_manifest_template.csv
  results/seed_metrics_template.csv
```

Required data content:

```text
ta-lif-msresnet/data/
  README.md
  manifests/
    cifar100_seed2024.json
    cifar10dvs_seed2024.json
  cifar100/
    cifar-100-python.tar.gz
    cifar-100-python/
      train
      test
      meta
  cifar10dvs/
    processed/
      index.csv
      conversion_manifest.json
      source_manifest.json
      split_manifest.json
      verification_receipt.json
      samples/                 # exactly 10,000 .pt files
```

Expected upload payload:

- CIFAR-100: 355,302,535 bytes, about 0.331 GiB;
- CIFAR10-DVS processed: 1,863,740,132 bytes, about 1.736 GiB;
- DVS tensors: exactly 10,000 files totaling 1,858,970,000 bytes;
- generated configs: exactly 40 YAML files plus `matrix_manifest.json` and
  `run_manifest.csv`.

Do not upload:

- `.venv`, `.pytest_cache`, `.ruff_cache`, `.tmp-aedat-wheel`, `.git`;
- `data/cifar10dvs/raw` or the approximately 15 GB AEDAT4 source tree;
- `data/cifar10`, old checkpoints, old smoke results;
- `configs/smoke_generated`;
- Windows `.whl` files or the old Windows virtual environment.

## 4. Create one upload archive on Windows

Open Windows PowerShell. Run the following exactly:

```powershell
Set-Location -LiteralPath 'D:\论文解构\王'

$archive = 'D:\论文解构\王\ta-lif-msresnet-smoke-upload-20260722.tar.gz'

tar.exe -czf $archive `
  --exclude '*/__pycache__' `
  --exclude '*.pyc' `
  --exclude '*.pyo' `
  --exclude '*.egg-info' `
  'ta-lif-msresnet/.gitignore' `
  'ta-lif-msresnet/pyproject.toml' `
  'ta-lif-msresnet/README.md' `
  'ta-lif-msresnet/HENGYUAN_5090_SMOKE_RUNBOOK.md' `
  'ta-lif-msresnet/CITATION.cff.template' `
  'ta-lif-msresnet/IMPLEMENTATION_DECISIONS.md' `
  'ta-lif-msresnet/LICENSE_CHOICE_REQUIRED.md' `
  'ta-lif-msresnet/PREREGISTRATION_SIGNOFF.md' `
  'ta-lif-msresnet/THIRD_PARTY_NOTICES.md' `
  'ta-lif-msresnet/configs/protocol.yaml' `
  'ta-lif-msresnet/configs/generated' `
  'ta-lif-msresnet/data/README.md' `
  'ta-lif-msresnet/data/manifests' `
  'ta-lif-msresnet/data/cifar100' `
  'ta-lif-msresnet/data/cifar10dvs/processed' `
  'ta-lif-msresnet/environment' `
  'ta-lif-msresnet/examples' `
  'ta-lif-msresnet/scripts' `
  'ta-lif-msresnet/src' `
  'ta-lif-msresnet/tests' `
  'ta-lif-msresnet/checkpoints/README.md' `
  'ta-lif-msresnet/results/README.md' `
  'ta-lif-msresnet/results/run_manifest_template.csv' `
  'ta-lif-msresnet/results/seed_metrics_template.csv'

Get-Item -LiteralPath $archive | Select-Object FullName,Length,LastWriteTime
Get-FileHash -Algorithm SHA256 -LiteralPath $archive

$entries = tar.exe -tf $archive
($entries | Where-Object { $_ -match '/data/cifar10dvs/processed/samples/.+\.pt$' }).Count
($entries | Where-Object { $_ -match '/configs/generated/E1_.+\.yaml$' }).Count
```

The last two lines must print `10000` and `40`. Upload only this `.tar.gz` file
through the Hengyuan Cloud file manager. Binary archive upload avoids newline
conversion and avoids uploading 10,000 tensors individually.

## 5. Create the Hengyuan instance

Choose:

- one RTX 5090 32 GB GPU;
- an Ubuntu PyTorch image explicitly marked compatible with RTX 5090 or
  Blackwell;
- Python 3.10 through 3.12;
- at least 50 GB free on the mounted data disk.

Do not select a blank CUDA image unless you are prepared to build the full
environment. Do not install or replace the NVIDIA driver or CUDA toolkit with
`apt`. The `CUDA Version` printed by `nvidia-smi` is the driver's supported
maximum, not necessarily PyTorch's bundled CUDA runtime.

## 6. Extract the archive on Hengyuan Cloud

Open the Hengyuan Linux terminal, not local PowerShell. Locate the uploaded
archive and data disk:

```bash
pwd
df -h
nvidia-smi
```

If the archive is under `/hy-tmp`, run:

```bash
cd /hy-tmp
tar -xzf ta-lif-msresnet-smoke-upload-20260722.tar.gz
cd /hy-tmp/ta-lif-msresnet

test -f pyproject.toml
test -f configs/protocol.yaml
test -f configs/generated/matrix_manifest.json
test -f data/cifar100/cifar-100-python/train
test -f data/cifar10dvs/processed/index.csv
test "$(find data/cifar10dvs/processed/samples -maxdepth 1 -type f -name '*.pt' | wc -l)" -eq 10000
test "$(find configs/generated -maxdepth 1 -type f -name 'E1_*.yaml' | wc -l)" -eq 40
du -sh data/cifar100 data/cifar10dvs/processed
```

If the archive is elsewhere, replace `/hy-tmp` in both `cd` commands with the
actual mounted data path. Keep the project path ASCII-only.

## 7. Start a persistent terminal session

Smoke is short, but DVS verification may take several minutes. Use `tmux` so a
browser disconnect does not stop it:

```bash
tmux new -s talif-smoke
```

Detach with `Ctrl-B`, then `D`. Reconnect with:

```bash
tmux attach -t talif-smoke
```

Run all remaining commands inside the same tmux shell.

## 8. Create the cloud Python environment

The preinstalled image must already import a 5090-compatible PyTorch and
torchvision pair:

```bash
cd /hy-tmp/ta-lif-msresnet
set -euo pipefail
export CUDA_VISIBLE_DEVICES=0
export PYTHONUNBUFFERED=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8

python -c "import torch, torchvision; print(torch.__version__, torchvision.__version__)"

mkdir -p "$HOME/venvs"
python -m venv --system-site-packages "$HOME/venvs/talif-msresnet"
source "$HOME/venvs/talif-msresnet/bin/activate"

python -m pip install --upgrade pip setuptools wheel
python -m pip install \
  'numpy>=1.26' 'PyYAML>=6.0' 'pandas>=2.1' 'scipy>=1.11' \
  'statsmodels>=0.14' 'tqdm>=4.66' 'pytest>=8.0' 'ruff>=0.5' \
  'tonic==1.6.0' 'aedat==2.2.0'
python -m pip install --no-deps -e .

python -m pip check
python -m compileall -q src scripts
python -m ruff check .
python -m pytest -q
```

The current local baseline is `149 passed`. The event packages are retained
even though smoke reads only preprocessed tensors: the conversion-integrity
tests require installed `aedat` version metadata and deliberately reject an
`unknown` decoder version. Stop if any command fails. Do not install
`environment/requirements-lock.txt` on Linux; it records the earlier Windows
RTX 4070 environment and Windows wheel hashes.

## 9. Create a new smoke evidence directory

Still in the project root and the activated virtual environment:

```bash
SMOKE_BASE="results/smoke/cloud_5090_01"

if [ -e "$SMOKE_BASE" ]; then
  echo "Smoke evidence directory already exists; use cloud_5090_02 instead."
  exit 1
fi

mkdir -p "$SMOKE_BASE/meta"
printf '%s\n' "$SMOKE_BASE" > "$SMOKE_BASE/meta/smoke-root.txt"
uname -a > "$SMOKE_BASE/meta/uname.txt"
nvidia-smi -q > "$SMOKE_BASE/meta/nvidia-smi.txt"
python -m pip freeze --exclude-editable > "$SMOKE_BASE/meta/pip-freeze.txt"
```

Do not reuse a smoke evidence directory after a failed attempt. Preserve the
old directory and use the next numbered suffix only after the cause is known.

## 10. Verify RTX 5090 and PyTorch CUDA compatibility

```bash
python - <<'PY' | tee "$SMOKE_BASE/meta/torch-check.txt"
import os
import platform
import torch
import torchvision
from talif_msresnet.utils import seed_everything
from torchvision.ops import nms

print("OS:", platform.platform())
print("Python:", platform.python_version())
print("PyTorch:", torch.__version__)
print("torchvision:", torchvision.__version__)
print("PyTorch CUDA runtime:", torch.version.cuda)
print("cuDNN:", torch.backends.cudnn.version())
print("CUDA available:", torch.cuda.is_available())
assert torch.cuda.is_available(), "CUDA unavailable or a CPU-only PyTorch wheel is installed"

p = torch.cuda.get_device_properties(0)
print("GPU:", p.name)
print("Compute capability:", f"{p.major}.{p.minor}")
print("VRAM GiB:", p.total_memory / 2**30)
print("Compiled architectures:", torch.cuda.get_arch_list())
assert "5090" in p.name, f"Expected RTX 5090, found {p.name}"

seed_everything(11, deterministic=True)
print("CUBLAS_WORKSPACE_CONFIG:", os.environ.get("CUBLAS_WORKSPACE_CONFIG"))
print("Torch deterministic algorithms:", torch.are_deterministic_algorithms_enabled())
print("Torch deterministic warn-only:", torch.is_deterministic_algorithms_warn_only_enabled())
print("cuDNN benchmark:", torch.backends.cudnn.benchmark)
print("cuDNN deterministic:", torch.backends.cudnn.deterministic)
assert torch.are_deterministic_algorithms_enabled()
assert not torch.is_deterministic_algorithms_warn_only_enabled()
assert not torch.backends.cudnn.benchmark
assert torch.backends.cudnn.deterministic

x = torch.randn(8, 3, 32, 32, device="cuda", requires_grad=True)
model = torch.nn.Conv2d(3, 16, 3, padding=1).cuda()
model(x).square().mean().backward()

boxes = torch.tensor([[0., 0., 10., 10.], [1., 1., 9., 9.]], device="cuda")
scores = torch.tensor([0.9, 0.8], device="cuda")
print("CUDA NMS:", nms(boxes, scores, 0.5))
torch.cuda.synchronize()
print("GPU_SANITY_PASS")
PY
```

Stop if this does not end with `GPU_SANITY_PASS`. A failure means the selected
image's PyTorch/torchvision pair is not a verified 5090 environment. Do not try
to repair it by installing a random CUDA toolkit or an older wheel.

## 11. Verify uploaded data

The following hashes bind the exact local CIFAR-100 files and fixed split
manifests:

```bash
md5sum -c <<'EOF' | tee "$SMOKE_BASE/meta/cifar100-md5.txt"
eb9058c3a382ffc7106e4002c42a8d85  data/cifar100/cifar-100-python.tar.gz
EOF

sha256sum -c <<'EOF' | tee "$SMOKE_BASE/meta/static-data-sha256.txt"
85cd44d02ba6437773c5bbd22e183051d648de2e7d6b014e1ef29b855ba677a7  data/cifar100/cifar-100-python.tar.gz
735e79b04f092ca3d2e6d07f368c0a7d70d48c48d28865950cc24454cf45129b  data/cifar100/cifar-100-python/train
4b67687d9933c4db8f0831104447f15b93774f4f464bd0516f0f0f2ac83b7864  data/cifar100/cifar-100-python/test
a5d4786345c961390f865e93b434dbd5c6904ce880667e0cb888c97d449f28b9  data/cifar100/cifar-100-python/meta
ade5f378ea8864b8c36711c4c3bc0e4e7016adb1eaded30d2dc5cf7b0d998f26  data/manifests/cifar100_seed2024.json
c9aa1856311feb944529304c6583cef344acea2447a39f8f6e37222346ec7e65  data/manifests/cifar10dvs_seed2024.json
EOF

python - <<'PY' | tee "$SMOKE_BASE/meta/cifar100-load.txt"
from torchvision.datasets import CIFAR100
d = CIFAR100("data/cifar100", train=True, download=False)
assert len(d) == 50000 and len(d.classes) == 100
print("CIFAR100_TRAIN_LOAD_PASS", len(d), len(d.classes))
PY

python scripts/verify_cifar10dvs.py \
  --root data/cifar10dvs/processed \
  --progress-every 500 \
  | tee "$SMOKE_BASE/meta/cifar10dvs-verify.txt"

cp data/cifar10dvs/processed/verification_receipt.json "$SMOKE_BASE/meta/"
```

The DVS verifier hashes and loads all 10,000 tensors. It normally rewrites the
verification receipt with a new verification time; this is allowed before
freeze. Do not manually edit the receipt or any DVS manifest.

## 12. Run strict preflight checks

```bash
python scripts/preflight.py --mode smoke \
  | tee "$SMOKE_BASE/meta/smoke-preflight.txt"

python scripts/preflight.py --mode pilot \
  | tee "$SMOKE_BASE/meta/pilot-preflight.txt"

python scripts/generate_run_configs.py --dry-run \
  | tee "$SMOKE_BASE/meta/config-dry-run.txt"
```

Both preflights must end in `PASS`. The expected canonical protocol hash is:

```text
34e2d640836d3f545168660c30a48d96170d32778c1c1f718c9cb7d43c43273b
```

Warnings that the protocol is unsigned/unfrozen and that AutoAugment, CutMix,
and label smoothing are disabled are expected. `--mode pilot` here performs a
stricter check only; it does not start pilot training.

The config dry run must print:

```text
Validated 40 unique configurations; no files written
```

Do not run `generate_run_configs.py --output` at this stage. The uploaded
provisional configs are sufficient for smoke and must be regenerated only after
the signed Phase A freeze commit.

## 13. Run the eight smoke jobs

Use DVS first and run C2/C4 first so memory failures are exposed early:

```bash
for dataset in cifar10dvs cifar100; do
  for condition in C2 C4 C1 C3; do
    echo "===== smoke dataset=$dataset condition=$condition =====" \
      | tee -a "$SMOKE_BASE/launcher.log"

    python scripts/run_matrix.py \
      --config-dir configs/generated \
      --dataset "$dataset" \
      --experiment E1 \
      --condition "$condition" \
      --limit 1 \
      --dry-run \
      --limit-batches 2 \
      --device cuda:0 \
      --output-root "$SMOKE_BASE" \
      --stop-on-error \
      2>&1 | tee -a "$SMOKE_BASE/launcher.log"
  done
done
```

Each invocation selects seed 11. The expected run directories are under
`$SMOKE_BASE/smoke/` and cover exactly these pairs:

```text
cifar10dvs: C1 C2 C3 C4
cifar100:   C1 C2 C3 C4
```

Open a second Hengyuan terminal for live monitoring if desired:

```bash
watch -n 1 nvidia-smi
```

Do not act only because GPU utilization briefly reaches zero; the process may
be loading data or writing checkpoints. A nonzero process exit, OOM, NaN/Inf,
or `failure.json` is the failure criterion.

## 14. Validate the smoke outputs

```bash
python - "$SMOKE_BASE" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1]) / "smoke"
expected = {
    (dataset, condition)
    for dataset in ("cifar100", "cifar10dvs")
    for condition in ("C1", "C2", "C3", "C4")
}
runs = sorted(path for path in root.iterdir() if path.is_dir() and path.name.startswith("E1_"))
found = set()
environment_hashes = set()

assert len(runs) == 8, f"Expected 8 run directories, found {len(runs)}"
for run in runs:
    required = (
        "seed_metrics.json", "run_manifest.json", "resolved_config.json",
        "events.jsonl", "best.pt", "last.pt",
        "attempt_001.stdout.log", "attempt_001.stderr.log",
    )
    missing = [name for name in required if not (run / name).is_file()]
    assert not missing, f"{run.name}: missing {missing}"
    assert not (run / "failure.json").exists(), f"{run.name}: failure.json exists"

    metrics = json.loads((run / "seed_metrics.json").read_text())
    config = json.loads((run / "resolved_config.json").read_text())
    identity = json.loads(metrics["training_environment_identity"])
    events = [json.loads(line) for line in (run / "events.jsonl").read_text().splitlines() if line]
    epochs = [event for event in events if event.get("event") == "epoch_completed"]

    found.add((metrics["dataset"], metrics["condition"]))
    environment_hashes.add(metrics["training_environment_sha256"])
    assert metrics["seed"] == 11
    assert metrics["status"] == "dry_run" and metrics["failed"] in (0, False)
    assert metrics["peak_training_memory_bytes"] > 0
    assert metrics.get("test_accuracy") in ("", None)
    assert config["optimizer"]["batch_size"] == 64
    assert config["runtime"]["dry_run"] is True
    assert config["runtime"]["limit_batches"] == 2
    assert "5090" in identity["hardware"]["name"]
    assert identity["precision"] == "float32"
    determinism = identity["determinism"]
    assert determinism["requested"] is True
    assert determinism["torch_algorithms_enabled"] is True
    assert determinism["torch_warn_only"] is False
    assert determinism["cudnn_benchmark"] is False
    assert determinism["cudnn_deterministic"] is True
    assert determinism["cublas_workspace_config"] in (":4096:8", ":16:8")
    assert len(epochs) == 1
    assert epochs[0]["train"]["samples"] == 128
    assert epochs[0]["val"]["samples"] == 128
    print(run.name, "PASS", metrics["peak_training_memory_bytes"] / 2**30, "GiB")

assert found == expected, f"Coverage mismatch: {found}"
assert len(environment_hashes) == 1, "The eight runs used different environments"
print("ALL_8_SMOKE_RUNS_PASS")
PY

if grep -R -n -E \
  'not deterministic|CUBLAS_WORKSPACE_CONFIG|Traceback|RuntimeError' \
  "$SMOKE_BASE/smoke"/*/attempt_*.stderr.log; then
  echo "Smoke stderr contains a determinism or runtime warning/error"
  exit 1
fi
```

The final line must be `ALL_8_SMOKE_RUNS_PASS`.

## 15. Re-run after the determinism fix

The first returned archive (`cloud_5090_01`) is retained as an engineering
record, but it must not be used as deterministic evidence. Upload
`ta_lif_determinism_fix_20260723.tar.gz` and its `.sha256` sidecar next to the
existing project archive. On the cloud terminal, verify and apply the patch:

```bash
cd /hy-tmp
sha256sum -c ta_lif_determinism_fix_20260723.tar.gz.sha256
tar -xzf ta_lif_determinism_fix_20260723.tar.gz -C /hy-tmp/ta-lif-msresnet
cd /hy-tmp/ta-lif-msresnet
set -euo pipefail
export CUDA_VISIBLE_DEVICES=0
export PYTHONUNBUFFERED=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8
python -m pytest -q
```

The patched baseline should report `152 passed`. Create a fresh evidence root;
never append to or rename `cloud_5090_01`:

```bash
SMOKE_BASE="results/smoke/cloud_5090_02"
test ! -e "$SMOKE_BASE"
mkdir -p "$SMOKE_BASE/meta"
printf '%s\n' "$SMOKE_BASE" > "$SMOKE_BASE/meta/smoke-root.txt"
uname -a > "$SMOKE_BASE/meta/uname.txt"
nvidia-smi -q > "$SMOKE_BASE/meta/nvidia-smi.txt"
python -m pip freeze --exclude-editable > "$SMOKE_BASE/meta/pip-freeze.txt"
python - <<'PY' | tee "$SMOKE_BASE/meta/determinism-env.txt"
import os
import torch
print("CUBLAS_WORKSPACE_CONFIG:", os.environ.get("CUBLAS_WORKSPACE_CONFIG"))
print("torch algorithms before trainer initialization:", torch.are_deterministic_algorithms_enabled())
assert os.environ.get("CUBLAS_WORKSPACE_CONFIG") in (":4096:8", ":16:8")
PY
```

`torch algorithms before trainer initialization` may print `False`; this is
expected because strict PyTorch determinism is enabled by the trainer after it
starts. Section 14 verifies `torch_algorithms_enabled=true` and
`torch_warn_only=false` from every actual smoke run.

Run Sections 10-14 again, replacing every `$SMOKE_BASE` value with
`results/smoke/cloud_5090_02`. The updated validator requires the determinism
fields in each run manifest and rejects any stderr containing a CuBLAS
determinism warning, traceback, or runtime error. Package the new root using
Section 16 and return both the archive and sidecar.

## 16. Package the evidence for return

```bash
ARCHIVE="../ta_lif_5090_smoke_$(date +%Y%m%d_%H%M%S).tar.gz"
tar -czf "$ARCHIVE" "$SMOKE_BASE"
sha256sum "$ARCHIVE" | tee "${ARCHIVE}.sha256"
echo "Download this file: $ARCHIVE"
echo "Download this checksum: ${ARCHIVE}.sha256"
```

Download the archive through the Hengyuan file manager and place it under:

```text
D:\论文解构\王
```

Then report the exact local filename. The archive contains environment identity,
preflight output, all eight run records, logs, metrics, and smoke checkpoints.
The adjacent `.sha256` file allows the returned archive to be checked after
download without changing the archive itself.

## 17. Failure handling

If any step fails:

1. Stop. Do not continue to the next job.
2. Preserve the complete `$SMOKE_BASE` directory.
3. Do not lower batch size, enable AMP, change workers, edit the protocol, or
   silently retry.
4. Do not use `--resume-matrix` for dry runs.
5. Package the partial `$SMOKE_BASE` directory using Section 15 and return it
   together with the terminal error text.
6. If the failure occurs before `$SMOKE_BASE` is created, return the output of
   `nvidia-smi`, Python version, PyTorch/torchvision versions, and the full error.

Do not run `scripts/preflight.py --mode full` or any formal 40-run command yet.
Full execution remains intentionally blocked until the authors sign the six
scientific confirmations and the two-stage freeze is complete.
