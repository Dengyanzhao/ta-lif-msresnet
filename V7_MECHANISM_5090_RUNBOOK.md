# V7 机制实验：新 RTX 5090 实例运行手册

## 研究记录

- 用途：在一台全新的单卡 RTX 5090 实例上执行冻结的 V7 机制研究。
- 协议：`configs/protocol_v7_mechanism.yaml`。
- 条件顺序：`M0, M1, M2, M3, M4, PLIF`。
- 核心正式矩阵：CIFAR-100，8 个配对种子，48 个 run。
- health seed 和 pilot seed 只用于 non-reportable 门禁，不得写入论文结果。
- CIFAR-10 是条件性扩展。核心 CIFAR-100 结果没有通过预注册激活门时，不得启动它。

V5 的代码、数据、测试访问和 DOI 不在本手册的实验范围内。不要把 V5 输出复制到任何 V7 目录。

## 绝对规则

1. 只有最终提交中的配置矩阵才可以运行。云端不得重新生成或手工编辑 V7 YAML。
2. `--precheck-only` 是唯一可以在未消耗 health seed 前验证新实例的方法；失败时停止并修复环境。
3. health、pilot、正式训练、benchmark、final-test 和 analysis 必须按顺序执行。
4. 失败或中断的 V7 run 不得 fresh retry；只有同一环境下存在有效 `last.pt` 时才可 resume。
5. 不要使用 `--limit`、`--condition`、`--output-root` 等参数绕过 V7 矩阵门禁；唯一例外是 D2 冻结 pilot 命令中用于选择预注册 pilot 矩阵的 `--dataset cifar100`。
6. benchmark 只读 validation 数据，禁止任何 energy/功耗结论。
7. final-test 每个正式 checkpoint 只能访问一次；看到 `final_test.in_progress.json` 时停止，不要重试。

## A. 本地封存（租实例前）

这一节在 Windows 本地仓库执行。只有代码和协议最终不再修改后才执行。

```powershell
Set-Location -LiteralPath 'D:\论文解构\王\ta-lif-msresnet'
$PYTHON = (Get-Command python).Source
$PROTOCOL = 'configs/protocol_v7_mechanism.yaml'
$EXPECTED_BRANCH = 'v7-mechanism-study'

# V7 必须在独立分支上封存；存在同名分支时只做安全切换，不覆盖任何改动。
git show-ref --verify --quiet "refs/heads/$EXPECTED_BRANCH"
if ($LASTEXITCODE -eq 0) {
    git switch $EXPECTED_BRANCH
} else {
    git switch -c $EXPECTED_BRANCH
}
if ($LASTEXITCODE -ne 0) { throw 'Cannot create or switch to the V7 release branch' }
if ((git branch --show-current).Trim() -ne $EXPECTED_BRANCH) {
    throw 'V7 release branch binding failed'
}

& $PYTHON scripts/preflight.py --protocol $PROTOCOL --mode pilot --skip-dependency-check
if ($LASTEXITCODE -ne 0) { throw 'V7 pilot preflight failed' }

# 这两步是唯一的矩阵生成步骤；生成后必须纳入最终提交。
& $PYTHON scripts/generate_run_configs.py --protocol $PROTOCOL --stage pilot
if ($LASTEXITCODE -ne 0) { throw 'V7 pilot matrix generation failed' }
& $PYTHON scripts/generate_run_configs.py --protocol $PROTOCOL --stage formal
if ($LASTEXITCODE -ne 0) { throw 'V7 formal matrix generation failed' }

Get-ChildItem configs/v7_mechanism_pilot_generated -Filter '*.yaml' | Measure-Object
Get-ChildItem configs/v7_mechanism_generated -Filter '*.yaml' | Measure-Object

$env:PYTHONPATH = (Join-Path (Get-Location) 'src')
& $PYTHON -m compileall -q src scripts
if ($LASTEXITCODE -ne 0) { throw 'V7 source compilation failed' }
& $PYTHON -m pytest -q -p no:cacheprovider tests
if ($LASTEXITCODE -ne 0) { throw 'V7 test suite failed' }
```

计数必须分别为 6 和 48，完整测试必须全部通过。此时把 V7 源码、测试、两个矩阵目录和本手册一起提交到一个新的 V7 commit；不要把 health/pilot/results 文件提交进去。提交后执行：

```powershell
$PYTHON = (Get-Command python).Source
if ((git branch --show-current).Trim() -ne 'v7-mechanism-study') {
    throw 'Wrong branch: V7 release must be sealed on v7-mechanism-study'
}
& $PYTHON scripts/validate_v7_source_release.py --require-data
if ($LASTEXITCODE -ne 0) { throw 'V7 source release is not ready' }

$PACKAGE = 'D:\论文解构\王\ta-lif-msresnet-v7-instance-input.tar.gz'
& $PYTHON scripts/create_v7_instance_package.py --output $PACKAGE
if ($LASTEXITCODE -ne 0) { throw 'V7 offline package creation failed' }
& $PYTHON scripts/validate_v7_source_release.py --require-data --package $PACKAGE
if ($LASTEXITCODE -ne 0) { throw 'V7 offline package validation failed' }
Get-FileHash -Algorithm SHA256 -LiteralPath $PACKAGE
Get-FileHash -Algorithm SHA256 -LiteralPath ($PACKAGE + '.sha256')
```

只上传 `ta-lif-msresnet-v7-instance-input.tar.gz` 和同名 `.sha256` sidecar。包内包含 Git bundle、CIFAR-100 原始文件、固定切分清单和 provenance；不包含任何实验结果。

## B. 新实例恢复与环境检查

以下命令在新实例的 Linux 终端执行。优先把实例数据盘挂载到 `/hy-tmp`，并在 tmux 中执行。

```bash
tmux new -s v7-setup
set +e
set -o pipefail
(
  set -euo pipefail
  cd /hy-tmp
  df -h /hy-tmp
  nvidia-smi
  sha256sum -c ta-lif-msresnet-v7-instance-input.tar.gz.sha256
  test ! -e /hy-tmp/ta-lif-msresnet-v7-instance-input
  tar -xzf ta-lif-msresnet-v7-instance-input.tar.gz
  INPUT=/hy-tmp/ta-lif-msresnet-v7-instance-input
  (cd "$INPUT" && sha256sum -c SHA256SUMS.txt)
  test ! -e /hy-tmp/ta-lif-msresnet
  git clone "$INPUT/source/ta-lif-msresnet-v7.bundle" /hy-tmp/ta-lif-msresnet
  cd /hy-tmp/ta-lif-msresnet
  git switch v7-mechanism-study
  test "$(git branch --show-current)" = v7-mechanism-study
  mkdir -p data environment
  cp -a "$INPUT/payload/data/." data/
  cp -a "$INPUT/payload/environment/CIFAR100_SOURCE_PROVENANCE.json" environment/
  git status --short --branch
) 2>&1 | tee /hy-tmp/v7-restore.log
rc=${PIPESTATUS[0]}
printf '%s\n' "$rc" > /hy-tmp/v7-restore.exit
echo "V7_RESTORE_EXIT=$rc"
```

必须看到 `V7_RESTORE_EXIT=0`。失败时 pane 会保留在命令提示符，不会因内部 `set -e` 直接关闭；查看 `/hy-tmp/v7-restore.log` 修复后再继续。

`git status --short --branch` 只能显示分支名，不能显示源代码改动。若出现任何 tracked 改动，立即停止；不要执行 `git reset` 或覆盖文件。记录：

```bash
cd /hy-tmp/ta-lif-msresnet
git rev-parse HEAD
git branch --show-current
test "$(git branch --show-current)" = v7-mechanism-study
test "$(find configs/v7_mechanism_pilot_generated -maxdepth 1 -name '*.yaml' -type f | wc -l)" -eq 6
test "$(find configs/v7_mechanism_generated -maxdepth 1 -name '*.yaml' -type f | wc -l)" -eq 48
```

创建或复用实例自带的 Python 环境。不要安装或替换 NVIDIA 驱动、系统 CUDA 或 PyTorch 主版本：

```bash
set +e
set -o pipefail
(
  set -euo pipefail
  cd /hy-tmp/ta-lif-msresnet
  export CUDA_VISIBLE_DEVICES=0
  export PYTHONUNBUFFERED=1
  export CUBLAS_WORKSPACE_CONFIG=:4096:8

  IMAGE_PYTHON=$(command -v python3.11 || command -v python3)
  "$IMAGE_PYTHON" - <<'PY'
import torch
assert torch.__version__ == "2.9.1+cu128", torch.__version__
assert torch.version.cuda == "12.8", torch.version.cuda
assert torch.cuda.is_available()
assert "RTX 5090" in torch.cuda.get_device_name(0)
print("V7_IMAGE_TORCH_PASS", torch.__version__, torch.version.cuda, torch.cuda.get_device_name(0))
PY
  SYSTEM_TORCH_FILE=$("$IMAGE_PYTHON" -c 'from pathlib import Path; import torch; print(Path(torch.__file__).resolve())')

  # --without-pip 绕过该镜像已知的 ensurepip 失败；system-site-packages 只继承镜像中已验证的 Torch。
  "$IMAGE_PYTHON" -m venv --without-pip --system-site-packages /root/venvs/talif-msresnet
  PYTHON=/root/venvs/talif-msresnet/bin/python
  VENV_TORCH_FILE=$("$PYTHON" -c 'from pathlib import Path; import torch; print(Path(torch.__file__).resolve())')
  test "$VENV_TORCH_FILE" = "$SYSTEM_TORCH_FILE"
  "$PYTHON" -c 'import torch; assert torch.__version__ == "2.9.1+cu128" and torch.version.cuda == "12.8"'

  TV_VERSION=$("$PYTHON" -c 'from importlib.metadata import version; print(version("torchvision"))' 2>/dev/null || true)
  if [ "$TV_VERSION" != "0.24.1+cu128" ]; then
    "$PYTHON" -m pip install --no-deps \
      --index-url https://download.pytorch.org/whl/cu128 \
      'torchvision==0.24.1+cu128'
  fi

  "$PYTHON" -m pip install --index-url https://mirrors.aliyun.com/pypi/simple \
    'numpy>=1.26' 'PyYAML>=6.0' 'pandas>=2.1' 'scipy>=1.11' \
    'statsmodels>=0.14' 'tqdm>=4.66' 'pytest>=8.0' 'ruff>=0.5'
  "$PYTHON" -m pip install --no-deps -e .

  PIP_CHECK_RC=0
  PIP_CHECK_OUTPUT=$("$PYTHON" -m pip check 2>&1) || PIP_CHECK_RC=$?
  printf '%s\n' "$PIP_CHECK_OUTPUT"
  if [ "$PIP_CHECK_RC" -ne 0 ]; then
    KNOWN_GUI=$(printf '%s\n' "$PIP_CHECK_OUTPUT" | grep -E '^pygobject [^ ]+ requires pycairo, which is not installed\.$' || true)
    UNEXPECTED=$(printf '%s\n' "$PIP_CHECK_OUTPUT" | grep -Ev '^pygobject [^ ]+ requires pycairo, which is not installed\.$' || true)
    test -n "$KNOWN_GUI"
    test -z "$UNEXPECTED"
    echo 'KNOWN_IMAGE_GUI_DEPENDENCY_IGNORED: pygobject -> pycairo'
  fi

  "$PYTHON" - <<'PY'
from importlib.metadata import version
from pathlib import Path
import talif_msresnet
import torch
import torchvision

assert torch.__version__ == "2.9.1+cu128", torch.__version__
assert torchvision.__version__ == "0.24.1+cu128", torchvision.__version__
assert version("torchvision") == "0.24.1+cu128", version("torchvision")
assert torch.version.cuda == "12.8", torch.version.cuda
assert torch.cuda.is_available()
assert "RTX 5090" in torch.cuda.get_device_name(0)
assert torchvision.extension._has_ops(), torchvision.__file__
assert str(Path(talif_msresnet.__file__).resolve()).startswith("/hy-tmp/ta-lif-msresnet/src/")
assert str(Path(torchvision.__file__).resolve()).startswith("/root/venvs/talif-msresnet/")
print("V7_EXACT_RUNTIME_PASS")
print("python_package =", Path(talif_msresnet.__file__).resolve())
print("torch =", torch.__version__)
print("torchvision =", torchvision.__version__)
print("cuda =", torch.version.cuda)
print("gpu =", torch.cuda.get_device_name(0))
PY
  "$PYTHON" -m compileall -q src scripts
) 2>&1 | tee /hy-tmp/v7-environment.log
rc=${PIPESTATUS[0]}
printf '%s\n' "$rc" > /hy-tmp/v7-environment.exit
echo "V7_ENVIRONMENT_REPAIR_EXIT=$rc"
```

必须看到 `V7_EXACT_RUNTIME_PASS` 和 `V7_ENVIRONMENT_REPAIR_EXIT=0`。这里允许的唯一 `pip check` 例外是镜像 GUI 包 `pygobject -> pycairo`；出现任何其他依赖错误、版本断言失败或导入路径异常时，不要继续、不要消耗 health seed。

## C. 不消耗 seed 的 V7 预检

先验证源发布和协议，再运行可重复的 health precheck。两个命令都不会启动训练：

```bash
tmux new -s v7-precheck
set +e
set -o pipefail
(
  set -euo pipefail
  cd /hy-tmp/ta-lif-msresnet
  INPUT=/hy-tmp/ta-lif-msresnet-v7-instance-input
  PYTHON=/root/venvs/talif-msresnet/bin/python
  export CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1 CUBLAS_WORKSPACE_CONFIG=:4096:8
  test "$(cat /hy-tmp/v7-restore.exit)" = 0
  test "$(cat /hy-tmp/v7-environment.exit)" = 0

  "$PYTHON" scripts/validate_v7_source_release.py \
    --protocol configs/protocol_v7_mechanism.yaml --require-data \
    --package /hy-tmp/ta-lif-msresnet-v7-instance-input.tar.gz
  "$PYTHON" scripts/preflight.py --protocol configs/protocol_v7_mechanism.yaml --mode pilot
  "$PYTHON" scripts/generate_run_configs.py --protocol configs/protocol_v7_mechanism.yaml --stage pilot --dry-run
  "$PYTHON" scripts/generate_run_configs.py --protocol configs/protocol_v7_mechanism.yaml --stage formal --dry-run
  "$PYTHON" scripts/pilot_health_gate_v7.py \
    --protocol configs/protocol_v7_mechanism.yaml \
    --config-dir configs/v7_mechanism_pilot_generated \
    --device cuda:0 --precheck-only
) 2>&1 | tee /hy-tmp/v7-precheck.log
rc=${PIPESTATUS[0]}
printf '%s\n' "$rc" > /hy-tmp/v7-precheck.exit
echo "V7_ZERO_SEED_PREFLIGHT_EXIT=$rc"
```

必须看到 `V7_HEALTH_PRECHECK_ONLY_PASS`、`NO_HEALTH_SEED_WAS_CLAIMED` 和 `V7_ZERO_SEED_PREFLIGHT_EXIT=0`，且以下文件仍不存在：

```text
results/pilot/v7_mechanism/health_cifar100_s1068798027.attempt.json
results/pilot/v7_mechanism/health_cifar100_s1068798027.json
```

任一预检失败都可以修复后重跑这一节；不要执行 full health 命令。

## D. 固定执行顺序

### D1. Health（唯一一次）

确认 C 节全部 PASS 后，才领取 health seed：

```bash
tmux new -s v7-health
set +e
set -o pipefail
(
  set -euo pipefail
  cd /hy-tmp/ta-lif-msresnet
  PYTHON=/root/venvs/talif-msresnet/bin/python
  export CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1 CUBLAS_WORKSPACE_CONFIG=:4096:8
  test "$(cat /hy-tmp/v7-precheck.exit)" = 0
  "$PYTHON" scripts/pilot_health_gate_v7.py \
    --protocol configs/protocol_v7_mechanism.yaml \
    --config-dir configs/v7_mechanism_pilot_generated \
    --device cuda:0
) 2>&1 | tee /hy-tmp/v7-health.log
rc=${PIPESTATUS[0]}
printf '%s\n' "$rc" > /hy-tmp/v7-health.exit
echo "V7_HEALTH_COMMAND_EXIT=$rc"
```

必须同时看到 `V7_HEALTH_GATE_PASS` 和 `V7_HEALTH_COMMAND_EXIT=0`。无论成功还是失败，health attempt receipt 都表示 seed 已消耗；失败时不要重试，下载 health report、attempt receipt、`/hy-tmp/v7-health.log` 和 `/hy-tmp/v7-health.exit` 后停止。

### D1R. 已消耗 health PASS 的密封兼容恢复

本节只适用于 health 已 PASS、health seed 已消耗、但 pilot 在 `run_started` 之前被兼容性校验器阻断的单次恢复。不得重跑 health，也不得在密封恢复提交安装并验证前再次启动 pilot。原 health report 与 attempt receipt 必须保持逐字节不变。

安装经过校验的增量 bundle 后，先保存会被成功重跑覆盖的首次阻断日志，再执行唯一允许的零 seed 兼容预检：

```bash
set +e
set -o pipefail
(
  set -euo pipefail
  test -s /hy-tmp/v7-pilot.log
  test -s /hy-tmp/v7-pilot.exit
  test "$(cat /hy-tmp/v7-pilot.exit)" = 1
  test ! -e /hy-tmp/v7-pilot-blocked-launch.log
  test ! -e /hy-tmp/v7-pilot-blocked-launch.exit
  test ! -e /hy-tmp/v7-pilot-blocked-launch.sha256
  cp --preserve=timestamps /hy-tmp/v7-pilot.log \
    /hy-tmp/v7-pilot-blocked-launch.log
  cp --preserve=timestamps /hy-tmp/v7-pilot.exit \
    /hy-tmp/v7-pilot-blocked-launch.exit
  cd /hy-tmp
  sha256sum v7-pilot-blocked-launch.log v7-pilot-blocked-launch.exit \
    > v7-pilot-blocked-launch.sha256
  sha256sum -c v7-pilot-blocked-launch.sha256
  echo V7_BLOCKED_LAUNCH_EVIDENCE_PRESERVED
  cd /hy-tmp/ta-lif-msresnet
  PYTHON=/root/venvs/talif-msresnet/bin/python
  export CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1 CUBLAS_WORKSPACE_CONFIG=:4096:8
  test "$(cat /hy-tmp/v7-health.exit)" = 0
  "$PYTHON" scripts/pilot_health_gate_v7.py \
    --protocol configs/protocol_v7_mechanism.yaml \
    --config-dir configs/v7_mechanism_pilot_generated \
    --device cuda:0 --pilot-compatibility-precheck-only
) 2>&1 | tee /hy-tmp/v7-compatibility-precheck.log
rc=${PIPESTATUS[0]}
printf '%s\n' "$rc" > /hy-tmp/v7-compatibility-precheck.exit
echo "V7_PILOT_COMPATIBILITY_PRECHECK_EXIT=$rc"
```

必须同时看到 `V7_BLOCKED_LAUNCH_EVIDENCE_PRESERVED`、`V7_PILOT_COMPATIBILITY_PRECHECK_PASS`、两个 `SHA256_UNCHANGED` 标记、`NO_HEALTH_OR_PILOT_SEED_WAS_CLAIMED` 和 `V7_PILOT_COMPATIBILITY_PRECHECK_EXIT=0`。并再次确认 pilot plan、pilot 输出目录和 pilot validation 均不存在。任一检查失败都停止，不得重跑 health 或启动 pilot。

### D2. 六条件 Pilot

Health PASS 后运行完整、不可报告的六条件 pilot。下面的 `--dataset cifar100` 是冻结 pilot 选择器，不得再增删任何筛选参数：

```bash
tmux new -s v7-pilot
set +e
set -o pipefail
(
  set -euo pipefail
  cd /hy-tmp/ta-lif-msresnet
  PYTHON=/root/venvs/talif-msresnet/bin/python
  export CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1 CUBLAS_WORKSPACE_CONFIG=:4096:8
  test "$(cat /hy-tmp/v7-health.exit)" = 0
  "$PYTHON" scripts/run_matrix.py \
    --protocol configs/protocol_v7_mechanism.yaml \
    --allow-unfrozen-pilot --dataset cifar100 --device cuda:0 \
    --stop-on-error
  "$PYTHON" scripts/validate_v7_pilot.py \
    --protocol configs/protocol_v7_mechanism.yaml \
    --config-dir configs/v7_mechanism_pilot_generated
) 2>&1 | tee /hy-tmp/v7-pilot.log
rc=${PIPESTATUS[0]}
printf '%s\n' "$rc" > /hy-tmp/v7-pilot.exit
echo "V7_PILOT_CHAIN_EXIT=$rc"
```

可另开一个只读监控会话；每个新 tmux 都要重新定义 `PYTHON`：

```bash
tmux new -s v7-pilot-monitor
cd /hy-tmp/ta-lif-msresnet
PYTHON=/root/venvs/talif-msresnet/bin/python
"$PYTHON" scripts/v7_monitor_cn.py --stage pilot --watch-seconds 60
```

必须看到 `V7_PILOT_VALIDATION_PASS`、决策 `ACCEPT_V7_SIX_CONDITION_120_EPOCH_PILOT_RELEASE_FORMAL_FREEZE` 和 `V7_PILOT_CHAIN_EXIT=0`。若训练或验证失败，保存 pilot 目录、日志和 exit sidecar，不得进入 formal。

只有在训练进程被中断、仍是同一实例与环境、且未完成 run 存在有效 `last.pt` 时，才允许执行下面的完整恢复链。若中断发生在 validation，或无法证明 checkpoint 有效，停止并保存证据，不得使用此命令：

```bash
tmux new -s v7-pilot-resume
set +e
set -o pipefail
(
  set -euo pipefail
  cd /hy-tmp/ta-lif-msresnet
  PYTHON=/root/venvs/talif-msresnet/bin/python
  export CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1 CUBLAS_WORKSPACE_CONFIG=:4096:8
  test "$(cat /hy-tmp/v7-health.exit)" = 0
  "$PYTHON" scripts/run_matrix.py \
    --protocol configs/protocol_v7_mechanism.yaml \
    --allow-unfrozen-pilot --dataset cifar100 --device cuda:0 \
    --resume-matrix --stop-on-error
  "$PYTHON" scripts/validate_v7_pilot.py \
    --protocol configs/protocol_v7_mechanism.yaml \
    --config-dir configs/v7_mechanism_pilot_generated
) 2>&1 | tee -a /hy-tmp/v7-pilot.log
rc=${PIPESTATUS[0]}
printf '%s\n' "$rc" > /hy-tmp/v7-pilot.exit
echo "V7_PILOT_RESUME_CHAIN_EXIT=$rc"
```

### D3. Formal freeze

只有 pilot validation PASS 后创建正式冻结清单；创建和验证都必须在同一干净 checkout：

```bash
set +e
set -o pipefail
(
  set -euo pipefail
  cd /hy-tmp/ta-lif-msresnet
  PYTHON=/root/venvs/talif-msresnet/bin/python
  export CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1 CUBLAS_WORKSPACE_CONFIG=:4096:8
  test "$(cat /hy-tmp/v7-pilot.exit)" = 0
  COMMIT=$(git rev-parse HEAD)
  "$PYTHON" scripts/create_freeze_manifest.py \
    --protocol configs/protocol_v7_mechanism.yaml \
    --matrix-dir configs/v7_mechanism_generated \
    --freeze-commit "$COMMIT"
  "$PYTHON" scripts/create_freeze_manifest.py \
    --protocol configs/protocol_v7_mechanism.yaml --verify
  "$PYTHON" scripts/preflight.py --protocol configs/protocol_v7_mechanism.yaml --mode full
) 2>&1 | tee /hy-tmp/v7-freeze.log
rc=${PIPESTATUS[0]}
printf '%s\n' "$rc" > /hy-tmp/v7-freeze.exit
echo "V7_FORMAL_FREEZE_EXIT=$rc"
```

三步都必须 PASS，且必须看到 `V7_FORMAL_FREEZE_EXIT=0`；不要修改 protocol、signoff 或矩阵文件。

### D4. 48 个正式 run

在独立 tmux 会话中启动，断开浏览器不会结束进程：

```bash
tmux new -s v7-formal
set +e
set -o pipefail
(
  set -euo pipefail
  cd /hy-tmp/ta-lif-msresnet
  PYTHON=/root/venvs/talif-msresnet/bin/python
  export CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1 CUBLAS_WORKSPACE_CONFIG=:4096:8
  test "$(cat /hy-tmp/v7-freeze.exit)" = 0
  "$PYTHON" scripts/run_matrix.py \
    --protocol configs/protocol_v7_mechanism.yaml \
    --device cuda:0 --stop-on-error
) 2>&1 | tee /hy-tmp/v7-formal.log
rc=${PIPESTATUS[0]}
printf '%s\n' "$rc" > /hy-tmp/v7-formal.exit
echo "V7_FORMAL_MATRIX_EXIT=$rc"
```

另开一个 tmux 会话实时监控：

```bash
tmux new -s v7-formal-monitor
cd /hy-tmp/ta-lif-msresnet
PYTHON=/root/venvs/talif-msresnet/bin/python
"$PYTHON" scripts/v7_monitor_cn.py --stage formal --watch-seconds 60
```

监控器会显示每个 run 的 epoch、单 run 剩余时间和全矩阵预计完成时间。它只读日志，不会触碰训练进程。正式 run 被中断时，先保留对应目录和日志；只有确认仍是同一实例与环境且该目录有有效 `last.pt`，才执行下面的完整恢复命令：

```bash
tmux new -s v7-formal-resume
set +e
set -o pipefail
(
  set -euo pipefail
  cd /hy-tmp/ta-lif-msresnet
  PYTHON=/root/venvs/talif-msresnet/bin/python
  export CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1 CUBLAS_WORKSPACE_CONFIG=:4096:8
  test "$(cat /hy-tmp/v7-freeze.exit)" = 0
  "$PYTHON" scripts/run_matrix.py \
    --protocol configs/protocol_v7_mechanism.yaml \
    --device cuda:0 --resume-matrix --stop-on-error
) 2>&1 | tee -a /hy-tmp/v7-formal.log
rc=${PIPESTATUS[0]}
printf '%s\n' "$rc" > /hy-tmp/v7-formal.exit
echo "V7_FORMAL_MATRIX_RESUME_EXIT=$rc"
```

若无法证明 `last.pt` 有效，或实例、代码、环境、协议任一项已经变化，停止并保存证据，不得恢复。

## E. Formal 完成后的验证链

不要先访问 test。先导出固定 validation batch，再做 48 个 checkpoint 的资源审计：

```bash
tmux new -s v7-benchmark
set +e
set -o pipefail
(
  set -euo pipefail
  cd /hy-tmp/ta-lif-msresnet
  PYTHON=/root/venvs/talif-msresnet/bin/python
  export CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1 CUBLAS_WORKSPACE_CONFIG=:4096:8
  test "$(cat /hy-tmp/v7-formal.exit)" = 0
  "$PYTHON" scripts/export_v7_validation_batch.py \
    --protocol configs/protocol_v7_mechanism.yaml
  "$PYTHON" scripts/run_v7_benchmarks.py \
    --protocol configs/protocol_v7_mechanism.yaml --device cuda:0
) 2>&1 | tee /hy-tmp/v7-benchmark.log
rc=${PIPESTATUS[0]}
printf '%s\n' "$rc" > /hy-tmp/v7-benchmark.exit
echo "V7_BENCHMARK_CHAIN_EXIT=$rc"
```

必须看到 `V7_BENCHMARK_RECEIPT_PASS` 和 `V7_BENCHMARK_CHAIN_EXIT=0`。若 benchmark 中断，保留已完成的 `results/benchmark/v7_mechanism/runs/*.json`，不要换 GPU 或修改输入；只有同一实例、同一环境、同一固定 validation batch 才可继续执行同一命令。

资源审计通过后，才允许 one-time final-test：

```bash
tmux new -s v7-final-test
set +e
set -o pipefail
(
  set -euo pipefail
  cd /hy-tmp/ta-lif-msresnet
  PYTHON=/root/venvs/talif-msresnet/bin/python
  export CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1 CUBLAS_WORKSPACE_CONFIG=:4096:8
  test "$(cat /hy-tmp/v7-benchmark.exit)" = 0
  "$PYTHON" scripts/evaluate_checkpoints.py \
    --protocol configs/protocol_v7_mechanism.yaml \
    --config-dir configs/v7_mechanism_generated \
    --results-root results/formal_v7_mechanism --device cuda:0
  "$PYTHON" - <<'PY'
from pathlib import Path

root = Path("results/formal_v7_mechanism")
count = sum(path.is_file() for path in root.glob("*/final_test.json"))
print(f"V7_FINAL_TEST_MARKERS={count}/48")
assert count == 48, count
PY
) 2>&1 | tee -a /hy-tmp/v7-final-test.log
rc=${PIPESTATUS[0]}
printf '%s\n' "$rc" > /hy-tmp/v7-final-test.exit
echo "V7_FINAL_TEST_EXIT=$rc"
```

首次无中断执行应输出 `Final-test pass complete: 48 evaluated, 0 failed`；最终无论是否经过允许的事务恢复，都必须看到 `V7_FINAL_TEST_MARKERS=48/48` 和 `V7_FINAL_TEST_EXIT=0`。出现 `final_test.in_progress.json` 或非零退出时，停止并保存证据，不要删除 journal/lock，也不要直接重跑整条 final-test 命令。

只读检查每个中断事务的阶段：

```bash
cd /hy-tmp/ta-lif-msresnet
PYTHON=/root/venvs/talif-msresnet/bin/python
"$PYTHON" - <<'PY'
import json
from pathlib import Path

root = Path("results/formal_v7_mechanism")
for path in sorted(root.glob("*/final_test.in_progress.json")):
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        marker_exists = (path.parent / "final_test.json").is_file()
        print(
            path.parent.name,
            "stage=", payload.get("stage"),
            "marker_exists=", marker_exists,
        )
    except Exception as exc:
        print(path.parent.name, "UNREADABLE", type(exc).__name__, exc)
PY
```

只有 journal 明确为 `stage=results_ready` 时，才可按程序打印的那个 `RUN_ID` 执行以下恢复。`marker_exists=False` 时，它只提交已经写入 journal 的结果；`marker_exists=True` 时，它只核验已提交 marker 并清理残留事务。两条路径都不会再次构造或读取 test loader：

```bash
tmux new -s v7-final-test-recovery
set +e
set -o pipefail
RUN_ID='把程序打印的完整_RUN_ID_填在这里'
test "$RUN_ID" != '把程序打印的完整_RUN_ID_填在这里' || {
  echo 'ERROR: 必须先填写唯一允许恢复的 RUN_ID'
  exit 2
}
RECOVERY_LOG="/hy-tmp/v7-final-test-recovery-${RUN_ID}.log"
RECOVERY_EXIT="/hy-tmp/v7-final-test-recovery-${RUN_ID}.exit"
(
set -euo pipefail
cd /hy-tmp/ta-lif-msresnet
PYTHON=/root/venvs/talif-msresnet/bin/python
export CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1 CUBLAS_WORKSPACE_CONFIG=:4096:8
"$PYTHON" scripts/evaluate_checkpoints.py \
  --protocol configs/protocol_v7_mechanism.yaml \
  --config-dir configs/v7_mechanism_generated \
  --results-root results/formal_v7_mechanism --device cuda:0 \
  --recover-run "$RUN_ID"
) 2>&1 | tee "$RECOVERY_LOG"
rc=${PIPESTATUS[0]}
printf '%s\n' "$rc" > "$RECOVERY_EXIT"
echo "V7_FINAL_TEST_RECOVERY_EXIT=$rc RUN_ID=$RUN_ID"
```

恢复成功判据取决于只读检查结果：`marker_exists=False` 时必须打印 `recovered_without_test_access`；`marker_exists=True` 时必须打印 `cleaned_committed_transaction`。两种情况都必须打印 `V7_FINAL_TEST_RECOVERY_EXIT=0`。`stage` 为任何其他值、journal 无法解析、只有 lock 没有 journal，或 journal 与 marker 不一致时，都属于 test 访问状态不明确：停止并先下载全部证据，不得自行恢复或重试。成功恢复后，新建名为 `v7-final-test-resume` 的 tmux，并重新执行上一段完整 final-test 命令；其中 `tee -a` 会保留首次中断日志，程序会审计已有 marker 并仅评估其余 pending run。恢复日志及其 exit sidecar 会被最终 session-log 包一并封存。

最后运行冻结统计分析：

```bash
set +e
set -o pipefail
(
  set -euo pipefail
  cd /hy-tmp/ta-lif-msresnet
  PYTHON=/root/venvs/talif-msresnet/bin/python
  export PYTHONUNBUFFERED=1
  test "$(cat /hy-tmp/v7-final-test.exit)" = 0
  "$PYTHON" scripts/analyze_v7_results.py --protocol configs/protocol_v7_mechanism.yaml
) 2>&1 | tee /hy-tmp/v7-analysis.log
rc=${PIPESTATUS[0]}
printf '%s\n' "$rc" > /hy-tmp/v7-analysis.exit
echo "V7_ANALYSIS_EXIT=$rc"
```

分析输出必须同时包含 `v7_mechanism_accuracy_analysis.json`、`v7_seed_condition_test_accuracies.csv` 和 `v7_paired_seed_differences.csv`，分析报告的 `n_final_test_markers` 为 48，且必须看到 `V7_ANALYSIS_EXIT=0`。

## F. 下载与归档

确认没有 V7 进程后，在实例上生成只读证据包：

```bash
tmux new -s v7-archive
set +e
set -o pipefail
(
  set -euo pipefail
  cd /hy-tmp/ta-lif-msresnet
  PYTHON=/root/venvs/talif-msresnet/bin/python
  export PYTHONUNBUFFERED=1
  test "$(cat /hy-tmp/v7-analysis.exit)" = 0
  "$PYTHON" scripts/archive_v7_evidence.py \
    --protocol configs/protocol_v7_mechanism.yaml \
    --output /hy-tmp/ta-lif-msresnet-v7-evidence.tar.gz
  cd /hy-tmp
  sha256sum -c ta-lif-msresnet-v7-evidence.tar.gz.sha256
) 2>&1 | tee /hy-tmp/v7-archive.log
rc=${PIPESTATUS[0]}
printf '%s\n' "$rc" > /hy-tmp/v7-archive.exit
echo "V7_EVIDENCE_ARCHIVE_EXIT=$rc"

if [ "$rc" -eq 0 ]; then
  cd /hy-tmp
  test ! -e ta-lif-msresnet-v7-session-logs.tar.gz
  test ! -e ta-lif-msresnet-v7-session-logs.tar.gz.sha256
  for stage in restore environment precheck health compatibility-precheck pilot freeze formal benchmark final-test analysis archive; do
    test -s "v7-${stage}.log"
    test -s "v7-${stage}.exit"
    test "$(cat "v7-${stage}.exit")" = 0
  done
  test -s v7-pilot-blocked-launch.log
  test "$(cat v7-pilot-blocked-launch.exit)" = 1
  test -s v7-pilot-blocked-launch.sha256
  sha256sum -c v7-pilot-blocked-launch.sha256
  tar -czf ta-lif-msresnet-v7-session-logs.tar.gz \
    v7-*.log v7-*.exit v7-*.sha256
  sha256sum ta-lif-msresnet-v7-session-logs.tar.gz > ta-lif-msresnet-v7-session-logs.tar.gz.sha256
  sha256sum -c ta-lif-msresnet-v7-session-logs.tar.gz.sha256
  echo V7_SESSION_LOG_ARCHIVE_PASS
fi
```

必须看到科学证据包的校验 `OK`、`V7_EVIDENCE_ARCHIVE_EXIT=0` 和 `V7_SESSION_LOG_ARCHIVE_PASS`。下载以下四个文件，并保留本地 V7 instance input 包：

```text
/hy-tmp/ta-lif-msresnet-v7-evidence.tar.gz
/hy-tmp/ta-lif-msresnet-v7-evidence.tar.gz.sha256
/hy-tmp/ta-lif-msresnet-v7-session-logs.tar.gz
/hy-tmp/ta-lif-msresnet-v7-session-logs.tar.gz.sha256
```

科学证据归档器不包含原始 CIFAR-100 数据；单独的 session-log 包保存各阶段终端日志和退出码。数据包、科学证据包和运行日志包分开保存，既能完整追溯，也避免把大容量数据误当作论文结果。

## 失败处理

- 预检失败：修复后重跑预检，不消耗 seed。
- health 失败：health seed 已消耗，停止，不重试；需要新协议版本和新 seed。
- pilot 失败：不进入 formal，保留全部 pilot 日志。
- formal 单 run 中断：仅同环境 `last.pt` resume；不能 fresh retry 或替换 seed。
- benchmark/final-test/analysis 失败：保留现有 receipt、journal 和日志，先让作者确认，再决定是否恢复。

任何阶段出现路径绝对化、协议 hash 不一致、GPU 不是 RTX 5090、PyTorch 不是 `2.9.1+cu128`、torchvision 不是 `0.24.1+cu128` 或 CUDA 不是 `12.8`，都按失败处理。
