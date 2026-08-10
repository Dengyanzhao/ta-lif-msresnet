# V6 机制实验：新 RTX 5090 实例运行手册

## 研究记录

- 用途：在一台全新的单卡 RTX 5090 实例上执行冻结的 V6 机制研究。
- 协议：`configs/protocol_v6_mechanism.yaml`。
- 条件顺序：`M0, M1, M2, M3, M4, PLIF`。
- 核心正式矩阵：CIFAR-100，8 个配对种子，48 个 run。
- health seed 和 pilot seed 只用于 non-reportable 门禁，不得写入论文结果。
- CIFAR-10 是条件性扩展。核心 CIFAR-100 结果没有通过预注册激活门时，不得启动它。

V5 的代码、数据、测试访问和 DOI 不在本手册的实验范围内。不要把 V5 输出复制到任何 V6 目录。

## 绝对规则

1. 只有最终提交中的配置矩阵才可以运行。云端不得重新生成或手工编辑 V6 YAML。
2. `--precheck-only` 是唯一可以在未消耗 health seed 前验证新实例的方法；失败时停止并修复环境。
3. health、pilot、正式训练、benchmark、final-test 和 analysis 必须按顺序执行。
4. 失败或中断的 V6 run 不得 fresh retry；只有同一环境下存在有效 `last.pt` 时才可 resume。
5. 不要使用 `--limit`、`--condition`、`--dataset`、`--output-root` 等参数绕过 V6 矩阵门禁。
6. benchmark 只读 validation 数据，禁止任何 energy/功耗结论。
7. final-test 每个正式 checkpoint 只能访问一次；看到 `final_test.in_progress.json` 时停止，不要重试。

## A. 本地封存（租实例前）

这一节在 Windows 本地仓库执行。只有代码和协议最终不再修改后才执行。

```powershell
Set-Location -LiteralPath 'D:\论文解构\王\ta-lif-msresnet'
$PYTHON = (Get-Command python).Source
$PROTOCOL = 'configs/protocol_v6_mechanism.yaml'

& $PYTHON scripts/preflight.py --protocol $PROTOCOL --mode pilot --skip-dependency-check
if ($LASTEXITCODE -ne 0) { throw 'V6 pilot preflight failed' }

# 这两步是唯一的矩阵生成步骤；生成后必须纳入最终提交。
& $PYTHON scripts/generate_run_configs.py --protocol $PROTOCOL --stage pilot
if ($LASTEXITCODE -ne 0) { throw 'V6 pilot matrix generation failed' }
& $PYTHON scripts/generate_run_configs.py --protocol $PROTOCOL --stage formal
if ($LASTEXITCODE -ne 0) { throw 'V6 formal matrix generation failed' }

Get-ChildItem configs/v6_mechanism_pilot_generated -Filter '*.yaml' | Measure-Object
Get-ChildItem configs/v6_mechanism_generated -Filter '*.yaml' | Measure-Object
```

计数必须分别为 6 和 48。此时把 V6 源码、测试、两个矩阵目录和本手册一起提交到一个新的 V6 commit；不要把 health/pilot/results 文件提交进去。提交后执行：

```powershell
& $PYTHON scripts/validate_v6_source_release.py --require-data
if ($LASTEXITCODE -ne 0) { throw 'V6 source release is not ready' }

$PACKAGE = 'D:\论文解构\王\ta-lif-msresnet-v6-instance-input.tar.gz'
& $PYTHON scripts/create_v6_instance_package.py --output $PACKAGE
if ($LASTEXITCODE -ne 0) { throw 'V6 offline package creation failed' }
& $PYTHON scripts/validate_v6_source_release.py --require-data --package $PACKAGE
if ($LASTEXITCODE -ne 0) { throw 'V6 offline package validation failed' }
Get-FileHash -Algorithm SHA256 -LiteralPath $PACKAGE
Get-FileHash -Algorithm SHA256 -LiteralPath ($PACKAGE + '.sha256')
```

只上传 `ta-lif-msresnet-v6-instance-input.tar.gz` 和同名 `.sha256` sidecar。包内包含 Git bundle、CIFAR-100 原始文件、固定切分清单和 provenance；不包含任何实验结果。

## B. 新实例恢复与环境检查

以下命令在新实例的 Linux 终端执行。优先把实例数据盘挂载到 `/hy-tmp`，并在 tmux 中执行。

```bash
set -euo pipefail
cd /hy-tmp
df -h /hy-tmp
nvidia-smi
sha256sum -c ta-lif-msresnet-v6-instance-input.tar.gz.sha256
tar -xzf ta-lif-msresnet-v6-instance-input.tar.gz
INPUT=/hy-tmp/ta-lif-msresnet-v6-instance-input
(cd "$INPUT" && sha256sum -c SHA256SUMS.txt)
test ! -e /hy-tmp/ta-lif-msresnet
git clone "$INPUT/source/ta-lif-msresnet-v6.bundle" /hy-tmp/ta-lif-msresnet
cd /hy-tmp/ta-lif-msresnet
git switch v6-mechanism-study
mkdir -p data environment
cp -a "$INPUT/payload/data/." data/
cp -a "$INPUT/payload/environment/CIFAR100_SOURCE_PROVENANCE.json" environment/
git status --short --branch
```

`git status --short --branch` 只能显示分支名，不能显示源代码改动。若出现任何 tracked 改动，立即停止；不要执行 `git reset` 或覆盖文件。记录：

```bash
git rev-parse HEAD
git branch --show-current
test "$(git branch --show-current)" = v6-mechanism-study
test "$(find configs/v6_mechanism_pilot_generated -maxdepth 1 -name '*.yaml' -type f | wc -l)" -eq 6
test "$(find configs/v6_mechanism_generated -maxdepth 1 -name '*.yaml' -type f | wc -l)" -eq 48
```

创建或复用实例自带的 Python 环境。不要安装或替换 NVIDIA 驱动、系统 CUDA 或 PyTorch 主版本：

```bash
export CUDA_VISIBLE_DEVICES=0
export PYTHONUNBUFFERED=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8
PYTHON=/root/venvs/talif-msresnet/bin/python
if [ ! -x "$PYTHON" ]; then
  python3 -m venv --system-site-packages /root/venvs/talif-msresnet
fi
PYTHON=/root/venvs/talif-msresnet/bin/python

"$PYTHON" - <<'PY'
from importlib.metadata import version
import torch
assert torch.__version__ == "2.9.1+cu128", torch.__version__
assert version("torchvision") == "0.24.1+cu128", version("torchvision")
assert torch.version.cuda == "12.8", torch.version.cuda
assert torch.cuda.is_available()
assert "RTX 5090" in torch.cuda.get_device_name(0)
print("V6_GPU_TORCHVISION_PASS", torch.__version__, version("torchvision"), torch.version.cuda, torch.cuda.get_device_name(0))
PY

# 只安装项目声明的 Python 依赖；--no-deps 防止意外替换已验证的 PyTorch。
"$PYTHON" -m pip install --index-url https://mirrors.aliyun.com/pypi/simple \
  'numpy>=1.26' 'PyYAML>=6.0' 'pandas>=2.1' 'scipy>=1.11' \
  'statsmodels>=0.14' 'tqdm>=4.66' 'pytest>=8.0' 'ruff>=0.5'
"$PYTHON" -m pip install --no-deps -e .
"$PYTHON" -m pip check
"$PYTHON" -m compileall -q src scripts
```

如果版本断言失败，不要继续、不要消耗 health seed；换用明确提供 PyTorch `2.9.1+cu128`/CUDA `12.8` 的镜像或实例。

## C. 不消耗 seed 的 V6 预检

先验证源发布和协议，再运行可重复的 health precheck。两个命令都不会启动训练：

```bash
"$PYTHON" scripts/validate_v6_source_release.py \
  --protocol configs/protocol_v6_mechanism.yaml --require-data \
  --package "$INPUT/../ta-lif-msresnet-v6-instance-input.tar.gz"
"$PYTHON" scripts/preflight.py --protocol configs/protocol_v6_mechanism.yaml --mode pilot
"$PYTHON" scripts/generate_run_configs.py --protocol configs/protocol_v6_mechanism.yaml --stage pilot --dry-run
"$PYTHON" scripts/generate_run_configs.py --protocol configs/protocol_v6_mechanism.yaml --stage formal --dry-run

"$PYTHON" scripts/pilot_health_gate_v6.py \
  --protocol configs/protocol_v6_mechanism.yaml \
  --config-dir configs/v6_mechanism_pilot_generated \
  --device cuda:0 --precheck-only
```

最后一行必须打印 `V6_HEALTH_PRECHECK_ONLY_PASS` 和 `NO_HEALTH_SEED_WAS_CLAIMED`，且以下文件仍不存在：

```text
results/pilot/v6_mechanism/health_cifar100_s1707261715.attempt.json
results/pilot/v6_mechanism/health_cifar100_s1707261715.json
```

任一预检失败都可以修复后重跑这一节；不要执行 full health 命令。

## D. 固定执行顺序

### D1. Health（唯一一次）

确认 C 节全部 PASS 后，才领取 health seed：

```bash
"$PYTHON" scripts/pilot_health_gate_v6.py \
  --protocol configs/protocol_v6_mechanism.yaml \
  --config-dir configs/v6_mechanism_pilot_generated \
  --device cuda:0 2>&1 | tee /hy-tmp/v6-health.log
```

必须看到 `V6_HEALTH_GATE_PASS`。无论成功还是失败，health attempt receipt 都表示 seed 已消耗；失败时不要重试，保存日志并停止。

### D2. 六条件 Pilot

Health PASS 后运行完整、不可报告的六条件 pilot。不要加筛选参数：

```bash
set -o pipefail
"$PYTHON" scripts/run_matrix.py \
  --protocol configs/protocol_v6_mechanism.yaml \
  --allow-unfrozen-pilot --dataset cifar100 --device cuda:0 \
  --stop-on-error 2>&1 | tee /hy-tmp/v6-pilot.log
test "${PIPESTATUS[0]}" -eq 0
"$PYTHON" scripts/validate_v6_pilot.py \
  --protocol configs/protocol_v6_mechanism.yaml \
  --config-dir configs/v6_mechanism_pilot_generated
```

必须看到 `V6_PILOT_VALIDATION_PASS` 和决策 `ACCEPT_V6_SIX_CONDITION_120_EPOCH_PILOT_RELEASE_FORMAL_FREEZE`。

### D3. Formal freeze

只有 pilot validation PASS 后创建正式冻结清单；创建和验证都必须在同一干净 checkout：

```bash
COMMIT=$(git rev-parse HEAD)
"$PYTHON" scripts/create_freeze_manifest.py \
  --protocol configs/protocol_v6_mechanism.yaml \
  --matrix-dir configs/v6_mechanism_generated \
  --freeze-commit "$COMMIT"
"$PYTHON" scripts/create_freeze_manifest.py \
  --protocol configs/protocol_v6_mechanism.yaml --verify
"$PYTHON" scripts/preflight.py --protocol configs/protocol_v6_mechanism.yaml --mode full
```

三步都必须 PASS；不要修改 protocol、signoff 或矩阵文件。

### D4. 48 个正式 run

在独立 tmux 会话中启动，断开浏览器不会结束进程：

```bash
tmux new -s v6-formal
cd /hy-tmp/ta-lif-msresnet
export CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1 CUBLAS_WORKSPACE_CONFIG=:4096:8
set -o pipefail
"$PYTHON" scripts/run_matrix.py \
  --protocol configs/protocol_v6_mechanism.yaml \
  --device cuda:0 --stop-on-error 2>&1 | tee /hy-tmp/v6-formal.log
rc=${PIPESTATUS[0]}
printf '%s\n' "$rc" > /hy-tmp/v6-formal.exit
echo "V6_FORMAL_MATRIX_EXIT=$rc"
```

另开一个 tmux 会话实时监控：

```bash
cd /hy-tmp/ta-lif-msresnet
"$PYTHON" scripts/v6_monitor_cn.py --stage formal --watch-seconds 60
```

监控器会显示每个 run 的 epoch、单 run 剩余时间和全矩阵预计完成时间。它只读日志，不会触碰训练进程。正式 run 被中断时，先保留对应目录和日志；只有确认仍是同一环境且该目录有有效 `last.pt`，才使用同一命令加 `--resume-matrix` 恢复。

## E. Formal 完成后的验证链

不要先访问 test。先导出固定 validation batch，再做 48 个 checkpoint 的资源审计：

```bash
"$PYTHON" scripts/export_v6_validation_batch.py \
  --protocol configs/protocol_v6_mechanism.yaml
"$PYTHON" scripts/run_v6_benchmarks.py \
  --protocol configs/protocol_v6_mechanism.yaml --device cuda:0
```

必须看到 `V6_BENCHMARK_RECEIPT_PASS`。若 benchmark 中断，保留已完成的 `results/benchmark/v6_mechanism/runs/*.json`，不要换 GPU 或修改输入；按相同环境继续执行同一命令。

资源审计通过后，才允许 one-time final-test：

```bash
"$PYTHON" scripts/evaluate_checkpoints.py \
  --protocol configs/protocol_v6_mechanism.yaml \
  --config-dir configs/v6_mechanism_generated \
  --results-root results/formal_v6_mechanism --device cuda:0
```

必须输出 `Final-test pass complete: 48 evaluated, 0 failed`。出现 `final_test.in_progress.json` 或非零退出时，停止并保存证据，不要自行重试。

最后运行冻结统计分析：

```bash
"$PYTHON" scripts/analyze_v6_results.py --protocol configs/protocol_v6_mechanism.yaml
```

分析输出必须同时包含 `v6_mechanism_accuracy_analysis.json`、`v6_seed_condition_test_accuracies.csv` 和 `v6_paired_seed_differences.csv`，且分析报告的 `n_final_test_markers` 为 48。

## F. 下载与归档

确认没有 V6 进程后，在实例上生成只读证据包：

```bash
"$PYTHON" scripts/archive_v6_evidence.py \
  --protocol configs/protocol_v6_mechanism.yaml \
  --output /hy-tmp/ta-lif-msresnet-v6-evidence.tar.gz
sha256sum /hy-tmp/ta-lif-msresnet-v6-evidence.tar.gz \
  /hy-tmp/ta-lif-msresnet-v6-evidence.tar.gz.sha256
```

下载 `.tar.gz` 和 `.sha256` 两个文件，并保留本地 V6 instance input 包。归档器不包含原始 CIFAR-100 数据；数据包和证据包分开保存更容易核对，也避免把大容量数据误当作论文结果。

## 失败处理

- 预检失败：修复后重跑预检，不消耗 seed。
- health 失败：health seed 已消耗，停止，不重试；需要新协议版本和新 seed。
- pilot 失败：不进入 formal，保留全部 pilot 日志。
- formal 单 run 中断：仅同环境 `last.pt` resume；不能 fresh retry 或替换 seed。
- benchmark/final-test/analysis 失败：保留现有 receipt、journal 和日志，先让作者确认，再决定是否恢复。

任何阶段出现路径绝对化、协议 hash 不一致、GPU 不是 RTX 5090、PyTorch 不是 `2.9.1+cu128`、torchvision 不是 `0.24.1+cu128` 或 CUDA 不是 `12.8`，都按失败处理。
