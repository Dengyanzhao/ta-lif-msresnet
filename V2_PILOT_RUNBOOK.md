# v2 非报告性 pilot 运行手册

本手册用于修复后的 TA-LIF / MS-ResNet 实现。所有 v1 正式结果均已撤回，
只能作为事故审计材料保存。v2 pilot 也不得写入论文结果、不得进入正式统计分析。

## 0. 不可跨越的边界

- v1 身份：commit `369a96b142532e9084bffbae008942477e05fcfd`。
- v1 结果根：`results/runs`。不得删除、覆盖、续跑或与 v2 混用。
- v2 pilot 协议：`configs/protocol_v2_pilot.yaml`。
- v2 pilot 固定结果根：`results/pilot/v2_seed77_e120`。
- v2 pilot 固定 seed：`77`，固定顺序：`C1 -> C2 -> C3 -> C4`。
- v2 pilot 必须在同一 RTX 5090、PyTorch `2.9.1+cu128`、CUDA `12.8`、
  FP32 严格确定性环境中完成。
- 运行健康门或 pilot 后不得修改 tracked 文件。代码有任何修改都必须重新提交、
  重新生成配置、重新运行健康门并使用新的空 pilot 结果根。

## 1. 停止并归档 v1（在保存 v1 的旧实例操作）

先做只读检查：

```bash
cd /hy-tmp/ta-lif-msresnet
python scripts/archive_formal_v1.py
```

若输出明确识别到 v1 writer，再使用唯一确认令牌发送 `SIGTERM`：

```bash
python scripts/archive_formal_v1.py --stop \
  --confirm-stop STOP_FORMAL_V1_369A96B \
  --stop-receipt environment/formal_v1_stop_20260727.json
```

再次运行只读检查。只有确认不存在 active 或 ambiguous writer 后才归档：

```bash
python scripts/archive_formal_v1.py \
  --archive /hy-tmp/ta_lif_formal_v1_incident_20260727.tar.gz

sha256sum /hy-tmp/ta_lif_formal_v1_incident_20260727.tar.gz \
  | tee /hy-tmp/ta_lif_formal_v1_incident_20260727.tar.gz.sha256
```

下载 `.tar.gz` 和 `.sha256` 后在本地复核。归档工具不会删除云端源文件。
若旧实例已不可访问，只能保留此前下载的原始结果与操作记录，并把“无法生成完整
云端事故归档”记入研究日志；不得据此恢复 v1。

## 2. 准备 v2 pilot 实例

以下命令在新的 RTX 5090 实例执行。`<V2_COMMIT>` 必须替换为本地最终验证并推送后
得到的完整 40 位提交号。

```bash
cd /hy-tmp
git clone https://github.com/Dengyanzhao/ta-lif-msresnet.git
cd /hy-tmp/ta-lif-msresnet
git checkout --detach <V2_COMMIT>

test "$(git rev-parse HEAD)" = "<V2_COMMIT>" \
  && echo V2_COMMIT_PASS
test -z "$(git status --porcelain --untracked-files=no)" \
  && echo TRACKED_WORKTREE_CLEAN
```

建立环境：

```bash
export VENV="$HOME/venvs/talif-msresnet-v2"
mkdir -p "$HOME/venvs"

if [ ! -e "$VENV" ]; then
  python -m venv --system-site-packages --without-pip "$VENV"
fi

source "$VENV/bin/activate"
python -m pip install --no-deps --no-build-isolation -e .

export CUDA_VISIBLE_DEVICES=0
export PYTHONUNBUFFERED=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8
```

只恢复训练数据和数据清单，不恢复 v1 的 `results/runs` 或 checkpoint：

```bash
test -f data/cifar100/cifar-100-python.tar.gz
test -f data/manifests/cifar100_seed2024.json

md5sum -c <<'EOF'
eb9058c3a382ffc7106e4002c42a8d85  data/cifar100/cifar-100-python.tar.gz
EOF

python - <<'PY'
from torchvision.datasets import CIFAR100

dataset = CIFAR100("data/cifar100", train=True, download=False)
assert len(dataset) == 50000
assert len(dataset.classes) == 100
print("CIFAR100_TRAIN_LOAD_PASS", len(dataset), len(dataset.classes))
PY
```

## 3. 测试、配置生成和 pilot preflight

```bash
mkdir -p results/pilot/evidence_meta

python -m pytest -q \
  | tee results/pilot/evidence_meta/pytest.txt

python scripts/generate_run_configs.py \
  --protocol configs/protocol_v2_pilot.yaml \
  --output configs/v2_pilot_generated

python scripts/preflight.py \
  --protocol configs/protocol_v2_pilot.yaml \
  --mode pilot \
  | tee results/pilot/evidence_meta/pilot-preflight.txt

test "$(find configs/v2_pilot_generated -maxdepth 1 -type f -name '*.yaml' | wc -l)" -eq 4 \
  && echo V2_PILOT_4_CONFIGS_PASS

test -z "$(git status --porcelain --untracked-files=no)" \
  && echo TRACKED_WORKTREE_CLEAN_BEFORE_HEALTH
```

保存环境证据：

```bash
git rev-parse HEAD > results/pilot/evidence_meta/git-commit.txt
git status --short > results/pilot/evidence_meta/git-status-short.txt
python -m pip freeze --exclude-editable > results/pilot/evidence_meta/pip-freeze.txt
uname -a > results/pilot/evidence_meta/uname.txt
nvidia-smi -q > results/pilot/evidence_meta/nvidia-smi-q.txt
```

## 4. RTX 5090 健康门

健康报告路径和全部阈值已写入协议。不要增加任何用于改变 seed、batch size、steps
或阈值的命令行参数。

```bash
test ! -e results/pilot/v2_seed77_e120_health.json
test ! -e results/pilot/v2_seed77_e120
test ! -e environment/unfrozen_pilot_plan_v2_seed77_e120.json

python scripts/pilot_health_gate.py \
  --output results/pilot/v2_seed77_e120_health.json \
  --device cuda:0 \
  | tee results/pilot/evidence_meta/health-gate-console.txt
```

必须看到：

```text
PILOT_HEALTH_GATE_PASS
```

失败时不要启动 120-epoch pilot。保存 JSON 与控制台输出，先分析失败原因；代码一旦
修改，必须产生新 commit，并从本节之前重新开始。

## 5. 启动完整四项 120-epoch pilot

推荐在单独的 tmux 会话中一次启动完整 block：

```bash
tmux new -s talif-v2-pilot
```

在 tmux 内：

```bash
cd /hy-tmp/ta-lif-msresnet
source "$HOME/venvs/talif-msresnet-v2/bin/activate"

export CUDA_VISIBLE_DEVICES=0
export PYTHONUNBUFFERED=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8

python scripts/run_matrix.py \
  --protocol configs/protocol_v2_pilot.yaml \
  --config-dir configs/v2_pilot_generated \
  --allow-unfrozen-pilot \
  --device cuda:0 \
  --stop-on-error \
  | tee results/pilot/evidence_meta/pilot-matrix-console.txt
```

启动训练前必须先看到：

```text
V2_PILOT_CURRENT_CONTEXT_PASS
```

编排器还会在每个子实验启动前重新核对 GPU UUID、主机与软件环境、CIFAR-100
划分清单和固定 batch。健康门之后换实例、换 GPU、修改数据或改变软件环境都会在训练前被阻止；
此时必须在新实例重新运行健康门，不能沿用旧实例的健康报告。

不要传 `--condition`、`--config`、其他 `--output-root` 或其他 `--pilot-plan`。
编排器会强制使用协议中的固定路径，并验证：

- 健康报告为 `PASS`；
- 健康报告 protocol hash、acceptance hash、Git commit 均与当前代码一致；
- tracked worktree 仍然干净；
- 选择的是唯一完整 C1-C4 block；
- 训练子进程再次核验 pilot plan 与健康报告 SHA-256。

若发生纯技术性中断，只允许在**同一实例、同一 GPU UUID、同一软件和数据环境**中，
从该 run 自己的 `last.pt` 继续。重新执行完整四项命令并增加 `--resume-matrix`：

```bash
python scripts/run_matrix.py \
  --protocol configs/protocol_v2_pilot.yaml \
  --config-dir configs/v2_pilot_generated \
  --allow-unfrozen-pilot \
  --device cuda:0 \
  --resume-matrix \
  --stop-on-error \
  | tee -a results/pilot/evidence_meta/pilot-matrix-console.txt
```

已完成的条件会被跳过；未完成条件只会从其本目录的 `last.pt` 继续。不要删除或改写
`failure.json`、`failed.pt`、旧 attempt 日志或 `matrix_events.jsonl`。跨实例续跑被禁止；
120-epoch schedule 判定失败后也禁止在其 checkpoint 上追加到 160 epochs。

退出 tmux 显示但不中止任务：按 `Ctrl+B`，再按 `D`。重新进入：

```bash
tmux attach -t talif-v2-pilot
```

另开终端监控：

```bash
watch -n 30 nvidia-smi
```

```bash
watch -n 60 'find results/pilot/v2_seed77_e120 -name seed_metrics.json -type f | wc -l'
```

不要因短时间没有 `epoch_completed` 就自动终止。先检查 trainer 进程、GPU 利用率和
当前 run 的 `events.jsonl` 是否继续增长。

## 6. 120-epoch pilot 验收

只有矩阵命令退出为 0 且四项均完成后执行一次：

```bash
test ! -e results/pilot/v2_seed77_e120_validation.json

python scripts/validate_v2_pilot.py \
  | tee results/pilot/evidence_meta/pilot-validation-console.txt
```

验收器检查：

- 恰好四个 C1-C4 结果目录，且不存在未解决失败；历史失败工件仅在同一 run 后续通过
  同环境 `last.pt` 续跑成功、matrix 事件和 `resume_history` 完整时允许保留；
- protocol/config/file hashes 一致；
- health report、pilot plan、每次启动的 runtime/data context 证据链一致；
- 共享权重和数据划分 hash 一致；
- 四项训练环境完全一致且符合 RTX 5090 / PyTorch / CUDA / FP32 确定性约束；
- 每项恰好有 `0..119` 共 120 个 epoch 事件；
- C1/C3 始终记录 `ta_enabled=false`；
- C2/C4 按预定激活 epoch 切换；
- 每项 best validation accuracy 均达到 `0.60`；
- 最佳准确率、最佳损失、最佳 epoch 和收敛 epoch 均由 `epoch_completed` 事件重算，
  并与 `seed_metrics.json`、汇总 CSV、`best.pt` 和 `last.pt` 一致；
- C2/C4 的 TA 启用后/冻结期 median epoch-time ratio 均不超过 `3.0`。

### PASS

必须同时看到：

```text
V2_PILOT_PASS
DECISION=ACCEPT_120_EPOCH_SCHEDULE_FOR_FORMAL_V2_DESIGN
```

这只允许进入“正式 v2 方案起草”，并不允许直接跑正式实验。下一步是建立 32-run
正式协议、三位作者重新确认、Phase A、Phase B、full preflight，然后才能正式启动。

### FAIL

若看到：

```text
DECISION=REQUIRE_FRESH_160_EPOCH_PILOT
```

必须新建 160-epoch 协议、新 protocol hash、新配置目录、新健康报告、新 pilot plan
和新结果根。禁止从 120-epoch pilot 的任何 checkpoint resume。

### INVALID

环境混用、工件缺失、hash 不一致或日志语义错误会得到 `INVALID`。此时禁止正式 v2，
先保全全部工件并调查完整性问题。

## 7. 打包和下载 pilot 证据

```bash
cd /hy-tmp/ta-lif-msresnet

tar -czf /hy-tmp/ta_lif_v2_pilot_e120_$(date +%Y%m%d_%H%M%S).tar.gz \
  configs/protocol_v2_pilot.yaml \
  configs/v2_pilot_generated \
  environment/unfrozen_pilot_plan_v2_seed77_e120.json \
  results/pilot/v2_seed77_e120_health.json \
  results/pilot/v2_seed77_e120_validation.json \
  results/pilot/v2_seed77_e120 \
  results/pilot/evidence_meta

sha256sum /hy-tmp/ta_lif_v2_pilot_e120_*.tar.gz \
  | tee /hy-tmp/ta_lif_v2_pilot_e120_SHA256SUMS.txt
```

下载压缩包和 `SHA256SUMS`，在本地独立复核后再决定是否释放实例。
