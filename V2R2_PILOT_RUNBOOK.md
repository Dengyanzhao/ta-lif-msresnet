# v2r2 非报告性 pilot 运行手册

本手册只适用于身份 `v2r2_seed88_of80_e120`。它用于验证修复后的
TA-LIF 实现和 120-epoch 候选调度，不得作为论文结果、正式统计样本或正式种子。
设计依据、旧失败证据和新工程 benchmark 哈希见 `V2R2_PILOT_DECISION.md`。

## 0. 固定身份与不可跨越边界

- 协议：`configs/protocol_v2r2_seed88_of80_e120.yaml`。
- 生成配置：`configs/v2r2_seed88_of80_e120_generated`。
- 唯一 seed：`88`；唯一顺序：`C1 -> C2 -> C3 -> C4`。
- 健康门：固定批次过拟合 80 步；准确率、损失、梯度和计时阈值均不得修改。
- pilot：四个条件各 120 epochs，结果根为
  `results/pilot/v2r2_seed88_of80_e120`。
- 环境：RTX 5090、PyTorch `2.9.1+cu128`、CUDA `12.8`、FP32、严格确定性。
- seed-77 v2r1 协议、失败健康报告和工程 benchmark 是只读历史证据，不得删除、
  覆盖、重跑、续跑或混入本轮结果。
- 新健康报告 JSON 一旦生成，无论 PASS 或 FAIL，seed 88 即视为已消费。
  FAIL 时不得自动重试、降阈值或继续 120-epoch pilot；必须保全证据并另建 r3 身份。
- `PILOT_HEALTH_GATE_BLOCKED` 且未生成 JSON 表示 GPU 工作尚未启动，可修正命令或
  环境后重新进行启动前检查。其他中断不得自行重试，须先审计现有进程和工件。

## 1. 切换到固定提交

以下命令在 RTX 5090 实例执行。把 `<V2R2_COMMIT>` 替换为本地验证并推送后的完整
40 位提交号，不要使用分支尖端作为实验身份。

```bash
cd /hy-tmp/ta-lif-msresnet

git -c http.version=HTTP/1.1 fetch origin v2-fix-pilot
git checkout --detach <V2R2_COMMIT>

test "$(git rev-parse HEAD)" = "<V2R2_COMMIT>" \
  && echo V2R2_COMMIT_PASS
test -z "$(git status --porcelain --untracked-files=no)" \
  && echo TRACKED_WORKTREE_CLEAN
```

建立或复用独立环境：

```bash
export VENV="$HOME/venvs/talif-msresnet-v2r2"
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

确认数据，但不要恢复任何 v1 或 seed-77 checkpoint：

```bash
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

## 2. 测试、重生成校验和 preflight

这一节可以重复执行，但必须保持 tracked 工作树干净。生成器只写入新的 v2r2 配置目录。

```bash
export V2R2_ID="v2r2_seed88_of80_e120"
export V2R2_PROTOCOL="configs/protocol_${V2R2_ID}.yaml"
export V2R2_CONFIG_DIR="configs/${V2R2_ID}_generated"
export V2R2_EVIDENCE="results/pilot/evidence_meta_${V2R2_ID}"

mkdir -p "$V2R2_EVIDENCE"

python -m pytest -q \
  | tee "$V2R2_EVIDENCE/pytest.txt"

python scripts/generate_run_configs.py \
  --protocol "$V2R2_PROTOCOL" \
  --output "$V2R2_CONFIG_DIR"

git diff --exit-code -- "$V2R2_CONFIG_DIR" \
  && echo V2R2_GENERATED_CONFIGS_MATCH_COMMIT

python scripts/preflight.py \
  --protocol "$V2R2_PROTOCOL" \
  --mode pilot \
  | tee "$V2R2_EVIDENCE/pilot-preflight.txt"

test "$(find "$V2R2_CONFIG_DIR" -maxdepth 1 -type f -name '*.yaml' | wc -l)" -eq 4 \
  && echo V2R2_4_CONFIGS_PASS
test "$(find "$V2R2_CONFIG_DIR" -mindepth 1 -maxdepth 1 -type f | wc -l)" -eq 6 \
  && echo V2R2_6_GENERATED_FILES_PASS
test -z "$(git status --porcelain --untracked-files=no)" \
  && echo TRACKED_WORKTREE_CLEAN_BEFORE_HEALTH
```

保存启动前环境证据：

```bash
git rev-parse HEAD > "$V2R2_EVIDENCE/git-commit.txt"
git status --short > "$V2R2_EVIDENCE/git-status-short.txt"
python -m pip freeze --exclude-editable > "$V2R2_EVIDENCE/pip-freeze.txt"
uname -a > "$V2R2_EVIDENCE/uname.txt"
nvidia-smi -q > "$V2R2_EVIDENCE/nvidia-smi-q.txt"

nvidia-smi \
  --query-compute-apps=pid,process_name,used_gpu_memory \
  --format=csv,noheader
```

最后一条应无计算进程输出。若 GPU 被占用，不要运行健康门。

## 3. 唯一一次 seed-88 健康门

先确认所有新路径不存在。任何一个已存在都必须停止并审计，不能覆盖。

```bash
test ! -e results/pilot/v2r2_seed88_of80_e120_health.json
test ! -e results/pilot/v2r2_seed88_of80_e120_validation.json
test ! -e results/pilot/v2r2_seed88_of80_e120
test ! -e environment/unfrozen_pilot_plan_v2r2_seed88_of80_e120.json
```

在普通终端或 tmux 中只执行一次：

```bash
python scripts/pilot_health_gate.py \
  --config configs/v2r2_seed88_of80_e120_generated/E1_cifar100_d20_t6_C1_s88.yaml \
  --protocol configs/protocol_v2r2_seed88_of80_e120.yaml \
  --output results/pilot/v2r2_seed88_of80_e120_health.json \
  --device cuda:0 \
  --expected-gpu-substring "RTX 5090" \
  --seed 88 \
  --overfit-batch-size 8 \
  --timing-batch-size 64 \
  --overfit-steps 80 \
  --min-overfit-accuracy 0.50 \
  --max-overfit-loss-fraction 0.90 \
  --min-ta-routed-gradient-coverage 0.90 \
  --timing-warmup 2 \
  --timing-iterations 5 \
  --max-ta-step-ratio 3.0 \
  2>&1 | tee "$V2R2_EVIDENCE/health-gate-console.txt"

HEALTH_RC=${PIPESTATUS[0]}
echo "HEALTH_RC=$HEALTH_RC"
test "$HEALTH_RC" -eq 0 \
  && grep -qx 'PILOT_HEALTH_GATE_PASS' "$V2R2_EVIDENCE/health-gate-console.txt" \
  && echo V2R2_HEALTH_GATE_PASS
```

只有同时得到 `HEALTH_RC=0` 和 `V2R2_HEALTH_GATE_PASS` 才可进入第 4 节。
`FAIL` 必须停止；不得重跑 seed 88，也不得启动矩阵。

健康门通过后立即保存小证据包：

```bash
tar -czf "/hy-tmp/ta_lif_${V2R2_ID}_health_$(date +%Y%m%d_%H%M%S).tar.gz" \
  "$V2R2_PROTOCOL" \
  "$V2R2_CONFIG_DIR" \
  results/pilot/v2r2_seed88_of80_e120_health.json \
  "$V2R2_EVIDENCE"

sha256sum /hy-tmp/ta_lif_v2r2_seed88_of80_e120_health_*.tar.gz \
  | tee /hy-tmp/ta_lif_v2r2_seed88_of80_e120_health_SHA256SUMS.txt
```

## 4. 启动完整四项 120-epoch pilot

推荐新建 tmux：

```bash
tmux new -s talif-v2r2-pilot
```

在 tmux 内重新设置变量和环境，然后一次启动完整 block：

```bash
cd /hy-tmp/ta-lif-msresnet
source "$HOME/venvs/talif-msresnet-v2r2/bin/activate"

export CUDA_VISIBLE_DEVICES=0
export PYTHONUNBUFFERED=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export V2R2_ID="v2r2_seed88_of80_e120"
export V2R2_PROTOCOL="configs/protocol_${V2R2_ID}.yaml"
export V2R2_CONFIG_DIR="configs/${V2R2_ID}_generated"
export V2R2_EVIDENCE="results/pilot/evidence_meta_${V2R2_ID}"

python scripts/run_matrix.py \
  --protocol "$V2R2_PROTOCOL" \
  --config-dir "$V2R2_CONFIG_DIR" \
  --output-root results/pilot/v2r2_seed88_of80_e120 \
  --pilot-plan environment/unfrozen_pilot_plan_v2r2_seed88_of80_e120.json \
  --allow-unfrozen-pilot \
  --device cuda:0 \
  --stop-on-error \
  2>&1 | tee "$V2R2_EVIDENCE/pilot-matrix-console.txt"

MATRIX_RC=${PIPESTATUS[0]}
echo "MATRIX_RC=$MATRIX_RC"
```

训练前必须出现 `V2_PILOT_CURRENT_CONTEXT_PASS`。完整结束必须有
`MATRIX_RC=0`、四个完成结果且零失败。不要传 `--condition`、`--config`、`--limit`、
`--dry-run` 或其他输出路径。

纯技术中断只允许在同一实例、同一 GPU UUID、同一软件和数据环境中，从该 run 自己的
`last.pt` 恢复：

```bash
python scripts/run_matrix.py \
  --protocol "$V2R2_PROTOCOL" \
  --config-dir "$V2R2_CONFIG_DIR" \
  --output-root results/pilot/v2r2_seed88_of80_e120 \
  --pilot-plan environment/unfrozen_pilot_plan_v2r2_seed88_of80_e120.json \
  --allow-unfrozen-pilot \
  --device cuda:0 \
  --resume-matrix \
  --stop-on-error \
  2>&1 | tee -a "$V2R2_EVIDENCE/pilot-matrix-console.txt"
```

不要自动使用这条恢复命令。先确认是技术中断而非训练失败；失败实验不自动重试。

退出 tmux 但保持任务：按 `Ctrl+B`，再按 `D`。重新进入：

```bash
tmux attach -t talif-v2r2-pilot
```

另开终端监控：

```bash
watch -n 30 nvidia-smi
```

```bash
watch -n 60 'find results/pilot/v2r2_seed88_of80_e120 -name seed_metrics.json -type f | wc -l'
```

## 5. pilot 验收

只有矩阵命令 `MATRIX_RC=0` 且四项都完成后执行一次：

```bash
test ! -e results/pilot/v2r2_seed88_of80_e120_validation.json

python scripts/validate_v2_pilot.py \
  --protocol configs/protocol_v2r2_seed88_of80_e120.yaml \
  --config-dir configs/v2r2_seed88_of80_e120_generated \
  --results-root results/pilot/v2r2_seed88_of80_e120 \
  --output results/pilot/v2r2_seed88_of80_e120_validation.json \
  2>&1 | tee "$V2R2_EVIDENCE/pilot-validation-console.txt"

VALIDATION_RC=${PIPESTATUS[0]}
echo "VALIDATION_RC=$VALIDATION_RC"
```

PASS 必须同时出现：

```text
V2_PILOT_PASS
DECISION=ACCEPT_120_EPOCH_SCHEDULE_FOR_FORMAL_V2_DESIGN
VALIDATION_RC=0
```

PASS 只允许起草新的正式协议，不允许直接把 pilot 写进论文或启动正式实验。
FAIL、INVALID 或任何非零退出码都必须停止并保全全部工件。若判定需要 160 epochs，
必须建立新协议、新 seed、新健康报告、新计划和新结果根，禁止从本轮 checkpoint 续跑。

## 6. 打包并下载完整证据

```bash
cd /hy-tmp/ta-lif-msresnet

tar -czf "/hy-tmp/ta_lif_${V2R2_ID}_complete_$(date +%Y%m%d_%H%M%S).tar.gz" \
  "$V2R2_PROTOCOL" \
  "$V2R2_CONFIG_DIR" \
  environment/unfrozen_pilot_plan_v2r2_seed88_of80_e120.json \
  results/pilot/v2r2_seed88_of80_e120_health.json \
  results/pilot/v2r2_seed88_of80_e120_validation.json \
  results/pilot/v2r2_seed88_of80_e120 \
  "$V2R2_EVIDENCE"

sha256sum /hy-tmp/ta_lif_v2r2_seed88_of80_e120_complete_*.tar.gz \
  | tee /hy-tmp/ta_lif_v2r2_seed88_of80_e120_complete_SHA256SUMS.txt
```

下载 `.tar.gz` 和 `SHA256SUMS`，在本地独立复核后再释放实例。
