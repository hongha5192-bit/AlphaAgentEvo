#!/bin/bash
# AlphaAgentEvo v2 — Verl v0.4.1 + sglang 0.4.6.post5
set -Eeuo pipefail

trap 'echo "[ERROR] train.sh failed at line $LINENO: $BASH_COMMAND" >&2; ls -lah /tmp/ray || true; ls -lah /tmp/ray/session_latest/logs 2>/dev/null || true; tail -n 200 /tmp/ray/session_latest/logs/gcs_server.out 2>/dev/null || true; tail -n 200 /tmp/ray/session_latest/logs/raylet.out 2>/dev/null || true' ERR

log() { echo "[$(date '+%F %T')] $*"; }

WORK=/workspace/v2
VERL=$WORK/verl
MODEL=/workspace/models/Qwen3-4B-Thinking-2507
DATA=$WORK/data
ENV_NAME=verl041
N_GPUS=$(nvidia-smi -L | wc -l)

# Batch sizes must be divisible by N_GPUS
BATCH_SIZE=$((N_GPUS * 7))
VAL_BATCH_SIZE=$N_GPUS

[[ -d "$VERL" ]] || { echo "Missing Verl checkout at $VERL" >&2; exit 1; }
[[ -d "$DATA" ]] || { echo "Missing data dir at $DATA" >&2; exit 1; }
[[ -d "$WORK/checkpoints" ]] || mkdir -p "$WORK/checkpoints"
[[ -d "$WORK/logs" ]] || mkdir -p "$WORK/logs"
[[ "$N_GPUS" -gt 0 ]] || { echo "No GPUs detected" >&2; exit 1; }

# Shell/env
set -x
eval "$('/workspace/miniconda/bin/conda' shell.bash hook)"
conda activate "$ENV_NAME"

# Shared libs that can disappear on restart
if ! ldconfig -p | grep -q 'libnuma.so.1'; then
  apt-get update -qq
  apt-get install -y -qq libnuma-dev >/dev/null 2>&1
  ldconfig
fi
ldconfig -p | grep -q 'libnuma.so.1'

export LD_LIBRARY_PATH=/lib/x86_64-linux-gnu:${LD_LIBRARY_PATH:-}
export PYTHONPATH="$WORK:$VERL:$WORK/backtest:$WORK/expression_manager:${PYTHONPATH:-}"
export HYDRA_FULL_ERROR=1
export CUDA_VISIBLE_DEVICES=$(seq -s, 0 $((N_GPUS-1)))
export NCCL_TIMEOUT=7200
export NCCL_IB_DISABLE=1
export NCCL_PROTO=Simple
export TORCH_DISTRIBUTED_TIMEOUT=7200
export TORCH_NCCL_TIMEOUT=7200
ulimit -n 65535

# Version / path gate
python - <<PY
import sys, torch, sglang, verl
print('python', sys.executable)
print('torch', torch.__version__)
print('sglang', sglang.__version__)
print('verl', verl.__file__)
assert torch.__version__.startswith('2.6.0'), f'Expected torch 2.6.0, got {torch.__version__}'
assert sglang.__version__ == '0.4.6.post5', f'Expected sglang 0.4.6.post5, got {sglang.__version__}'
assert 'v2/verl' in verl.__file__, verl.__file__
assert torch.cuda.is_available()
assert torch.cuda.device_count() == $N_GPUS
PY

git -C "$VERL" describe --tags --always | grep -qx 'v0.4.1'

# API gate with retries
for i in $(seq 1 20); do
  if curl -fsS http://localhost:8002/health >/dev/null; then
    break
  fi
  echo "Waiting for backtest API on :8002 (attempt $i/20)..." >&2
  sleep 3
done
curl -fsS http://localhost:8002/health >/dev/null || { echo "ERROR: Start backtest API first!" >&2; exit 1; }

# Clean stale Ray state before every launch
ray stop --force >/dev/null 2>&1 || true
pkill -9 -f 'ray|raylet|gcs_server|monitor.py|dashboard' >/dev/null 2>&1 || true
rm -rf /tmp/ray /dev/shm/ray* || true

log "============================================================"
log "Starting AlphaAgentEvo GRPO Training (Official Verl)"
log "  Model: $MODEL"
log "  GPUs: $N_GPUS"
log "  Batch: $BATCH_SIZE, Rollouts: 3, Turns: 2"
log "============================================================"

cd "$VERL"

python -m verl.trainer.main_ppo \
  --config-path="$VERL/examples/sglang_multiturn/config" \
  --config-name='search_multiturn_grpo' \
  algorithm.adv_estimator=grpo \
  data.train_batch_size="$BATCH_SIZE" \
  data.val_batch_size="$VAL_BATCH_SIZE" \
  data.max_prompt_length=4096 \
  data.max_response_length=10000 \
  data.filter_overlong_prompts=True \
  data.truncation='left' \
  data.return_raw_chat=True \
  +data.shuffle_train_dataloader=False \
  actor_rollout_ref.model.path="$MODEL" \
  actor_rollout_ref.actor.optim.lr=1e-6 \
  actor_rollout_ref.actor.optim.lr_warmup_steps_ratio=0.02 \
  actor_rollout_ref.model.use_remove_padding=False \
  actor_rollout_ref.actor.ppo_mini_batch_size="$BATCH_SIZE" \
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
  actor_rollout_ref.actor.use_kl_loss=True \
  actor_rollout_ref.actor.kl_loss_coef=0.001 \
  actor_rollout_ref.actor.kl_loss_type=low_var_kl \
  actor_rollout_ref.actor.entropy_coeff=0.00 \
  actor_rollout_ref.model.enable_gradient_checkpointing=True \
  actor_rollout_ref.actor.fsdp_config.param_offload=False \
  actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
  +actor_rollout_ref.actor.fsdp_config.mixed_precision.param_dtype=bfloat16 \
  +actor_rollout_ref.actor.fsdp_config.mixed_precision.reduce_dtype=bfloat16 \
  +actor_rollout_ref.actor.fsdp_config.mixed_precision.buffer_dtype=bfloat16 \
  actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
  actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
  actor_rollout_ref.rollout.name=sglang \
  actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
  actor_rollout_ref.rollout.n=3 \
  actor_rollout_ref.rollout.multi_turn.max_assistant_turns=2 \
  actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 \
  actor_rollout_ref.ref.fsdp_config.param_offload=False \
  custom_reward_function.path="$VERL/verl/utils/reward_score/factor.py" \
  custom_reward_function.name=compute_score \
  algorithm.use_kl_in_reward=False \
  trainer.critic_warmup=0 \
  trainer.val_before_train=False \
  'trainer.logger=["console","tensorboard"]' \
  trainer.project_name='alphaagentevo-vn' \
  trainer.experiment_name='qwen3-4b-vn-official-verl' \
  trainer.n_gpus_per_node="$N_GPUS" \
  trainer.nnodes=1 \
  trainer.save_freq=10 \
  trainer.test_freq=10 \
  data.train_files="$DATA/train.parquet" \
  data.val_files="$DATA/val.parquet" \
  actor_rollout_ref.rollout.multi_turn.tool_config_path="$VERL/examples/sglang_multiturn/config/tool_config/factor_tool_config.yaml" \
  trainer.total_training_steps=150 \
  "$@"
