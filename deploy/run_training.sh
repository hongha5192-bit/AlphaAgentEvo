#!/bin/bash
# AlphaAgentEvo — Verl GRPO Training (Fixed)
# Uses paper's MODIFIED Verl with custom factor_tool
# batch_size=20 matching paper exactly

set -x

# ── Paths ──
WORK_DIR="/root/AlphaAgentEvo"
VERL_DIR="/root/AlphaAgentEvo-paper-verl"
MODEL_DIR="/root/models/Qwen3-4B-Thinking-2507"
DATA_DIR="$WORK_DIR/data"
TRAIN_DATA="$DATA_DIR/train.parquet"
VAL_DATA="$DATA_DIR/val.parquet"

N_GPUS=$(nvidia-smi -L | wc -l)

# ── Batch size = 20 (matching paper exactly) ──
BATCH_SIZE=20
echo "Training: $N_GPUS GPUs, batch_size=$BATCH_SIZE (paper: 20)"

# ── Environment ──
export HYDRA_FULL_ERROR=1
export CUDA_VISIBLE_DEVICES=$(seq -s, 0 $((N_GPUS-1)))
export VERL_LOGGING_LEVEL=INFO
export VERL_PPO_LOGGING_LEVEL=INFO
export TORCH_USE_CUDA_DSA=1

# NCCL for Runpod
export NCCL_TIMEOUT=7200
export NCCL_IB_TIMEOUT=7200
export NCCL_IB_RETRY_CNT=7
export NCCL_IB_DISABLE=1
export NCCL_PROTO=Simple
export NCCL_MIN_NCHANNELS=2
export NCCL_MAX_NCHANNELS=4
export NCCL_BLOCKING_WAIT=1
export NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_DEBUG=WARN
export NCCL_P2P_DISABLE=0
export NCCL_P2P_LEVEL=SYS
export NCCL_SOCKET_IFNAME=lo

# PyTorch distributed
export TORCH_DISTRIBUTED_TIMEOUT=7200
export TORCH_NCCL_TIMEOUT=7200
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export TORCH_NCCL_BLOCKING_WAIT=1
export TORCH_NCCL_TRACE_BUFFER_SIZE=16384
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128
export CUDA_LAUNCH_BLOCKING=0

export PYTHONPATH=$PYTHONPATH:$WORK_DIR:$VERL_DIR
ulimit -n 65535

# ── GPU status ──
echo "=== GPU Memory Status ==="
nvidia-smi --query-gpu=index,name,memory.total,memory.used,memory.free --format=csv
echo "========================="

# ── Verify backtest API ──
if ! curl -s http://localhost:8001/health > /dev/null 2>&1; then
    echo "ERROR: Backtest API not running!"
    echo "Starting API..."
    cd $WORK_DIR
    nohup python3 deploy/api_server_verl.py > /tmp/api.log 2>&1 &
    sleep 10
    if ! curl -s http://localhost:8001/health > /dev/null 2>&1; then
        echo "FATAL: Cannot start backtest API"
        exit 1
    fi
fi
echo "✅ Backtest API ready"

# ── Config paths (from paper's modified Verl) ──
CONFIG_PATH="$VERL_DIR/examples/sglang_multiturn/config"
TOOL_CONFIG="$CONFIG_PATH/tool_config/factor_tool_config.yaml"
REWARD_FUNC="$VERL_DIR/verl/utils/reward_score/factor.py"

# Verify configs exist
for f in "$CONFIG_PATH/search_multiturn_grpo.yaml" "$TOOL_CONFIG" "$REWARD_FUNC"; do
    if [ ! -f "$f" ]; then
        echo "ERROR: Missing config: $f"
        exit 1
    fi
done
echo "✅ All configs found"

# ── Launch training ──
echo ""
echo "============================================================"
echo "Starting AlphaAgentEvo GRPO Training"
echo "  Model:      $MODEL_DIR"
echo "  GPUs:       $N_GPUS"
echo "  Batch:      $BATCH_SIZE seeds/step (paper: 20)"
echo "  Rollouts:   3 per seed (paper: 3)"
echo "  Turns:      2 per trajectory (paper: 2)"
echo "  LR:         1e-6 (paper: 1e-6)"
echo "  KL:         0.001 (paper: 0.001)"
echo "  Steps:      150 (paper: 150)"
echo "  Save:       every 10 steps"
echo "  Eval:       use step 80 (paper recommendation)"
echo "============================================================"

python3 -m verl.trainer.main_ppo \
    --config-path="$CONFIG_PATH" \
    --config-name='search_multiturn_grpo' \
    algorithm.adv_estimator=grpo \
    data.train_batch_size=$BATCH_SIZE \
    data.val_batch_size=10 \
    data.max_prompt_length=4096 \
    data.max_response_length=10000 \
    data.filter_overlong_prompts=True \
    data.truncation='left' \
    data.return_raw_chat=True \
    +data.shuffle_train_dataloader=False \
    actor_rollout_ref.model.path=$MODEL_DIR \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.optim.lr_warmup_steps_ratio=0.02 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=$BATCH_SIZE \
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
    +actor_rollout_ref.actor.fsdp_config.activation_checkpointing=False \
    +actor_rollout_ref.actor.fsdp_config.cpu_offload.offload_params=False \
    actor_rollout_ref.rollout.max_model_len=14500 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=sglang \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
    actor_rollout_ref.rollout.n=3 \
    actor_rollout_ref.rollout.multi_turn.max_assistant_turns=2 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.ref.fsdp_config.param_offload=False \
    +actor_rollout_ref.ref.fsdp_config.optimizer_offload=False \
    +actor_rollout_ref.ref.fsdp_config.mixed_precision.param_dtype=bfloat16 \
    +actor_rollout_ref.ref.fsdp_config.mixed_precision.reduce_dtype=bfloat16 \
    +actor_rollout_ref.ref.fsdp_config.mixed_precision.buffer_dtype=bfloat16 \
    custom_reward_function.path="$REWARD_FUNC" \
    custom_reward_function.name=compute_score \
    algorithm.use_kl_in_reward=False \
    trainer.critic_warmup=0 \
    trainer.val_before_train=False \
    trainer.logger='["console","tensorboard"]' \
    trainer.project_name='alphaagentevo-vn' \
    trainer.experiment_name="qwen3-4b-vn_bs${BATCH_SIZE}_rollout3_lr1e-6" \
    trainer.n_gpus_per_node=$N_GPUS \
    trainer.nnodes=1 \
    trainer.save_freq=10 \
    trainer.test_freq=10 \
    data.train_files="$TRAIN_DATA" \
    data.val_files="$VAL_DATA" \
    actor_rollout_ref.rollout.multi_turn.tool_config_path="$TOOL_CONFIG" \
    trainer.total_training_steps=150 \
    "$@"
