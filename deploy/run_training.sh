#!/bin/bash
# AlphaAgentEvo — Verl GRPO Training Launch Script
# Adapted from paper's run_qwen3-4b_instruct_alphaevo_multiturn.sh
# for Vietnam market data on Runpod

set -x

# ── Paths ──
WORK_DIR="/root/AlphaAgentEvo"
VERL_DIR="$WORK_DIR/verl"
MODEL_DIR="/root/models/Qwen3-4B-Thinking-2507"
DATA_DIR="$WORK_DIR/data"
TRAIN_DATA="$DATA_DIR/train.parquet"
VAL_DATA="$DATA_DIR/val.parquet"

# Detect GPUs
N_GPUS=$(nvidia-smi -L | wc -l)
echo "Training with $N_GPUS GPUs"

# ── Batch size ──
# Paper: batch=20 on 10 GPUs. Scale proportionally.
# Each GPU handles ~2 seeds. Minimum batch=4 for meaningful advantage estimation.
if [ $N_GPUS -ge 10 ]; then
    BATCH_SIZE=20
elif [ $N_GPUS -ge 8 ]; then
    BATCH_SIZE=16
elif [ $N_GPUS -ge 4 ]; then
    BATCH_SIZE=8
else
    BATCH_SIZE=4
fi
echo "Batch size: $BATCH_SIZE (paper: 20)"

# ── Environment ──
export HYDRA_FULL_ERROR=1
export CUDA_VISIBLE_DEVICES=$(seq -s, 0 $((N_GPUS-1)))
export VERL_LOGGING_LEVEL=INFO
export TORCH_USE_CUDA_DSA=1

# NCCL for Runpod
export NCCL_TIMEOUT=7200
export NCCL_IB_TIMEOUT=7200
export NCCL_IB_RETRY_CNT=7
export NCCL_IB_DISABLE=1
export NCCL_PROTO=Simple
export NCCL_BLOCKING_WAIT=1
export NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_DEBUG=WARN

# PyTorch distributed
export TORCH_DISTRIBUTED_TIMEOUT=7200
export TORCH_NCCL_TIMEOUT=7200
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export TORCH_NCCL_BLOCKING_WAIT=1
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128

export PYTHONPATH=$PYTHONPATH:$WORK_DIR:$VERL_DIR
ulimit -n 65535

# ── GPU status ──
echo "=== GPU Status ==="
nvidia-smi --query-gpu=index,name,memory.total,memory.used,memory.free --format=csv
echo "=================="

# ── Config paths ──
CONFIG_PATH="$VERL_DIR/examples/sglang_multiturn/config"
TOOL_CONFIG="$CONFIG_PATH/tool_config/factor_tool_config.yaml"
REWARD_FUNC="$VERL_DIR/verl/utils/reward_score/factor.py"

# ── Verify backtest API ──
if ! curl -s http://localhost:8001/health > /dev/null 2>&1; then
    echo "ERROR: Backtest API not running on port 8001!"
    echo "Start it first: cd $WORK_DIR && python3 backtest/api_server.py &"
    exit 1
fi
echo "Backtest API: OK"

# ── Launch training ──
echo ""
echo "============================================================"
echo "Starting AlphaAgentEvo GRPO Training"
echo "  Model: $MODEL_DIR"
echo "  GPUs: $N_GPUS"
echo "  Batch: $BATCH_SIZE seeds/step"
echo "  Steps: 150"
echo "  Save: every 10 steps"
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
    actor_rollout_ref.rollout.max_model_len=14500 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=sglang \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
    actor_rollout_ref.rollout.n=3 \
    actor_rollout_ref.rollout.multi_turn.max_assistant_turns=2 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.ref.fsdp_config.param_offload=False \
    custom_reward_function.path="$REWARD_FUNC" \
    custom_reward_function.name=compute_score \
    algorithm.use_kl_in_reward=False \
    trainer.critic_warmup=0 \
    trainer.val_before_train=False \
    trainer.logger='["console","tensorboard"]' \
    trainer.project_name='alphaagentevo-vn' \
    trainer.experiment_name="qwen3-4b-vn-bs${BATCH_SIZE}_rollout3_lr1e-6" \
    trainer.n_gpus_per_node=$N_GPUS \
    trainer.nnodes=1 \
    trainer.save_freq=10 \
    trainer.test_freq=10 \
    data.train_files="$TRAIN_DATA" \
    data.val_files="$VAL_DATA" \
    actor_rollout_ref.rollout.multi_turn.tool_config_path="$TOOL_CONFIG" \
    trainer.total_training_steps=150 \
    "$@"
