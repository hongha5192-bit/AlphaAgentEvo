#!/bin/bash
# AlphaAgentEvo v2 — Verl v0.4.1 + sglang 0.4.6.post5
set -x

WORK=/workspace/v2
VERL=$WORK/verl
MODEL=/workspace/models/Qwen3-4B-Thinking-2507
DATA=$WORK/data
N_GPUS=$(nvidia-smi -L | wc -l)

eval "$(/workspace/miniconda/bin/conda shell.bash hook)"
conda activate verl041

# Install libnuma if missing (lost on container restart)
ldconfig -p | grep -q libnuma || { apt-get update -qq && apt-get install -y -qq libnuma-dev > /dev/null 2>&1 && ldconfig; }

export LD_LIBRARY_PATH=/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH
export PYTHONPATH=$WORK:$VERL:$WORK/backtest:$WORK/expression_manager:$PYTHONPATH
export HYDRA_FULL_ERROR=1
export CUDA_VISIBLE_DEVICES=$(seq -s, 0 $((N_GPUS-1)))
export NCCL_TIMEOUT=7200
export NCCL_IB_DISABLE=1
export NCCL_PROTO=Simple
export TORCH_DISTRIBUTED_TIMEOUT=7200
export TORCH_NCCL_TIMEOUT=7200
ulimit -n 65535

# Verify API
curl -s http://localhost:8002/health || { echo "ERROR: Start backtest API first!"; exit 1; }

echo "============================================================"
echo "Starting AlphaAgentEvo GRPO Training (Official Verl)"
echo "  Model: $MODEL"
echo "  GPUs: $N_GPUS"
echo "  Batch: 20, Rollouts: 3, Turns: 2"
echo "============================================================"

cd $VERL

python -m verl.trainer.main_ppo \
    --config-path="$VERL/examples/sglang_multiturn/config" \
    --config-name='search_multiturn_grpo' \
    algorithm.adv_estimator=grpo \
    data.train_batch_size=$((N_GPUS * 7)) \
    data.val_batch_size=$N_GPUS \
    data.max_prompt_length=4096 \
    data.max_response_length=10000 \
    data.filter_overlong_prompts=True \
    data.truncation='left' \
    data.return_raw_chat=True \
    +data.shuffle_train_dataloader=False \
    actor_rollout_ref.model.path=$MODEL \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.optim.lr_warmup_steps_ratio=0.02 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=$((N_GPUS * 7)) \
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
    trainer.n_gpus_per_node=$N_GPUS \
    trainer.nnodes=1 \
    trainer.save_freq=10 \
    trainer.test_freq=10 \
    data.train_files=$DATA/train.parquet \
    data.val_files=$DATA/val.parquet \
    actor_rollout_ref.rollout.multi_turn.tool_config_path="$VERL/examples/sglang_multiturn/config/tool_config/factor_tool_config.yaml" \
    trainer.total_training_steps=150 \
    "$@"
