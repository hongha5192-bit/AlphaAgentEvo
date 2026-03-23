#!/bin/bash
# AlphaAgentEvo — Runpod Deployment Script
# Installs Verl, sets up data, starts backtest API, launches training
#
# Usage:
#   1. Rent Runpod with 4-10 GPUs (A100 80GB recommended)
#   2. SSH into pod
#   3. Run: bash setup_runpod.sh
#
# Prerequisites: CUDA, conda/python3.10, git

set -e
echo "============================================================"
echo "AlphaAgentEvo — Runpod Setup"
echo "============================================================"

# ── Config ──
WORK_DIR="/root/AlphaAgentEvo"
MODEL_DIR="/root/models/Qwen3-4B-Thinking-2507"
DATA_DIR="$WORK_DIR/data"
VERL_DIR="$WORK_DIR/verl"
N_GPUS=$(nvidia-smi -L | wc -l)

echo "GPUs detected: $N_GPUS"
echo "Work dir: $WORK_DIR"

# ── Step 1: Clone repos ──
echo ""
echo "[1/6] Cloning repos..."
if [ ! -d "$WORK_DIR" ]; then
    git clone https://github.com/hongha5192-bit/AlphaAgentEvo.git $WORK_DIR
fi

# Clone Verl (paper's framework)
if [ ! -d "$VERL_DIR" ]; then
    # Use the paper's Verl fork from AlphaAgent repo
    git clone --depth 1 https://github.com/volcengine/verl.git $VERL_DIR
fi

# ── Step 2: Install dependencies ──
echo ""
echo "[2/6] Installing dependencies..."
pip install -q torch transformers>=5.2.0 peft accelerate
pip install -q fastapi uvicorn requests pandas numpy pyarrow
pip install -q ray[default] hydra-core datasets
pip install -q Levenshtein jmespath
pip install -q sglang  # Inference engine for Verl rollout

cd $VERL_DIR
pip install -q --no-deps -e .

# ── Step 3: Download model ──
echo ""
echo "[3/6] Downloading Qwen3-4B-Thinking-2507..."
if [ ! -d "$MODEL_DIR" ]; then
    pip install -q huggingface_hub
    python3 -c "
from huggingface_hub import snapshot_download
snapshot_download('Qwen/Qwen3-4B', local_dir='$MODEL_DIR')
"
fi
echo "Model ready at $MODEL_DIR"

# ── Step 4: Prepare data ──
echo ""
echo "[4/6] Preparing data..."
# daily_pv_v2.h5 needs to be uploaded separately (9.6MB)
# Check if it exists
if [ ! -f "$WORK_DIR/backtest/data/daily_pv.h5" ]; then
    echo "WARNING: daily_pv.h5 not found!"
    echo "Upload daily_pv_v2.h5 to $WORK_DIR/backtest/data/daily_pv.h5"
    echo "Or create symlink from your data location"
fi

# Verify data files
echo "Checking data files:"
for f in train.parquet val.parquet test.parquet vn_seeds_300.jsonl; do
    if [ -f "$DATA_DIR/$f" ]; then
        echo "  ✅ $f"
    else
        echo "  ❌ $f MISSING"
    fi
done

# ── Step 5: Start backtest API ──
echo ""
echo "[5/6] Starting backtest API..."
cd $WORK_DIR
nohup python3 backtest/api_server.py > /tmp/api.log 2>&1 &
API_PID=$!
echo "API PID: $API_PID"

# Wait for API
for i in $(seq 1 30); do
    if curl -s http://localhost:8001/health > /dev/null 2>&1; then
        echo "Backtest API ready on port 8001"
        break
    fi
    sleep 2
done

# ── Step 6: Print launch command ──
echo ""
echo "[6/6] Setup complete!"
echo ""
echo "============================================================"
echo "To start training, run:"
echo "============================================================"
echo ""
echo "cd $WORK_DIR"
echo "bash deploy/run_training.sh"
echo ""
echo "Or manually:"
echo ""
echo "python3 -m verl.trainer.main_ppo \\"
echo "    --config-path=$VERL_DIR/examples/sglang_multiturn/config \\"
echo "    --config-name=search_multiturn_grpo \\"
echo "    data.train_files=$DATA_DIR/train.parquet \\"
echo "    data.val_files=$DATA_DIR/val.parquet \\"
echo "    data.train_batch_size=20 \\"
echo "    actor_rollout_ref.model.path=$MODEL_DIR \\"
echo "    trainer.n_gpus_per_node=$N_GPUS \\"
echo "    trainer.total_training_steps=150 \\"
echo "    trainer.save_freq=10"
echo ""
echo "============================================================"
