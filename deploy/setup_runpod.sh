#!/bin/bash
# AlphaAgentEvo — Runpod Deployment Script (Fixed)
# Uses paper's MODIFIED Verl framework (not stock volcengine/verl)
#
# Fixes applied:
#   1. Uses paper's Verl with custom tools (factor_tool, factor_ast)
#   2. Backtest API matches paper's contract (POST /backtest)
#   3. factor_tool.py patched for VN dates (2016-2023)
#   4. batch_size=20 to match paper
#   5. save_freq=10 to avoid checkpoint loss

set -e
echo "============================================================"
echo "AlphaAgentEvo — Runpod Setup (VN Market)"
echo "============================================================"

WORK_DIR="/root/AlphaAgentEvo"
PAPER_VERL_DIR="/root/AlphaAgentEvo-paper-verl"
MODEL_DIR="/root/models/Qwen3-4B-Thinking-2507"
DATA_DIR="$WORK_DIR/data"
N_GPUS=$(nvidia-smi -L | wc -l)

echo "GPUs detected: $N_GPUS"

# ── Step 1: Clone our repo (seeds, data, backtest, deploy) ──
echo ""
echo "[1/7] Cloning AlphaAgentEvo..."
if [ ! -d "$WORK_DIR" ]; then
    git clone https://github.com/hongha5192-bit/AlphaAgentEvo.git $WORK_DIR
fi

# ── Step 2: Clone paper's modified Verl (has custom tools) ──
echo ""
echo "[2/7] Setting up paper's modified Verl..."
# Paper's Verl is in the AlphaAgent repo under AlphaAgentEvo/verl/
if [ ! -d "$PAPER_VERL_DIR" ]; then
    git clone --depth 1 https://github.com/hongha5192-bit/AlphaAgent.git /tmp/AlphaAgent-clone
    cp -r /tmp/AlphaAgent-clone/AlphaAgentEvo/verl $PAPER_VERL_DIR
    rm -rf /tmp/AlphaAgent-clone
fi

# Patch factor_tool.py for VN dates
echo "Patching factor_tool.py for VN market..."
cp $WORK_DIR/deploy/factor_tool_vn.py $PAPER_VERL_DIR/verl/tools/factor_tool.py

echo "Paper's Verl tools:"
ls $PAPER_VERL_DIR/verl/tools/*.py

# ── Step 3: Install dependencies ──
echo ""
echo "[3/7] Installing dependencies..."
pip install -q torch transformers peft accelerate
pip install -q fastapi uvicorn requests pandas numpy pyarrow
pip install -q ray[default] hydra-core datasets
pip install -q Levenshtein jmespath
pip install -q sglang

cd $PAPER_VERL_DIR
pip install -q --no-deps -e . 2>/dev/null || pip install -q -e .

# ── Step 4: Download model ──
echo ""
echo "[4/7] Downloading Qwen3-4B-Thinking-2507..."
if [ ! -d "$MODEL_DIR" ]; then
    pip install -q huggingface_hub
    python3 -c "
from huggingface_hub import snapshot_download
snapshot_download('Qwen/Qwen3-4B', local_dir='$MODEL_DIR')
"
fi
echo "Model ready at $MODEL_DIR"

# ── Step 5: Setup data ──
echo ""
echo "[5/7] Setting up data..."
if [ ! -f "$WORK_DIR/backtest/data/daily_pv.h5" ]; then
    echo "⚠️  WARNING: daily_pv_v2.h5 not found!"
    echo "Upload it: scp daily_pv_v2.h5 root@<pod>:$WORK_DIR/backtest/data/daily_pv.h5"
fi

echo "Checking files:"
for f in train.parquet val.parquet test.parquet vn_seeds_300.jsonl seed_backtest_results.json sector_map.json; do
    [ -f "$DATA_DIR/$f" ] && echo "  ✅ $f" || echo "  ❌ $f MISSING"
done

# ── Step 6: Start Verl-compatible backtest API ──
echo ""
echo "[6/7] Starting backtest API (Verl-compatible)..."
cd $WORK_DIR
export PYTHONPATH=$PYTHONPATH:$WORK_DIR:$PAPER_VERL_DIR

nohup python3 deploy/api_server_verl.py > /tmp/api.log 2>&1 &
API_PID=$!
echo "API PID: $API_PID"

for i in $(seq 1 30); do
    if curl -s http://localhost:8001/health > /dev/null 2>&1; then
        echo "✅ Backtest API ready on port 8001 (/backtest endpoint)"
        break
    fi
    sleep 2
done

# Test the API with paper's expected format
echo "Testing API contract..."
RESULT=$(curl -s -X POST http://localhost:8001/backtest \
    -H "Content-Type: application/json" \
    -d '{"exprs": {"test": "RANK($close)"}}')
echo "API test result: $RESULT"

# ── Step 7: Print training command ──
echo ""
echo "[7/7] Setup complete!"
echo ""
echo "============================================================"
echo "READY TO TRAIN"
echo "============================================================"
echo ""
echo "GPUs: $N_GPUS"
echo "Batch size: 20 (matching paper)"
echo "Model: $MODEL_DIR"
echo "Verl: $PAPER_VERL_DIR (paper's modified version)"
echo "API: http://localhost:8001/backtest (Verl-compatible)"
echo ""
echo "To start training:"
echo "  cd $WORK_DIR && bash deploy/run_training.sh"
echo ""
echo "============================================================"
