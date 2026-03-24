#!/bin/bash
# AlphaAgentEvo v2 — Verl v0.4.1 + sglang 0.4.6.post5 + torch 2.6.0
# This is the stable, tested combo with explicit multi-turn tool calling support
set -e

echo "============================================================"
echo "AlphaAgentEvo v2 — Verl v0.4.1 (stable multi-turn)"
echo "============================================================"

WORK=/workspace/v2

# ── Step 1: Fresh conda env ──
echo "[1/7] Create fresh conda env"
eval "$(/workspace/miniconda/bin/conda shell.bash hook)"
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main 2>/dev/null || true
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r 2>/dev/null || true
rm -rf /workspace/miniconda/envs/verl310 2>/dev/null || true
conda create -n verl041 python=3.10 -y || true
conda activate verl041
echo "Python: $(python --version)"
echo "Env: $CONDA_DEFAULT_ENV"

mkdir -p $WORK/data $WORK/logs $WORK/checkpoints

# ── Step 2: Clone Verl v0.4.1 ──
echo "[2/7] Clone Verl v0.4.1"
rm -rf $WORK/verl
git clone --branch v0.4.1 --depth 1 https://github.com/verl-project/verl.git $WORK/verl

# ── Step 3: Install exact pinned versions ──
echo "[3/7] Install torch 2.6.0 + sglang 0.4.6.post5"
cd $WORK/verl
pip install --upgrade pip setuptools wheel
pip uninstall -y sglang sgl-kernel flashinfer-python verl 2>/dev/null || true

# Install the exact versions that v0.4.1 expects
pip install "torch==2.6.0" "tensordict<=0.6.2" "sglang[srt,openai]==0.4.6.post5" "torch-memory-saver>=0.0.5"

# Install flash-attn (hard requirement — dp_actor.py imports flash_attn.bert_padding)
pip install flash-attn --no-build-isolation

# Install verl itself
pip install --no-deps -e .
pip install setuptools

# Our extras
pip install fastapi uvicorn requests pyarrow tables h5py Levenshtein jmespath joblib scipy pyparsing tensorboard

# ── Step 4: Install libnuma ──
echo "[4/7] Install libnuma"
apt-get update -qq && apt-get install -y -qq libnuma-dev > /dev/null 2>&1 || true
ldconfig

# ── Step 5: Copy factor_tool + reward ──
echo "[5/7] Install factor_tool + reward"
cp /workspace/AlphaAgentEvo/deploy/v2/factor_tool.py $WORK/verl/verl/tools/factor_tool.py
mkdir -p $WORK/verl/examples/sglang_multiturn/config/tool_config
cp /workspace/AlphaAgentEvo/deploy/v2/factor_tool_config.yaml $WORK/verl/examples/sglang_multiturn/config/tool_config/factor_tool_config.yaml
mkdir -p $WORK/verl/verl/utils/reward_score
cp /workspace/AlphaAgentEvo/deploy/v2/factor_reward.py $WORK/verl/verl/utils/reward_score/factor.py

# ── Step 6: Copy data ──
echo "[6/7] Copy data"
cp /workspace/AlphaAgentEvo/deploy/v2/train.parquet $WORK/data/
cp /workspace/AlphaAgentEvo/deploy/v2/val.parquet $WORK/data/
cp /workspace/AlphaAgentEvo/deploy/v2/test.parquet $WORK/data/
ln -sf /workspace/AlphaAgentEvo/backtest $WORK/backtest
ln -sf /workspace/AlphaAgentEvo/expression_manager $WORK/expression_manager

# ── Step 7: Verify ──
echo "[7/7] Verify"
for f in $WORK/data/train.parquet $WORK/data/val.parquet $WORK/verl/verl/tools/factor_tool.py $WORK/verl/verl/tools/base_tool.py $WORK/verl/verl/utils/reward_score/factor.py; do
    [ -f "$f" ] && echo "  OK: $f" || { echo "  MISSING: $f"; exit 1; }
done
python -c "
import torch; print(f'torch {torch.__version__}')
import sglang; print(f'sglang {sglang.__version__}')
import tensordict; print(f'tensordict {tensordict.__version__}')
import verl; print(f'verl OK')
import ray; print(f'ray OK')
print('ALL OK')
"

echo ""
echo "============================================================"
echo "Setup complete. Next steps:"
echo "  1. Start API: nohup python /workspace/AlphaAgentEvo/deploy/api_server_verl.py > $WORK/logs/api.log 2>&1 &"
echo "  2. Run training: nohup bash /workspace/AlphaAgentEvo/deploy/v2/train.sh > $WORK/logs/train.log 2>&1 &"
echo "============================================================"
