#!/bin/bash
# AlphaAgentEvo v2 — Clean Setup on Official Verl
# Uses official Verl release with sglang multi-turn tool calling
# NOT the paper's fork
set -e

echo "============================================================"
echo "AlphaAgentEvo v2 — Clean Official Verl Setup"
echo "============================================================"

WORK=/workspace/v2
mkdir -p $WORK/data $WORK/logs $WORK/checkpoints

# ── Step 1: Conda env ──
echo "[1/6] Python 3.10 env"
eval "$(/workspace/miniconda/bin/conda shell.bash hook)"
conda activate verl310 || { echo "Create env first: conda create -n verl310 python=3.10 -y"; exit 1; }
echo "Python: $(python --version)"

# ── Step 2: Clone official Verl ──
echo "[2/6] Clone official Verl"
if [ ! -d "$WORK/verl" ]; then
    git clone --depth 1 https://github.com/volcengine/verl.git $WORK/verl
fi

# ── Step 3: Install with correct sglang ──
echo "[3/6] Install Verl + sglang"
cd $WORK/verl
# Use official install script but pin sglang to post5
sed -i 's/sglang\[all\]==0.4.6.post1/sglang[all]==0.4.6.post5/' scripts/install_vllm_sglang_mcore.sh 2>/dev/null || true
USE_MEGATRON=0 bash scripts/install_vllm_sglang_mcore.sh
pip install --no-deps -e .
pip install fastapi uvicorn requests pyarrow tables h5py Levenshtein jmespath joblib scipy pyparsing tensorboard

# ── Step 4: Copy our factor_tool + reward into Verl ──
echo "[4/6] Install factor_tool + reward"
cp /workspace/AlphaAgentEvo/deploy/v2/factor_tool.py $WORK/verl/verl/tools/factor_tool.py
mkdir -p $WORK/verl/examples/sglang_multiturn/config/tool_config
cp /workspace/AlphaAgentEvo/deploy/v2/factor_tool_config.yaml $WORK/verl/examples/sglang_multiturn/config/tool_config/factor_tool_config.yaml
mkdir -p $WORK/verl/verl/utils/reward_score
cp /workspace/AlphaAgentEvo/deploy/v2/factor_reward.py $WORK/verl/verl/utils/reward_score/factor.py

# ── Step 5: Copy data ──
echo "[5/6] Copy data"
cp /workspace/AlphaAgentEvo/deploy/v2/train.parquet $WORK/data/
cp /workspace/AlphaAgentEvo/deploy/v2/val.parquet $WORK/data/
cp /workspace/AlphaAgentEvo/deploy/v2/test.parquet $WORK/data/
ln -sf /workspace/AlphaAgentEvo/backtest $WORK/backtest
ln -sf /workspace/AlphaAgentEvo/expression_manager $WORK/expression_manager

# ── Step 6: Install libnuma (needed by sgl_kernel) ──
echo "[6/7] Install libnuma"
apt-get update -qq && apt-get install -y -qq libnuma-dev > /dev/null 2>&1 || true
ldconfig

# ── Step 7: Verify ──
echo "[7/7] Verify"
# Check critical files exist
for f in $WORK/data/train.parquet $WORK/data/val.parquet $WORK/verl/verl/tools/factor_tool.py $WORK/verl/verl/tools/base_tool.py $WORK/verl/verl/utils/reward_score/factor.py $WORK/verl/examples/sglang_multiturn/config/tool_config/factor_tool_config.yaml; do
    [ -f "$f" ] && echo "  OK: $f" || { echo "  MISSING: $f"; exit 1; }
done
python -c "
import sglang; print(f'sglang OK')
import verl; print(f'verl OK')
import flash_attn; print(f'flash_attn OK')
import ray; print(f'ray OK')
from sgl_kernel import common_ops; print(f'sgl_kernel OK (libnuma working)')
print('ALL OK')
"

echo ""
echo "============================================================"
echo "Setup complete. Next steps:"
echo "  1. Start API: cd $WORK && python /workspace/AlphaAgentEvo/deploy/api_server_verl.py &"
echo "  2. Test GSM8K example first (optional)"
echo "  3. Run training: bash /workspace/AlphaAgentEvo/deploy/v2/train.sh"
echo "============================================================"
