#!/bin/bash
# AlphaAgentEvo v2 — Clean Setup on Official Verl v0.6.1
# Uses verl v0.6.1 + sglang 0.5.5 (from setup.py, NOT install script)
set -e

echo "============================================================"
echo "AlphaAgentEvo v2 — Verl v0.6.1 + sglang 0.5.5 Setup"
echo "============================================================"

WORK=/workspace/v2
mkdir -p $WORK/data $WORK/logs $WORK/checkpoints

# ── Step 1: Conda env ──
echo "[1/7] Python 3.10 env"
eval "$(/workspace/miniconda/bin/conda shell.bash hook)"
conda activate verl310 || { echo "Create env first: conda create -n verl310 python=3.10 -y"; exit 1; }
echo "Python: $(python --version)"

# ── Step 2: Clone Verl v0.6.1 (pinned release, not HEAD) ──
echo "[2/7] Clone Verl v0.6.1"
if [ ! -d "$WORK/verl" ]; then
    git clone --branch v0.6.1 --depth 1 https://github.com/verl-project/verl.git $WORK/verl
fi

# ── Step 3: Install Verl + sglang (from setup.py, not install script) ──
echo "[3/7] Install Verl + sglang 0.5.5"
cd $WORK/verl
# Install non-sglang deps from script (skip sglang's stale 0.5.2 pin)
USE_MEGATRON=0 USE_SGLANG=0 bash scripts/install_vllm_sglang_mcore.sh
# Install verl + correct sglang from setup.py (pins sglang==0.5.5)
pip install -e ".[sglang]"
pip install fastapi uvicorn requests pyarrow tables h5py Levenshtein jmespath joblib scipy pyparsing tensorboard

# ── Step 4: Install libnuma (needed by sgl_kernel) ──
echo "[4/7] Install libnuma"
apt-get update -qq && apt-get install -y -qq libnuma-dev > /dev/null 2>&1 || true
ldconfig

# ── Step 5: Copy our factor_tool + reward into Verl ──
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
for f in $WORK/data/train.parquet $WORK/data/val.parquet $WORK/verl/verl/tools/factor_tool.py $WORK/verl/verl/tools/base_tool.py $WORK/verl/verl/utils/reward_score/factor.py $WORK/verl/examples/sglang_multiturn/config/tool_config/factor_tool_config.yaml; do
    [ -f "$f" ] && echo "  OK: $f" || { echo "  MISSING: $f"; exit 1; }
done
python -c "
import sglang; print(f'sglang {sglang.__version__} OK')
from sglang.srt.managers.io_struct import ContinueGenerationReqInput; print('ContinueGenerationReqInput OK')
import verl; print(f'verl OK')
import ray; print(f'ray OK')
from sgl_kernel import common_ops; print(f'sgl_kernel OK (libnuma working)')
print('ALL OK')
"

echo ""
echo "============================================================"
echo "Setup complete. Next steps:"
echo "  1. Start API: nohup python /workspace/AlphaAgentEvo/deploy/api_server_verl.py > $WORK/logs/api.log 2>&1 &"
echo "  2. Run training: nohup bash /workspace/AlphaAgentEvo/deploy/v2/train.sh > $WORK/logs/train.log 2>&1 &"
echo "============================================================"
