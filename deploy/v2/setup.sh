#!/bin/bash
# AlphaAgentEvo v2 — Verl v0.4.1 + sglang 0.4.6.post5 + torch 2.6.0
set -Eeuo pipefail

trap 'echo "[ERROR] setup.sh failed at line $LINENO: $BASH_COMMAND" >&2' ERR

log() { echo "[$(date '+%F %T')] $*"; }

WORK=/workspace/v2
ENV_NAME=verl041
VERL_DIR=$WORK/verl
SRC=/workspace/AlphaAgentEvo/deploy/v2
BACKTEST_SRC=/workspace/AlphaAgentEvo/backtest
EXPR_SRC=/workspace/AlphaAgentEvo/expression_manager

log "============================================================"
log "AlphaAgentEvo v2 — Verl v0.4.1 (stable multi-turn)"
log "============================================================"

# Step 1: Fresh conda env
log "[1/8] Create fresh conda env"
eval "$('/workspace/miniconda/bin/conda' shell.bash hook)"
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main >/dev/null 2>&1 || true
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r >/dev/null 2>&1 || true
conda deactivate >/dev/null 2>&1 || true
conda env remove -n "$ENV_NAME" -y >/dev/null 2>&1 || true
conda create -n "$ENV_NAME" python=3.10 -y
conda activate "$ENV_NAME"
log "Python: $(python --version 2>&1)"
log "Env: $CONDA_DEFAULT_ENV"

mkdir -p "$WORK/data" "$WORK/logs" "$WORK/checkpoints"

# Step 2: Clone pinned Verl tag
log "[2/8] Clone Verl v0.4.1"
rm -rf "$VERL_DIR"
git clone --branch v0.4.1 --depth 1 https://github.com/verl-project/verl.git "$VERL_DIR"
git -C "$VERL_DIR" describe --tags --always | tee "$WORK/verl.tag"

# Step 3: Install exact pinned versions
log "[3/8] Install torch 2.6.0 + sglang 0.4.6.post5"
cd "$VERL_DIR"
pip install --upgrade pip setuptools wheel
pip uninstall -y sglang sgl-kernel flashinfer-python verl >/dev/null 2>&1 || true
pip install "torch==2.6.0" "tensordict<=0.6.2" "sglang[srt,openai]==0.4.6.post5" "torch-memory-saver>=0.0.5"
# Install verl (--no-deps to prevent torch version drift!)
pip install --no-deps -e .
pip install setuptools

# Patch flash_attn imports (ABI mismatch on H100 with torch 2.6.0)
# flash_attn.bert_padding is only used for use_remove_padding=True
# We disable it and use use_remove_padding=False in train.sh instead
for f in verl/workers/actor/dp_actor.py verl/workers/fsdp_workers.py verl/workers/critic/dp_critic.py; do
    python -c "
lines = open('$f').readlines()
for i, line in enumerate(lines):
    if 'from flash_attn.bert_padding import' in line and 'if is_cuda_available' in (lines[i-1] if i > 0 else ''):
        lines[i] = '    pass  # flash_attn removed (ABI issue on H100)\n'
open('$f','w').writelines(lines)
"
done

# Clear stale .pyc files
find . -name "*.pyc" -delete
pip install fastapi uvicorn requests pyarrow tables h5py Levenshtein jmespath joblib scipy pyparsing tensorboard

# Step 4: Install libnuma
log "[4/8] Install libnuma"
apt-get update -qq
apt-get install -y -qq libnuma-dev >/dev/null 2>&1
ldconfig
ldconfig -p | grep -q 'libnuma.so.1'

# Step 5: Copy factor_tool + reward
log "[5/8] Install factor_tool + reward"
install -D "$SRC/factor_tool.py" "$VERL_DIR/verl/tools/factor_tool.py"
install -D "$SRC/factor_tool_config.yaml" "$VERL_DIR/examples/sglang_multiturn/config/tool_config/factor_tool_config.yaml"
install -D "$SRC/factor_reward.py" "$VERL_DIR/verl/utils/reward_score/factor.py"

# Step 6: Copy data + links
log "[6/8] Copy data"
install -D "$SRC/train.parquet" "$WORK/data/train.parquet"
install -D "$SRC/val.parquet" "$WORK/data/val.parquet"
install -D "$SRC/test.parquet" "$WORK/data/test.parquet"
ln -sfn "$BACKTEST_SRC" "$WORK/backtest"
ln -sfn "$EXPR_SRC" "$WORK/expression_manager"

# Step 7: Hard verification
log "[7/8] Verify files + imports + GPU"
for f in \
  "$WORK/data/train.parquet" \
  "$WORK/data/val.parquet" \
  "$VERL_DIR/verl/tools/factor_tool.py" \
  "$VERL_DIR/verl/tools/base_tool.py" \
  "$VERL_DIR/verl/utils/reward_score/factor.py"; do
  [[ -f "$f" ]] && log "OK: $f" || { echo "MISSING: $f" >&2; exit 1; }
done

python - <<PY
import os, sys
import torch, sglang, tensordict, verl, ray
print('python', sys.executable)
print('torch', torch.__version__)
print('sglang', sglang.__version__)
print('tensordict', tensordict.__version__)
print('verl', verl.__file__)
print('ray', ray.__version__)
print('cuda_available', torch.cuda.is_available())
print('gpu_count', torch.cuda.device_count())
assert torch.__version__.startswith('2.6.0'), f'Expected torch 2.6.0, got {torch.__version__}'
assert sglang.__version__ == '0.4.6.post5', f'Expected sglang 0.4.6.post5, got {sglang.__version__}'
assert 'v2/verl' in verl.__file__, f'Unexpected verl path: {verl.__file__}'
assert torch.cuda.is_available(), 'CUDA not available'
assert torch.cuda.device_count() > 0, 'No GPUs visible'
print('ALL OK')
PY

# Step 8: Freeze environment
log "[8/8] Freeze environment"
pip freeze | sort > "$WORK/requirements.lock.txt"
conda env export > "$WORK/conda.$ENV_NAME.lock.yml"

log ""
log "============================================================"
log "Setup complete. Next steps:"
log "  1. Start API: nohup python /workspace/AlphaAgentEvo/deploy/api_server_verl.py > $WORK/logs/api.log 2>&1 &"
log "  2. Run training: nohup bash /workspace/AlphaAgentEvo/deploy/v2/train.sh > $WORK/logs/train.log 2>&1 &"
log "============================================================"
