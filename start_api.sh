#!/bin/bash
# Launch the backtest API server
# Usage: bash start_api.sh

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "============================================"
echo "  AlphaAgentEvo Backtest Server"
echo "============================================"

# Activate conda environment if available
if command -v conda &> /dev/null; then
    eval "$(conda shell.bash hook)"
    conda activate alphaevo 2>/dev/null || echo "Warning: alphaevo env not found, using current env"
fi

echo "Starting FastAPI server on port 8001..."
python -m uvicorn backtest.api_server:app \
    --host 0.0.0.0 \
    --port 8001 \
    --workers 1 \
    --log-level info
