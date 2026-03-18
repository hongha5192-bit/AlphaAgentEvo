#!/bin/bash
# Launch GRPO training
# Usage: bash start_training.sh [--smoke-test]

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "============================================"
echo "  AlphaAgentEvo GRPO Training"
echo "============================================"

# Activate conda environment if available
if command -v conda &> /dev/null; then
    eval "$(conda shell.bash hook)"
    conda activate alphaevo 2>/dev/null || echo "Warning: alphaevo env not found, using current env"
fi

# Check if backtest server is running
echo "Checking backtest server..."
if ! curl -s http://localhost:8001/health > /dev/null 2>&1; then
    echo "ERROR: Backtest server not running on port 8001!"
    echo "Please start it first: bash start_api.sh"
    exit 1
fi
echo "Backtest server: OK"

# Generate dataset if not exists
if [ ! -f "data/train.parquet" ]; then
    echo ""
    echo "Generating training dataset..."
    python training/generate_dataset.py --augment
fi

# Start training
echo ""
echo "Starting GRPO training..."
python training/train.py --config configs/grpo_config.yaml "$@"
