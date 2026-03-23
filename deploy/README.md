# AlphaAgentEvo — Runpod Deployment

## Quick Start

```bash
# 1. Rent Runpod pod (4-10 GPUs, A100 80GB recommended)
# 2. SSH in
# 3. Run setup
bash deploy/setup_runpod.sh

# 4. Upload data (from local machine)
scp daily_pv_v2.h5 root@<pod>:/root/AlphaAgentEvo/backtest/data/daily_pv.h5

# 5. Start training
bash deploy/run_training.sh
```

## What This Trains

- **Model**: Qwen3-4B-Thinking-2507 (full fine-tuning, not LoRA)
- **Algorithm**: GRPO with multi-turn tool calling (Verl framework)
- **Data**: Vietnam stock market (219 stocks, 2016-2026)
- **Seeds**: 300 train / 30 val / 99 test (duplicated from paper's architecture)
- **Backtest**: Qlib-consistent (top_k=10, T+2, 0.13% costs, VNINDEX benchmark)
- **Reward**: 5-component hierarchical (paper Eq.5)

## GPU Scaling

| GPUs | Batch Size | Est. Time | Est. Cost |
|------|-----------|-----------|-----------|
| 10× RTX 4090 | 20 (paper) | ~55 hrs | ~$440 |
| 8× A100 80GB | 16 | ~65 hrs | ~$520 |
| 4× A100 80GB | 8 | ~90 hrs | ~$360 |

## Key Config (from paper)

```
batch_size=20, rollouts=3, turns=2, lr=1e-6, KL=0.001
save_freq=10, total_steps=150, use step 80 for evaluation
```

## Files

```
deploy/
├── setup_runpod.sh     ← Install & configure everything
├── run_training.sh     ← Launch Verl GRPO training
└── README.md           ← This file

data/
├── train.parquet       ← 300 seeds with prompts (GRPO training)
├── val.parquet         ← 30 seeds (validation during training)
├── test.parquet        ← 99 seeds (evaluation after training)
├── vn_seeds_300.jsonl  ← Raw seed expressions
├── val_seeds.jsonl     ← Raw val seeds
├── test_seeds.jsonl    ← Raw test seeds
└── seed_backtest_results.json ← IR for all 300 train seeds

backtest/
├── qlib_backtester.py  ← Qlib-consistent portfolio backtester
├── factor_executor.py  ← Expression parser + executor
├── api_server.py       ← FastAPI wrapper (port 8001)
└── data/daily_pv.h5    ← Market data (symlink, need to upload)
```
