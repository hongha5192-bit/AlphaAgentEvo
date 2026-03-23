# AlphaAgentEvo — Training Instructions for Phu

## Overview

Everything is ready to train Qwen3-4B on Vietnam stock data using the paper's exact GRPO approach (Verl framework). All 7 critical issues vs paper's codebase have been fixed and verified.

## What's Included

```
AlphaAgentEvo/
├── deploy/
│   ├── setup_runpod.sh          ← Run this first (installs everything)
│   ├── run_training.sh          ← Run this to start training
│   ├── api_server_verl.py       ← Backtest API (paper's contract)
│   └── factor_tool_vn.py        ← Patched for VN dates
├── data/
│   ├── train.parquet            ← 300 seeds with prompts
│   ├── val.parquet              ← 30 seeds
│   ├── test.parquet             ← 99 seeds
│   ├── vn_seeds_300.jsonl       ← Raw seed expressions
│   ├── seed_backtest_results.json ← IR for all 300 seeds
│   ├── daily_pv_v2.h5 (symlink) ← Market data (UPLOAD SEPARATELY)
│   └── sector_map.json          ← 476 tickers → 19 sectors
├── backtest/
│   ├── qlib_backtester.py       ← Qlib-consistent backtester
│   └── factor_executor.py       ← Expression executor
└── expression_manager/
    └── function_lib.py          ← All operators (SLOPE, ATR, INDUSTRY_NEUTRALIZE, etc.)
```

## Step-by-Step

### 1. Setup Server

Rent a server with **4-10 GPUs** (A100 80GB recommended).

```bash
# Clone repo
git clone https://github.com/hongha5192-bit/AlphaAgentEvo.git /root/AlphaAgentEvo
cd /root/AlphaAgentEvo

# Run setup (installs Verl, downloads model, starts API)
bash deploy/setup_runpod.sh
```

### 2. Upload Data

The market data file (9.6MB) is not in the repo. Upload it:

```bash
# From your local machine:
scp daily_pv_v2.h5 root@<server>:/root/AlphaAgentEvo/backtest/data/daily_pv.h5
```

This file contains: 219 VN stocks, 2016-2026, 12 columns:
`$open, $close, $high, $low, $volume, $net_foreign_val, $net_foreign_vol, $return, $bench_close, $bench_return, $amount, $industry`

### 3. Start Training

```bash
cd /root/AlphaAgentEvo
bash deploy/run_training.sh
```

Training will:
- Use paper's modified Verl framework (not stock volcengine/verl)
- Run GRPO with `batch_size=20, rollouts=3, turns=2` (matching paper)
- Save checkpoints every 10 steps
- Total: 150 steps
- Use step 80 checkpoint for final evaluation (paper's recommendation)

### 4. Monitor

```bash
# Training log
tail -f /tmp/train.log

# API log
tail -f /tmp/api.log

# GPU usage
watch nvidia-smi
```

## Training Config (matches paper's run_qwen3-4b_instruct_alphaevo_multiturn.sh)

| Parameter | Value | Source |
|-----------|-------|--------|
| `data.train_batch_size` | **20** | Paper codebase |
| `actor_rollout_ref.rollout.n` | **3** | Paper codebase |
| `actor_rollout_ref.rollout.multi_turn.max_assistant_turns` | **2** | Paper codebase |
| `actor_rollout_ref.actor.ppo_mini_batch_size` | **20** | Paper codebase |
| `actor_rollout_ref.actor.optim.lr` | **1e-6** | Paper codebase |
| `actor_rollout_ref.actor.kl_loss_coef` | **0.001** | Paper codebase |
| `trainer.total_training_steps` | **150** | Paper codebase |
| `trainer.save_freq` | **10** | Changed (paper: 10) |
| `actor_rollout_ref.rollout.gpu_memory_utilization` | **0.6** | Paper codebase |
| Mixed precision | **bf16** | Paper codebase |
| Rollout engine | **sglang** | Paper codebase |

## Backtest API

The API speaks the **exact same contract** as the paper's `api_server_fast.py`:

```
POST http://localhost:8001/backtest
Body: {"exprs": {"factor_name": "factor_expr"}, ...}
Returns: {"data": {"metrics": {"Information_Ratio_with_cost": 0.123}}}
```

Internally uses our Qlib-consistent backtester:
- top_k=10, n_drop=2, T+2 settlement
- Buy at OPEN@T, Sell at CLOSE@T+2
- Transaction costs: 0.13% buy + 0.13% sell
- Benchmark: VNINDEX excess returns
- IR formula: Qlib `mean/std * sqrt(252)` (verified exact match)

## Seeds

- **300 train**: 270 converted from paper + 30 VN foreign flow seeds
- **30 val**: 18 converted + 12 VN-specific
- **99 test**: 60 converted + 39 VN-specific
- All 429 seeds backtested successfully
- 67% conditional (paper: 71%), matching paper's 3-layer architecture

## GPU Requirements

| GPUs | VRAM | batch=20 possible? | Est. Time | Est. Cost |
|------|------|---------------------|-----------|-----------|
| 10× RTX 4090 | 240GB | ✅ Yes | ~55 hrs | ~$440 |
| 8× A100 80GB | 640GB | ✅ Yes | ~65 hrs | ~$520 |
| 4× A100 80GB | 320GB | ✅ Yes | ~90 hrs | ~$360 |
| 2× A100 80GB | 160GB | ⚠️ May need batch=10 | ~120 hrs | ~$480 |

**Minimum: 4× A100 80GB** to run batch_size=20 comfortably.

## After Training

1. Use **step 80 checkpoint** for evaluation (paper's recommendation, not step 150)
2. Evaluate on test set (99 seeds) with the trained model
3. Compare pass@3, pass@5, VR metrics vs baselines

## Known Issues

- If OOM: reduce `gpu_memory_utilization` from 0.6 to 0.5
- If NCCL timeout: increase `NCCL_TIMEOUT` and `TORCH_DISTRIBUTED_TIMEOUT`
- If API slow: the first request takes ~5s (data loading), subsequent ~1s
- Checkpoint at step 80 is recommended by paper (not final step 150)

## Contact

Questions → Ha (hongha5192-bit)
