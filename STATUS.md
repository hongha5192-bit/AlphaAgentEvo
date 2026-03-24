# AlphaAgentEvo — Project Status (2026-03-24)

## What We're Building

**Goal**: Train an LLM (Qwen3-4B) to autonomously discover profitable stock trading signals (alpha factors) for the Vietnam stock market, using reinforcement learning.

**Paper**: AlphaAgentEvo — uses GRPO (Group Relative Policy Optimization) with multi-turn tool calling. The model generates alpha factor expressions, calls a backtest tool to evaluate them, sees the results, and iterates to improve — learning from reward signals.

**Why it matters**: Traditional quant alpha discovery requires human experts. This system automates it — the model learns to write, test, and refine alpha factors through trial-and-error, like a junior quant analyst with unlimited patience.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    GRPO Training Loop                       │
│                                                             │
│  ┌──────────┐    ┌──────────┐    ┌──────────────────────┐  │
│  │  Qwen3   │───>│  sglang  │───>│  Tool Execution      │  │
│  │  4B      │    │  rollout  │    │  (multi-turn)        │  │
│  │  (FSDP)  │<───│  engine   │<───│                      │  │
│  └──────────┘    └──────────┘    │  1. Model generates   │  │
│       │                          │     factor expression  │  │
│       │ gradient                 │  2. Calls backtest API │  │
│       │ update                   │  3. Gets IR result     │  │
│       │                          │  4. Refines or submits │  │
│       v                          └──────────────────────┘  │
│  ┌──────────┐                              │               │
│  │  GRPO    │    ┌──────────────┐          │               │
│  │  reward  │<───│  Hierarchical│<─────────┘               │
│  │  update  │    │  Reward      │                          │
│  └──────────┘    │  (Paper Eq.5)│                          │
│                  └──────────────┘                          │
└─────────────────────────────────────────────────────────────┘
                           │
                           v
              ┌──────────────────────┐
              │  Backtest API        │
              │  (FastAPI, port 8002)│
              │                      │
              │  - Loads daily_pv.h5 │
              │  - Executes alpha    │
              │  - Returns IR        │
              │  - Qlib-consistent   │
              └──────────────────────┘
```

### Key Components

| Component | Description | Location |
|-----------|-------------|----------|
| **Verl** | Distributed RL training framework (Ray + FSDP + sglang) | `/workspace/v2/verl` |
| **sglang** | LLM inference engine with multi-turn tool calling support | pip package |
| **Qwen3-4B** | Base LLM being fine-tuned via GRPO | `/workspace/models/Qwen3-4B-Thinking-2507` |
| **Backtest API** | FastAPI server that evaluates alpha factor expressions | `deploy/api_server_verl.py` |
| **Factor Tool** | Verl BaseTool that connects model to backtest API | `deploy/v2/factor_tool.py` |
| **Reward Function** | Hierarchical reward (Eq.5) for RL training | `deploy/v2/factor_reward.py` |
| **Seeds** | 300 train + 30 val + 99 test expert-curated alpha factors | `deploy/v2/*.parquet` |

### Training Config (matching paper)

| Parameter | Value |
|-----------|-------|
| GPUs | 4x H100 80GB |
| Batch size | 20 seeds/step |
| Rollouts per seed | 3 |
| Max assistant turns | 2 (multi-turn) |
| Total steps | 150 |
| Checkpoint for eval | Step 80 |
| Learning rate | 1e-6 |
| KL beta | 0.001 |

---

## Infrastructure

### Runpod Pod
- **Pod ID**: `jwuvcy7bj9mepi`
- **GPUs**: 4x NVIDIA H100 80GB HBM3
- **Storage**: Network volume at `/workspace` (persists across restarts)
- **SSH**: `jwuvcy7bj9mepi-64411ff4@ssh.runpod.io` with key `~/Ha/.ssh/id_ed25519`
- **Limitation**: Web terminal cannot handle multi-line commands

### Target Software Stack (v0.4.1 — stable)

| Package | Version | Source of truth |
|---------|---------|-----------------|
| Python | 3.10 | conda env `verl041` |
| Verl | v0.4.1 | `git clone --branch v0.4.1` |
| sglang | 0.4.6.post5 | v0.4.1 `setup.py` |
| torch | 2.6.0 | v0.4.1 `setup.py` |
| tensordict | <=0.6.2 | v0.4.1 `setup.py` |

**Why v0.4.1?** Its release notes explicitly include sglang multi-turn rollout, the interaction system, and a fix for "tool call parser not found" — all for sglang 0.4.6.post5. All three packaging files (setup.py, requirements_sglang.txt, docs) agree on versions.

---

## What's Working

| Item | Status |
|------|--------|
| 300 VN seed alpha factors (train/val/test) | Done |
| Qlib-consistent backtester (top_k=10, T+2, 0.13% costs) | Done |
| Backtest API (FastAPI, port 8002) | Done |
| Factor tool (BaseTool interface) | Done |
| Hierarchical reward function (Paper Eq.5) | Done |
| Data file (daily_pv.h5, 219 stocks, 12 columns) | Done |
| Training config matching paper | Done |
| All files committed to GitHub | Done |

### Previous Local Training (trl, RTX 5090)
- LoRA r=64 on single RTX 5090
- Reached step 65/150, rewards ~0.22 (paper target: 0.38)
- Proved the concept works but LoRA limited capacity

---

## Problems Encountered & Resolved

### Phase 1: Paper's Fork (v1) — Abandoned
- Paper's modified Verl fork had API mismatches with public sglang
- vllm rollout engine doesn't support multi-turn tool calls (generates `<tool_call>` text but never executes)
- **Result**: Zero rewards, zero gradients, model learns nothing
- **Decision**: Pivot to official Verl release

### Phase 2: Official Verl (v2) — Dependency Hell

| # | Problem | Root Cause | Fix |
|---|---------|-----------|-----|
| 1 | `libnuma.so.1` not found | System package, lost on container restart | Auto-install in train.sh |
| 2 | `flash_attn` undefined symbol | Pre-built wheel ABI mismatch with torch | Uninstall (SDPA fallback) |
| 3 | Port mismatch (8001 vs 8002) | API and callers disagreed | Fixed API to 8002 |
| 4 | `factor_reward.py` not copied | setup.sh missed it | Added cp command |
| 5 | `tool_config/` dir missing | mkdir before cp | Added mkdir -p |
| 6 | `daily_pv.h5` broken symlink | Local-only symlink in git | Replaced with real file |
| 7 | API not running after restart | Container restart kills processes | Start API before training |
| 8 | `ContinueGenerationReqInput` missing | Verl HEAD needs newer sglang than installed | Pin to v0.4.1 |
| 9 | Verl install script inconsistent | setup.py says 0.5.8, script installs 0.5.2 | Use setup.py, skip script |
| 10 | Disk quota exceeded | Accumulated failed installs | Clean + fresh env |
| 11 | Wrong Python (3.11 vs 3.10) | pip installed to system Python | Ensure conda env active |
| 12 | flash_attn wheel wrong Python | cp312 wheel on cp310 env | Skip flash_attn |
| 13 | Conda TOS not accepted | New conda version requires TOS | Auto-accept in setup.sh |

### Root Cause of All Phase 2 Issues

**Building a complex ML stack from scratch on a bare pod.** Every version mismatch cascades. Every container restart loses system packages. Every `pip install` risks polluting the wrong Python.

---

## Current Status (2026-03-24)

**Setting up fresh environment with Verl v0.4.1**

Setup script (`setup.sh`) running on restarted pod with:
- Fresh conda env `verl041` (Python 3.10)
- Verl v0.4.1 (pinned tag, not HEAD)
- sglang 0.4.6.post5 (from setup.py)
- torch 2.6.0 (from setup.py)

### What Happens Next

1. **Setup completes** → verify "ALL OK"
2. **Start API** → `nohup python deploy/api_server_verl.py > logs/api.log 2>&1 &`
3. **Start training** → `nohup bash deploy/v2/train.sh > logs/train.log 2>&1 &`
4. **Monitor** → check `tail -20 logs/train.log` for `step` and `reward`
5. **Critical test** → are rewards non-zero? (proves multi-turn tool calling works)
6. **Train 150 steps** → ~5 hours at ~2 min/step on 4x H100
7. **Evaluate step 80 checkpoint** on val/test sets
8. **Transfer model** back to local machine

### Success Criteria
- `critic/score/mean` shows non-zero values (tool calls executing)
- Reward climbs from ~0.16 to ~0.38 over 150 steps (matching paper Fig.7)
- Step 80 checkpoint achieves VR > 0.95, Pass@3 > 0.90

---

## Lesson Learned

> **For production ML training on cloud GPUs, use the project's official Docker image — not a hand-built env.**
>
> Verl provides `verlai/verl:sgl055.latest` with everything pre-tested. We spent hours debugging dependency mismatches that wouldn't exist with the Docker image.
>
> If you must build from source, **always pin to a release tag** and install from `setup.py`, not the helper scripts.

---

## File Structure

```
AlphaAgentEvo/
├── deploy/
│   ├── api_server_verl.py      # Backtest API (port 8002)
│   └── v2/
│       ├── setup.sh            # One-command pod setup
│       ├── train.sh            # Training launch script
│       ├── factor_tool.py      # BaseTool for Verl
│       ├── factor_tool_config.yaml
│       ├── factor_reward.py    # compute_score for Verl
│       ├── train.parquet       # 300 seed prompts
│       ├── val.parquet         # 30 val prompts
│       └── test.parquet        # 99 test prompts
├── backtest/
│   ├── factor_executor.py      # Expression parser + executor
│   ├── qlib_backtester.py      # Qlib-consistent portfolio sim
│   └── data/
│       └── daily_pv.h5         # VN market data (219 stocks)
└── expression_manager/
    └── function_lib.py         # All operators (RANK, TS_MEAN, etc.)
```
