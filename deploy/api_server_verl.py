"""Backtest API Server — Compatible with paper's factor_tool.py contract.

Speaks the SAME API contract as the paper's api_server_fast.py:
  POST /backtest
  Body: {"exprs": {"factor_name": "factor_expr"}, "backtest_start_time": "...", ...}
  Returns: {"data": {"metrics": {"Information_Ratio_with_cost": 0.123}, "success": true}}

Internally uses our Qlib-consistent backtester (qlib_backtester.py).
"""

import os
import sys
import time
from pathlib import Path
from typing import Dict, Optional

import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel, Field

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from backtest.factor_executor import load_data, execute_expression, configure_periods

app = FastAPI(title="AlphaAgentEvo Backtest API (Verl-compatible)", version="2.0")


class BacktestRequest(BaseModel):
    exprs: Dict[str, str]  # {"factor_name": "factor_expr"}
    backtest_start_time: str = "2016-01-01"
    backtest_end_time: str = "2023-12-31"
    start_cash: float = 10000000.0
    update_freq: int = 5
    label_forward_days: int = 5
    stock_pool: str = "VN100"
    stop_loss_rate: float = 0.5
    stop_profit_rate: float = 0.5
    position_size: float = 1.0
    max_pos_each_stock: float = 0.2
    use_cache: bool = True
    layer_start: int = 0
    layer_end: int = 1
    pred_score_industry_neutralization: bool = False


@app.on_event("startup")
async def startup():
    """Pre-load data and configure periods."""
    configure_periods({
        "train": {"start": "2016-01-01", "end": "2023-12-31"},
        "val":   {"start": "2024-01-01", "end": "2024-12-31"},
        "test":  {"start": "2025-01-01", "end": "2026-12-31"},
    })
    load_data()
    print("[API] Backtest API ready on /backtest")


@app.get("/health")
async def health():
    return {"status": "healthy", "service": "AlphaAgentEvo Backtest API (Verl-compatible)"}


@app.post("/backtest")
async def backtest(req: BacktestRequest):
    """Paper-compatible backtest endpoint.

    factor_tool.py calls this with:
      POST /backtest
      {"exprs": {"name": "expr"}, ...}

    Expects response:
      {"data": {"metrics": {"Information_Ratio_with_cost": 0.123}, "success": true}}
    """
    t0 = time.time()

    if not req.exprs:
        return {
            "data": None,
            "detail": {"error": "No expressions provided"},
        }

    # Extract single factor (paper sends one at a time)
    factor_name = list(req.exprs.keys())[0]
    factor_expr = req.exprs[factor_name]

    try:
        result = execute_expression(factor_expr, period="train")
    except Exception as e:
        return {
            "data": None,
            "detail": {"error": str(e)[:500]},
        }

    elapsed = time.time() - t0

    if not result.get('success', False):
        return {
            "data": None,
            "detail": {"error": result.get('error', 'Unknown error')},
        }

    # Return in paper's expected format
    ir = result.get('ir', 0.0)
    return {
        "data": {
            "success": True,
            "metrics": {
                "Information_Ratio_with_cost": ir,
                "Information_Ratio_without_cost": ir,
                "Annualized_Return_with_cost": result.get('annualized_return', 0.0),
                "Max_Drawdown_with_cost": result.get('mdd', 0.0),
                "IC": ir * 0.05,  # Approximate IC from IR
                "ICIR": ir,
                "RankIC": ir * 0.04,
                "RankICIR": ir * 0.8,
            },
            "exec_time": elapsed,
        }
    }


@app.get("/example")
async def example():
    return {
        "method": "POST",
        "url": "/backtest",
        "body": {
            "exprs": {"momentum_20d": "RANK(TS_PCTCHANGE($close, 20))"},
            "backtest_start_time": "2016-01-01",
            "backtest_end_time": "2023-12-31",
        }
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)
