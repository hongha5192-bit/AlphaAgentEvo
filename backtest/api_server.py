"""FastAPI backtest server for factor expression evaluation.

Endpoints:
    POST /evaluate_factor — parse expression, execute, compute IC/IR
    GET  /health          — health check

Supports temporal splits: pass ?period=train to compute IC only on
the training date range (factor values still use full history for lookback).
"""

import os
import sys
from pathlib import Path
from typing import Optional

import yaml
from fastapi import FastAPI
from pydantic import BaseModel

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from backtest.factor_executor import (
    execute_expression,
    load_data,
    configure_periods,
)

app = FastAPI(title="AlphaAgentEvo Backtest Server", version="1.0")

# Default period used for /evaluate_factor when no ?period= is passed
_default_period: str | None = None


class EvaluateRequest(BaseModel):
    factor_name: Optional[str] = "unnamed"
    factor_expr: str
    period: Optional[str] = None  # "train", "val", "test", or None for all


class EvaluateResponse(BaseModel):
    success: bool
    score: float  # IR
    ic_mean: float
    ic_std: float
    ir: float
    error: Optional[str] = None
    exec_time: float


@app.on_event("startup")
async def startup():
    """Pre-load data and configure temporal periods from config."""
    global _default_period

    # Load config to get period definitions
    config_path = os.environ.get(
        "ALPHAEVO_CONFIG",
        str(PROJECT_ROOT / "configs" / "grpo_config.yaml"),
    )
    try:
        with open(config_path) as f:
            config = yaml.safe_load(f)
        periods = config.get("backtest", {}).get("periods", {})
        if periods:
            configure_periods(periods)
        _default_period = config.get("backtest", {}).get("default_period", None)
        if _default_period:
            print(f"[API] Default evaluation period: {_default_period}")
    except Exception as e:
        print(f"[API] Warning: could not load config for periods: {e}")

    load_data()


@app.get("/health")
async def health():
    return {"status": "ok", "service": "alphaagentevo-backtest"}


@app.post("/evaluate_factor", response_model=EvaluateResponse)
async def evaluate_factor(req: EvaluateRequest):
    """Evaluate a factor expression and return IC/IR metrics."""
    period = req.period or _default_period
    result = execute_expression(req.factor_expr, period=period)
    return EvaluateResponse(**result)


@app.post("/batch_evaluate")
async def batch_evaluate(expressions: list[EvaluateRequest]):
    """Evaluate multiple factor expressions."""
    results = []
    for req in expressions:
        period = req.period or _default_period
        result = execute_expression(req.factor_expr, period=period)
        result["factor_name"] = req.factor_name
        results.append(result)
    return results


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)
