"""FactorTool — BaseTool implementation for AlphaAgentEvo.

Follows the official Verl BaseTool interface (same as gsm8k_tool.py):
  - create(): init instance with seed factor info
  - execute(): call backtest API, compute hierarchical reward
  - calc_reward(): return final trajectory reward
  - release(): cleanup

The tool calls the backtest API at http://localhost:8002/backtest
using the paper's contract:
  POST {"exprs": {"name": "expr"}, ...}
  Returns {"data": {"metrics": {"Information_Ratio_with_cost": 0.123}}}
"""

import logging
import math
import os
from typing import Any, Optional
from uuid import uuid4

import numpy as np
import requests

from verl.utils.rollout_trace import rollout_trace_op
from .base_tool import BaseTool
from .schemas import OpenAIFunctionToolSchema

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))

# Reward coefficients (paper Appendix D)
ALPHA_SUCC = 0.1
ALPHA_FAIL = 0.2
ALPHA_CONS = 0.02
ALPHA_EXP = 0.02
ALPHA_PERF = 0.1
ALPHA_STREAK = 0.15

# Reward caps (paper Appendix D)
CAP_TOOL = 1.0
CAP_CONS = 0.2
CAP_EXPL = 0.3
CAP_PERF = 0.5
CAP_STREAK = 0.6

R_TOOL_FLOOR = 0.01


class FactorTool(BaseTool):
    """Tool for evaluating alpha factors via backtesting.

    Each instance tracks one evolution trajectory:
    - Seed factor (initial expression + IR)
    - All proposed variations and their IRs
    - Progressive improvement streak
    - Hierarchical reward (paper Eq.5)
    """

    def __init__(self, config: dict, tool_schema: OpenAIFunctionToolSchema):
        super().__init__(config, tool_schema)
        self._instance_dict = {}
        self.backtest_api_url = config.get("backtest_api_url", "http://localhost:8002/backtest")
        logger.info(f"Initializing FactorTool with name: {self.tool_schema.function.name}")
        logger.info(f"Tool config: {config}")

    def get_openai_tool_schema(self) -> OpenAIFunctionToolSchema:
        return self.tool_schema

    async def create(self, instance_id: Optional[str] = None, **kwargs) -> str:
        if instance_id is None:
            instance_id = str(uuid4())
        self._instance_dict[instance_id] = {
            "factor_name": "",
            "factor_expr": "",
            "metric": "Information_Ratio_with_cost",
            "init_metric": kwargs.get("init_metric", 0),
            "best_metric": kwargs.get("init_metric", 0),
            "init_factor_expr": kwargs.get("init_factor_expr", ""),
            "backtest_result": None,
            "reward": 0.0,
            "streak": 0,
            "tool_call_count": 0,
            "succ_tried_factors": [],
            "failed_count": 0,
        }
        return instance_id

    @rollout_trace_op
    async def execute(self, instance_id: str, parameters: dict[str, Any], **kwargs) -> tuple[str, float, dict]:
        factor_name = parameters.get("factor_name", "")
        factor_expr = parameters.get("factor_expr", "")
        metric = parameters.get("metric", "Information_Ratio_with_cost")

        self._instance_dict[instance_id]["factor_name"] = factor_name
        self._instance_dict[instance_id]["factor_expr"] = factor_expr
        self._instance_dict[instance_id]["metric"] = metric
        self._instance_dict[instance_id]["tool_call_count"] += 1

        metric_value, status = await self._call_backtest_api(instance_id)

        if status == "success":
            self._instance_dict[instance_id]["succ_tried_factors"].append({
                "factor_expr": factor_expr,
                "metric_value": metric_value,
            })
            if metric_value > self._instance_dict[instance_id]["best_metric"]:
                self._instance_dict[instance_id]["streak"] += 1
                self._instance_dict[instance_id]["best_metric"] = metric_value

            reward = await self.calc_reward(instance_id)
            return (
                f'success: Evaluated factor "{factor_name}" with expression "{factor_expr}", {metric}={metric_value}',
                reward,
                {self.tool_schema.function.name: reward},
            )
        else:
            self._instance_dict[instance_id]["failed_count"] += 1
            reward = await self.calc_reward(instance_id)
            return (
                f"failed: factor {factor_name} with expression {factor_expr}. Reason: {status}",
                reward,
                {self.tool_schema.function.name: reward},
            )

    async def _call_backtest_api(self, instance_id: str) -> tuple[float, str]:
        instance = self._instance_dict[instance_id]
        factor_name = instance["factor_name"]
        factor_expr = instance["factor_expr"]
        metric = instance["metric"]

        test_request = {
            "exprs": {factor_name: factor_expr},
            "backtest_start_time": "2016-01-01",
            "backtest_end_time": "2023-12-31",
            "stock_pool": "VN100",
        }

        try:
            response = requests.post(self.backtest_api_url, json=test_request, timeout=120)
            if response.status_code == 200:
                result = response.json()
                if result.get("data"):
                    metric_value = np.round(result["data"]["metrics"][metric], 4)
                    return float(metric_value), "success"
            return np.nan, f"HTTP {response.status_code}"
        except Exception as e:
            logger.error(f"Backtest exception: {e}")
            return np.nan, str(e)[:200]

    async def calc_reward(self, instance_id: str) -> float:
        """Hierarchical reward per paper Eq.5."""
        instance = self._instance_dict[instance_id]
        best_metric = instance["best_metric"]
        init_metric = instance["init_metric"]

        # R_tool
        base_reward = ALPHA_SUCC * instance["tool_call_count"] - ALPHA_FAIL * instance["failed_count"]
        base_reward = min(base_reward, CAP_TOOL)

        if instance["tool_call_count"] == 0:
            return 0.0

        # R_cons (simplified — no AST, just count successful)
        consistency_reward = ALPHA_CONS * len(instance["succ_tried_factors"])
        consistency_reward = min(consistency_reward, CAP_CONS)

        # R_expl (simplified — reward diversity)
        exploration_reward = ALPHA_EXP * len(set(f["factor_expr"] for f in instance["succ_tried_factors"]))
        exploration_reward = min(exploration_reward, CAP_EXPL)

        # R_perf
        baseline = max(0.0, init_metric)
        x = best_metric - baseline
        x_clamped = max(-20.0, min(20.0, x))
        performance_reward = ALPHA_PERF * math.log(1.0 + math.exp(x_clamped))
        performance_reward = min(performance_reward, CAP_PERF)

        # R_streak
        streak_reward = ALPHA_STREAK * instance["streak"]
        streak_reward = min(streak_reward, CAP_STREAK)

        # Eq.5
        r_tool_denom = max(base_reward, R_TOOL_FLOOR)
        total_reward = (consistency_reward + exploration_reward) / r_tool_denom + performance_reward * streak_reward
        instance["reward"] = total_reward

        logger.info(
            f"Reward for {instance_id}: {total_reward:.4f} "
            f"(base={base_reward:.3f}, perf={performance_reward:.3f}, "
            f"expl={exploration_reward:.3f}, streak={streak_reward:.3f})"
        )
        return total_reward

    async def release(self, instance_id: str, **kwargs) -> None:
        if instance_id in self._instance_dict:
            del self._instance_dict[instance_id]
