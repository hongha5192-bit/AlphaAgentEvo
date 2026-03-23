import logging
import os
from typing import Any, Optional
from uuid import uuid4

import numpy as np
import requests
import pdb
from Levenshtein import distance

from verl.utils.rollout_trace import rollout_trace_op
from .base_tool import BaseTool
from .schemas import OpenAIFunctionToolSchema
from .factor_ast import compute_similarity

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


class FactorTool(BaseTool):
    """A tool for evaluating factors with backtesting.

    - `to_openai_function_tool_schema`: return the tool schema in OpenAI format.
    - `create`: create a tool instance for a trajectory.
    - `execute`: execute the tool.
    - `calc_reward`: calculate the reward based on backtest results.
    - `release`: release the tool instance.
    """

    def __init__(self, config: dict, tool_schema: OpenAIFunctionToolSchema):
        """
        _tool_schema = OpenAIFunctionToolSchema.model_validate({
            "type": "function",
            "function": {
                "name": "evaluate_factor",
                "description": "A tool for evaluating factors with backtesting",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "factor_name": {
                            "type": "string",
                            "description": "The name of the factor",
                        },
                        "factor_expr": {
                            "type": "string",
                            "description": "The expression of the factor",
                        },
                    },
                    "required": ["factor_name", "factor_expr"],
                },
            }
        })
        """
        super().__init__(config, tool_schema)
        logger.info(f"Initializing FactorTool with name: {tool_schema.function.name}")
        logger.info(f"Tool config: {config}")
        self._instance_dict = {}
        self.backtest_api_url = config.get("backtest_api_url", "http://localhost:8001/backtest")
        self._tried_factors = set()  # Record tried factor expressions

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
            "streak": 0,  # Record consecutive improvement rounds
            "tool_call_count": 0,  # Record tool call count
            "succ_tried_factors": [],  # Record tried factor expressions
            "failed_count": 0  # Record failed attempts
        }
        return instance_id

    @rollout_trace_op
    async def execute(self, instance_id: str, parameters: dict[str, Any], **kwargs) -> tuple[str, float, dict]:
        factor_name = parameters.get("factor_name", "")
        factor_expr = parameters.get("factor_expr", "")
        metric = parameters.get("metric", "Information_Ratio_with_cost")
        logger.info(f"Sending API call request for {instance_id}: {factor_name}, {factor_expr}, {metric}")

        self._instance_dict[instance_id]["factor_name"] = factor_name
        self._instance_dict[instance_id]["factor_expr"] = factor_expr
        self._instance_dict[instance_id]["metric"] = metric
        self._instance_dict[instance_id]["tool_call_count"] += 1

        metric_value, status = await self.call_backtest_api(instance_id)
        logger.info(f"Received backtest result for {instance_id}: {metric_value}, {status}")
        

        if status == "success":
            self._instance_dict[instance_id]["succ_tried_factors"].append({"factor_expr": factor_expr, "metric_value": metric_value})
            # Update best_metric
            if metric_value > self._instance_dict[instance_id]["best_metric"]:
                self._instance_dict[instance_id]["streak"] += 1
                self._instance_dict[instance_id]["best_metric"] = metric_value

            # Calculate reward
            reward = await self.calc_reward(instance_id)
            return f"success: Evaluated factor \"{factor_name}\" with expression \"{factor_expr}\", {metric}={metric_value}", reward, {self.tool_schema.function.name: reward}
        else:
            self._instance_dict[instance_id]["failed_count"] += 1
            reward = await self.calc_reward(instance_id)
            return f"failed: factor {factor_name} with expression {factor_expr}. Reason: {status}", reward, {self.tool_schema.function.name: reward}

    async def call_backtest_api(self, instance_id: str, **kwargs) -> float:
        """Calculate reward based on backtest results.
        
        Reward consists of two parts:
        1. Basic reward for successful backtest (0.5)
        2. Metric-based reward (scaled by metric value)
        """
        instance = self._instance_dict[instance_id]
        
        # Ensure factor_name and factor_expr are strings
        factor_name = instance["factor_name"]
        factor_expr = instance["factor_expr"]
        
        # If it's a dictionary, extract string value
        if isinstance(factor_name, dict):
            factor_name = str(factor_name)
            logger.warning(f"factor_name is a dictionary, converting to string: {factor_name}")
        
        if isinstance(factor_expr, dict):
            factor_expr = str(factor_expr)
            logger.warning(f"factor_expr is a dictionary, converting to string: {factor_expr}")
        
        factor = {"name": factor_name, "expr": factor_expr}
        metric = instance["metric"]
        
        logger.debug(f"Calling backtest API: factor={factor}, metric={metric}")
        
        # Run backtest
        metric_value, status = self._run_backtest(factor, metric)
        return metric_value, status
    
    def _run_backtest(self, factor: dict, metric: str = 'IC') -> tuple[float, str]:
        """Run backtest and return metric value and status."""
        test_request = {
            "exprs": {
                factor["name"]: factor["expr"]
            },
            "backtest_start_time": "2016-01-01",
            "backtest_end_time": "2023-12-31",
            "start_cash": 10000000.0,
            "update_freq": 5,
            "label_forward_days": 5,
            "stock_pool": "VN100",
            "stop_loss_rate": 0.5,
            "stop_profit_rate": 0.5,
            "position_size": 1.0,
            "max_pos_each_stock": 0.2,
            "use_cache": True,
            "layer_start": 0,
            "layer_end": 1,
            "pred_score_industry_neutralization": False
        }
        
        try:
            response = requests.post(
                self.backtest_api_url,
                json=test_request,
                timeout=600,
            )
            
            if response.status_code == 200:
                result = response.json()
                if result['data']:
                    metric_value = np.round(result['data']['metrics'][metric], 4)
                    return metric_value, "success"
            else:
                logger.error(f"Backtest request failed: {response.text}")
                return np.nan, response.json().get('detail', {}).get('error', "unknown error")
                
        except Exception as e:
            logger.error(f"Backtest exception: {e}")
            return np.nan, str(e)

    def _factor_expr_similarity(self, expr: str, hist_exprs: list[str]) -> float:
        """Compute the similarity of two factor expressions"""
        try:
            simlarity, _, _ = compute_similarity(expr, hist_exprs)
            return simlarity
        except Exception as e:
            logger.error(f"Failed to compute factor expression similarity: {e}")
            return None

    async def calc_reward(self, instance_id: str) -> float:
        """
        Reward components:
        1. Base reward for successful tool calls
        2. Exploration reward for testing new factors
        3. Consistency reward for testing factors with similar expressions
        4. Performance reward for factors with higher metrics
        5. Streak reward for consecutive successful tests
        """
        instance = self._instance_dict[instance_id]
        best_metric = instance["best_metric"]
        init_metric = instance["init_metric"]

        base_reward = 0.1 * instance["tool_call_count"] - 0.2 * instance["failed_count"]
        base_reward = min(base_reward, 1)

        # Consistency reward
        consistency_reward = 0.0
        for i, tried_factor in enumerate(instance["succ_tried_factors"]):
            simlarity = self._factor_expr_similarity(tried_factor["factor_expr"], [instance["init_factor_expr"]])
            if tried_factor["metric_value"] != 0 and simlarity is not None and simlarity > 0.1 and simlarity < 0.9:
                consistency_reward += 0.02
        consistency_reward = min(consistency_reward, 0.2)

        # Exploration reward
        exploration_reward = 0.0
        for i, tried_factor in enumerate(instance["succ_tried_factors"]):
            simlarity = self._factor_expr_similarity(tried_factor["factor_expr"], [instance["init_factor_expr"]] + [factor["factor_expr"] for factor in instance["succ_tried_factors"][:i]])
            logger.info(f"Actor tested factor {tried_factor['factor_expr']}, similarity with history factors: {simlarity}")
            if (
                tried_factor["metric_value"] != 0
                and tried_factor["factor_expr"] != instance["init_factor_expr"]
                and simlarity is not None and simlarity < 1
            ):
                exploration_reward += 0.02 * (1 - simlarity)
        exploration_reward = min(exploration_reward, 0.3)

        
        # Performance reward
        performance_reward = np.log(1 + np.exp(best_metric - max(0, init_metric))) * 0.1  # V3 supports negative init_metric
        performance_reward = min(performance_reward, 0.5)

        # Streak reward
        streak_reward = instance["streak"] * 0.15
        streak_reward = min(streak_reward, 0.6)

        total_reward = (exploration_reward + consistency_reward) /  base_reward + performance_reward * streak_reward
        instance['reward'] = total_reward


        logger.info(f"Reward for {instance_id}: {total_reward}, (base reward: {base_reward}, performance reward: {performance_reward}, exploration reward: {exploration_reward}, streak reward: {streak_reward})")
        return total_reward

    async def release(self, instance_id: str, **kwargs) -> None:
        del self._instance_dict[instance_id] 