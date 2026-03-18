"""5-dimension hierarchical reward calculator for factor evolution.

Implements the reward function from the AlphaAgentEvo paper (Eq.5):

  R(τ) = [min(R_cons, C_cons) + min(R_expl, C_expl)] / min(R_tool, C_tool)
         + min(R_perf, C_perf) · min(R_streak, C_streak)

Components:
  R_tool:   α_succ·N_succ - α_fail·N_fail  (cap: 1.0, used as denominator)
  R_cons:   Σ α_cons·1[sim(fi, f_seed) > h_low]  (cap: 0.2)
  R_expl:   Σ α_exp·(1 - max sim to predecessors)  (cap: 0.3)
  R_perf:   α_perf·log(1 + exp(s(f*) - max(0, s(f_seed))))  (cap: 0.5)
  R_streak: α_streak·N_streak  (cap: 0.6)
"""

import math
import sys
from pathlib import Path

import requests

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from expression_manager.factor_ast import (
    parse_expression as parse_ast,
    find_largest_common_subtree,
    count_all_nodes,
)

# Reward caps (from paper Appendix D)
CAP_TOOL = 1.0
CAP_CONS = 0.2
CAP_EXPL = 0.3
CAP_PERF = 0.5
CAP_STREAK = 0.6

# Weighting coefficients (from paper Appendix D)
ALPHA_SUCC = 0.1
ALPHA_FAIL = 0.2
ALPHA_CONS = 0.02
ALPHA_EXP = 0.02
ALPHA_PERF = 0.1
ALPHA_STREAK = 0.15
H_LOW = 0.1   # consistency similarity lower threshold
H_HIGH = 0.9  # consistency similarity upper threshold (too similar = no credit)

# Floor for R_tool denominator to prevent division by zero
R_TOOL_FLOOR = 0.01

BACKTEST_URL = "http://localhost:8001"


def _ast_similarity(expr_a: str, expr_b: str) -> float:
    """Compute AST structural similarity per paper Eq.4.

    sim(fi, fj) = |AST(fi) ∩ AST(fj)| / max(|AST(fi)|, |AST(fj)|)
    """
    try:
        ast_a = parse_ast(expr_a)
        ast_b = parse_ast(expr_b)
        match = find_largest_common_subtree(ast_a, ast_b)
        size_a = count_all_nodes(expr_a)
        size_b = count_all_nodes(expr_b)
        if match is None or max(size_a, size_b) == 0:
            return 0.0
        return match.size / max(size_a, size_b)
    except Exception:
        return 0.0


class FactorTool:
    """Provides evaluate() and calc_reward() for GRPO training."""

    def __init__(self, backtest_url: str = BACKTEST_URL):
        self.backtest_url = backtest_url

    def evaluate(self, factor_name: str, factor_expr: str,
                 period: str | None = None) -> dict:
        """Call backtest API to evaluate a factor expression.

        Args:
            factor_name: Name for the factor.
            factor_expr: Expression to evaluate.
            period: Optional temporal period ("train", "val", "test").
                    If None, the server uses its default_period from config.

        Returns:
            dict with keys: success, ir, ic_mean, error
        """
        try:
            payload = {"factor_name": factor_name, "factor_expr": factor_expr}
            if period:
                payload["period"] = period
            resp = requests.post(
                f"{self.backtest_url}/evaluate_factor",
                json=payload,
                timeout=120,
            )
            data = resp.json()
            return {
                "success": data.get("success", False),
                "ir": data.get("ir", 0.0),
                "ic_mean": data.get("ic_mean", 0.0),
                "error": data.get("error"),
            }
        except Exception as e:
            return {
                "success": False,
                "ir": 0.0,
                "ic_mean": 0.0,
                "error": f"API call failed: {str(e)[:200]}",
            }

    def calc_reward(
        self,
        trajectory_results: list[dict],
        seed_expr: str,
        seed_ir: float,
    ) -> float:
        """Compute the hierarchical reward per paper Eq.5.

        R(τ) = (R_cons + R_expl) / R_tool + R_perf * R_streak

        Args:
            trajectory_results: List of evaluation results from the trajectory,
                each with keys: success, ir, factor_expr
            seed_expr: The original seed factor expression
            seed_ir: The seed factor's baseline IR

        Returns:
            Total reward (float)
        """
        if not trajectory_results:
            return 0.0

        n_calls = len(trajectory_results)
        n_success = sum(1 for r in trajectory_results if r.get("success", False))
        n_fail = n_calls - n_success

        # --- R_tool: α_succ·N_succ - α_fail·N_fail ---
        r_tool_raw = ALPHA_SUCC * n_success - ALPHA_FAIL * n_fail
        r_tool = min(CAP_TOOL, r_tool_raw)

        if n_success == 0:
            # No successful evaluations — return 0 (R_tool as denominator
            # would be ≤0, and there's nothing to compute for other terms)
            return 0.0

        # Collect successful factor expressions and IRs
        successful = [r for r in trajectory_results if r.get("success", False)]
        successful_exprs = [r.get("factor_expr", "") for r in successful]
        best_result = max(successful, key=lambda r: r.get("ir", 0.0))
        best_ir = best_result.get("ir", 0.0)

        # --- R_cons: Σ α_cons·1[sim(fi, f_seed) > h_low] ---
        r_cons = _calc_consistency(seed_expr, successful_exprs)

        # --- R_expl: Σ α_exp·(1 - max sim to predecessors) ---
        r_expl = _calc_exploration(seed_expr, successful_exprs)

        # --- R_perf: α_perf·log(1 + exp(best_ir - max(0, seed_ir))) ---
        r_perf = _calc_performance(best_ir, seed_ir)

        # --- R_streak: α_streak·N_streak ---
        r_streak = _calc_streak(successful, seed_ir)

        # Apply caps
        r_cons_capped = min(r_cons, CAP_CONS)
        r_expl_capped = min(r_expl, CAP_EXPL)
        r_tool_capped = min(r_tool, CAP_TOOL)
        r_perf_capped = min(r_perf, CAP_PERF)
        r_streak_capped = min(r_streak, CAP_STREAK)

        # Eq.5: hierarchical combination
        r_tool_denom = max(r_tool_capped, R_TOOL_FLOOR)
        direction_term = (r_cons_capped + r_expl_capped) / r_tool_denom
        quality_term = r_perf_capped * r_streak_capped

        total = direction_term + quality_term
        return round(total, 4)


def _calc_consistency(seed_expr: str, successful_exprs: list[str]) -> float:
    """R_cons: Sum α_cons for each factor with sim(fi, f_seed) > h_low.

    Paper Section 2.3: direction-aware consistency reward.
    Binary indicator: factors structurally similar to seed get credit.
    """
    total = 0.0
    for expr in successful_exprs:
        if not expr:
            continue
        sim = _ast_similarity(seed_expr, expr)
        if sim > H_LOW and sim < H_HIGH:
            total += ALPHA_CONS
    return total


def _calc_exploration(seed_expr: str, successful_exprs: list[str]) -> float:
    """R_expl: Sum α_exp·(1 - max_sim_to_predecessors) for each factor.

    Paper Eq.3: encourages diversity among proposals. Each new factor is
    compared to ALL previously proposed factors (including seed), rewarding
    novelty relative to the most-similar predecessor.
    """
    total = 0.0
    all_prior = [seed_expr]  # F_<i starts with seed
    for expr in successful_exprs:
        if not expr:
            all_prior.append(expr)
            continue
        # Max similarity to any predecessor
        max_sim = max(_ast_similarity(expr, prior) for prior in all_prior if prior)
        total += ALPHA_EXP * (1.0 - max_sim)
        all_prior.append(expr)
    return total


def _calc_performance(best_ir: float, seed_ir: float) -> float:
    """R_perf: α_perf·log(1 + exp(s(f*) - max(0, s(f_seed)))).

    Paper Section 2.3: softplus-based performance reward.
    Always positive (softplus(0) = log(2) ≈ 0.693).
    """
    baseline = max(0.0, seed_ir)
    x = best_ir - baseline
    # Clamp x to avoid overflow in exp
    x_clamped = max(-20.0, min(20.0, x))
    softplus = math.log(1.0 + math.exp(x_clamped))
    return ALPHA_PERF * softplus


def _calc_streak(successful_results: list[dict], seed_ir: float) -> float:
    """R_streak: α_streak·N_streak.

    Paper Section 2.3: cumulative count of new-best improvements.
    Original logic: increment streak each time a factor beats the running
    best_metric. Never resets — counts total breakthroughs, not consecutive.
    """
    if len(successful_results) < 2:
        return 0.0

    streak = 0
    best_so_far = seed_ir

    for r in successful_results:
        ir_val = r.get("ir", 0.0)
        if ir_val > best_so_far:
            streak += 1
            best_so_far = ir_val

    return ALPHA_STREAK * streak


if __name__ == "__main__":
    # Quick test of reward calculation (without API)
    tool = FactorTool()

    seed_expr = "RANK(TS_MEAN($return, 5) / (TS_STD($return, 20) + 1e-8)) * SIGN(DELTA($close, 5) / ($close + 1e-8))"
    seed_ir = 1.15

    # Simulate trajectory
    trajectory = [
        {"success": True, "ir": 1.10, "factor_expr": "RANK(TS_MEAN($return, 5) / (TS_STD($return, 20) + 1e-8))"},
        {"success": True, "ir": 1.25, "factor_expr": "RANK(TS_MEAN($return, 10) / (TS_STD($return, 20) + 1e-8)) * SIGN(DELTA($close, 10) / ($close + 1e-8))"},
        {"success": False, "ir": 0.0, "factor_expr": "INVALID()"},
    ]

    reward = tool.calc_reward(trajectory, seed_expr, seed_ir)

    # Manual verification
    n_succ, n_fail = 2, 1
    r_tool = min(1.0, 0.1 * n_succ - 0.2 * n_fail)  # 0.2 - 0.2 = 0.0 -> floor 0.01
    print(f"R_tool raw: {0.1 * n_succ - 0.2 * n_fail}, capped: {r_tool}, denom: {max(r_tool, 0.01)}")
    print(f"Total reward: {reward}")
    print()

    # Test with all successes
    trajectory2 = [
        {"success": True, "ir": 1.10, "factor_expr": "RANK(TS_MEAN($return, 5))"},
        {"success": True, "ir": 1.20, "factor_expr": "RANK(TS_MEAN($return, 10))"},
        {"success": True, "ir": 1.30, "factor_expr": "RANK(TS_MEAN($return, 20))"},
    ]
    reward2 = tool.calc_reward(trajectory2, seed_expr, seed_ir)
    r_tool2 = min(1.0, 0.1 * 3 - 0.2 * 0)  # 0.3
    print(f"All-success: R_tool={r_tool2}, reward={reward2}")
