"""Factor expression execution engine.

Loads daily_pv.h5 into memory, parses factor expressions via expr_parser,
executes them using function_lib operators, and computes IC/IR metrics.

Supports temporal data splits for proper train/val/test evaluation:
- Factor computation uses ALL data (needs lookback for TS operators)
- IC/IR metrics are computed only on the specified period's date range
"""

import sys
import time
import traceback
from pathlib import Path

import numpy as np
import pandas as pd

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from expression_manager.expr_parser import parse_expression, parse_symbol
from expression_manager import function_lib

# Default data path
DATA_PATH = Path(__file__).resolve().parent / "data" / "daily_pv.h5"

# Python keywords that can't be used as variable names
_PYTHON_KEYWORDS = {'return', 'class', 'import', 'from', 'def', 'if', 'else',
                    'for', 'while', 'try', 'except', 'with', 'as', 'in', 'is',
                    'not', 'and', 'or', 'pass', 'break', 'continue', 'yield',
                    'lambda', 'global', 'nonlocal', 'del', 'raise', 'assert'}

# Cached dataframe (full data — used for factor computation)
_cached_df: pd.DataFrame | None = None
_cached_columns: dict[str, pd.Series] = {}

# Temporal split config: period_name -> (start_date, end_date)
# Dates are inclusive. Set via configure_periods().
_period_ranges: dict[str, tuple[str, str]] = {}

# Cached boolean masks per period (index-aligned with _cached_df)
_period_masks: dict[str, np.ndarray] = {}


def configure_periods(periods: dict[str, dict[str, str]]) -> None:
    """Configure temporal date ranges for train/val/test splits.

    Args:
        periods: dict like {"train": {"start": "2016-01-01", "end": "2023-12-31"},
                            "val":   {"start": "2024-01-01", "end": "2024-12-31"},
                            "test":  {"start": "2025-01-01", "end": "2026-12-31"}}
    """
    global _period_ranges, _period_masks
    _period_ranges.clear()
    _period_masks.clear()
    for name, cfg in periods.items():
        _period_ranges[name] = (cfg["start"], cfg["end"])
    print(f"[FactorExecutor] Configured periods: {_period_ranges}")

    # Rebuild masks if data already loaded
    if _cached_df is not None:
        _build_period_masks()


def _build_period_masks() -> None:
    """Build boolean masks for each configured period."""
    global _period_masks
    _period_masks.clear()
    if _cached_df is None or not _period_ranges:
        return
    dates = _cached_df.index.get_level_values("datetime")
    for name, (start, end) in _period_ranges.items():
        mask = (dates >= pd.Timestamp(start)) & (dates <= pd.Timestamp(end))
        mask_arr = mask.values if hasattr(mask, 'values') else np.asarray(mask)
        _period_masks[name] = mask_arr
        n_rows = mask_arr.sum()
        n_days = dates[mask_arr].nunique()
        print(f"[FactorExecutor] Period '{name}': {start} to {end} — {n_days} days, {n_rows} rows")


def load_data(data_path: str | Path | None = None) -> pd.DataFrame:
    """Load daily_pv.h5 into memory (cached on first call)."""
    global _cached_df, _cached_columns
    if _cached_df is not None:
        return _cached_df

    path = Path(data_path) if data_path else DATA_PATH
    print(f"[FactorExecutor] Loading data from {path}...")
    _cached_df = pd.read_hdf(str(path))

    # Pre-extract column Series for fast access during expression execution
    for col in _cached_df.columns:
        _cached_columns[col] = _cached_df[col]

    print(f"[FactorExecutor] Loaded {_cached_df.shape[0]} rows, "
          f"{len(_cached_df.columns)} columns, "
          f"instruments: {_cached_df.index.get_level_values('instrument').nunique()}")

    # Build period masks if periods already configured
    if _period_ranges:
        _build_period_masks()

    return _cached_df


def execute_expression(factor_expr: str, data_path: str | Path | None = None,
                       period: str | None = None) -> dict:
    """Execute a factor expression and compute IC/IR metrics.

    Args:
        factor_expr: Factor expression string, e.g. "RANK($close)"
        data_path: Optional path to daily_pv.h5
        period: Optional period name ("train", "val", "test"). If set,
                IC/IR is computed only on that period's date range.
                Factor values are still computed on ALL data (TS operators
                need lookback). If None, uses all data.

    Returns:
        dict with keys: success, score (IR), ic_mean, ic_std, ir, error, exec_time
    """
    start_time = time.time()

    try:
        # Load data
        df = load_data(data_path)

        # Step 1: Parse expression into executable Python code
        parsed_code = parse_expression(factor_expr)

        # Step 2: Replace $variables with safe Python variable names
        # Build a safe name mapping: $return → col_return (avoid Python keywords)
        columns = [col for col in df.columns]
        safe_name_map = {}
        for col in columns:
            clean = col.replace('$', '')
            if clean in _PYTHON_KEYWORDS:
                safe_name_map[col] = f'col_{clean}'
            else:
                safe_name_map[col] = clean

        # First do standard parse_symbol to strip $
        executable_code = parse_symbol(parsed_code, columns)

        # Then fix any Python keyword collisions
        for col in columns:
            clean = col.replace('$', '')
            safe = safe_name_map[col]
            if clean != safe:
                # Replace the bare keyword with the safe name
                # Use word-boundary replacement to avoid partial matches
                import re as _re
                executable_code = _re.sub(
                    r'\b' + _re.escape(clean) + r'\b',
                    safe,
                    executable_code
                )

        # Step 3: Build execution namespace with function_lib functions + data columns
        exec_namespace = {}

        # Import all functions from function_lib
        for name in dir(function_lib):
            obj = getattr(function_lib, name)
            if callable(obj) and not name.startswith('_'):
                exec_namespace[name] = obj

        # Add numpy and pandas
        exec_namespace['np'] = np
        exec_namespace['pd'] = pd

        # Add column data using safe variable names
        for col in columns:
            safe_name = safe_name_map[col]
            exec_namespace[safe_name] = _cached_columns[col].copy()

        # Step 4: Execute the parsed expression
        factor_values = eval(executable_code, exec_namespace)

        # Ensure result is a Series with the same index as data
        if isinstance(factor_values, pd.DataFrame):
            factor_values = factor_values.iloc[:, 0]
        elif isinstance(factor_values, np.ndarray):
            factor_values = pd.Series(factor_values, index=df.index)

        # Step 5: Compute portfolio-based IR (Qlib-style)
        from backtest.qlib_backtester import compute_portfolio_ir

        # Determine evaluation period
        start_date = None
        end_date = None
        if period and period in _period_ranges:
            start_date, end_date = _period_ranges[period]

        # Get benchmark return
        bench_return = None
        if '$bench_return' in _cached_columns:
            bench_series = _cached_columns['$bench_return']
            # Extract unique datetime-level benchmark (same for all stocks)
            bench_return = bench_series.groupby('datetime').first()

        result = compute_portfolio_ir(
            factor_values=factor_values,
            price_df=df[['$open', '$close']],
            bench_return=bench_return,
            top_k=10,
            n_drop=2,
            rebalance_freq=5,
            cost_buy=0.0013,
            cost_sell=0.0013,
            hold_thresh=2,
            ann_scaler=252,
            start_date=start_date,
            end_date=end_date,
        )

        exec_time = time.time() - start_time

        return {
            'success': result['success'],
            'score': result['ir'],
            'ir': result['ir'],
            'annualized_return': result.get('annualized_return', 0.0),
            'annualized_volatility': result.get('annualized_volatility', 0.0),
            'sharpe': result.get('sharpe', 0.0),
            'mdd': result.get('mdd', 0.0),
            'total_return': result.get('total_return', 0.0),
            'n_days': result.get('n_days', 0),
            'error': result.get('error'),
            'exec_time': round(exec_time, 3),
        }

    except Exception as e:
        exec_time = time.time() - start_time
        return {
            'success': False,
            'score': 0.0,
            'ic_mean': 0.0,
            'ic_std': 0.0,
            'ir': 0.0,
            'error': f"{type(e).__name__}: {str(e)[:500]}",
            'exec_time': round(exec_time, 3),
        }


if __name__ == "__main__":
    # Quick test
    test_exprs = [
        "RANK($close)",
        "RANK(TS_MEAN($return, 5) / (TS_STD($return, 20) + 1e-8)) * SIGN(DELTA($close, 5) / ($close + 1e-8))",
        "TS_CORR(RANK($volume), RANK($close), 10) - TS_CORR(RANK($volume), RANK($close), 40)",
    ]
    for expr in test_exprs:
        print(f"\nExpression: {expr}")
        result = execute_expression(expr)
        print(f"Result: {result}")
