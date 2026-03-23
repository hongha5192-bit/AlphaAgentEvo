"""Qlib-consistent single-factor portfolio backtester for Vietnam market.

Strictly matches AlphaAgent's conf_vn_combined_kdd_ver.yaml:

Strategy: ForceExitTopkDropoutStrategy
  - topk: 10 (absolute, not percentage)
  - n_drop: 2
  - hold_thresh: 2 (T+2 settlement)

Execution:
  - Buy at OPEN price at T
  - Sell at CLOSE price at T+2
  - Signal uses data up to T-1

Costs:
  - open_cost: 0.13% (buy)
  - close_cost: 0.13% (sell)

Metrics (Qlib "sum" mode):
  - IR = mean(daily_excess_return) / std(daily_excess_return) * sqrt(252)
  - annualized_return = mean * 252
  - max_drawdown = (cumsum - cummax(cumsum)).min()
  - ann_scaler: 252
"""

import numpy as np
import pandas as pd
from typing import Optional


def compute_portfolio_ir(
    factor_values: pd.Series,
    price_df: pd.DataFrame,
    bench_return: pd.Series | None = None,
    top_k: int = 10,
    n_drop: int = 2,
    rebalance_freq: int = 5,
    cost_buy: float = 0.0013,
    cost_sell: float = 0.0013,
    hold_thresh: int = 2,
    ann_scaler: int = 252,
    start_date: str | None = None,
    end_date: str | None = None,
) -> dict:
    """Compute portfolio-based IR matching AlphaAgent's Qlib config.

    Args:
        factor_values: Factor signal (datetime × instrument multi-index)
        price_df: DataFrame with $open, $close columns (same multi-index)
        bench_return: Benchmark daily return series indexed by datetime
        top_k: Number of stocks to long (default 10, absolute)
        n_drop: Stocks to dropout per rebalance (default 2)
        rebalance_freq: Rebalance every N trading days (default 5)
        cost_buy: Buy transaction cost at open (default 0.13%)
        cost_sell: Sell transaction cost at close (default 0.13%)
        hold_thresh: Minimum holding days / T+N settlement (default 2)
        ann_scaler: Annualization factor (default 252)
        start_date: Evaluation start date
        end_date: Evaluation end date

    Returns:
        dict with Qlib-consistent metrics
    """
    N = ann_scaler

    if isinstance(factor_values, pd.DataFrame):
        factor_values = factor_values.iloc[:, 0]

    # ── Prepare dates ──
    all_dates = sorted(factor_values.index.get_level_values('datetime').unique())
    if start_date:
        all_dates = [d for d in all_dates if d >= pd.Timestamp(start_date)]
    if end_date:
        all_dates = [d for d in all_dates if d <= pd.Timestamp(end_date)]

    if len(all_dates) < rebalance_freq + hold_thresh + 1:
        return _empty_result("Not enough dates for backtesting")

    # ── Build price lookups ──
    # Unstack to datetime × instrument for fast access
    open_prices = price_df['$open'].unstack('instrument') if '$open' in price_df.columns else None
    close_prices = price_df['$close'].unstack('instrument') if '$close' in price_df.columns else None

    if open_prices is None or close_prices is None:
        return _empty_result("Missing $open or $close in price_df")

    # ── Rebalance schedule ──
    # T+N: ensure rebalance freq respects settlement
    effective_freq = max(rebalance_freq, hold_thresh)
    rebalance_indices = list(range(0, len(all_dates), effective_freq))
    rebalance_set = set(all_dates[i] for i in rebalance_indices)

    # ── Portfolio state ──
    # holdings: {instrument: {'buy_price': float, 'buy_date': Timestamp, 'buy_day_idx': int}}
    holdings = {}
    cash = 1.0  # Normalized to 1.0 for return computation

    daily_portfolio_values = []
    daily_dates_out = []

    prev_portfolio_value = 1.0

    for day_idx, date in enumerate(all_dates):
        is_rebalance = date in rebalance_set

        # ── Mark-to-market: value portfolio at CLOSE today ──
        portfolio_value = cash
        for inst, pos in holdings.items():
            try:
                close_px = close_prices.loc[date, inst]
                if pd.notna(close_px) and close_px > 0:
                    # Position value = qty_normalized * close_price
                    # qty_normalized = investment / buy_price
                    portfolio_value += pos['investment'] * (close_px / pos['buy_price'])
                else:
                    portfolio_value += pos['investment']  # fallback to cost
            except (KeyError, TypeError):
                portfolio_value += pos['investment']

        daily_portfolio_values.append(portfolio_value)
        daily_dates_out.append(date)

        # ── Rebalance ──
        if is_rebalance:
            # Signal from T-1: use prev_date factor values
            if day_idx == 0:
                prev_portfolio_value = portfolio_value
                continue
            prev_date = all_dates[day_idx - 1]

            try:
                signal = factor_values.xs(prev_date, level='datetime')
            except KeyError:
                prev_portfolio_value = portfolio_value
                continue

            signal = signal.dropna()
            if len(signal) < top_k:
                prev_portfolio_value = portfolio_value
                continue

            # ── SELL: force exit holdings past hold_thresh ──
            to_sell = []
            for inst, pos in list(holdings.items()):
                days_held = day_idx - pos['buy_day_idx']
                if days_held >= hold_thresh:
                    to_sell.append(inst)

            for inst in to_sell:
                pos = holdings[inst]
                # Sell at CLOSE price at T+2 (today's close, since we held >= 2 days)
                try:
                    sell_px = close_prices.loc[date, inst]
                    if pd.isna(sell_px) or sell_px <= 0:
                        sell_px = pos['buy_price']
                except (KeyError, TypeError):
                    sell_px = pos['buy_price']

                sell_px = float(sell_px)
                # Proceeds after sell cost
                proceeds = pos['investment'] * (sell_px / pos['buy_price']) * (1.0 - cost_sell)
                cash += proceeds
                del holdings[inst]

            # ── TopkDropout: select new stocks ──
            # Score current holdings
            current_insts = set(holdings.keys())

            # New top-k from signal
            new_top = set(signal.nlargest(top_k).index.tolist())

            # Stocks to potentially sell (lowest scored in current holdings)
            if current_insts:
                scored_current = signal.reindex(list(current_insts)).dropna()
                if len(scored_current) > 0:
                    dropout_candidates = set(
                        scored_current.nsmallest(min(n_drop, len(scored_current))).index.tolist()
                    )
                    # Also drop stocks not in new top-k at all
                    dropout_candidates |= (current_insts - new_top)
                else:
                    dropout_candidates = current_insts.copy()

                # Sell dropout candidates (if they meet hold_thresh)
                for inst in list(dropout_candidates):
                    if inst in holdings:
                        days_held = day_idx - holdings[inst]['buy_day_idx']
                        if days_held >= hold_thresh:
                            pos = holdings[inst]
                            try:
                                sell_px = close_prices.loc[date, inst]
                                if pd.isna(sell_px) or sell_px <= 0:
                                    sell_px = pos['buy_price']
                            except (KeyError, TypeError):
                                sell_px = pos['buy_price']

                            sell_px = float(sell_px)
                            proceeds = pos['investment'] * (sell_px / pos['buy_price']) * (1.0 - cost_sell)
                            cash += proceeds
                            del holdings[inst]

            # ── BUY: fill up to top_k ──
            slots_available = top_k - len(holdings)
            if slots_available > 0 and cash > 0:
                to_buy = [s for s in signal.nlargest(top_k + n_drop).index.tolist()
                          if s not in holdings][:slots_available]

                if to_buy:
                    investment_per_stock = cash / len(to_buy)

                    for inst in to_buy:
                        try:
                            buy_px = open_prices.loc[date, inst]
                            if pd.isna(buy_px) or buy_px <= 0:
                                continue
                        except (KeyError, TypeError):
                            continue

                        buy_px = float(buy_px)
                        # Investment after buy cost
                        actual_investment = investment_per_stock * (1.0 - cost_buy)

                        holdings[inst] = {
                            'buy_price': buy_px,
                            'buy_date': date,
                            'buy_day_idx': day_idx,
                            'investment': actual_investment,
                        }
                        cash -= investment_per_stock

        prev_portfolio_value = portfolio_value

    # ── Compute Qlib-consistent metrics ──
    if len(daily_portfolio_values) < 10:
        return _empty_result("Not enough daily returns")

    pv = np.array(daily_portfolio_values)
    # Daily returns from portfolio value changes
    daily_returns = np.diff(pv) / pv[:-1]
    return_dates = daily_dates_out[1:]

    # Excess returns (vs benchmark)
    if bench_return is not None:
        bench_aligned = bench_return.reindex(pd.DatetimeIndex(return_dates)).fillna(0).values
        excess_returns = daily_returns - bench_aligned
    else:
        excess_returns = daily_returns

    # Qlib "sum" mode metrics
    mean_r = float(np.mean(excess_returns))
    std_r = float(np.std(excess_returns, ddof=1)) if len(excess_returns) > 1 else 1e-8

    annualized_return = mean_r * N
    information_ratio = mean_r / (std_r + 1e-12) * np.sqrt(N)
    max_drawdown = float((np.cumsum(excess_returns) - np.maximum.accumulate(np.cumsum(excess_returns))).min())
    annualized_volatility = std_r * np.sqrt(N)
    total_return = float(np.sum(daily_returns))

    return {
        'success': True,
        'ir': round(information_ratio, 6),
        'annualized_return': round(annualized_return, 6),
        'annualized_volatility': round(annualized_volatility, 6),
        'sharpe': round(information_ratio, 6),
        'mdd': round(max_drawdown, 6),
        'total_return': round(total_return, 6),
        'mean_daily_return': round(mean_r, 8),
        'std_daily_return': round(std_r, 8),
        'n_days': len(daily_returns),
        'final_value': round(float(pv[-1]), 6),
        'error': None,
    }


def _empty_result(error: str) -> dict:
    return {
        'success': False,
        'ir': 0.0,
        'annualized_return': 0.0,
        'annualized_volatility': 0.0,
        'sharpe': 0.0,
        'mdd': 0.0,
        'total_return': 0.0,
        'mean_daily_return': 0.0,
        'std_daily_return': 0.0,
        'n_days': 0,
        'final_value': 1.0,
        'error': error,
    }
