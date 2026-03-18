"""Generate training dataset for GRPO from seed factors.

Creates conversation-format training data where each example contains:
- System prompt (from system_prompt.md)
- User query with seed factor expression and baseline IR
- Output as parquet with columns: prompt, seed_expr, seed_ir

Usage:
    python training/generate_dataset.py [--augment] [--backtest-url http://localhost:8001]
"""

import argparse
import json
import random
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def load_system_prompt() -> str:
    """Load the system prompt from system_prompt.md."""
    prompt_path = Path(__file__).parent / "system_prompt.md"
    return prompt_path.read_text().strip()


def load_seed_factors(path: str | Path | None = None) -> list[dict]:
    """Load seed factors from JSON file."""
    if path is None:
        path = PROJECT_ROOT / "data" / "seed_factors.json"
    with open(path) as f:
        return json.load(f)


def generate_mutations(seed_factors: list[dict]) -> list[dict]:
    """Generate additional seed factors by mutating existing ones.

    Mutations include:
    - Window parameter changes (e.g., 5→10, 20→30)
    - Adding RANK/ZSCORE wrapper
    - Swapping variables (e.g., $close→$open)
    - Combining two seeds
    """
    mutations = []

    # Window parameter mutations
    window_changes = {
        '5': ['3', '7', '10'],
        '10': ['5', '15', '20'],
        '15': ['10', '20', '30'],
        '20': ['10', '15', '30'],
        '40': ['20', '30', '60'],
    }

    for seed in seed_factors:
        expr = seed['expression']

        # Window mutations
        for old_w, new_ws in window_changes.items():
            for new_w in new_ws:
                # Only replace if it looks like a window parameter (after comma)
                mutated = expr.replace(f', {old_w})', f', {new_w})')
                if mutated != expr:
                    mutations.append({
                        'name': f"{seed['name']}_w{old_w}to{new_w}",
                        'expression': mutated,
                        'hypothesis': f"Window adjustment of {seed['name']}: {old_w}→{new_w}",
                        'ir': seed.get('ir', 0.0) * random.uniform(0.8, 1.2),
                        'source': 'mutation_window',
                    })

        # RANK wrapper mutation
        if not expr.startswith('RANK('):
            mutations.append({
                'name': f"{seed['name']}_ranked",
                'expression': f"RANK({expr})",
                'hypothesis': f"Cross-sectional rank normalization of {seed['name']}",
                'ir': seed.get('ir', 0.0) * random.uniform(0.9, 1.1),
                'source': 'mutation_rank',
            })

        # ZSCORE wrapper mutation
        if not expr.startswith('ZSCORE('):
            mutations.append({
                'name': f"{seed['name']}_zscored",
                'expression': f"ZSCORE({expr})",
                'hypothesis': f"Cross-sectional z-score normalization of {seed['name']}",
                'ir': seed.get('ir', 0.0) * random.uniform(0.9, 1.1),
                'source': 'mutation_zscore',
            })

        # Variable swap mutations
        var_swaps = [
            ('$close', '$open'),
            ('$close', '$high'),
            ('$volume', '$net_foreign_vol'),
        ]
        for old_var, new_var in var_swaps:
            if old_var in expr:
                mutated = expr.replace(old_var, new_var)
                mutations.append({
                    'name': f"{seed['name']}_swap_{old_var.replace('$','')}_{new_var.replace('$','')}",
                    'expression': mutated,
                    'hypothesis': f"Variable swap in {seed['name']}: {old_var}→{new_var}",
                    'ir': seed.get('ir', 0.0) * random.uniform(0.7, 1.3),
                    'source': 'mutation_swap',
                })

    # Combination mutations: combine two seed expressions
    if len(seed_factors) >= 2:
        for i in range(len(seed_factors)):
            for j in range(i + 1, len(seed_factors)):
                # Additive combination
                mutations.append({
                    'name': f"combo_{seed_factors[i]['name']}_plus_{seed_factors[j]['name']}",
                    'expression': f"RANK({seed_factors[i]['expression']}) + RANK({seed_factors[j]['expression']})",
                    'hypothesis': f"Additive combination of {seed_factors[i]['name']} and {seed_factors[j]['name']}",
                    'ir': max(seed_factors[i].get('ir', 0), seed_factors[j].get('ir', 0)) * random.uniform(0.9, 1.2),
                    'source': 'mutation_combine',
                })
                # Multiplicative combination
                mutations.append({
                    'name': f"combo_{seed_factors[i]['name']}_times_{seed_factors[j]['name']}",
                    'expression': f"RANK({seed_factors[i]['expression']}) * RANK({seed_factors[j]['expression']})",
                    'hypothesis': f"Multiplicative combination of {seed_factors[i]['name']} and {seed_factors[j]['name']}",
                    'ir': max(seed_factors[i].get('ir', 0), seed_factors[j].get('ir', 0)) * random.uniform(0.9, 1.2),
                    'source': 'mutation_combine',
                })

    return mutations


def generate_programmatic_factors() -> list[dict]:
    """Generate factor expressions programmatically using all available operators."""
    factors = []

    def add(name, expr, hyp):
        factors.append({
            'name': name, 'expression': expr,
            'hypothesis': hyp, 'ir': 0.0, 'source': 'programmatic',
        })

    # =========================================================================
    # 1. MOMENTUM (basic + enhanced)
    # =========================================================================
    for w in [3, 5, 10, 15, 20, 30]:
        add(f'momentum_{w}d',
            f'RANK(DELTA($close, {w}) / ($close + 1e-8))',
            f'{w}-day price momentum')

    for w in [3, 5, 10, 20, 40]:
        add(f'ts_rank_return_{w}d',
            f'TS_RANK($return, {w})',
            f'{w}-day time-series rank of return')

    for w in [5, 10, 20]:
        add(f'decay_momentum_{w}d',
            f'RANK(DECAYLINEAR($return, {w}))',
            f'{w}-day decay-weighted momentum')

    for w in [5, 10, 20]:
        add(f'ema_momentum_{w}d',
            f'RANK(EMA($close, {w}) - $close)',
            f'EMA({w}) vs price deviation')

    for s, l in [(5, 20), (10, 30), (5, 40)]:
        add(f'ema_crossover_{s}_{l}',
            f'RANK(EMA($close, {s}) - EMA($close, {l}))',
            f'EMA crossover {s}d vs {l}d')

    for w in [5, 10, 20]:
        add(f'wma_momentum_{w}d',
            f'RANK(WMA($return, {w}))',
            f'{w}-day weighted moving average momentum')

    for w in [5, 10, 20]:
        add(f'pctchange_{w}d',
            f'RANK(TS_PCTCHANGE($close, {w}))',
            f'{w}-day percentage change')

    # =========================================================================
    # 2. MEAN REVERSION
    # =========================================================================
    for s, l in [(5, 20), (3, 10), (5, 30), (10, 40)]:
        add(f'mean_rev_{s}_{l}',
            f'RANK(TS_MEAN($close, {s}) / (TS_MEAN($close, {l}) + 1e-8) - 1)',
            f'Price mean reversion ({s}d vs {l}d)')

    for w in [5, 10, 20, 40]:
        add(f'ts_zscore_close_{w}d',
            f'TS_ZSCORE($close, {w})',
            f'{w}-day time-series z-score of price')

    for w in [10, 20, 30]:
        add(f'ts_zscore_return_{w}d',
            f'RANK(TS_ZSCORE($return, {w}))',
            f'{w}-day return z-score extremes')

    for w in [10, 20, 30]:
        add(f'bollinger_pctb_{w}d',
            f'($close - BB_LOWER($close, {w})) / (BB_UPPER($close, {w}) - BB_LOWER($close, {w}) + 1e-8)',
            f'{w}-day Bollinger %B position')

    for w in [10, 20, 30]:
        add(f'bollinger_width_{w}d',
            f'RANK(BB_UPPER($close, {w}) - BB_LOWER($close, {w}))',
            f'{w}-day Bollinger bandwidth')

    for w in [10, 20]:
        add(f'regresi_detrend_{w}d',
            f'RANK(REGRESI($close, SEQUENCE({w}), {w}))',
            f'{w}-day detrended price (regression residual)')

    # =========================================================================
    # 3. VOLATILITY
    # =========================================================================
    for w in [5, 10, 20]:
        add(f'volatility_{w}d',
            f'RANK(TS_STD($return, {w}))',
            f'{w}-day return volatility rank')

    for w in [5, 10, 20]:
        add(f'mad_vol_{w}d',
            f'RANK(TS_MAD($return, {w}))',
            f'{w}-day MAD-based robust volatility')

    for s, l in [(5, 20), (5, 40), (10, 30)]:
        add(f'vol_ratio_{s}_{l}',
            f'RANK(TS_STD($return, {s}) / (TS_STD($return, {l}) + 1e-8))',
            f'Volatility ratio {s}d/{l}d')

    for w in [10, 20]:
        add(f'tail_width_{w}d',
            f'RANK(TS_QUANTILE($return, {w}, 0.95) - TS_QUANTILE($return, {w}, 0.05))',
            f'{w}-day return tail width (5th-95th pctile)')

    for w in [5, 10, 20]:
        add(f'inv_vol_{w}d',
            f'RANK(INV(TS_STD($return, {w}) + 1e-8))',
            f'{w}-day inverse volatility (low-vol factor)')

    for w in [5, 10, 20]:
        add(f'intraday_range_{w}d',
            f'RANK(TS_MEAN(($high - $low) / ($close + 1e-8), {w}))',
            f'{w}-day average intraday range')

    for w in [10, 20]:
        add(f'log_variance_{w}d',
            f'RANK(LOG(TS_VAR($return, {w}) + 1e-8))',
            f'{w}-day log variance of returns')

    # =========================================================================
    # 4. VOLUME-PRICE RELATIONSHIP
    # =========================================================================
    for w in [5, 10, 20]:
        add(f'vol_price_corr_{w}d',
            f'TS_CORR(RANK($volume), RANK($close), {w})',
            f'{w}-day volume-price correlation')

    for w in [5, 10, 20]:
        add(f'ret_vol_corr_{w}d',
            f'TS_CORR($return, $volume, {w})',
            f'{w}-day return-volume correlation')

    for w in [10, 20]:
        add(f'ret_vol_beta_{w}d',
            f'REGBETA($return, $volume, {w})',
            f'{w}-day return sensitivity to volume')

    for w in [5, 10, 20]:
        add(f'relative_volume_{w}d',
            f'RANK($volume / (TS_MEAN($volume, {w}) + 1e-8))',
            f'{w}-day relative volume')

    for w in [10, 20]:
        add(f'vol_price_covar_{w}d',
            f'RANK(TS_COVARIANCE($close, $volume, {w}))',
            f'{w}-day price-volume covariance')

    for w in [5, 10, 20]:
        add(f'signed_volume_{w}d',
            f'RANK(TS_SUM($volume * SIGN($return), {w}))',
            f'{w}-day signed volume (OBV-like)')

    # =========================================================================
    # 5. FOREIGN FLOW
    # =========================================================================
    for w in [5, 10, 20]:
        add(f'foreign_flow_{w}d',
            f'RANK(TS_SUM($net_foreign_val, {w}) / (TS_SUM($volume, {w}) + 1e-8))',
            f'{w}-day foreign flow ratio')

    for w in [5, 10, 20]:
        add(f'foreign_zscore_{w}d',
            f'RANK(TS_ZSCORE($net_foreign_val, {w}))',
            f'{w}-day z-scored foreign flow')

    for w in [10, 20]:
        add(f'foreign_ret_corr_{w}d',
            f'TS_CORR($net_foreign_val, $return, {w})',
            f'{w}-day foreign flow return predictability')

    for w in [10, 20]:
        add(f'foreign_ret_beta_{w}d',
            f'REGBETA($return, $net_foreign_val, {w})',
            f'{w}-day return sensitivity to foreign flow')

    for w in [10, 20]:
        add(f'foreign_decay_{w}d',
            f'RANK(DECAYLINEAR($net_foreign_val, {w}))',
            f'{w}-day decay-weighted foreign accumulation')

    for s, l in [(5, 20), (5, 40)]:
        add(f'foreign_short_long_{s}_{l}',
            f'RANK(TS_SUM($net_foreign_val, {s}) / (TS_SUM($net_foreign_val, {l}) + 1e-8))',
            f'Foreign flow short({s}d) vs long({l}d)')

    for w in [10, 20]:
        add(f'foreign_resi_{w}d',
            f'RANK(REGRESI($close, $net_foreign_val, {w}))',
            f'{w}-day price residual after foreign flow')

    # =========================================================================
    # 6. TECHNICAL INDICATORS
    # =========================================================================
    for w in [7, 14, 21]:
        add(f'rsi_{w}d',
            f'RSI($close, {w})',
            f'{w}-day RSI')

    for w in [7, 14]:
        add(f'rsi_momentum_{w}d',
            f'RANK(RSI($close, {w}) - DELAY(RSI($close, {w}), 3))',
            f'{w}-day RSI momentum (3d change)')

    for s, l in [(5, 20), (12, 26), (7, 21)]:
        add(f'macd_{s}_{l}',
            f'RANK(MACD($close, {s}, {l}))',
            f'MACD({s},{l}) cross-sectional rank')

    for s, l in [(5, 20), (12, 26)]:
        add(f'macd_dir_{s}_{l}',
            f'RANK(MACD($close, {s}, {l})) * SIGN(DELTA(MACD($close, {s}, {l}), 3))',
            f'MACD({s},{l}) with direction')

    for w in [10, 20, 30]:
        add(f'bb_dist_{w}d',
            f'RANK($close - BB_MIDDLE($close, {w}))',
            f'{w}-day distance from Bollinger middle band')

    # =========================================================================
    # 7. RECENCY / TIMING
    # =========================================================================
    for w in [10, 20, 40]:
        add(f'days_since_high_{w}d',
            f'RANK(TS_ARGMAX($close, {w}))',
            f'{w}-day days since price high')

    for w in [10, 20, 40]:
        add(f'days_since_low_{w}d',
            f'RANK(TS_ARGMIN($close, {w}))',
            f'{w}-day days since price low')

    for w in [10, 20, 40]:
        add(f'high_low_timing_{w}d',
            f'RANK(HIGHDAY($close, {w}) - LOWDAY($close, {w}))',
            f'{w}-day high-low timing asymmetry')

    for w in [10, 20]:
        add(f'days_since_vol_peak_{w}d',
            f'RANK(TS_ARGMAX($volume, {w}))',
            f'{w}-day days since volume peak')

    # =========================================================================
    # 8. REGRESSION-BASED
    # =========================================================================
    for w in [10, 20, 30]:
        add(f'price_trend_slope_{w}d',
            f'REGBETA($close, SEQUENCE({w}), {w})',
            f'{w}-day price trend slope')

    for w in [10, 20]:
        add(f'vol_trend_slope_{w}d',
            f'REGBETA($volume, SEQUENCE({w}), {w})',
            f'{w}-day volume trend slope')

    for w in [10, 20]:
        add(f'ret_vol_resi_{w}d',
            f'RANK(REGRESI($return, $volume, {w}))',
            f'{w}-day return orthogonal to volume')

    # =========================================================================
    # 9. NONLINEAR TRANSFORMS
    # =========================================================================
    for w in [10, 20]:
        add(f'log_volume_{w}d',
            f'RANK(LOG(TS_SUM($volume, {w}) + 1e-8))',
            f'{w}-day log cumulative volume')

    for w in [10, 20]:
        add(f'sqrt_vol_{w}d',
            f'RANK(SQRT(TS_STD($return, {w}) + 1e-8))',
            f'{w}-day sqrt volatility')

    for w in [10, 20]:
        add(f'exp_neg_vol_{w}d',
            f'RANK(EXP(-TS_STD($return, {w})))',
            f'{w}-day exp-dampened volatility')

    for w in [5, 10]:
        add(f'sq_rank_momentum_{w}d',
            f'RANK(POW(TS_RANK($return, {w}), 2))',
            f'{w}-day squared rank momentum')

    # =========================================================================
    # 10. MULTI-FACTOR COMPOSITES
    # =========================================================================
    add('mom_invvol',
        'RANK(TS_MEAN($return, 5)) * RANK(INV(TS_STD($return, 20) + 1e-8))',
        'Momentum * inverse volatility')

    add('foreign_vol_confirm',
        'RANK($net_foreign_val) * RANK($volume / (TS_MEAN($volume, 20) + 1e-8))',
        'Foreign flow confirmed by relative volume')

    add('meanrev_vol_filter',
        'TS_ZSCORE($close, 20) * RANK(TS_STD($return, 5))',
        'Mean reversion filtered by short-term volatility')

    add('mom_trend',
        'RANK(DELTA($close, 10)) + RANK(REGBETA($close, SEQUENCE(20), 20))',
        'Momentum plus trend slope')

    add('vol_weighted_mom_10d',
        'RANK(DECAYLINEAR($return * $volume, 10))',
        'Volume-weighted decay momentum 10d')

    add('foreign_mom_10d',
        'RANK(TS_SUM($net_foreign_val, 5)) * RANK(DELTA($close, 10))',
        'Foreign flow momentum interaction')

    add('rsi_bollinger',
        'RANK(RSI($close, 14) - 50) * ($close - BB_MIDDLE($close, 20)) / (BB_UPPER($close, 20) - BB_LOWER($close, 20) + 1e-8)',
        'RSI-Bollinger composite signal')

    add('vol_breakout',
        'RANK($volume / (TS_MEAN($volume, 20) + 1e-8)) * RANK(DELTA($close, 5) / ($close + 1e-8))',
        'Volume breakout momentum')

    add('foreign_trend_resi',
        'RANK(REGRESI($return, $net_foreign_val, 20)) * SIGN(REGBETA($close, SEQUENCE(20), 20))',
        'Return residual (after foreign flow) with trend direction')

    add('high_low_momentum',
        'RANK(($close - TS_MIN($low, 20)) / (TS_MAX($high, 20) - TS_MIN($low, 20) + 1e-8))',
        'Price position within 20d high-low range')

    # =========================================================================
    # 11. CROSS-SECTIONAL HIGHER MOMENTS
    # =========================================================================
    for w in [10, 20]:
        add(f'return_skew_{w}d',
            f'RANK(SKEW(TS_MEAN($return, {w})))',
            f'{w}-day cross-sectional return skewness')

    for w in [10, 20]:
        add(f'volume_kurt_{w}d',
            f'RANK(KURT(TS_MEAN($volume, {w})))',
            f'{w}-day cross-sectional volume kurtosis')

    # =========================================================================
    # 12. SCALE-NORMALIZED
    # =========================================================================
    add('scaled_momentum_10d',
        'SCALE(DELTA($close, 10) / ($close + 1e-8))',
        'Scale-normalized 10d momentum')

    add('scaled_foreign_flow_10d',
        'SCALE(TS_SUM($net_foreign_val, 10))',
        'Scale-normalized 10d foreign flow')

    return factors


def evaluate_seeds_via_api(factors: list[dict], backtest_url: str) -> list[dict]:
    """Optionally evaluate seed IR via backtest API."""
    import requests

    evaluated = []
    for f in factors:
        try:
            resp = requests.post(
                f"{backtest_url}/evaluate_factor",
                json={"factor_name": f["name"], "factor_expr": f["expression"]},
                timeout=120,
            )
            data = resp.json()
            if data.get("success"):
                f["ir"] = data["ir"]
                f["ic"] = data.get("ic_mean", 0.0)
            evaluated.append(f)
        except Exception as e:
            print(f"[WARN] Failed to evaluate {f['name']}: {e}")
            evaluated.append(f)
    return evaluated


def build_conversation(seed: dict, system_prompt: str) -> list[dict]:
    """Build a conversation prompt for one seed factor."""
    user_msg = (
        f"Evolve this seed alpha factor to achieve a higher IR.\n\n"
        f"Factor: {seed['name']}\n"
        f"Expression: {seed['expression']}\n"
        f"Hypothesis: {seed.get('hypothesis', 'N/A')}\n"
        f"Baseline IR: {seed.get('ir', 0.0):.4f}\n\n"
        f"Evaluate the seed first, then try at least 2 variations. Report the best result."
    )

    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_msg},
    ]


def main():
    parser = argparse.ArgumentParser(description="Generate GRPO training dataset")
    parser.add_argument("--augment", action="store_true", help="Generate augmented factors")
    parser.add_argument("--backtest-url", default="http://localhost:8001", help="Backtest API URL")
    parser.add_argument("--evaluate", action="store_true", help="Evaluate IRs via API")
    parser.add_argument("--output-dir", default=str(PROJECT_ROOT / "data"), help="Output directory")
    args = parser.parse_args()

    system_prompt = load_system_prompt()
    seed_factors = load_seed_factors()
    print(f"Loaded {len(seed_factors)} seed factors")

    all_factors = list(seed_factors)

    if args.augment:
        programmatic = generate_programmatic_factors()
        print(f"  Programmatic: {len(programmatic)} factors")

        # Mutations from base seeds
        mutations_base = generate_mutations(seed_factors)
        print(f"  Mutations from base seeds: {len(mutations_base)}")

        # Also mutate a random sample of programmatic factors for more diversity
        prog_sample = random.sample(programmatic, min(50, len(programmatic)))
        mutations_prog = generate_mutations(prog_sample)
        # Cap programmatic mutations to keep dataset manageable
        if len(mutations_prog) > 200:
            mutations_prog = random.sample(mutations_prog, 200)
        print(f"  Mutations from programmatic (sampled): {len(mutations_prog)}")

        all_factors.extend(programmatic)
        all_factors.extend(mutations_base)
        all_factors.extend(mutations_prog)
        print(f"After augmentation: {len(all_factors)} total factors")

    if args.evaluate:
        print("Evaluating factors via backtest API...")
        all_factors = evaluate_seeds_via_api(all_factors, args.backtest_url)

    # Build training examples
    records = []
    for seed in all_factors:
        conversation = build_conversation(seed, system_prompt)
        records.append({
            "prompt": conversation,
            "seed_expr": seed["expression"],
            "seed_ir": seed.get("ir", 0.0),
            "seed_name": seed["name"],
        })

    # Shuffle and split into train/val (90/10)
    random.shuffle(records)
    split_idx = max(1, int(len(records) * 0.9))
    train_records = records[:split_idx]
    val_records = records[split_idx:]

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_df = pd.DataFrame(train_records)
    val_df = pd.DataFrame(val_records)

    train_path = output_dir / "train.parquet"
    val_path = output_dir / "val.parquet"

    train_df.to_parquet(train_path, index=False)
    val_df.to_parquet(val_path, index=False)

    print(f"Train: {len(train_df)} examples → {train_path}")
    print(f"Val:   {len(val_df)} examples → {val_path}")

    # Also save all factors as JSONL for reference
    jsonl_path = output_dir / "seed_factors.jsonl"
    with open(jsonl_path, "w") as f:
        for factor in all_factors:
            f.write(json.dumps(factor, ensure_ascii=False) + "\n")
    print(f"All factors: {len(all_factors)} → {jsonl_path}")


if __name__ == "__main__":
    main()
