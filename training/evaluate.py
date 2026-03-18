"""Evaluate trained AlphaAgentEvo checkpoints on val/test seed sets.

Metrics (matching paper):
  - VR (Valid Ratio): % of generated factors that are syntactically valid
  - pass@T: fraction of seeds where at least one evolved factor beats max(0, seed_ir)
  - Mean best IR improvement over seed
"""

import argparse
import json
import re
import time
import sys
from pathlib import Path

import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from training.factor_tool import FactorTool


def load_model(base_model_path: str, checkpoint_path: str | None, device: str = "cuda"):
    """Load base model + optional LoRA adapter."""
    print(f"Loading tokenizer from {base_model_path}...")
    tokenizer = AutoTokenizer.from_pretrained(
        base_model_path,
        trust_remote_code=True,
        padding_side="left",
    )

    # Swap chat template for Thinking model (same as train.py)
    if "Thinking" in base_model_path or "thinking" in base_model_path:
        base_name = base_model_path.replace("-Thinking-2507", "").replace("-thinking", "")
        base_tok = AutoTokenizer.from_pretrained(base_name, trust_remote_code=True)
        tokenizer.chat_template = base_tok.chat_template
        del base_tok

    print(f"Loading model from {base_model_path}...")
    model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
        attn_implementation="sdpa",
    )

    if checkpoint_path:
        print(f"Loading LoRA adapter from {checkpoint_path}...")
        model = PeftModel.from_pretrained(model, checkpoint_path)
        model = model.merge_and_unload()  # Merge for faster inference
        print("LoRA merged.")

    model.eval()
    return model, tokenizer


def build_tool_schema():
    """Build the tool schema for evaluate_factor (matching train.py)."""
    return {
        "type": "function",
        "function": {
            "name": "evaluate_factor",
            "description": "Evaluate a factor expression by backtesting against historical Vietnam stock market data. Returns the Information Ratio (IR), mean IC, and success status.",
            "parameters": {
                "type": "object",
                "properties": {
                    "factor_name": {
                        "type": "string",
                        "description": "A descriptive name for the factor"
                    },
                    "factor_expr": {
                        "type": "string",
                        "description": "The factor expression using available variables and operators"
                    }
                },
                "required": ["factor_name", "factor_expr"]
            }
        }
    }


def run_inference(model, tokenizer, messages, tool_schema, max_turns=3, max_new_tokens=5000):
    """Run multi-turn inference with tool calling.

    Returns:
        all_results: list of {success, ir, factor_expr} from each tool call
        full_messages: the complete conversation
    """
    factor_tool = FactorTool()
    all_results = []
    full_messages = list(messages)

    for turn in range(max_turns):
        # Apply chat template with tools
        text = tokenizer.apply_chat_template(
            full_messages,
            tools=[tool_schema],
            tokenize=False,
            add_generation_prompt=True,
        )

        inputs = tokenizer(text, return_tensors="pt").to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
            )

        # Decode only the new tokens
        new_tokens = outputs[0][inputs["input_ids"].shape[1]:]
        response_text = tokenizer.decode(new_tokens, skip_special_tokens=False)

        # Parse tool calls from response
        tool_calls = parse_tool_calls(response_text)

        if not tool_calls:
            # No tool calls — model is done
            full_messages.append({"role": "assistant", "content": response_text})
            break

        # Execute each tool call
        assistant_msg = {
            "role": "assistant",
            "content": response_text.split("<tool_call>")[0].strip() if "<tool_call>" in response_text else "",
            "tool_calls": []
        }

        tool_responses = []
        for i, tc in enumerate(tool_calls):
            call_id = f"call_{turn}_{i}"
            assistant_msg["tool_calls"].append({
                "id": call_id,
                "type": "function",
                "function": {
                    "name": "evaluate_factor",
                    "arguments": json.dumps(tc)
                }
            })

            # Execute via API
            result = factor_tool.evaluate(
                factor_name=tc.get("factor_name", "unnamed"),
                factor_expr=tc.get("factor_expr", ""),
            )
            all_results.append({
                "success": result.get("success", False),
                "ir": result.get("ir", 0.0),
                "factor_expr": tc.get("factor_expr", ""),
                "factor_name": tc.get("factor_name", ""),
                "turn": turn,
            })

            tool_responses.append({
                "role": "tool",
                "tool_call_id": call_id,
                "content": json.dumps(result),
            })

        full_messages.append(assistant_msg)
        full_messages.extend(tool_responses)

    return all_results, full_messages


def parse_tool_calls(text: str) -> list[dict]:
    """Extract tool call arguments from model output."""
    calls = []
    pattern = re.compile(r'<tool_call>\s*(.*?)\s*</tool_call>', re.DOTALL)
    for match in pattern.finditer(text):
        try:
            data = json.loads(match.group(1))
            args = data.get("arguments", data)
            if isinstance(args, str):
                args = json.loads(args)
            if "factor_expr" in args:
                calls.append(args)
        except (json.JSONDecodeError, TypeError):
            continue
    return calls


def evaluate_checkpoint(model, tokenizer, seeds_df, tool_schema, max_turns=3, max_new_tokens=5000):
    """Evaluate a model on a set of seeds.

    Returns dict with metrics: vr, pass_at_3, pass_at_5, per-seed details
    """
    results = []

    for idx, row in seeds_df.iterrows():
        seed_name = row["seed_name"]
        seed_expr = row["seed_expr"]
        seed_ir = row["seed_ir"]
        messages = row["prompt"]  # Already a list of message dicts

        print(f"  [{idx+1}/{len(seeds_df)}] {seed_name} (seed IR={seed_ir:.4f})...", end="", flush=True)
        t0 = time.time()

        try:
            all_tool_results, _ = run_inference(
                model, tokenizer, messages, tool_schema,
                max_turns=max_turns, max_new_tokens=max_new_tokens,
            )
        except Exception as e:
            print(f" ERROR: {e}")
            results.append({
                "seed_name": seed_name,
                "seed_ir": seed_ir,
                "n_calls": 0,
                "n_valid": 0,
                "best_ir": None,
                "beat_seed": False,
                "all_irs": [],
                "error": str(e),
            })
            continue

        n_calls = len(all_tool_results)
        valid = [r for r in all_tool_results if r["success"]]
        n_valid = len(valid)
        irs = [r["ir"] for r in valid]
        best_ir = max(irs) if irs else None
        baseline = max(0.0, seed_ir)
        beat_seed = best_ir is not None and best_ir > baseline

        elapsed = time.time() - t0

        # Compute pass at different turn counts
        irs_by_turn = {}
        for r in all_tool_results:
            t = r["turn"]
            if r["success"]:
                irs_by_turn.setdefault(t, []).append(r["ir"])

        # Cumulative best IR at each turn
        cum_best = None
        pass_at = {}
        for t in range(max_turns):
            if t in irs_by_turn:
                turn_best = max(irs_by_turn[t])
                cum_best = max(cum_best, turn_best) if cum_best is not None else turn_best
            pass_at[t] = cum_best is not None and cum_best > baseline

        status = "BEAT" if beat_seed else "miss"
        ir_str = f"{best_ir:.4f}" if best_ir is not None else "N/A"
        print(f" {status} | best_ir={ir_str} | {n_valid}/{n_calls} valid | {elapsed:.1f}s")

        results.append({
            "seed_name": seed_name,
            "seed_ir": seed_ir,
            "n_calls": n_calls,
            "n_valid": n_valid,
            "best_ir": best_ir,
            "beat_seed": beat_seed,
            "pass_at": pass_at,
            "all_irs": irs,
            "all_results": all_tool_results,
        })

    # Compute aggregate metrics
    n_seeds = len(results)
    vr_per_seed = [r["n_valid"] / r["n_calls"] if r["n_calls"] > 0 else 0 for r in results]
    vr = sum(vr_per_seed) / len(vr_per_seed) if vr_per_seed else 0

    pass_at_3 = sum(1 for r in results if r.get("pass_at", {}).get(2, False)) / n_seeds if n_seeds > 0 else 0
    # pass_at_5 not applicable since we only do 3 turns, use overall beat rate
    beat_rate = sum(1 for r in results if r["beat_seed"]) / n_seeds if n_seeds > 0 else 0

    # Mean IR improvement for seeds that were beaten
    beaten = [r for r in results if r["beat_seed"]]
    mean_ir_improvement = 0
    if beaten:
        improvements = [r["best_ir"] - max(0, r["seed_ir"]) for r in beaten]
        mean_ir_improvement = sum(improvements) / len(improvements)

    metrics = {
        "n_seeds": n_seeds,
        "vr": round(vr, 4),
        "pass_at_3": round(pass_at_3, 4),
        "beat_rate": round(beat_rate, 4),
        "mean_ir_improvement": round(mean_ir_improvement, 4),
        "n_beaten": len(beaten),
    }

    return metrics, results


def main():
    parser = argparse.ArgumentParser(description="Evaluate AlphaAgentEvo checkpoints")
    parser.add_argument("--base-model", default="/home/dc_analyst/models/Qwen3-4B-Thinking-2507")
    parser.add_argument("--checkpoint", default=None, help="Path to LoRA checkpoint dir")
    parser.add_argument("--data", required=True, help="Path to parquet file (val or test)")
    parser.add_argument("--max-turns", type=int, default=3)
    parser.add_argument("--max-new-tokens", type=int, default=5000)
    parser.add_argument("--label", default="", help="Label for this evaluation run")
    args = parser.parse_args()

    label = args.label or (Path(args.checkpoint).name if args.checkpoint else "base")
    print(f"\n{'='*60}")
    print(f"EVALUATION: {label}")
    print(f"{'='*60}")
    print(f"Base model: {args.base_model}")
    print(f"Checkpoint: {args.checkpoint or 'NONE (base model)'}")
    print(f"Data: {args.data}")
    print(f"Max turns: {args.max_turns}")
    print()

    # Load data
    seeds_df = pd.read_parquet(args.data)
    print(f"Loaded {len(seeds_df)} seeds")

    # Load model
    model, tokenizer = load_model(args.base_model, args.checkpoint)

    # Build tool schema
    tool_schema = build_tool_schema()

    # Evaluate
    t0 = time.time()
    metrics, details = evaluate_checkpoint(
        model, tokenizer, seeds_df, tool_schema,
        max_turns=args.max_turns,
        max_new_tokens=args.max_new_tokens,
    )
    elapsed = time.time() - t0

    # Print results
    print(f"\n{'='*60}")
    print(f"RESULTS: {label}")
    print(f"{'='*60}")
    print(f"  Seeds:              {metrics['n_seeds']}")
    print(f"  Valid Ratio (VR):   {metrics['vr']:.4f}")
    print(f"  Pass@3:             {metrics['pass_at_3']:.4f}")
    print(f"  Beat Rate (overall):{metrics['beat_rate']:.4f}")
    print(f"  Seeds beaten:       {metrics['n_beaten']}/{metrics['n_seeds']}")
    print(f"  Mean IR improvement:{metrics['mean_ir_improvement']:.4f}")
    print(f"  Total time:         {elapsed:.1f}s ({elapsed/len(seeds_df):.1f}s/seed)")
    print()

    # Per-seed details
    print("Per-seed results:")
    print(f"  {'Seed':<20} {'Seed IR':>8} {'Best IR':>8} {'Valid':>6} {'Beat':>5}")
    print(f"  {'-'*20} {'-'*8} {'-'*8} {'-'*6} {'-'*5}")
    for r in details:
        best = f"{r['best_ir']:.4f}" if r['best_ir'] is not None else "N/A"
        valid = f"{r['n_valid']}/{r['n_calls']}" if r['n_calls'] > 0 else "0/0"
        beat = "YES" if r['beat_seed'] else "no"
        print(f"  {r['seed_name']:<20} {r['seed_ir']:>8.4f} {best:>8} {valid:>6} {beat:>5}")

    # Save results
    output_dir = PROJECT_ROOT / "eval_results"
    output_dir.mkdir(exist_ok=True)
    data_name = Path(args.data).stem
    out_file = output_dir / f"eval_{label}_{data_name}.json"
    with open(out_file, "w") as f:
        # Convert non-serializable items
        save_details = []
        for r in details:
            save_r = {k: v for k, v in r.items() if k != "all_results"}
            save_r["pass_at"] = {str(k): v for k, v in r.get("pass_at", {}).items()}
            save_details.append(save_r)
        json.dump({"metrics": metrics, "details": save_details, "config": vars(args)}, f, indent=2, default=str)
    print(f"\nResults saved to {out_file}")

    return metrics


if __name__ == "__main__":
    main()
