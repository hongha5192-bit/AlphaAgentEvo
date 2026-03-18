"""GRPO training script for AlphaAgentEvo.

Uses trl.GRPOTrainer with Qwen3-4B + LoRA on a single RTX 5090 (32GB).
Leverages trl 0.28.0's native multi-turn tool calling for factor evolution.

Usage:
    python training/train.py [--config configs/grpo_config.yaml]
"""

import argparse
import json
import re
import sys
import time
from pathlib import Path

import pandas as pd
import torch
import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from training.factor_tool import FactorTool

# ---------------------------------------------------------------------------
# Tool function — called by trl during multi-turn generation
# ---------------------------------------------------------------------------

_factor_tool: FactorTool | None = None


def _get_factor_tool(backtest_url: str = "http://localhost:8001") -> FactorTool:
    global _factor_tool
    if _factor_tool is None:
        _factor_tool = FactorTool(backtest_url=backtest_url)
    return _factor_tool


def evaluate_factor(factor_name: str = "unnamed", factor_expr: str = "") -> dict:
    """Evaluate a factor expression by backtesting against historical Vietnam stock market data. Returns the Information Ratio (IR), mean IC, and success status.

    Args:
        factor_name: A descriptive name for the factor being evaluated.
        factor_expr: The factor expression to evaluate using the operator DSL with variables like $close, $volume, etc. Example: RANK(TS_MEAN($return, 5)).
    """
    tool = _get_factor_tool()
    result = tool.evaluate(factor_name, factor_expr)
    return {
        "success": result["success"],
        "ir": round(result["ir"], 6),
        "ic_mean": round(result["ic_mean"], 6),
        "error": result.get("error"),
    }


# ---------------------------------------------------------------------------
# Config + data loading
# ---------------------------------------------------------------------------

def load_config(config_path: str | None = None) -> dict:
    """Load training configuration from YAML."""
    if config_path is None:
        config_path = PROJECT_ROOT / "configs" / "grpo_config.yaml"
    with open(config_path) as f:
        return yaml.safe_load(f)


def load_dataset(path: str | Path) -> list[dict]:
    """Load training dataset from parquet."""
    df = pd.read_parquet(path)
    records = []
    for _, row in df.iterrows():
        prompt = row["prompt"]
        if hasattr(prompt, 'tolist'):
            prompt = prompt.tolist()
        if isinstance(prompt, str):
            prompt = json.loads(prompt)
        records.append({
            "prompt": prompt,
            "seed_expr": row["seed_expr"],
            "seed_ir": float(row["seed_ir"]),
        })
    return records


# ---------------------------------------------------------------------------
# Reward function
# ---------------------------------------------------------------------------

def parse_trajectory_from_completion(completion) -> list[dict]:
    """Extract tool call + response pairs from a completion.

    Handles multiple formats:
    1. List of message dicts (structured): [{role: "assistant", tool_calls: [...]}, {role: "tool", content: ...}]
    2. String with serialized multi-turn (trl native tool calling log format):
       <tool_call>...</tool_call>\\nuser\\n<tool_response>...</tool_response>\\nassistant\\n...
    3. String with <tool_call> tags only (no responses — needs re-evaluation)

    Returns list of dicts with: success, ir, factor_expr
    """
    # Convert to text if it's a list of message dicts
    if isinstance(completion, list):
        # Structured messages — extract from role-based format
        return _parse_structured_messages(completion)
    elif isinstance(completion, dict):
        text = completion.get("content", "")
    else:
        text = str(completion)

    if not text:
        return []

    # Parse paired <tool_call> + <tool_response> from text
    return _parse_text_with_responses(text)


def _parse_structured_messages(messages: list[dict]) -> list[dict]:
    """Parse structured message dicts (list of {role, content, ...})."""
    results = []
    pending_exprs = []

    for msg in messages:
        role = msg.get("role", "")
        content = msg.get("content", "")

        if role == "assistant":
            # Check for structured tool_calls
            for tc in msg.get("tool_calls", []):
                func = tc.get("function", {})
                args = func.get("arguments", {})
                if isinstance(args, str):
                    try:
                        args = json.loads(args)
                    except json.JSONDecodeError:
                        args = {}
                expr = args.get("factor_expr", "")
                if expr:
                    pending_exprs.append(expr)

            # Also check text content for <tool_call> tags
            if isinstance(content, str):
                for match in re.finditer(r'<tool_call>\s*(.*?)\s*</tool_call>', content, re.DOTALL):
                    try:
                        data = json.loads(match.group(1))
                        args = data.get("arguments", data)
                        expr = args.get("factor_expr", "")
                        if expr:
                            pending_exprs.append(expr)
                    except json.JSONDecodeError:
                        continue

        elif role == "tool":
            # Parse tool response — may be JSON, Python repr, or dict
            if isinstance(content, dict):
                result_data = content
            elif isinstance(content, str):
                try:
                    result_data = json.loads(content)
                except json.JSONDecodeError:
                    # Handle Python repr format (single quotes, True/False/None)
                    try:
                        fixed = (content
                            .replace("'", '"')
                            .replace("True", "true")
                            .replace("False", "false")
                            .replace("None", "null")
                        )
                        result_data = json.loads(fixed)
                    except json.JSONDecodeError:
                        result_data = {}
            else:
                result_data = {}

            expr = pending_exprs.pop(0) if pending_exprs else ""
            results.append({
                "success": result_data.get("success", False),
                "ir": float(result_data.get("ir", 0.0)),
                "factor_expr": expr,
            })

    return results


def _parse_text_with_responses(text: str) -> list[dict]:
    """Parse text containing interleaved <tool_call> and <tool_response> blocks.

    trl serializes multi-turn as:
      <tool_call>{"name": "...", "arguments": {...}}</tool_call>
      user
      <tool_response>{'success': True, 'ir': 0.123, ...}</tool_response>
      assistant
      ...next tool call or final text...
    """
    results = []

    # Find all tool_call blocks
    call_pattern = re.compile(r'<tool_call>\s*(.*?)\s*</tool_call>', re.DOTALL)
    # Find all tool_response blocks (may use single quotes from Python repr)
    resp_pattern = re.compile(r'<tool_response>\s*(.*?)\s*</tool_response>', re.DOTALL)

    calls = list(call_pattern.finditer(text))
    responses = list(resp_pattern.finditer(text))

    # Pair calls with responses by position
    for i, call_match in enumerate(calls):
        # Extract expression from call
        expr = ""
        try:
            data = json.loads(call_match.group(1))
            args = data.get("arguments", data)
            expr = args.get("factor_expr", "")
        except json.JSONDecodeError:
            continue

        if not expr:
            continue

        # Find matching response
        if i < len(responses):
            resp_text = responses[i].group(1).strip()
            # Handle Python repr format (single quotes, True/False/None)
            resp_text = (resp_text
                .replace("'", '"')
                .replace("True", "true")
                .replace("False", "false")
                .replace("None", "null")
            )
            try:
                result_data = json.loads(resp_text)
                results.append({
                    "success": result_data.get("success", False),
                    "ir": float(result_data.get("ir", 0.0)),
                    "factor_expr": expr,
                })
            except json.JSONDecodeError:
                # Response couldn't be parsed — re-evaluate
                tool = _get_factor_tool()
                result = tool.evaluate("evolved", expr)
                result["factor_expr"] = expr
                results.append(result)
        else:
            # No response available — re-evaluate
            tool = _get_factor_tool()
            result = tool.evaluate("evolved", expr)
            result["factor_expr"] = expr
            results.append(result)

    return results


def create_reward_function(factor_tool: FactorTool):
    """Create a reward function compatible with trl GRPOTrainer.

    trl 0.28.0 calls: reward_func(prompts=..., completions=..., **dataset_kwargs)
    where dataset_kwargs includes seed_expr, seed_ir from the dataset columns.
    """

    def reward_func(prompts, completions, **kwargs):
        """Compute 5-dim hierarchical reward for each completion.

        Args:
            prompts: list of prompt conversations (list of message dicts)
            completions: list of completion conversations (list of message dicts)
            **kwargs: includes seed_expr (list[str]), seed_ir (list[float]) from dataset
        """
        seed_exprs = kwargs.get("seed_expr", [])
        seed_irs = kwargs.get("seed_ir", [])

        rewards = []
        for i, completion in enumerate(completions):
            # Get seed info for this example
            seed_expr = seed_exprs[i] if i < len(seed_exprs) else ""
            seed_ir = float(seed_irs[i]) if i < len(seed_irs) else 0.0

            try:
                # Parse tool results from the multi-turn conversation
                trajectory = parse_trajectory_from_completion(completion)

                if not trajectory:
                    rewards.append(0.0)
                    continue

                # Compute 5-dim reward
                reward = factor_tool.calc_reward(trajectory, seed_expr, seed_ir)
                rewards.append(reward)
            except Exception as e:
                print(f"[REWARD ERROR] completion {i}: {e}")
                rewards.append(0.0)

        return rewards

    return reward_func


# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="GRPO Training for AlphaAgentEvo")
    parser.add_argument("--config", default=str(PROJECT_ROOT / "configs" / "grpo_config.yaml"))
    parser.add_argument("--smoke-test", action="store_true", help="Run 5 steps only")
    args = parser.parse_args()

    config = load_config(args.config)
    model_config = config["model"]
    training_config = config["training"]
    lora_config = config.get("lora", {})

    print("=" * 60)
    print("AlphaAgentEvo GRPO Training")
    print("=" * 60)
    print(f"Model: {model_config['name']}")
    print(f"LoRA rank: {lora_config.get('r', 16)}")
    print(f"Batch size: {training_config.get('per_device_train_batch_size', 1)}")
    print(f"Num generations: {training_config.get('num_generations', 3)}")
    print(f"Tool calling iterations: {training_config.get('max_tool_calling_iterations', 3)}")
    print("=" * 60)

    # --- Import heavy dependencies ---
    from datasets import Dataset
    from peft import LoraConfig
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import GRPOConfig, GRPOTrainer

    # --- Load model + tokenizer ---
    model_name = model_config["name"]
    print(f"\n[1/5] Loading model: {model_name}")

    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True,
        padding_side="left",
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Qwen3 Thinking variants have a non-prefix-preserving chat template
    # (forces <think>\n on every assistant turn) which trl can't handle.
    # Use the base Qwen3 template instead — it already parses <think> blocks
    # in responses, and the model weights naturally produce them anyway.
    if "Thinking" in model_name or "thinking" in model_name:
        base_tokenizer = AutoTokenizer.from_pretrained(
            model_name.replace("-Thinking-2507", "").replace("-thinking", ""),
            trust_remote_code=True,
        )
        tokenizer.chat_template = base_tokenizer.chat_template
        del base_tokenizer

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
        attn_implementation="flash_attention_2" if model_config.get("flash_attn", False) else "sdpa",
    )

    # --- Apply LoRA ---
    print("[2/5] Applying LoRA...")
    peft_config = LoraConfig(
        r=lora_config.get("r", 16),
        lora_alpha=lora_config.get("alpha", 32),
        lora_dropout=lora_config.get("dropout", 0.05),
        target_modules=lora_config.get("target_modules", ["q_proj", "k_proj", "v_proj", "o_proj"]),
        bias="none",
        task_type="CAUSAL_LM",
    )

    # --- Load dataset ---
    print("[3/5] Loading dataset...")
    train_path = config.get("data", {}).get("train_path", str(PROJECT_ROOT / "data" / "train.parquet"))

    train_records = load_dataset(train_path)
    print(f"  Train: {len(train_records)} examples")

    if args.smoke_test:
        train_records = train_records[:5]
        training_config["max_steps"] = 5
        training_config["num_train_epochs"] = 1
        print("  [SMOKE TEST] Using 5 examples, 5 steps")

    # Convert to HuggingFace Dataset
    hf_dataset = Dataset.from_list([
        {
            "prompt": rec["prompt"],
            "seed_expr": rec["seed_expr"],
            "seed_ir": rec["seed_ir"],
        }
        for rec in train_records
    ])

    # --- Setup tools + reward function ---
    print("[4/5] Setting up tool calling + reward function...")
    backtest_url = config.get("backtest", {}).get("url", "http://localhost:8001")
    _get_factor_tool(backtest_url)  # Initialize the global tool
    factor_tool = _get_factor_tool()
    reward_fn = create_reward_function(factor_tool)

    # --- Configure and run GRPO ---
    print("[5/5] Starting GRPO training...")

    output_dir = str(PROJECT_ROOT / "outputs" / f"grpo_{int(time.time())}")

    num_gens = training_config.get("num_generations", 3)
    grpo_config = GRPOConfig(
        output_dir=output_dir,
        num_generations=num_gens,
        generation_batch_size=num_gens,
        max_completion_length=training_config.get("max_new_tokens", 3072),
        max_tool_calling_iterations=training_config.get("max_tool_calling_iterations", 3),
        per_device_train_batch_size=training_config.get("per_device_train_batch_size", 1),
        gradient_accumulation_steps=training_config.get("gradient_accumulation_steps", 4),
        num_train_epochs=training_config.get("num_train_epochs", 1),
        max_steps=training_config.get("max_steps", -1),
        learning_rate=float(training_config.get("learning_rate", 1e-6)),
        bf16=training_config.get("bf16", True),
        logging_steps=training_config.get("logging_steps", 1),
        save_steps=training_config.get("save_steps", 50),
        save_total_limit=training_config.get("save_total_limit", 3),
        report_to=training_config.get("report_to", "tensorboard"),
        log_completions=True,
        beta=float(training_config.get("beta", 0.001)),
        seed=training_config.get("seed", 42),
    )

    trainer = GRPOTrainer(
        model=model,
        args=grpo_config,
        train_dataset=hf_dataset,
        reward_funcs=reward_fn,
        processing_class=tokenizer,
        peft_config=peft_config,
        tools=[evaluate_factor],
    )

    print(f"\nOutput directory: {output_dir}")
    print("Training started...\n")

    trainer.train()

    # Save final model
    final_path = Path(output_dir) / "final"
    trainer.save_model(str(final_path))
    tokenizer.save_pretrained(str(final_path))
    print(f"\nTraining complete! Model saved to {final_path}")


if __name__ == "__main__":
    main()
