# AlphaAgentEvo v2: Close 4 Gaps with Original Codebase

## Summary

Implement 4 upgrades to match the original AlphaAgentEvo paper/codebase:
1. Multi-turn tool calling with real tool execution between turns
2. Loss masking for tool/observation tokens
3. Scale dataset from 66 → 500+ factors
4. Reward formula matching paper equation (5) with exact coefficients

## Gap 1: Multi-turn Tool Calling

**Goal**: Replace single-turn generation with real multi-turn: generate → parse tool_call → execute tool → inject tool_response → resume generation (max 2 turns, up to 4 tool calls per turn).

**Approach**: Write a custom GRPO training loop instead of using trl's GRPOTrainer. This gives us full control over generation (multi-turn), loss masking, and advantage computation.

**New file: `training/multi_turn_gen.py`**

```
generate_multi_turn(model, tokenizer, prompt_ids, max_turns=2, max_calls_per_turn=4):
    full_ids = prompt_ids
    completion_ids = []
    loss_mask = []  # 1=model token, 0=injected token

    for turn in range(max_turns):
        # Generate until </tool_call> or EOS
        output_ids = model.generate(full_ids, stopping_criteria=[ToolCallStopper])
        new_tokens = output_ids[len(full_ids):]
        completion_ids.extend(new_tokens)
        loss_mask.extend([1] * len(new_tokens))  # model-generated: train

        # Check if tool call was produced
        text = tokenizer.decode(new_tokens)
        tool_calls = parse_tool_calls(text)

        if not tool_calls:
            break  # EOS or no tool call → done

        # Execute each tool call and build response
        for call in tool_calls:
            result = factor_tool.evaluate(call["factor_name"], call["factor_expr"])
            tool_response = format_tool_response(result)

        # Inject tool response tokens (masked from loss)
        response_text = f"\n<tool_response>\n{tool_response}\n</tool_response>\n"
        response_ids = tokenizer.encode(response_text, add_special_tokens=False)
        completion_ids.extend(response_ids)
        loss_mask.extend([0] * len(response_ids))  # injected: mask

        full_ids = prompt_ids + completion_ids  # full context for next turn

    return completion_ids, loss_mask, trajectory_results
```

**Stopping strategy**: Use a custom `StoppingCriteria` that detects `</tool_call>` token sequence in the generated output.

## Gap 2: Loss Masking

**Goal**: Only train on model-generated tokens. Mask out tool_response tokens injected between turns.

**Integrated into the custom GRPO loop** (new file: `training/grpo_trainer.py`):

```
# Per-token policy loss with mask
log_probs = compute_log_probs(model, prompt_ids + completion_ids)
ref_log_probs = compute_log_probs(ref_model, prompt_ids + completion_ids)  # LoRA disabled

# Apply loss mask: only train on model-generated tokens
masked_log_probs = log_probs * loss_mask
masked_ref_log_probs = ref_log_probs * loss_mask

# GRPO loss with KL penalty
ratio = exp(masked_log_probs - masked_ref_log_probs)
clipped_ratio = clip(ratio, 1-epsilon, 1+epsilon)
loss = -min(ratio * advantage, clipped_ratio * advantage) + beta * KL
loss = (loss * loss_mask).sum() / loss_mask.sum()
```

**Reference model**: With LoRA, disable adapters to get base model logits (no extra VRAM).

## Gap 3: Dataset Scale (66 → 500+)

**Goal**: Build AlphaEvo500-equivalent dataset with 300 train / 30 val / 100 test.

**Source**: `/home/dc_analyst/Ha/AlphaAgent/QuantaAlpha-main/data/factorlib/all_factors_library.json` (836 unique factors with expressions).

**New file: `training/generate_dataset_v2.py`**

Steps:
1. Load 836 factors from all_factors_library.json
2. Parse each expression to validate it works with our parser
3. Optionally evaluate against Vietnam market via backtest API
4. Deduplicate using AST similarity (remove sim > 0.8)
5. Select 500 factors:
   - Prioritize diversity (low avg pairwise AST similarity)
   - Include variety of complexity levels
6. Split: 300 train / 30 val / 100 test (+ 70 reserve)
7. Generate conversation prompts using system_prompt.md template

**Update system prompt** to support multi-turn by adding instructions about receiving tool responses and iterating.

## Gap 4: Reward Formula (Paper Equation 5)

**Goal**: Replace our additive reward with the paper's hierarchical formula.

**Paper formula** (equation 5):
```
R(τ) = [min(R_cons, C_cons) + min(R_expl, C_expl)] / min(R_tool, C_tool)
      + min(R_perf, C_perf) · min(R_streak, C_streak)
```

**Exact coefficients** (Appendix D):
- Caps: C_tool=1, C_cons=0.2, C_expl=0.3, C_perf=0.5, C_streak=0.6
- Weights: α_succ=0.1, α_fail=0.2, α_cons=0.02, α_exp=0.02, α_perf=0.1, α_streak=0.15

**Component formulas** (from paper Section 2.3):
- R_tool = α_succ · N_succ - α_fail · N_fail
- R_cons = Σ_{f_i ∈ F_succ} α_cons · 1[sim(f_i, f_seed) > h_low]  (h_low=0.1)
- R_expl = α_exp · (1 - Σ_{f_i ∈ F_succ} max_{f_j ∈ F_{<i}} sim(f_i, f_j))
- R_perf = α_perf · log(1 + exp(s(f*) - max(0, s(f_seed))))
- R_streak = α_streak · N_streak
- sim(f_i, f_j) = |AST(f_i) ∩ AST(f_j)| / max(|AST(f_i)|, |AST(f_j)|)

**Safety**: Add floor `max(R_tool, 0.01)` to prevent div-by-zero (code bug noted in BigPicture doc).

**Rewrite**: `training/factor_tool.py` → update `calc_reward()` method.

## Gap 5 (bonus): Custom GRPO Training Loop

**Goal**: Replace trl.GRPOTrainer with custom loop for full control.

**New file: `training/grpo_trainer.py`**

Core loop per training step:
```
1. Sample batch of prompts (batch_size prompts)
2. For each prompt, generate num_generations multi-turn completions
   → Returns: completion_ids[], loss_mask[], trajectory_results[]
3. Compute rewards using updated calc_reward()
4. Compute GRPO advantages (group-normalize by prompt):
   advantages[i] = (reward[i] - group_mean) / (group_std + eps)
5. Compute per-token log_probs for completions (model + ref_model)
6. Compute clipped policy gradient loss with KL penalty:
   loss = -min(ratio * adv, clip(ratio) * adv) * mask + β * KL * mask
   loss = loss.sum() / mask.sum()
7. Backward pass + gradient accumulation + optimizer step
```

**Training config additions** (grpo_config.yaml):
```yaml
training:
  max_turns: 2              # max assistant turns per trajectory
  max_calls_per_turn: 4     # max tool calls per turn
  kl_coef: 0.001            # β for KL penalty
  clip_range: 0.2           # PPO clipping
  warmup_ratio: 0.1         # LR warmup
```

**Memory management** (RTX 5090 32GB):
- Model ~8GB (bf16) + LoRA ~200MB
- KV cache ~4-6GB for multi-turn (longer contexts)
- Gradient ~4GB with accumulation
- Total ~18-20GB → fits in 32GB
- Use gradient checkpointing if needed

## File Changes Summary

| File | Action | Description |
|------|--------|-------------|
| `training/multi_turn_gen.py` | CREATE | Multi-turn generation with tool execution |
| `training/grpo_trainer.py` | CREATE | Custom GRPO training loop with loss masking |
| `training/factor_tool.py` | REWRITE | Paper equation (5) reward formula |
| `training/generate_dataset_v2.py` | CREATE | Dataset generator for 500+ factors |
| `training/train.py` | REWRITE | Use custom GRPO trainer instead of trl |
| `training/system_prompt.md` | EDIT | Add multi-turn instructions |
| `configs/grpo_config.yaml` | EDIT | Add multi-turn + KL + warmup params |

## Execution Order

1. **Reward formula** (factor_tool.py) — self-contained, easy to test
2. **Dataset generation** (generate_dataset_v2.py) — needs backtest API running
3. **Multi-turn generation** (multi_turn_gen.py) — core new capability
4. **Custom GRPO trainer** (grpo_trainer.py + train.py rewrite) — integrates everything
5. **Config + prompt updates** — finalize parameters
6. **Smoke test** — verify 5-step training works end-to-end
