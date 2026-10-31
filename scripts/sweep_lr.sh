#!/bin/sh
# Short learning-rate sweep for LLM GRPO (br_q), after lr=2e-4 collapsed to a pure strategy in 10 steps.
cd "$(dirname "$0")/.."
export PYTHONPATH=.
for lr in 1e-5 3e-5; do
  python3 scripts/train_llm.py --reward br_q --steps 30 --refresh 10 --lr $lr --beta 0.04 --tag "_lr$lr" \
    > "logs/sweep_br_q_lr$lr.log" 2>&1
  mv "results/llm/Qwen2.5-0.5B-Instruct_br_q_poker_seed0_lr$lr" "results/llm_sweep/br_q_lr$lr"
done
