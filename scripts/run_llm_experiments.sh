#!/bin/sh
# The four reward designs on Qwen2.5-0.5B-Instruct (LoRA, top 8 layers), then the
# remaining base-model evaluations. Runs sequentially; each writes its own results.
set -e
cd "$(dirname "$0")/.."
export PYTHONPATH=.
for reward in br_q nash_q selfplay distill; do
  python3 scripts/train_llm.py --reward "$reward" --steps 100 --refresh 10 --batch-size 32 --lr 2e-4 \
    > "logs/train_${reward}.log" 2>&1
done
python3 scripts/eval_models.py Qwen/Qwen3-0.6B allenai/OLMo-2-0425-1B-Instruct Qwen/Qwen3-1.7B Qwen/Qwen2.5-3B-Instruct \
  > logs/eval_models_2.log 2>&1
