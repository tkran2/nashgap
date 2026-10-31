#!/bin/sh
# LLM experiments on Qwen2.5-0.5B-Instruct (LoRA rank 16 on the top 8 layers), CPU only.
# br_q at lr 1e-5 already ran as part of the learning-rate sweep (lr 2e-4 collapsed within 10 steps).
# Then the remaining base-model evaluations.
cd "$(dirname "$0")/.."
export PYTHONPATH=.
python3 scripts/train_llm.py --reward distill --steps 40 --refresh 10 --lr 2e-4 > logs/train_distill.log 2>&1
for reward in nash_q selfplay; do
  python3 scripts/train_llm.py --reward "$reward" --steps 30 --refresh 10 --lr 1e-5 --beta 0.04 > "logs/train_${reward}.log" 2>&1
done
python3 scripts/eval_models.py Qwen/Qwen3-0.6B allenai/OLMo-2-0425-1B-Instruct Qwen/Qwen3-1.7B Qwen/Qwen2.5-3B-Instruct \
  > logs/eval_models_2.log 2>&1
