"""GRPO fine-tuning (LoRA) of a small open model with solver-computed rewards.

Example:
  python scripts/train_llm.py --reward br_q --steps 150
Writes results/llm/<model>_<reward>_<framing>_seed<k>/{config.json,log.jsonl,final_strategy.npz}
"""

import argparse
from pathlib import Path

import numpy as np
import torch
from peft import LoraConfig, get_peft_model

from nashgap.games import build_tree
from nashgap.llm_policy import LLMPolicy, load_model
from nashgap.lp import lp_equilibrium
from nashgap.prompts import FRAMINGS
from nashgap.rl import TrainConfig, train

OUT = Path(__file__).resolve().parents[1] / "results" / "llm"

ap = argparse.ArgumentParser()
ap.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
ap.add_argument("--reward", default="br_q", choices=["nash_q", "br_q", "selfplay", "distill"])
ap.add_argument("--game", default="leduc")
ap.add_argument("--framing", default="poker")
ap.add_argument("--steps", type=int, default=150)
ap.add_argument("--batch-size", type=int, default=32)
ap.add_argument("--group-size", type=int, default=8)
ap.add_argument("--lr", type=float, default=2e-4)
ap.add_argument("--beta", type=float, default=0.0)
ap.add_argument("--refresh", type=int, default=10)
ap.add_argument("--lora-rank", type=int, default=16)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--tag", default="")
args = ap.parse_args()

torch.set_num_threads(4)
model, tok = load_model(args.model, "float32")
lora = LoraConfig(r=args.lora_rank, lora_alpha=2 * args.lora_rank, lora_dropout=0.0,
                  target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"])
model = get_peft_model(model, lora)
model.train()
tree = build_tree(args.game)
_, nash = lp_equilibrium(tree)
chat_kwargs = {"enable_thinking": False} if "Qwen3" in args.model else {}
policy = LLMPolicy(model, tok, tree, FRAMINGS[args.framing], batch_size=args.batch_size, chat_kwargs=chat_kwargs)

name = f"{args.model.split('/')[-1]}_{args.reward}_{args.framing}_seed{args.seed}{args.tag}"
out = OUT / name
cfg = TrainConfig(reward=args.reward, steps=args.steps, batch_size=args.batch_size, group_size=args.group_size,
                  lr=args.lr, beta=args.beta, refresh=args.refresh, eval_every=args.refresh, seed=args.seed,
                  extra={"policy": "llm", "model": args.model, "game": args.game, "framing": args.framing,
                         "lora_rank": args.lora_rank})
params = [p for p in model.parameters() if p.requires_grad]
history, final = train(policy, params, tree, nash, cfg, out, log=lambda s: print(s, flush=True))
np.savez_compressed(out / "final_strategy.npz", strategy=final)
model.save_pretrained(out / "adapter")
