"""GRPO with each reward design on the idealized tabular policy (one free logit per action).

Runs in seconds and shows what each reward does when the policy class is not a
constraint. Writes results/tabular/<reward>_seed<k>/log.jsonl.
"""

import argparse
from pathlib import Path

import numpy as np

from nashgap.games import build_tree
from nashgap.lp import lp_equilibrium
from nashgap.rl import REWARDS, TabularPolicy, TrainConfig, train

OUT = Path(__file__).resolve().parents[1] / "results" / "tabular"

ap = argparse.ArgumentParser()
ap.add_argument("--game", default="leduc")
ap.add_argument("--rewards", nargs="+", default=list(REWARDS))
ap.add_argument("--seeds", type=int, default=3)
ap.add_argument("--steps", type=int, default=3000)
ap.add_argument("--lr", type=float, default=0.05)
ap.add_argument("--init", default="", help="path to an .npz strategy to start from (key=leduc_poker)")
args = ap.parse_args()

tree = build_tree(args.game)
_, nash = lp_equilibrium(tree)
init = None
if args.init:
    init = np.load(args.init)[f"{args.game}_poker"]
for reward in args.rewards:
    for seed in range(args.seeds):
        cfg = TrainConfig(reward=reward, steps=args.steps, lr=args.lr, eval_every=100, refresh=5, seed=seed,
                          extra={"policy": "tabular", "game": args.game, "init": args.init or "uniform"})
        policy = TabularPolicy(tree, init)
        hist, _ = train(policy, [policy.logits], tree, nash, cfg, OUT / f"{reward}_seed{seed}", log=lambda s: None)
        e = [h["exploitability"] for h in hist]
        print(f"{reward:9s} seed {seed}: exploitability {e[0]:7.1f} -> min {min(e):7.1f} -> final {e[-1]:7.1f}  "
              f"nash_q_reward {hist[-1]['nash_q_reward']:+.4f}  entropy {hist[-1]['entropy']:.3f}", flush=True)
