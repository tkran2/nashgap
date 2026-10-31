"""Exact exploitability of open-weight models in Kuhn and Leduc poker, in both framings.

Usage: python scripts/eval_models.py Qwen/Qwen2.5-0.5B-Instruct [more models...]
Writes results/eval/<model>.json and results/eval/<model>.npz (full strategies).
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from nashgap.analysis import behavior_summary
from nashgap.games import build_tree
from nashgap.llm_policy import LLMPolicy, greedy, load_model
from nashgap.lp import lp_equilibrium
from nashgap.prompts import FRAMINGS
from nashgap.solvers import combine, expected_value, exploitability

OUT = Path(__file__).resolve().parents[1] / "results" / "eval"


def slug(name: str) -> str:
    return name.split("/")[-1]


def evaluate(name: str, dtype: str, games: list[str], framings: list[str]) -> dict:
    model, tok = load_model(name, dtype)
    chat_kwargs = {"enable_thinking": False} if "Qwen3" in name else {}
    report = {"model": name, "dtype": dtype, "results": {}}
    arrays = {}
    for game in games:
        tree = build_tree(game)
        nash_value, nash = lp_equilibrium(tree)
        for fname in framings:
            t0 = time.time()
            pol = LLMPolicy(model, tok, tree, FRAMINGS[fname], chat_kwargs=chat_kwargs)
            strat, mass = pol.strategy(with_mass=True)
            g = greedy(strat)
            e, eg = exploitability(tree, strat), exploitability(tree, g)
            # Seat-averaged result of the model playing against the equilibrium.
            vs_nash = 0.5 * (expected_value(tree, combine(tree, strat, nash)) - nash_value) + \
                0.5 * (nash_value - expected_value(tree, combine(tree, nash, strat)))
            key = f"{game}/{fname}"
            report["results"][key] = {
                "exploitability_milli_antes": e.milli_antes,
                "greedy_exploitability_milli_antes": eg.milli_antes,
                "br_value_vs_model_as_p1": e.br_value_p0,
                "br_value_vs_model_as_p0": e.br_value_p1,
                "ev_vs_nash_milli_antes": 1000 * vs_nash,
                "format_mass_mean": float(mass.mean()),
                "format_mass_min": float(mass.min()),
                "prompt_tokens": pol.prompt_tokens,
                "behavior": behavior_summary(tree, strat),
                "seconds": time.time() - t0,
            }
            arrays[key.replace("/", "_")] = strat
            print(f"{slug(name):28s} {key:16s} expl={e.milli_antes:8.1f}  greedy={eg.milli_antes:8.1f}  "
                  f"mass={mass.mean():.3f}  ({time.time() - t0:.0f}s)", flush=True)
    del model
    return report, arrays


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("models", nargs="+")
    ap.add_argument("--dtype", default="auto", help="float32, bfloat16, or auto (bf16 above 2B params)")
    ap.add_argument("--games", nargs="+", default=["kuhn", "leduc"])
    ap.add_argument("--framings", nargs="+", default=["poker", "abstract"])
    args = ap.parse_args()
    torch.set_num_threads(4)
    OUT.mkdir(parents=True, exist_ok=True)
    for name in args.models:
        dtype = args.dtype
        if dtype == "auto":
            big = any(s in name for s in ("3B", "4B", "Phi-4-mini", "7B", "8B"))
            dtype = "bfloat16" if big else "float32"
        report, arrays = evaluate(name, dtype, args.games, args.framings)
        (OUT / f"{slug(name)}.json").write_text(json.dumps(report, indent=2))
        np.savez_compressed(OUT / f"{slug(name)}.npz", **arrays)


if __name__ == "__main__":
    main()
