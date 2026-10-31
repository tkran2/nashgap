"""Is the private card linearly decodable from the model's state at the answer position?

For every Leduc information set, take the final hidden state where the model emits its
action word, and fit a cross-validated logistic-regression probe to predict the
player's own rank (3 classes). High probe accuracy with near-zero card sensitivity
means the information is present but unused by the action readout.

Usage: python scripts/probe_card.py [--model Qwen/Qwen2.5-0.5B-Instruct] [--adapter results/llm/.../adapter]
Writes results/probe_<name>.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from transformers import DynamicCache

from nashgap.games import build_tree
from nashgap.llm_policy import LLMPolicy, load_model
from nashgap.prompts import FRAMINGS

ROOT = Path(__file__).resolve().parents[1]


def hidden_states(policy: LLMPolicy) -> np.ndarray:
    out = []
    with torch.no_grad():
        prefix_kv = policy._prefix_cache()
        P = len(policy.prefix)
        for i in range(policy.tree.num_infosets):
            ids = policy.suffixes[i][None]
            cache = DynamicCache(ddp_cache_data=[(k, v) for k, v in prefix_kv])
            pos = (P + torch.arange(ids.shape[1]))[None]
            h = policy._decoder()(input_ids=ids, position_ids=pos, past_key_values=cache, use_cache=True).last_hidden_state
            out.append(h[0, -1].float().numpy())
    return np.stack(out)


def probe(X: np.ndarray, y: np.ndarray, folds: int = 6, seed: int = 0, l2: float = 1.0) -> float:
    """Multinomial logistic regression by gradient descent, k-fold cross-validated accuracy."""
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(y))
    X = (X - X.mean(0)) / (X.std(0) + 1e-6)
    k = y.max() + 1
    correct = 0
    for f in range(folds):
        test = idx[f::folds]
        train = np.setdiff1d(idx, test)
        W = np.zeros((X.shape[1], k))
        b = np.zeros(k)
        Y = np.eye(k)[y[train]]
        for _ in range(500):
            z = X[train] @ W + b
            p = np.exp(z - z.max(1, keepdims=True))
            p /= p.sum(1, keepdims=True)
            g = (p - Y) / len(train)
            W -= 0.1 * (X[train].T @ g + l2 * W / len(train))
            b -= 0.1 * g.sum(0)
        correct += int(((X[test] @ W + b).argmax(1) == y[test]).sum())
    return correct / len(y)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    ap.add_argument("--adapter", default="")
    ap.add_argument("--framing", default="poker")
    args = ap.parse_args()
    torch.set_num_threads(4)
    model, tok = load_model(args.model, "float32")
    if args.adapter:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, args.adapter)
    tree = build_tree("leduc")
    policy = LLMPolicy(model, tok, tree, FRAMINGS[args.framing])
    X = hidden_states(policy)
    own = np.array([s.own_rank for s in tree.infosets])
    paired = np.array([int(s.public_rank is not None and s.public_rank == s.own_rank) for s in tree.infosets])
    shuffled = np.random.default_rng(1).permutation(own)
    res = {
        "model": args.model,
        "adapter": args.adapter or None,
        "framing": args.framing,
        "probe_accuracy_own_rank": probe(X, own),
        "probe_accuracy_shuffled_labels": probe(X, shuffled),
        "chance_own_rank": float(np.bincount(own).max() / len(own)),
        "probe_accuracy_paired": probe(X, paired),
        "chance_paired": float(np.bincount(paired).max() / len(paired)),
    }
    name = (Path(args.adapter).parent.name if args.adapter else args.model.split("/")[-1]) + f"_{args.framing}"
    (ROOT / "results" / f"probe_{name}.json").write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
