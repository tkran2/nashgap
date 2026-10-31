"""GRPO with rewards computed exactly by the game solver.

A prompt is an information set; a completion is one action token. For each prompt in a
batch, G actions are sampled from the current policy, each is rewarded with its exact
expected payoff Q(I, a) in chips, and advantages are the rewards normalized within
the group, as in GRPO. The loss adds a KL penalty to the initial model, computed
exactly over the legal actions.

What changes between runs is only which opponent and continuation define Q:

  nash_q    Q under the equilibrium profile (a fixed table). The obvious "use the
            solver's values as reward" design. At equilibrium every action in the
            support has the same value, so this reward cannot teach mixing.
  br_q      Q against a best response to the current policy, with the policy's own
            continuation, recomputed every `refresh` steps. Its gradient is the
            exploitability gradient (exploitability descent, Lockhart et al. 2019).
  selfplay  Q against a frozen copy of the current policy, refreshed every
            `refresh` steps. The standard self-play setup.
  distill   Not RL: cross-entropy to the equilibrium strategy (an upper reference).
"""

from __future__ import annotations

import json
import math
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import torch

from .games import GameTree
from .solvers import action_values, best_response, combine, exploitability

REWARDS = ("nash_q", "br_q", "selfplay", "distill")


@dataclass
class TrainConfig:
    reward: str = "br_q"
    steps: int = 200
    batch_size: int = 32
    group_size: int = 8
    lr: float = 2e-4
    beta: float = 0.0  # KL penalty to the initial policy
    clip: float = 0.2
    refresh: int = 5
    eval_every: int = 10
    seed: int = 0
    extra: dict = field(default_factory=dict)


def q_table(tree: GameTree, reward: str, current: np.ndarray, nash: np.ndarray):
    """Exact Q(I, a) for every infoset's owner, and which infosets have a defined Q."""
    q = np.zeros_like(current)
    valid = np.zeros(tree.num_infosets, dtype=bool)
    owner = tree.infoset_player
    for p in (0, 1):
        if reward == "nash_q":
            profile = nash
        elif reward == "selfplay":
            profile = current
        elif reward == "br_q":
            br, _ = best_response(tree, current, 1 - p)
            profile = combine(tree, current, br) if p == 0 else combine(tree, br, current)
        else:
            raise ValueError(reward)
        av = action_values(tree, profile, p)
        rows = owner == p
        q[rows] = av.q[rows]
        valid[rows] = av.reachable[rows]
    return q, valid


class TabularPolicy(torch.nn.Module):
    """Softmax over a free logit per (infoset, action): the idealized policy class."""

    def __init__(self, tree: GameTree, init: np.ndarray | None = None):
        super().__init__()
        self.tree = tree
        self.legal = torch.from_numpy(tree.legal_mask)
        start = np.log(np.clip(init, 1e-6, 1.0)) if init is not None else np.zeros((tree.num_infosets, 3))
        self.logits = torch.nn.Parameter(torch.tensor(start, dtype=torch.float32))

    def action_logprobs(self, indices, grad: bool = False, full_vocab: bool = False):
        idx = torch.as_tensor(list(indices))
        ctx = torch.enable_grad() if grad else torch.no_grad()
        with ctx:
            z = self.logits[idx].masked_fill(~self.legal[idx], float("-inf"))
            return torch.log_softmax(z, dim=-1), None

    def strategy(self, with_mass: bool = False):
        logp, _ = self.action_logprobs(range(self.tree.num_infosets))
        s = torch.exp(logp).double().numpy()
        return (s, np.ones(len(s))) if with_mass else s


def grpo_loss(logp: torch.Tensor, ref_logp: torch.Tensor, q: torch.Tensor, cfg: TrainConfig, gen: torch.Generator):
    """GRPO surrogate for single-token completions. Shapes: [B, 3]."""
    probs_old = torch.exp(logp.detach()).nan_to_num(0.0)
    actions = torch.multinomial(probs_old, cfg.group_size, replacement=True, generator=gen)  # [B, G]
    rewards = torch.gather(q, 1, actions)
    adv = (rewards - rewards.mean(1, keepdim=True)) / (rewards.std(1, keepdim=True) + 1e-6)
    lp = torch.gather(logp, 1, actions)
    ratio = torch.exp(lp - lp.detach())
    surrogate = torch.minimum(ratio * adv, torch.clamp(ratio, 1 - cfg.clip, 1 + cfg.clip) * adv)
    legal = torch.isfinite(logp) & torch.isfinite(ref_logp)
    lp_safe = torch.where(legal, logp, 0.0)
    ref_safe = torch.where(legal, ref_logp, 0.0)
    kl = (torch.exp(lp_safe) * (lp_safe - ref_safe) * legal).sum(-1)  # exact KL(pi || ref) over legal actions
    loss = -surrogate.mean() + cfg.beta * kl.mean()
    zero_var = (rewards.std(1) < 1e-9).float().mean()
    return loss, {"reward": rewards.mean().item(), "kl": kl.mean().item(), "zero_var_groups": zero_var.item()}


def distill_loss(logp: torch.Tensor, target: torch.Tensor):
    safe = torch.where(torch.isfinite(logp), logp, 0.0)
    return -(target * safe).sum(-1).mean(), {}


def entropy(strategy: np.ndarray) -> float:
    s = np.clip(strategy, 1e-12, 1.0)
    return float(-(strategy * np.log(s)).sum(1).mean())


def train(policy, params, tree: GameTree, nash: np.ndarray, cfg: TrainConfig, out_dir: Path | None = None,
          log=print):
    torch.manual_seed(cfg.seed)
    gen = torch.Generator().manual_seed(cfg.seed)
    rng = np.random.default_rng(cfg.seed)
    opt = torch.optim.AdamW(params, lr=cfg.lr, weight_decay=0.0)
    nash_q, nash_valid = q_table(tree, "nash_q", nash, nash)
    nash_t = torch.from_numpy(nash).float()

    def evaluate(step: int, strat: np.ndarray, extra: dict):
        e = exploitability(tree, strat)
        g = np.zeros_like(strat)
        g[np.arange(len(strat)), strat.argmax(1)] = 1.0
        rec = {
            "step": step,
            "exploitability": e.milli_antes,
            "greedy_exploitability": exploitability(tree, g).milli_antes,
            "nash_q_reward": float((strat * nash_q).sum(1)[nash_valid].mean()),
            "entropy": entropy(strat),
            "time": time.time() - t0,
            **extra,
        }
        log(json.dumps({k: (round(v, 5) if isinstance(v, float) else v) for k, v in rec.items()}))
        if out_dir is not None:
            with open(out_dir / "log.jsonl", "a") as f:
                f.write(json.dumps(rec) + "\n")
        return rec

    if out_dir is not None:
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "config.json").write_text(json.dumps(asdict(cfg), indent=2))
        (out_dir / "log.jsonl").write_text("")

    t0 = time.time()
    strat = policy.strategy()
    ref_logp_all, _ = policy.action_logprobs(range(tree.num_infosets))
    ref_logp_all = ref_logp_all.detach()
    history = [evaluate(0, strat, {})]
    strat_step = 0  # the step whose weights produced `strat`
    q = valid = None
    running = {}
    for step in range(1, cfg.steps + 1):
        if cfg.reward != "distill" and (q is None or (cfg.reward != "nash_q" and (step - 1) % cfg.refresh == 0)):
            if strat_step != step - 1:
                strat, strat_step = policy.strategy(), step - 1
            q, valid = q_table(tree, cfg.reward, strat, nash)
            q_t = torch.from_numpy(q).float()
        pool = np.flatnonzero(valid) if cfg.reward != "distill" else np.arange(tree.num_infosets)
        idx = rng.choice(pool, size=min(cfg.batch_size, len(pool)), replace=False)
        logp, _ = policy.action_logprobs(idx, grad=True)
        if cfg.reward == "distill":
            loss, stats = distill_loss(logp, nash_t[idx])
        else:
            loss, stats = grpo_loss(logp, ref_logp_all[idx], q_t[idx], cfg, gen)
        opt.zero_grad()
        loss.backward()
        opt.step()
        for k, v in stats.items():
            running[k] = running.get(k, 0.0) + v
        running["n"] = running.get("n", 0) + 1
        if step % cfg.eval_every == 0 or step == cfg.steps:
            strat, strat_step = policy.strategy(), step
            n = running.pop("n")
            history.append(evaluate(step, strat, {f"train_{k}": v / n for k, v in running.items()}))
            running = {}
    return history, policy.strategy()
