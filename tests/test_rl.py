import numpy as np
import pytest

torch = pytest.importorskip("torch")

from nashgap.games import build_tree  # noqa: E402
from nashgap.lp import lp_equilibrium  # noqa: E402
from nashgap.rl import TabularPolicy, TrainConfig, grpo_loss, q_table, train  # noqa: E402
from nashgap.solvers import action_values, exploitability, uniform_strategy  # noqa: E402


@pytest.fixture(scope="module")
def leduc():
    tree = build_tree("leduc")
    _, nash = lp_equilibrium(tree)
    return tree, nash


def test_nash_q_is_indifferent_on_the_support(leduc):
    """The reason solver values fail as a reward: every supported action scores the same."""
    tree, nash = leduc
    q, valid = q_table(tree, "nash_q", nash, nash)
    own = np.zeros(tree.num_infosets)
    for p in (0, 1):  # indifference holds where the equilibrium itself can arrive
        rows = tree.infoset_player == p
        own[rows] = action_values(tree, nash, p).own_reach[rows]
    spread = [np.ptp(q[i, nash[i] > 1e-6]) for i in range(tree.num_infosets)
              if valid[i] and own[i] > 1e-9 and (nash[i] > 1e-6).sum() > 1]
    assert spread and max(spread) < 1e-6


def test_zero_variance_groups_give_no_gradient():
    """A deterministic policy samples identical groups, so every advantage is zero."""
    logits = torch.tensor([[0.0, 30.0, -30.0]] * 4, requires_grad=True)
    logp = torch.log_softmax(logits, dim=-1)
    q = torch.tensor([[0.0, -1.0, 5.0]] * 4)
    loss, stats = grpo_loss(logp, logp.detach(), q, TrainConfig(group_size=8), torch.Generator().manual_seed(0))
    loss.backward()
    assert stats["zero_var_groups"] == 1.0
    assert logits.grad.abs().max().item() < 1e-6


def run(tree, nash, reward, steps):
    policy = TabularPolicy(tree)
    cfg = TrainConfig(reward=reward, steps=steps, lr=0.05, eval_every=steps, refresh=5, seed=0)
    hist, final = train(policy, [policy.logits], tree, nash, cfg, None, log=lambda s: None)
    return hist, final


def test_best_response_reward_reduces_exploitability(leduc):
    tree, nash = leduc
    start = exploitability(tree, uniform_strategy(tree)).milli_antes
    hist, final = run(tree, nash, "br_q", 200)
    assert exploitability(tree, final).milli_antes < 0.4 * start


def test_nash_q_reward_rises_without_fixing_exploitability(leduc):
    tree, nash = leduc
    hist, final = run(tree, nash, "nash_q", 300)
    assert hist[-1]["nash_q_reward"] > hist[0]["nash_q_reward"] + 0.5
    assert exploitability(tree, final).milli_antes > 1000


def test_distillation_approaches_equilibrium(leduc):
    tree, nash = leduc
    hist, final = run(tree, nash, "distill", 300)
    assert exploitability(tree, final).milli_antes < 200
