"""Exact solvers on explicit game trees: values, best responses, exploitability, CFR.

A strategy is an array of shape [num_infosets, NUM_ACTIONS] whose rows are action
distributions (zero on illegal actions). One array holds both players' rows; which
player owns a row is `tree.infoset_player`.

Every pass is a level-by-level array operation over the tree (nodes are in preorder,
so a top-down pass visits levels in order and a bottom-up pass visits them in
reverse). Leduc has about 9,500 nodes, so one exact pass takes about a millisecond.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .games import CHANCE, DECISION, NUM_ACTIONS, TERMINAL, GameTree


def _decision_edges(tree: GameTree):
    """Nodes whose parent is a decision node, with that parent's infoset and the action taken."""
    cache = getattr(tree, "_edges", None)
    if cache is None:
        nodes = np.flatnonzero((tree.parent >= 0) & (tree.kind[np.maximum(tree.parent, 0)] == DECISION))
        parents = tree.parent[nodes]
        cache = (nodes, tree.infoset[parents], tree.action[nodes], tree.player[parents])
        tree._edges = cache
    return cache


def uniform_strategy(tree: GameTree) -> np.ndarray:
    legal = tree.legal_mask.astype(np.float64)
    return legal / legal.sum(axis=1, keepdims=True)


def normalize(tree: GameTree, strategy: np.ndarray) -> np.ndarray:
    """Zero illegal actions and renormalize; rows with no mass become uniform."""
    s = np.where(tree.legal_mask, np.maximum(strategy, 0.0), 0.0)
    total = s.sum(axis=1, keepdims=True)
    uni = uniform_strategy(tree)
    return np.where(total > 0, s / np.where(total > 0, total, 1.0), uni)


def combine(tree: GameTree, player0: np.ndarray, player1: np.ndarray) -> np.ndarray:
    """Rows of player 0 from `player0` and rows of player 1 from `player1`."""
    p = tree.infoset_player[:, None]
    return np.where(p == 0, player0, player1)


def edge_weights(tree: GameTree, strategy: np.ndarray) -> np.ndarray:
    w = tree.chance_prob.copy()
    nodes, infosets, actions, _ = _decision_edges(tree)
    w[nodes] = strategy[infosets, actions]
    return w


def reach_probabilities(tree: GameTree, strategy: np.ndarray):
    """Per-node reach split into chance, player 0 and player 1 factors."""
    n = tree.num_nodes
    chance = np.ones(n)
    player = np.ones((2, n))
    nodes, infosets, actions, owners = _decision_edges(tree)
    own_factor = np.ones((2, n))
    own_factor[owners, nodes] = strategy[infosets, actions]
    for lvl in tree.levels[1:]:
        par = tree.parent[lvl]
        chance[lvl] = chance[par] * tree.chance_prob[lvl]
        player[:, lvl] = player[:, par] * own_factor[:, lvl]
    return chance, player


def node_values(tree: GameTree, strategy: np.ndarray) -> np.ndarray:
    """Expected payoff to player 0 from every node when both players follow `strategy`."""
    w = edge_weights(tree, strategy)
    v = tree.utility.copy()
    for lvl in reversed(tree.levels[1:]):
        np.add.at(v, tree.parent[lvl], w[lvl] * v[lvl])
    return v


def expected_value(tree: GameTree, strategy: np.ndarray) -> float:
    """Expected payoff per hand to player 0."""
    return float(node_values(tree, strategy)[0])


@dataclass
class ActionValues:
    """Counterfactual action values for one player's infosets.

    cf[I, a] sums, over histories h in I, the probability that chance and the opponent
    bring play to h times the player's expected payoff after taking a at h.
    `opp_reach[I]` is the same sum without the payoff, so `q = cf / opp_reach` is the
    expected payoff of taking a at I, averaged over the hidden cards consistent with I.
    """

    cf: np.ndarray
    opp_reach: np.ndarray
    own_reach: np.ndarray

    @property
    def q(self) -> np.ndarray:
        safe = np.where(self.opp_reach > 0, self.opp_reach, 1.0)
        return self.cf / safe[:, None]

    @property
    def reachable(self) -> np.ndarray:
        return self.opp_reach > 0


def action_values(tree: GameTree, strategy: np.ndarray, player: int) -> ActionValues:
    chance, reach = reach_probabilities(tree, strategy)
    v = node_values(tree, strategy)
    if player == 1:
        v = -v
    opp = chance * reach[1 - player]
    dec = np.flatnonzero((tree.kind == DECISION) & (tree.player == player))
    I = tree.infoset[dec]
    cf = np.zeros((tree.num_infosets, NUM_ACTIONS))
    for a in range(NUM_ACTIONS):
        ch = tree.child[dec, a]
        ok = ch >= 0
        np.add.at(cf[:, a], I[ok], opp[dec[ok]] * v[ch[ok]])
    opp_reach = np.zeros(tree.num_infosets)
    np.add.at(opp_reach, I, opp[dec])
    own_reach = np.zeros(tree.num_infosets)
    own_reach[I] = reach[player][dec]  # identical across histories in an infoset (perfect recall)
    return ActionValues(cf=cf, opp_reach=opp_reach, own_reach=own_reach)


def best_response(tree: GameTree, strategy: np.ndarray, player: int) -> tuple[np.ndarray, float]:
    """Exact best response of `player` to the opponent's rows of `strategy`.

    Returns a pure strategy (only `player`'s rows are meaningful) and its expected
    payoff per hand. Histories in one infoset share a depth, so each level settles
    the responder's choices before any shallower level needs them.
    """
    chance, reach = reach_probabilities(tree, strategy)
    opp = chance * reach[1 - player]
    w = edge_weights(tree, strategy)
    sign = 1.0 if player == 0 else -1.0
    v = sign * tree.utility
    legal = tree.legal_mask
    br = np.zeros_like(strategy)
    is_mine = (tree.kind == DECISION) & (tree.player == player)
    parent_is_mine = np.zeros(tree.num_nodes, dtype=bool)
    parent_is_mine[1:] = is_mine[tree.parent[1:]]

    for lvl in reversed(tree.levels):
        mine = lvl[is_mine[lvl]]
        if mine.size:
            I = tree.infoset[mine]
            q = np.zeros((tree.num_infosets, NUM_ACTIONS))
            child_v = np.full((mine.size, NUM_ACTIONS), -np.inf)
            for a in range(NUM_ACTIONS):
                ch = tree.child[mine, a]
                ok = ch >= 0
                child_v[ok, a] = v[ch[ok]]
                np.add.at(q[:, a], I[ok], opp[mine[ok]] * v[ch[ok]])
            touched = np.unique(I)
            scores = np.where(legal[touched], q[touched], -np.inf)
            best = np.argmax(scores, axis=1)
            br[touched] = 0.0
            br[touched, best] = 1.0
            choice = np.zeros(tree.num_infosets, dtype=np.int64)
            choice[touched] = best
            v[mine] = child_v[np.arange(mine.size), choice[I]]
        if lvl[0] == 0:
            break
        push = lvl[~parent_is_mine[lvl]]
        np.add.at(v, tree.parent[push], w[push] * v[push])
    return br, float(v[0])


@dataclass
class Exploitability:
    br_value_p0: float  # what a best-responding player 0 wins against the strategy's player 1
    br_value_p1: float
    game_value_p0: float

    @property
    def nash_conv(self) -> float:
        return self.br_value_p0 + self.br_value_p1

    @property
    def exploitability(self) -> float:
        """Average gain of a best responder over the two seats, in chips (antes) per hand."""
        return self.nash_conv / 2.0

    @property
    def milli_antes(self) -> float:
        return 1000.0 * self.exploitability


def exploitability(tree: GameTree, strategy: np.ndarray) -> Exploitability:
    _, b0 = best_response(tree, strategy, 0)
    _, b1 = best_response(tree, strategy, 1)
    return Exploitability(b0, b1, expected_value(tree, strategy))


def regret_matching(tree: GameTree, regrets: np.ndarray) -> np.ndarray:
    pos = np.where(tree.legal_mask, np.maximum(regrets, 0.0), 0.0)
    total = pos.sum(axis=1, keepdims=True)
    return np.where(total > 0, pos / np.where(total > 0, total, 1.0), uniform_strategy(tree))


@dataclass
class SolveResult:
    average: np.ndarray
    current: np.ndarray
    history: list[tuple[int, float]]  # (iteration, exploitability in milli-antes)


def solve(
    tree: GameTree,
    iterations: int = 2000,
    method: str = "dcfr",
    alpha: float = 1.5,
    beta: float = 0.0,
    gamma: float = 2.0,
    log_every: int = 0,
) -> SolveResult:
    """Counterfactual regret minimization with alternating updates.

    method="cfr": vanilla CFR with uniform averaging.
    method="cfr+": regrets floored at zero, linearly weighted average.
    method="dcfr": discounted CFR (Brown & Sandholm 2019) with (alpha, beta, gamma).
    """
    legal = tree.legal_mask
    owner = tree.infoset_player
    regrets = np.zeros((tree.num_infosets, NUM_ACTIONS))
    avg_sum = np.zeros_like(regrets)
    history: list[tuple[int, float]] = []
    for t in range(1, iterations + 1):
        for p in (0, 1):
            rows = owner == p
            sigma = regret_matching(tree, regrets)
            av = action_values(tree, sigma, p)
            node_value = (sigma * av.cf).sum(axis=1, keepdims=True)
            inst = np.where(legal, av.cf - node_value, 0.0)
            if method == "dcfr":
                pos_w = t**alpha / (t**alpha + 1.0)
                neg_w = t**beta / (t**beta + 1.0)
                r = regrets[rows]
                regrets[rows] = np.where(r > 0, r * pos_w, r * neg_w) + inst[rows]
                avg_sum[rows] *= (t / (t + 1.0)) ** gamma
                avg_sum[rows] += av.own_reach[rows, None] * sigma[rows]
            elif method == "cfr+":
                regrets[rows] = np.maximum(regrets[rows] + inst[rows], 0.0)
                avg_sum[rows] += t * av.own_reach[rows, None] * sigma[rows]
            elif method == "cfr":
                regrets[rows] += inst[rows]
                avg_sum[rows] += av.own_reach[rows, None] * sigma[rows]
            else:
                raise ValueError(f"unknown method {method!r}")
        if log_every and (t % log_every == 0 or t == iterations):
            history.append((t, exploitability(tree, normalize(tree, avg_sum)).milli_antes))
    return SolveResult(
        average=normalize(tree, avg_sum),
        current=regret_matching(tree, regrets),
        history=history,
    )
