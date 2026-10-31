"""Sequence-form linear program: an exact equilibrium independent of CFR.

Used to cross-check the CFR solver and the best-response code: the LP's game value
must match CFR's, and the LP's strategies must have (numerically) zero exploitability
under `solvers.exploitability`.
"""

from __future__ import annotations

import numpy as np
from scipy import sparse
from scipy.optimize import linprog

from .games import DECISION, NUM_ACTIONS, TERMINAL, GameTree
from .solvers import normalize, reach_probabilities


def _sequences(tree: GameTree, player: int):
    """Index the player's sequences (empty sequence = 0) and each infoset's parent sequence."""
    seq_index: dict[tuple[int, int], int] = {}
    for s in tree.infosets:
        if s.player != player:
            continue
        for a in range(NUM_ACTIONS):
            if s.legal[a]:
                seq_index[(s.index, a)] = len(seq_index) + 1
    node_seq = np.zeros(tree.num_nodes, dtype=np.int64)
    parent_seq = {}
    for n in range(1, tree.num_nodes):
        p = tree.parent[n]
        node_seq[n] = node_seq[p]
        if tree.kind[p] == DECISION and tree.player[p] == player:
            node_seq[n] = seq_index[(tree.infoset[p], tree.action[n])]
    for n in np.flatnonzero((tree.kind == DECISION) & (tree.player == player)):
        parent_seq.setdefault(int(tree.infoset[n]), int(node_seq[n]))
    return seq_index, node_seq, parent_seq


def _constraints(tree: GameTree, player: int, seq_index, parent_seq):
    infos = [s.index for s in tree.infosets if s.player == player]
    rows, cols, vals = [0], [0], [1.0]
    for r, I in enumerate(infos, start=1):
        rows.append(r)
        cols.append(parent_seq[I])
        vals.append(-1.0)
        for a in range(NUM_ACTIONS):
            if (I, a) in seq_index:
                rows.append(r)
                cols.append(seq_index[(I, a)])
                vals.append(1.0)
    shape = (len(infos) + 1, len(seq_index) + 1)
    rhs = np.zeros(shape[0])
    rhs[0] = 1.0
    return sparse.csr_matrix((vals, (rows, cols)), shape=shape), rhs


def solve_lp(tree: GameTree, player: int = 0) -> tuple[float, np.ndarray]:
    """Equilibrium strategy for `player` and the game value to player 0."""
    seq = [_sequences(tree, p) for p in (0, 1)]
    E, e = _constraints(tree, 0, seq[0][0], seq[0][2])
    F, f = _constraints(tree, 1, seq[1][0], seq[1][2])
    chance, _ = reach_probabilities(tree, np.ones((tree.num_infosets, NUM_ACTIONS)))
    leaves = np.flatnonzero(tree.kind == TERMINAL)
    A = sparse.csr_matrix(
        (chance[leaves] * tree.utility[leaves], (seq[0][1][leaves], seq[1][1][leaves])),
        shape=(E.shape[1], F.shape[1]),
    )
    if player == 1:
        # Player 1 maximizes -x^T A y; swap roles.
        E, e, F, f, A = F, f, E, e, (-A).T.tocsr()
    nx, nq = E.shape[1], F.shape[0]
    # max f^T q  s.t.  F^T q - A^T x <= 0,  E x = e,  x >= 0.
    c = np.concatenate([np.zeros(nx), -f])
    A_ub = sparse.hstack([-A.T, F.T]).tocsr()
    A_eq = sparse.hstack([E, sparse.csr_matrix((E.shape[0], nq))]).tocsr()
    bounds = [(0, None)] * nx + [(None, None)] * nq
    res = linprog(c, A_ub=A_ub, b_ub=np.zeros(A_ub.shape[0]), A_eq=A_eq, b_eq=e,
                  bounds=bounds, method="highs")
    if res.status != 0:
        raise RuntimeError(f"LP failed: {res.message}")
    x = res.x[:nx]
    value = -res.fun if player == 0 else res.fun
    seq_index, _, parent_seq = seq[player]
    strat = np.zeros((tree.num_infosets, NUM_ACTIONS))
    for (I, a), k in seq_index.items():
        denom = x[parent_seq[I]]
        strat[I, a] = x[k] / denom if denom > 1e-12 else 0.0
    return float(value), normalize(tree, strat)


def lp_equilibrium(tree: GameTree) -> tuple[float, np.ndarray]:
    v0, s0 = solve_lp(tree, 0)
    _, s1 = solve_lp(tree, 1)
    owner = tree.infoset_player[:, None]
    return v0, np.where(owner == 0, s0, s1)
