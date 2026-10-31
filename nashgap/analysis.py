"""Descriptive statistics of a strategy: how it bets, folds and uses its private card."""

from __future__ import annotations

import itertools
from collections import defaultdict

import numpy as np

from .games import FOLD, RAISE, GameTree


def hand_class(tree: GameTree, info) -> str:
    names = tree.rules.rank_names
    if info.public_rank is not None and info.public_rank == info.own_rank:
        return "pair"
    return names[info.own_rank]


def behavior_summary(tree: GameTree, strategy: np.ndarray) -> dict:
    """Unweighted averages over information sets.

    bet_rate[class]:  P(bet) when no bet is pending, by hand class
    fold_rate[class]: P(fold) when facing a bet, by hand class
    card_sensitivity: mean total-variation distance between the strategies at infosets
        that differ only in the player's private card (0 means the card is ignored).
    """
    bet = defaultdict(list)
    fold = defaultdict(list)
    groups = defaultdict(list)
    for s in tree.infosets:
        cls = hand_class(tree, s)
        if s.facing_bet:
            fold[cls].append(strategy[s.index, FOLD])
        else:
            bet[cls].append(strategy[s.index, RAISE])
        groups[(s.player, s.public_rank, s.histories)].append(s.index)
    tv = []
    for idx in groups.values():
        for i, j in itertools.combinations(idx, 2):
            tv.append(0.5 * np.abs(strategy[i] - strategy[j]).sum())
    return {
        "bet_rate": {k: float(np.mean(v)) for k, v in sorted(bet.items())},
        "fold_rate": {k: float(np.mean(v)) for k, v in sorted(fold.items())},
        "card_sensitivity": float(np.mean(tv)) if tv else 0.0,
    }
