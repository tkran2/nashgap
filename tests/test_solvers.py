import itertools

import numpy as np
import pytest

from nashgap.games import CALL, FOLD, RAISE, TERMINAL, build_tree
from nashgap.lp import lp_equilibrium
from nashgap.solvers import (
    action_values,
    best_response,
    combine,
    expected_value,
    exploitability,
    reach_probabilities,
    solve,
    uniform_strategy,
)


@pytest.fixture(scope="module")
def kuhn():
    return build_tree("kuhn")


@pytest.fixture(scope="module")
def leduc():
    return build_tree("leduc")


@pytest.fixture(scope="module")
def leduc_eq(leduc):
    return lp_equilibrium(leduc)


def test_infoset_counts(kuhn, leduc):
    assert kuhn.num_infosets == 12
    assert leduc.num_infosets == 288
    assert len(leduc.infosets_of(0)) == len(leduc.infosets_of(1)) == 144


def test_terminal_reach_sums_to_one(leduc):
    rng = np.random.default_rng(0)
    strat = rng.random((leduc.num_infosets, 3)) * leduc.legal_mask
    strat /= strat.sum(1, keepdims=True)
    chance, reach = reach_probabilities(leduc, strat)
    leaves = leduc.kind == TERMINAL
    assert np.isclose((chance * reach[0] * reach[1])[leaves].sum(), 1.0)


def test_legal_actions(leduc):
    for s in leduc.infosets:
        assert s.legal[CALL]
        if not s.facing_bet:
            assert not s.legal[FOLD] and s.legal[RAISE]
        raises = s.histories[-1].count("r")
        assert s.legal[RAISE] == (raises < 2)


def test_kuhn_game_value(kuhn):
    res = solve(kuhn, iterations=4000, method="dcfr")
    assert abs(expected_value(kuhn, res.average) - (-1 / 18)) < 1e-3
    assert exploitability(kuhn, res.average).milli_antes < 1.0
    v_lp, s_lp = lp_equilibrium(kuhn)
    assert abs(v_lp - (-1 / 18)) < 1e-9
    assert exploitability(kuhn, s_lp).milli_antes < 1e-6


def _pure_strategies(tree, player):
    infos = tree.infosets_of(player)
    choices = [[a for a in range(3) if s.legal[a]] for s in infos]
    for combo in itertools.product(*choices):
        strat = np.zeros((tree.num_infosets, 3))
        for s, a in zip(infos, combo):
            strat[s.index, a] = 1.0
        yield strat


@pytest.mark.parametrize("player", [0, 1])
def test_best_response_matches_brute_force(kuhn, player):
    rng = np.random.default_rng(player)
    for _ in range(5):
        opp = rng.random((kuhn.num_infosets, 3)) * kuhn.legal_mask
        opp /= opp.sum(1, keepdims=True)
        sign = 1.0 if player == 0 else -1.0
        best = max(
            sign * expected_value(kuhn, combine(kuhn, pure, opp) if player == 0 else combine(kuhn, opp, pure))
            for pure in _pure_strategies(kuhn, player)
        )
        _, value = best_response(kuhn, opp, player)
        assert abs(best - value) < 1e-12


def test_best_response_value_is_achieved(leduc):
    rng = np.random.default_rng(7)
    opp = rng.random((leduc.num_infosets, 3)) * leduc.legal_mask
    opp /= opp.sum(1, keepdims=True)
    for player in (0, 1):
        br, value = best_response(leduc, opp, player)
        prof = combine(leduc, br, opp) if player == 0 else combine(leduc, opp, br)
        ev = expected_value(leduc, prof)
        assert abs((ev if player == 0 else -ev) - value) < 1e-12


def test_leduc_lp_equilibrium(leduc, leduc_eq):
    value, strat = leduc_eq
    assert exploitability(leduc, strat).milli_antes < 1e-5
    assert -0.09 < value < -0.08  # first player's disadvantage in Leduc is about -0.0856


def test_leduc_cfr_matches_lp(leduc, leduc_eq):
    res = solve(leduc, iterations=1500, method="dcfr")
    expl = exploitability(leduc, res.average)
    assert expl.milli_antes < 1.0
    assert abs(expected_value(leduc, res.average) - leduc_eq[0]) < 2e-3


def test_equilibrium_indifference(leduc, leduc_eq):
    """At equilibrium, every action played with positive probability has the same value."""
    _, strat = leduc_eq
    for p in (0, 1):
        av = action_values(leduc, strat, p)
        q = av.q
        for s in leduc.infosets_of(p):
            if av.opp_reach[s.index] < 1e-9 or av.own_reach[s.index] < 1e-9:
                continue
            support = [a for a in range(3) if strat[s.index, a] > 1e-6]
            vals = q[s.index, support]
            assert np.ptp(vals) < 1e-6, (s.key, vals)


def test_uniform_is_exploitable(leduc):
    assert exploitability(leduc, uniform_strategy(leduc)).milli_antes > 1000
