"""The solvers must reproduce Kuhn poker's known equilibrium family (Kuhn, 1950).

With J < Q < K: the first player bets J with some alpha in [0, 1/3], never bets Q,
bets K with 3 * alpha, and after checking into a bet calls Q with alpha + 1/3. The
second player's strategy is unique: facing a bet, call Q with 1/3 and always fold J;
when checked to, bet J with 1/3 and never bet Q; always bet and call with K.
"""

import pytest

from nashgap.games import CALL, RAISE, build_tree
from nashgap.lp import lp_equilibrium
from nashgap.solvers import solve

J, Q, K = 0, 1, 2


def table(tree, strat):
    return {(s.player, s.own_rank, s.histories[-1]): strat[s.index] for s in tree.infosets}


@pytest.fixture(scope="module", params=["lp", "dcfr"])
def eq(request):
    tree = build_tree("kuhn")
    if request.param == "lp":
        return table(tree, lp_equilibrium(tree)[1]), 1e-6
    return table(tree, solve(tree, iterations=20000, method="dcfr").average), 2e-2


def test_second_player_is_unique(eq):
    s, tol = eq
    assert s[(1, Q, "r")][CALL] == pytest.approx(1 / 3, abs=tol)
    assert s[(1, J, "r")][CALL] == pytest.approx(0.0, abs=tol)
    assert s[(1, K, "r")][CALL] == pytest.approx(1.0, abs=tol)
    assert s[(1, J, "c")][RAISE] == pytest.approx(1 / 3, abs=tol)
    assert s[(1, Q, "c")][RAISE] == pytest.approx(0.0, abs=tol)
    assert s[(1, K, "c")][RAISE] == pytest.approx(1.0, abs=tol)


def test_first_player_family(eq):
    s, tol = eq
    alpha = s[(0, J, "")][RAISE]
    assert -tol <= alpha <= 1 / 3 + tol
    assert s[(0, Q, "")][RAISE] == pytest.approx(0.0, abs=tol)
    assert s[(0, K, "")][RAISE] == pytest.approx(3 * alpha, abs=3 * tol)
    assert s[(0, Q, "cr")][CALL] == pytest.approx(alpha + 1 / 3, abs=2 * tol)
    assert s[(0, J, "cr")][CALL] == pytest.approx(0.0, abs=tol)
    assert s[(0, K, "cr")][CALL] == pytest.approx(1.0, abs=tol)
