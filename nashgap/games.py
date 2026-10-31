"""Two-player limit poker games (Kuhn and Leduc hold'em) expanded into explicit game trees.

Both games are instances of one rule family: a small deck of ranked cards, an ante,
one private card per player, one or two fixed-limit betting rounds, and (for two
rounds) a public card dealt between them. Suits never matter, so information sets
are keyed by ranks, which gives the standard counts: 12 information sets for Kuhn
and 288 for Leduc.

Nodes are emitted in depth-first preorder, so every parent index is smaller than its
children's. The solvers rely on this to run top-down and bottom-up passes as
level-by-level array operations instead of recursion.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

FOLD, CALL, RAISE = 0, 1, 2
NUM_ACTIONS = 3
ACTION_CHARS = "fcr"

TERMINAL, CHANCE, DECISION = 0, 1, 2


@dataclass(frozen=True)
class PokerRules:
    name: str
    num_ranks: int
    copies: int
    num_rounds: int
    raise_sizes: tuple[int, ...]
    max_raises: int
    ante: int = 1
    rank_names: tuple[str, ...] = ("Jack", "Queen", "King")

    @property
    def deck(self) -> list[int]:
        return list(range(self.num_ranks * self.copies))

    def rank(self, card: int) -> int:
        return card // self.copies


KUHN = PokerRules("kuhn", num_ranks=3, copies=1, num_rounds=1, raise_sizes=(1,), max_raises=1)
LEDUC = PokerRules("leduc", num_ranks=3, copies=2, num_rounds=2, raise_sizes=(2, 4), max_raises=2)
GAMES = {"kuhn": KUHN, "leduc": LEDUC}


@dataclass(frozen=True)
class Infoset:
    """What the acting player knows: their own rank, the public rank, and the betting."""

    index: int
    player: int
    own_rank: int
    public_rank: int | None
    histories: tuple[str, ...]  # one string per round so far, over "fcr"
    legal: tuple[bool, bool, bool]
    contributions: tuple[int, int]  # chips put in by (acting player, opponent)

    @property
    def key(self) -> str:
        pub = "-" if self.public_rank is None else str(self.public_rank)
        return f"P{self.player}:{self.own_rank}:{pub}:{'/'.join(self.histories)}"

    @property
    def facing_bet(self) -> bool:
        return self.legal[FOLD]

    @property
    def round(self) -> int:
        return len(self.histories) - 1


@dataclass
class GameTree:
    rules: PokerRules
    kind: np.ndarray
    player: np.ndarray
    infoset: np.ndarray
    parent: np.ndarray
    action: np.ndarray
    chance_prob: np.ndarray
    depth: np.ndarray
    utility: np.ndarray  # payoff to player 0 at terminal nodes
    child: np.ndarray  # [num_nodes, NUM_ACTIONS] child of a decision node per action, or -1
    infosets: list[Infoset]
    levels: list[np.ndarray] = field(default_factory=list)

    @property
    def num_nodes(self) -> int:
        return len(self.kind)

    @property
    def num_infosets(self) -> int:
        return len(self.infosets)

    @property
    def legal_mask(self) -> np.ndarray:
        return np.array([s.legal for s in self.infosets], dtype=bool)

    @property
    def infoset_player(self) -> np.ndarray:
        return np.array([s.player for s in self.infosets], dtype=np.int8)

    def infosets_of(self, player: int) -> list[Infoset]:
        return [s for s in self.infosets if s.player == player]


class _Builder:
    def __init__(self, rules: PokerRules):
        self.rules = rules
        self.kind: list[int] = []
        self.player: list[int] = []
        self.infoset: list[int] = []
        self.parent: list[int] = []
        self.action: list[int] = []
        self.chance_prob: list[float] = []
        self.depth: list[int] = []
        self.utility: list[float] = []
        self.child: list[list[int]] = []
        self.infoset_index: dict[tuple, int] = {}
        self.infosets: list[Infoset] = []

    def _node(self, kind: int, parent: int, action: int, prob: float, depth: int) -> int:
        n = len(self.kind)
        self.kind.append(kind)
        self.player.append(-1)
        self.infoset.append(-1)
        self.parent.append(parent)
        self.action.append(action)
        self.chance_prob.append(prob)
        self.depth.append(depth)
        self.utility.append(0.0)
        self.child.append([-1] * NUM_ACTIONS)
        return n

    def build(self) -> GameTree:
        root = self._node(CHANCE, -1, -1, 1.0, 0)
        deck = self.rules.deck
        for c0 in deck:
            n0 = self._node(CHANCE, root, -1, 1.0 / len(deck), 1)
            rest = [c for c in deck if c != c0]
            for c1 in rest:
                n1 = self._node(DECISION, n0, -1, 1.0 / len(rest), 2)
                self._betting(n1, (c0, c1), None, [""], [self.rules.ante] * 2, 2)
        tree = GameTree(
            rules=self.rules,
            kind=np.array(self.kind, dtype=np.int8),
            player=np.array(self.player, dtype=np.int8),
            infoset=np.array(self.infoset, dtype=np.int32),
            parent=np.array(self.parent, dtype=np.int32),
            action=np.array(self.action, dtype=np.int8),
            chance_prob=np.array(self.chance_prob, dtype=np.float64),
            depth=np.array(self.depth, dtype=np.int16),
            utility=np.array(self.utility, dtype=np.float64),
            child=np.array(self.child, dtype=np.int32),
            infosets=self.infosets,
        )
        tree.levels = [np.flatnonzero(tree.depth == d) for d in range(int(tree.depth.max()) + 1)]
        return tree

    def _betting(self, node: int, cards, public, histories, contrib, depth) -> None:
        """Fill in `node`, which was created as a placeholder for the next state."""
        rules = self.rules
        r = len(histories) - 1
        h = histories[-1]
        raises = h.count("r")
        to_act = len(h) % 2  # player 0 opens every round
        facing = contrib[to_act] < contrib[1 - to_act]
        legal = (facing, True, raises < rules.max_raises)

        own, opp = to_act, 1 - to_act
        pub_rank = None if public is None else rules.rank(public)
        key = (to_act, rules.rank(cards[to_act]), pub_rank, tuple(histories))
        if key not in self.infoset_index:
            self.infoset_index[key] = len(self.infosets)
            self.infosets.append(
                Infoset(
                    index=len(self.infosets),
                    player=to_act,
                    own_rank=key[1],
                    public_rank=pub_rank,
                    histories=tuple(histories),
                    legal=legal,
                    contributions=(contrib[own], contrib[opp]),
                )
            )
        self.kind[node] = DECISION
        self.player[node] = to_act
        self.infoset[node] = self.infoset_index[key]

        for a in range(NUM_ACTIONS):
            if not legal[a]:
                continue
            new_h = h + ACTION_CHARS[a]
            new_hist = histories[:-1] + [new_h]
            new_contrib = list(contrib)
            if a == FOLD:
                child = self._node(TERMINAL, node, a, 1.0, depth + 1)
                winner = opp
                self.utility[child] = self._payoff(winner, new_contrib)
            elif a == CALL:
                new_contrib[to_act] = contrib[opp]
                round_over = facing or new_h == "cc"
                if not round_over:
                    child = self._node(DECISION, node, a, 1.0, depth + 1)
                    self._betting(child, cards, public, new_hist, new_contrib, depth + 1)
                elif r + 1 < rules.num_rounds:
                    child = self._node(CHANCE, node, a, 1.0, depth + 1)
                    rest = [c for c in rules.deck if c not in cards]
                    for pc in rest:
                        g = self._node(DECISION, child, -1, 1.0 / len(rest), depth + 2)
                        self._betting(g, cards, pc, new_hist + [""], new_contrib, depth + 2)
                else:
                    child = self._node(TERMINAL, node, a, 1.0, depth + 1)
                    winner = self._showdown(cards, public)
                    self.utility[child] = self._payoff(winner, new_contrib)
            else:
                new_contrib[to_act] = contrib[opp] + rules.raise_sizes[r]
                child = self._node(DECISION, node, a, 1.0, depth + 1)
                self._betting(child, cards, public, new_hist, new_contrib, depth + 1)
            self.child[node][a] = child

    def _showdown(self, cards, public) -> int | None:
        rk = self.rules.rank
        r0, r1 = rk(cards[0]), rk(cards[1])
        if public is not None:
            pr = rk(public)
            if r0 == pr:
                return 0
            if r1 == pr:
                return 1
        if r0 == r1:
            return None
        return 0 if r0 > r1 else 1

    @staticmethod
    def _payoff(winner: int | None, contrib) -> float:
        if winner is None:
            return 0.0
        return float(contrib[1]) if winner == 0 else -float(contrib[0])


_CACHE: dict[str, GameTree] = {}


def build_tree(game: str | PokerRules) -> GameTree:
    rules = GAMES[game] if isinstance(game, str) else game
    if rules.name not in _CACHE:
        _CACHE[rules.name] = _Builder(rules).build()
    return _CACHE[rules.name]
