"""Render information sets as natural-language prompts.

Two framings describe the same game tree:

* "poker": the game as poker (cards, chips, check/bet/call/raise/fold).
* "abstract": an isomorphic tile game with every poker word removed (tiles marked
  1-3, tokens, stay/push/match/quit). The sentences are parallel, so differences in a
  model's play between framings measure how much it leans on poker knowledge rather
  than on the rules it was given.

Every prompt ends with the information set, so all prompts of one game and framing
share a long common prefix (system message and rules). `llm_policy` computes that
prefix once and reuses its KV cache.
"""

from __future__ import annotations

from dataclasses import dataclass

from .games import CALL, FOLD, RAISE, GameTree, Infoset, PokerRules


@dataclass(frozen=True)
class Framing:
    name: str
    system: str
    # action words: [when facing a bet][action] -> word
    words_no_bet: dict[int, str]
    words_facing: dict[int, str]
    # third-person verbs for narrating the opponent / second person for "you"
    verbs_no_bet: dict[int, tuple[str, str]]
    verbs_facing: dict[int, tuple[str, str]]

    def action_word(self, info: Infoset, action: int) -> str:
        return (self.words_facing if info.facing_bet else self.words_no_bet)[action]


POKER = Framing(
    name="poker",
    system="You are playing a two-player poker game. Choose your action.",
    words_no_bet={CALL: "check", RAISE: "bet"},
    words_facing={FOLD: "fold", CALL: "call", RAISE: "raise"},
    verbs_no_bet={CALL: ("check", "checks"), RAISE: ("bet", "bets")},
    verbs_facing={FOLD: ("fold", "folds"), CALL: ("call", "calls"), RAISE: ("raise", "raises")},
)

ABSTRACT = Framing(
    name="abstract",
    system="You are playing a two-player tile game. Choose your move.",
    words_no_bet={CALL: "stay", RAISE: "push"},
    words_facing={FOLD: "quit", CALL: "match", RAISE: "push"},
    verbs_no_bet={CALL: ("stay", "stays"), RAISE: ("push", "pushes")},
    verbs_facing={FOLD: ("quit", "quits"), CALL: ("match", "matches"), RAISE: ("push", "pushes")},
)

FRAMINGS = {"poker": POKER, "abstract": ABSTRACT}


def _join(words: list[str]) -> str:
    return words[0] if len(words) == 1 else ", ".join(words[:-1]) + " or " + words[-1]


def rules_text(rules: PokerRules, framing: Framing) -> str:
    two_rounds = rules.num_rounds == 2
    sizes = rules.raise_sizes
    if framing.name == "poker":
        names = [f"{rules.rank_names[r]}s" for r in range(rules.num_ranks)]
        ranks_desc = " > ".join(rules.rank_names[r] for r in reversed(range(rules.num_ranks)))
        if rules.copies == 1:
            deck = "The deck has three cards: a " + ", a ".join(rules.rank_names[:-1]) + f" and a {rules.rank_names[-1]}."
        else:
            deck = "The deck has six cards: " + ", ".join(f"two {n}" for n in names[:-1]) + f" and two {names[-1]}."
        lines = [
            f"Game: {'Leduc' if two_rounds else 'Kuhn'} poker for two players.",
            "Rules:",
            f"- {deck}",
            f"- Each player antes {rules.ante} chip and is dealt one private card.",
        ]
        if two_rounds:
            lines.append("- There are two betting rounds. Player 1 acts first in each round.")
        else:
            lines.append("- There is one betting round. Player 1 acts first.")
        limit = "one bet" if rules.max_raises == 1 else f"{rules.max_raises} bets or raises"
        lines.append(
            "- With no bet pending you may check or bet. Facing a bet you may "
            + ("fold, call or raise" if rules.max_raises > 1 else "fold or call")
            + f". At most {limit} per round."
        )
        if two_rounds:
            lines.append(f"- A bet or raise is {sizes[0]} chips more than your opponent has put in during round 1, and {sizes[1]} chips more during round 2.")
            lines.append("- After round 1, one public card is revealed.")
            lines.append(f"- At showdown, a player whose card matches the public card wins the pot. Otherwise the higher card wins ({ranks_desc}). Equal cards split the pot.")
        else:
            lines.append(f"- A bet is {sizes[0]} chip.")
            lines.append(f"- At showdown the higher card wins the pot ({ranks_desc}).")
        lines.append("- Folding gives the pot to your opponent.")
    else:
        if rules.copies == 1:
            deck = "There are three tiles, marked 1, 2 and 3."
        else:
            deck = "There are six tiles: two marked 1, two marked 2 and two marked 3."
        lines = [
            "Game: a tile game for two players.",
            "Rules:",
            f"- {deck}",
            f"- Each player puts {rules.ante} token into a shared pool and secretly draws one tile.",
        ]
        if two_rounds:
            lines.append("- There are two phases. Player 1 moves first in each phase.")
        else:
            lines.append("- There is one phase. Player 1 moves first.")
        limit = "one push" if rules.max_raises == 1 else f"{rules.max_raises} pushes"
        lines.append(
            "- With no push pending you may stay or push. Facing a push you may "
            + ("quit, match or push" if rules.max_raises > 1 else "quit or match")
            + f". At most {limit} per phase."
        )
        if two_rounds:
            lines.append(f"- A push puts in {sizes[0]} tokens more than your opponent has put in during phase 1, and {sizes[1]} tokens more during phase 2. Matching puts in the same total as your opponent.")
            lines.append("- After phase 1, one shared tile is turned face up.")
            lines.append("- At the end, a player whose tile has the same mark as the shared tile takes the pool. Otherwise the higher mark takes it. Equal marks split the pool.")
        else:
            lines.append(f"- A push puts in {sizes[0]} token more. Matching puts in the same total as your opponent.")
            lines.append("- At the end the higher mark takes the pool.")
        lines.append("- Quitting gives the pool to your opponent.")
    return "\n".join(lines)


def _rank_name(rules: PokerRules, framing: Framing, rank: int) -> str:
    return rules.rank_names[rank] if framing.name == "poker" else str(rank + 1)


def infoset_text(rules: PokerRules, framing: Framing, info: Infoset) -> str:
    poker = framing.name == "poker"
    me = info.player
    lines = [f"You are Player {me + 1}."]
    lines.append(f"{'Your card' if poker else 'Your tile'}: {_rank_name(rules, framing, info.own_rank)}")
    if rules.num_rounds == 2:
        shared = "Public card" if poker else "Shared tile"
        lines.append(f"{shared}: " + ("not revealed yet" if info.public_rank is None else _rank_name(rules, framing, info.public_rank)))
    contrib = [rules.ante, rules.ante]
    for r, h in enumerate(info.histories):
        label = (f"Round {r + 1}" if poker else f"Phase {r + 1}") if rules.num_rounds == 2 else ("Betting" if poker else "Moves")
        events = []
        for i, ch in enumerate(h):
            actor = i % 2
            facing = contrib[actor] < contrib[1 - actor]
            a = "fcr".index(ch)
            verbs = (framing.verbs_facing if facing else framing.verbs_no_bet)[a]
            who = "You" if actor == me else "Your opponent"
            events.append(f"{who} {verbs[0] if actor == me else verbs[1]}.")
            if a == CALL:
                contrib[actor] = contrib[1 - actor]
            elif a == RAISE:
                contrib[actor] = contrib[1 - actor] + rules.raise_sizes[r]
        lines.append(f"{label}: " + (" ".join(events) if events else "no actions yet."))
    mine, theirs = info.contributions
    unit = "chips" if poker else "tokens"
    lines.append(f"{'Pot' if poker else 'Pool'}: {mine + theirs} {unit}. You have put in {mine} and your opponent has put in {theirs}.")
    legal = [framing.action_word(info, a) for a in range(3) if info.legal[a]]
    lines.append(f"Legal {'actions' if poker else 'moves'}: {', '.join(legal)}")
    lines.append(f"Reply with one word: {_join(legal)}.")
    return "\n".join(lines)


def render(rules: PokerRules, framing: Framing, info: Infoset) -> list[dict]:
    """Chat messages for one information set."""
    return [
        {"role": "system", "content": framing.system},
        {"role": "user", "content": rules_text(rules, framing) + "\n\n" + infoset_text(rules, framing, info)},
    ]


def all_prompts(tree: GameTree, framing: Framing) -> list[list[dict]]:
    return [render(tree.rules, framing, s) for s in tree.infosets]
