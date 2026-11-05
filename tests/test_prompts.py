import re

import pytest

from nashgap.games import build_tree
from nashgap.prompts import ABSTRACT, POKER, infoset_text, render

POKER_WORDS = ("poker", "card", "chip", "bet", "fold", "call", "raise", "check", "jack", "queen", "king",
               "pot", "ante", "showdown", "bluff")


@pytest.mark.parametrize("game", ["kuhn", "leduc"])
@pytest.mark.parametrize("framing", [POKER, ABSTRACT], ids=["poker", "abstract"])
def test_every_infoset_renders_with_exactly_its_legal_actions(game, framing):
    tree = build_tree(game)
    for info in tree.infosets:
        text = infoset_text(tree.rules, framing, info)
        legal = [framing.action_word(info, a) for a in range(3) if info.legal[a]]
        assert f": {', '.join(legal)}\n" in text + "\n"
        mine, theirs = info.contributions
        assert re.search(rf"\b{mine + theirs} (chips|tokens)\. You have put in {mine} and your opponent has put in {theirs}", text)


@pytest.mark.parametrize("game", ["kuhn", "leduc"])
def test_abstract_framing_has_no_poker_vocabulary(game):
    tree = build_tree(game)
    for info in tree.infosets:
        msgs = render(tree.rules, ABSTRACT, info)
        words = set(re.findall(r"[a-z]+", " ".join(m["content"] for m in msgs).lower()))
        assert not words & set(POKER_WORDS), (info.key, words & set(POKER_WORDS))


def test_framings_are_parallel():
    """Both framings describe the same infoset with the same number of lines."""
    tree = build_tree("leduc")
    for info in tree.infosets:
        assert len(infoset_text(tree.rules, POKER, info).splitlines()) == \
            len(infoset_text(tree.rules, ABSTRACT, info).splitlines())


def test_prompts_share_a_long_prefix():
    tree = build_tree("leduc")
    contents = [render(tree.rules, POKER, s)[1]["content"] for s in tree.infosets]
    prefix = contents[0]
    for c in contents[1:]:
        while not c.startswith(prefix):
            prefix = prefix[:-1]
    assert len(prefix) > 0.6 * min(len(c) for c in contents)
