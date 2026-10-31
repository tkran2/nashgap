"""Read a language model's full behavioral strategy from its next-token distribution.

For every information set the model sees the rendered prompt, and the probability of
each legal action is the probability mass of that action's word as the first token of
the reply (lowercase and capitalized variants are summed). Renormalizing over legal
actions gives the model's mixed strategy at that information set, exactly and without
sampling noise. The mass that landed on any legal action word before renormalizing is
reported as `format_mass`.

All prompts of one game and framing share a long prefix (system message and rules).
Its KV cache is computed once per call and broadcast across the batch, so each prompt
costs only its own suffix tokens. With gradients enabled, the prefix is recomputed
under the current weights and gradients flow through the broadcast.
"""

from __future__ import annotations

import numpy as np
import torch
from transformers import DynamicCache

from .games import NUM_ACTIONS, GameTree
from .prompts import Framing, all_prompts


def _common_prefix_len(seqs: list[list[int]]) -> int:
    n = min(len(s) for s in seqs) - 1  # keep at least one suffix token per prompt
    first = seqs[0]
    for i in range(n):
        tok = first[i]
        if any(s[i] != tok for s in seqs):
            return i
    return n


class LLMPolicy:
    def __init__(self, model, tokenizer, tree: GameTree, framing: Framing, batch_size: int = 48,
                 chat_kwargs: dict | None = None):
        self.model = model
        self.tree = tree
        self.framing = framing
        self.batch_size = batch_size
        texts = [
            tokenizer.apply_chat_template(m, tokenize=False, add_generation_prompt=True, **(chat_kwargs or {}))
            for m in all_prompts(tree, framing)
        ]
        ids = [tokenizer(t, add_special_tokens=False).input_ids for t in texts]
        p = _common_prefix_len(ids)
        self.prefix = torch.tensor(ids[0][:p], dtype=torch.long)
        self.suffixes = [torch.tensor(x[p:], dtype=torch.long) for x in ids]
        self.pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0

        variants = 2
        cand = np.zeros((tree.num_infosets, NUM_ACTIONS, variants), dtype=np.int64)
        for s in tree.infosets:
            for a in range(NUM_ACTIONS):
                if not s.legal[a]:
                    continue
                word = framing.action_word(s, a)
                for v, form in enumerate((word, word.capitalize())):
                    tok = tokenizer.encode(form, add_special_tokens=False)
                    if len(tok) != 1:
                        raise ValueError(f"action word {form!r} is not a single token for this tokenizer")
                    cand[s.index, a, v] = tok[0]
        self.cand = torch.from_numpy(cand)
        self.legal = torch.from_numpy(tree.legal_mask)

    @property
    def prompt_tokens(self) -> tuple[int, float]:
        return len(self.prefix), float(np.mean([len(s) for s in self.suffixes]))

    def _decoder(self):
        m = self.model
        return m.get_decoder() if hasattr(m, "get_decoder") else m.model

    def _prefix_cache(self):
        out = self._decoder()(input_ids=self.prefix[None], use_cache=True)
        cache = out.past_key_values
        return [(layer.keys, layer.values) for layer in cache.layers]

    def action_logprobs(self, indices, grad: bool = False, full_vocab: bool = False):
        """Log-probabilities over the 3 action slots (illegal = -inf) for the given infosets.

        Returns (logp [B, 3], format_mass [B] or None).
        """
        indices = list(indices)
        ctx = torch.enable_grad() if grad else torch.no_grad()
        with ctx:
            prefix_kv = self._prefix_cache()
            P = len(self.prefix)
            head = self.model.get_output_embeddings().weight
            out_lp, out_mass = [], []
            for start in range(0, len(indices), self.batch_size):
                chunk = indices[start:start + self.batch_size]
                B = len(chunk)
                lens = [len(self.suffixes[i]) for i in chunk]
                S = max(lens)
                ids = torch.full((B, S), self.pad_id, dtype=torch.long)
                mask = torch.zeros((B, P + S), dtype=torch.long)
                mask[:, :P] = 1
                for r, i in enumerate(chunk):
                    ids[r, :lens[r]] = self.suffixes[i]
                    mask[r, P:P + lens[r]] = 1
                cache = DynamicCache(ddp_cache_data=[(k.expand(B, -1, -1, -1), v.expand(B, -1, -1, -1)) for k, v in prefix_kv])
                pos = (P + torch.arange(S)).unsqueeze(0).expand(B, -1)
                hidden = self._decoder()(input_ids=ids, attention_mask=mask, position_ids=pos,
                                         past_key_values=cache, use_cache=True).last_hidden_state
                h = hidden[torch.arange(B), torch.tensor(lens) - 1]  # [B, H]
                cand = self.cand[chunk]  # [B, 3, V]
                tok_logits = torch.einsum("bh,bavh->bav", h, head[cand].to(h.dtype)).float()
                legal = self.legal[chunk]
                if full_vocab:
                    lse = torch.logsumexp((h @ head.T.to(h.dtype)).float(), dim=-1)  # [B]
                    p_tok = torch.exp(tok_logits - lse[:, None, None])
                    out_mass.append((p_tok.sum(-1) * legal).sum(-1).detach())
                z = torch.logsumexp(tok_logits, dim=-1).masked_fill(~legal, float("-inf"))
                out_lp.append(torch.log_softmax(z, dim=-1))
            logp = torch.cat(out_lp)
            mass = torch.cat(out_mass) if full_vocab else None
        return logp, mass

    def strategy(self, with_mass: bool = False):
        logp, mass = self.action_logprobs(range(self.tree.num_infosets), grad=False, full_vocab=with_mass)
        strat = torch.exp(logp).nan_to_num(0.0).double().numpy()
        strat = np.where(self.tree.legal_mask, strat, 0.0)
        strat /= strat.sum(axis=1, keepdims=True)
        if with_mass:
            return strat, mass.double().numpy()
        return strat


def greedy(strategy: np.ndarray) -> np.ndarray:
    out = np.zeros_like(strategy)
    out[np.arange(len(strategy)), strategy.argmax(axis=1)] = 1.0
    return out


def load_model(name: str, dtype: str = "float32"):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(name)
    model = AutoModelForCausalLM.from_pretrained(name, dtype=getattr(torch, dtype))
    model.eval()
    return model, tok
