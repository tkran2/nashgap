"""Figures for the README, from results/ (writes PNGs to figures/)."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from nashgap.games import build_tree  # noqa: E402
from nashgap.solvers import exploitability, uniform_strategy  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "figures"

# Reference categorical palette (fixed order), light surface.
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
SURFACE, INK, INK2, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9"
REWARD_LABEL = {
    "nash_q": "Solver values as reward (nash_q)",
    "br_q": "Best response to current policy (br_q)",
    "selfplay": "Self-play (selfplay)",
    "distill": "Distill equilibrium (supervised)",
}
REWARD_COLOR = {"nash_q": SERIES[1], "br_q": SERIES[0], "selfplay": SERIES[3], "distill": SERIES[2]}


def style(ax, title, xlabel, ylabel):
    ax.set_facecolor(SURFACE)
    ax.figure.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(MUTED)
    ax.tick_params(colors=INK2, labelsize=9)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.set_title(title, color=INK, fontsize=11, loc="left", pad=10)
    ax.set_xlabel(xlabel, color=INK2, fontsize=9)
    ax.set_ylabel(ylabel, color=INK2, fontsize=9)


def load_runs(folder: Path):
    runs = {}
    for d in sorted(folder.glob("*")):
        log = d / "log.jsonl"
        if not log.exists() or not log.read_text().strip():
            continue
        cfg = json.loads((d / "config.json").read_text())
        rows = [json.loads(line) for line in log.read_text().splitlines() if line.strip()]
        runs.setdefault(cfg["reward"], []).append(rows)
    return runs


def end_labels(ax, ends, fmt, lo, hi):
    """Direct value labels at line ends, nudged apart so they never collide."""
    gap = 0.055 * (hi - lo)
    placed = []
    for x, y, color in sorted(ends, key=lambda e: e[1]):
        y_lab = max(y, placed[-1] + gap) if placed else y
        placed.append(y_lab)
        ax.annotate(fmt.format(y), (x, y), xytext=(x, y_lab), textcoords="data", color=INK2, fontsize=8,
                    va="center", ha="left", annotation_clip=False)


def training_figure(folder: Path, out: Path, title: str, metric="exploitability",
                    ylabel="Exploitability (milli-antes per hand)", ylim=None, fmt="{:.0f}", reference=None):
    runs = load_runs(folder)
    if not runs:
        return
    tree = build_tree("leduc")
    uniform = exploitability(tree, uniform_strategy(tree)).milli_antes
    fig, ax = plt.subplots(figsize=(7.2, 4.6), dpi=150)
    ends = []
    for reward in ("nash_q", "selfplay", "br_q", "distill"):
        if reward not in runs:
            continue
        seeds = runs[reward]
        steps = np.array([r["step"] for r in seeds[0]])
        n = min(len(s) for s in seeds)
        vals = np.array([[r[metric] for r in s[:n]] for s in seeds])
        mean = vals.mean(0)
        label = REWARD_LABEL[reward] + (f", {len(seeds)} seeds" if len(seeds) > 1 else "")
        sparse = n <= 12  # few evaluation points: mark them
        ax.plot(steps[:n], mean, color=REWARD_COLOR[reward], linewidth=2, label=label,
                linestyle=(0, (5, 2)) if reward == "nash_q" else "-",
                marker="o" if sparse else None, markersize=4 if sparse else 0, zorder=3 if reward == "nash_q" else 2)
        if len(seeds) > 1:
            ax.fill_between(steps[:n], vals.min(0), vals.max(0), color=REWARD_COLOR[reward], alpha=0.15, linewidth=0)
        ends.append((steps[n - 1] * 1.01, mean[-1], REWARD_COLOR[reward]))
    if metric == "exploitability":
        ax.axhline(uniform, color=MUTED, linewidth=1, linestyle=(0, (4, 3)))
        ax.annotate("uniform random play", (0, uniform), xytext=(2, 4), textcoords="offset points", color=MUTED, fontsize=8)
        ax.set_ylim(bottom=0)
    if reference is not None:
        value, text = reference
        ax.axhline(value, color=MUTED, linewidth=1, linestyle=(0, (4, 3)))
        x_text = 0.45 * max(e[0] for e in ends) if ends else 0
        ax.annotate(text, (x_text, value), xytext=(0, -12), textcoords="offset points", color=MUTED, fontsize=8)
    if ylim:
        ax.set_ylim(*ylim)
    lo, hi = ax.get_ylim()
    end_labels(ax, ends, fmt, lo, hi)
    style(ax, title, "Training step", ylabel)
    ax.legend(frameon=False, fontsize=8, labelcolor=INK2, loc="upper center", bbox_to_anchor=(0.5, -0.14), ncol=2)
    fig.tight_layout()
    fig.savefig(out, facecolor=SURFACE)
    plt.close(fig)


def base_models_figure(out: Path):
    files = sorted((ROOT / "results" / "eval").glob("*.json"))
    if not files:
        return
    tree = build_tree("leduc")
    uniform = exploitability(tree, uniform_strategy(tree)).milli_antes
    names, poker, abstract = [], [], []
    for f in files:
        r = json.loads(f.read_text())
        if "leduc/poker" not in r["results"]:
            continue
        names.append(r["model"].split("/")[-1])
        poker.append(r["results"]["leduc/poker"]["exploitability_milli_antes"])
        abstract.append(r["results"].get("leduc/abstract", {}).get("exploitability_milli_antes", np.nan))
    order = np.argsort(poker)
    names = [names[i] for i in order]
    poker = np.array(poker)[order]
    abstract = np.array(abstract)[order]
    y = np.arange(len(names))
    fig, ax = plt.subplots(figsize=(7.2, 0.5 * len(names) + 2.2), dpi=150)
    h = 0.36
    ax.barh(y - h / 2 - 0.01, poker, height=h, color=SERIES[0], label="Poker framing")
    ax.barh(y + h / 2 + 0.01, abstract, height=h, color=SERIES[1], label="Same game, no poker words")
    for yy, v in zip(y - h / 2 - 0.01, poker):
        ax.annotate(f"{v:.0f}", (v, yy), xytext=(3, 0), textcoords="offset points", color=INK2, fontsize=8, va="center")
    for yy, v in zip(y + h / 2 + 0.01, abstract):
        if np.isfinite(v):
            ax.annotate(f"{v:.0f}", (v, yy), xytext=(3, 0), textcoords="offset points", color=INK2, fontsize=8, va="center")
    ax.axvline(uniform, color=MUTED, linewidth=1, linestyle=(0, (4, 3)))
    ax.annotate(f"uniform random play ({uniform:.0f})", (uniform, -0.75), xytext=(-4, 0), textcoords="offset points",
                color=MUTED, fontsize=8, ha="right", va="center")
    ax.set_yticks(y, names)
    ax.set_xlim(0, max(np.nanmax(poker), np.nanmax(abstract), uniform) * 1.12)
    ax.set_ylim(-1.1, len(names) - 0.4)
    style(ax, "Exact exploitability in Leduc hold'em (equilibrium = 0)", "Exploitability (milli-antes per hand)", "")
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    ax.grid(axis="y", visible=False)
    fig.legend(*ax.get_legend_handles_labels(), frameon=False, fontsize=8, labelcolor=INK2, loc="lower center", ncol=2)
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    fig.savefig(out, facecolor=SURFACE)
    plt.close(fig)


def equilibrium_reward() -> float:
    from nashgap.lp import lp_equilibrium
    from nashgap.rl import q_table

    tree = build_tree("leduc")
    _, nash = lp_equilibrium(tree)
    q, valid = q_table(tree, "nash_q", nash, nash)
    return float((nash * q).sum(1)[valid].mean())


if __name__ == "__main__":
    FIG.mkdir(exist_ok=True)
    base_models_figure(FIG / "base_models.png")
    training_figure(ROOT / "results" / "tabular", FIG / "tabular_exploitability.png",
                    "Tabular policy: exploitability during GRPO, by reward design")
    eq_reward = equilibrium_reward()
    training_figure(ROOT / "results" / "tabular", FIG / "tabular_reward.png",
                    "Tabular policy: the naive reward ends above the equilibrium's", metric="nash_q_reward",
                    ylabel="Mean solver-value reward (chips)", ylim=(-0.45, 0.2), fmt="{:+.2f}",
                    reference=(eq_reward, f"equilibrium strategy's own reward ({eq_reward:+.2f})"))
    training_figure(ROOT / "results" / "llm", FIG / "llm_exploitability.png",
                    "Qwen2.5-0.5B, LoRA on the top 8 layers: exploitability during training")
    training_figure(ROOT / "results" / "llm", FIG / "llm_reward.png",
                    "Qwen2.5-0.5B: the naive rewards rise while exploitability does not move", metric="nash_q_reward",
                    ylabel="Mean solver-value reward (chips)", fmt="{:+.2f}",
                    reference=(eq_reward, f"equilibrium strategy's own reward ({eq_reward:+.2f})"))
    print("figures written to", FIG)
