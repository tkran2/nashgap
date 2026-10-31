"""Markdown table of base-model results (from results/eval) for the README."""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

rows = []
for f in sorted((ROOT / "results" / "eval").glob("*.json")):
    r = json.loads(f.read_text())["results"]
    if "leduc/poker" not in r:
        continue
    lp, la = r["leduc/poker"], r.get("leduc/abstract", {})
    kp = r.get("kuhn/poker", {})
    b = lp["behavior"]
    rows.append((lp["exploitability_milli_antes"], f.stem, lp, la, kp, b))
print("| Model | Leduc, poker framing | Leduc, no poker words | Kuhn, poker framing | Card sensitivity | Bets when unopposed | Folds to a bet |")
print("| --- | ---: | ---: | ---: | ---: | ---: | ---: |")
for _, name, lp, la, kp, b in sorted(rows):
    bet = sum(b["bet_rate"].values()) / len(b["bet_rate"])
    fold = sum(b["fold_rate"].values()) / len(b["fold_rate"])
    print(f"| {name} | {lp['exploitability_milli_antes']:,.0f} | {la.get('exploitability_milli_antes', float('nan')):,.0f} | "
          f"{kp.get('exploitability_milli_antes', float('nan')):,.0f} | {b['card_sensitivity']:.3f} | {bet:.0%} | {fold:.0%} |")
print("| *equilibrium (reference)* | 0 | 0 | 0 | 0.569 | 46% | 48% |")
print("| *uniform random play (reference)* | 2,374 | 2,374 | 458 | 0.000 | 50% | 42% |")
