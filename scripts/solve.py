"""Solve Kuhn or Leduc poker and print the game value, exploitability and convergence.

Usage: python scripts/solve.py leduc --method dcfr --iterations 2000
"""

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from nashgap.games import build_tree  # noqa: E402
from nashgap.lp import lp_equilibrium  # noqa: E402
from nashgap.solvers import expected_value, exploitability, solve  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("game", choices=["kuhn", "leduc"])
ap.add_argument("--method", default="dcfr", choices=["cfr", "cfr+", "dcfr"])
ap.add_argument("--iterations", type=int, default=2000)
args = ap.parse_args()

tree = build_tree(args.game)
print(f"{args.game}: {tree.num_nodes:,} nodes, {tree.num_infosets} information sets")
t0 = time.time()
value, eq = lp_equilibrium(tree)
print(f"LP equilibrium: value to first player {value:+.6f}, exploitability "
      f"{exploitability(tree, eq).milli_antes:.2e} milli-antes ({time.time() - t0:.1f}s)")
t0 = time.time()
res = solve(tree, iterations=args.iterations, method=args.method, log_every=max(1, args.iterations // 8))
for it, e in res.history:
    print(f"  {args.method} iteration {it:6d}: exploitability {e:10.4f} milli-antes")
print(f"{args.method}: value {expected_value(tree, res.average):+.6f} after {args.iterations} iterations "
      f"({time.time() - t0:.1f}s)")
