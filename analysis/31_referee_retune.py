"""Phase 6 #1: re-tune the referee on the current 10-match bench (nested leave-one-match-out).

REFEREE_SETTINGS were chosen in Step 8b (D27) on the 7 Bundesliga matches alone, before Metrica,
stoppage-linked blackouts, wrong-team noise and jitter were added to the bench. This script asks
whether a different setting is better NOW, using the current out-of-fold probabilities
(bench_mixed_probs.npz, the same cache the frozen system is scored from).

Protocol (same as Step 8b): for each held-out match, pick the grid setting with the best mean
spellF1 on the other 9 matches, apply it to the held-out match. The nested result is then compared
with the current settings per match (paired). A setting is only a candidate if it wins on most
matches AND the folds agree on it (a different pick per fold = noise).

    python analysis/31_referee_retune.py          # current team model
    python analysis/31_referee_retune.py ens      # Phase 6 #2 ensemble: average of current and edge3:3 team models
    python analysis/31_referee_retune.py current r2   # round 2: widened grid around the round-1 pick
"""
import itertools
import os
import sys
import time
import warnings
from collections import Counter

import numpy as np

warnings.filterwarnings("ignore", category=RuntimeWarning)
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from analysis.bench import bench, our_mess_profile, score, BENCH_JITTER  # noqa: E402
from analysis.referee import viterbi, REFEREE_SETTINGS  # noqa: E402
from analysis.paired import compare  # noqa: E402

DERIVED = os.path.join(ROOT, "external", "idsse", "derived")
KEYS = ("switch", "stop", "restart_other", "restart_same", "beta")
CURRENT = tuple(REFEREE_SETTINGS[k] for k in KEYS)
ROUND = sys.argv[2] if len(sys.argv) > 2 else "r1"
if ROUND == "r1":
    GRID = itertools.product(
        [5, 7, 10, 14, 20],        # switch: change of possession in open play
        [80, 120, 160, 240, 320],  # stop: play stops
        [2, 5, 10],                # restart_other
        [2, 5, 10],                # restart_same
        [1.5, 2.0, 3.0],           # beta: trust in the team model
    )
else:
    # Round 2 (15 Nov): round 1 picked (20, 80, 2, 5, 3.0) in 6/10 folds, at the grid edge for
    # switch (top), stop (bottom), restart_other (bottom) and beta (top). Widen around it.
    GRID = itertools.product([14, 20, 28, 40], [40, 60, 80, 120], [1, 2, 5], [5], [2.0, 3.0, 4.0, 5.0])
R1_PICK = (20, 80, 2, 5, 3.0)
GRID = sorted(set(GRID) | {CURRENT, R1_PICK})

halves, _ = bench(our_mess_profile(), jitter=BENCH_JITTER, sources=("dfl", "metrica"), dead_link=True, wrong_team=True)
matches = sorted({h["match"] for h in halves})
by_match = {m: [k for k, h in enumerate(halves) if h["match"] == m] for m in matches}
z = np.load(os.path.join(DERIVED, "bench_mixed_probs.npz"), allow_pickle=True)
p_dead, p_home = list(z["d"]), list(z["h"])
MIX = sys.argv[1] if len(sys.argv) > 1 else "current"
if MIX == "ens":
    e = np.load(os.path.join(DERIVED, "bench_tw_edge3_3_probs.npz"), allow_pickle=True)["h"]
    p_home = [(a + b) / 2 for a, b in zip(p_home, e)]
elif MIX == "gru":        # Phase 7 #3: the GRU sequence judge alone (bench_seq_base, 3 seeds x 12 epochs)
    p_home = list(np.load(os.path.join(DERIVED, "bench_seq_base_probs.npz"), allow_pickle=True)["h"])
elif MIX == "gruavg":     # Phase 7 #3: average of the GRU and the boosted judge
    e = np.load(os.path.join(DERIVED, "bench_seq_base_probs.npz"), allow_pickle=True)["h"]
    p_home = [(a + b) / 2 for a, b in zip(p_home, e)]
print("bench: {} halves, {} matches; grid: {} settings ({}); team probabilities: {}".format(len(halves), len(matches), len(GRID), ROUND, MIX), flush=True)


def decode(params):
    return [viterbi(d, h, *params) for d, h in zip(p_dead, p_home)]


def per_match(preds, metric="spell_f1_mean"):
    return {m: score([halves[k] for k in ks], [preds[k] for k in ks])[metric] for m, ks in by_match.items()}


cache = os.path.join(DERIVED, "referee_retune_table{}{}.npz".format("" if MIX == "current" else "_" + MIX, "" if ROUND == "r1" else "_" + ROUND))
if os.path.exists(cache):
    zz = np.load(cache, allow_pickle=True)
    table = {tuple(k): v for k, v in zip(zz["keys"], zz["vals"])}
    table = {k: dict(zip(matches, v)) for k, v in table.items()}
else:
    t0, table = time.time(), {}
    for i, params in enumerate(GRID):
        table[params] = per_match(decode(params))
        if i == 0:
            print("one setting takes {:.1f} s".format(time.time() - t0), flush=True)
        if (i + 1) % 100 == 0:
            print("  {} / {} settings ({:.0f} s)".format(i + 1, len(GRID), time.time() - t0), flush=True)
    np.savez(cache, keys=np.array(list(table.keys()), dtype=float),
             vals=np.array([[table[k][m] for m in matches] for k in table]))

current = decode(CURRENT)
nested, chosen = [None] * len(halves), {}
for m, ks in by_match.items():
    best = max(GRID, key=lambda g: np.mean([table[g][o] for o in matches if o != m]))
    chosen[m] = best
    preds = decode(best)
    for k in ks:
        nested[k] = preds[k]

print("\nsetting chosen per held-out match (switch, stop, restart_other, restart_same, beta); current =", CURRENT)
for m in matches:
    print("   {:<16} {}{}".format(m, chosen[m], "   (= current)" if chosen[m] == CURRENT else ""))
print("agreement:", Counter(chosen.values()).most_common(3))

pooled = max(GRID, key=lambda g: np.mean(list(table[g].values())))
rank = sorted(GRID, key=lambda g: -np.mean(list(table[g].values())))
print("\nbest on all 10 pooled (for information only, not a selection rule):", pooled,
      "mean spellF1 {:.4f}; current {:.4f} (rank {} / {})".format(
          np.mean(list(table[pooled].values())), np.mean(list(table[CURRENT].values())), rank.index(CURRENT) + 1, len(GRID)))
print("\nsensitivity: best mean spellF1 when one setting is changed, others as current:")
for i, k in enumerate(KEYS):
    vals = sorted({g[i] for g in GRID})
    row = []
    for v in vals:
        g = tuple(v if j == i else CURRENT[j] for j in range(5))
        row.append("{}: {:.4f}".format(v, np.mean(list(table[g].values()))) if g in table else "{}: -".format(v))
    print("   {:<14} ".format(k) + "  ".join(row))

for src in ("metrica", "dfl"):
    ks = [k for k, h in enumerate(halves) if h["source"] == src]
    for name, pr in (("current settings", current), ("nested re-tune", nested)):
        r = score([halves[k] for k in ks], [pr[k] for k in ks])
        print("   {:8s} {:18s} spellF1 {:.4f}  macroF1 {:.4f}  poss {}/{}".format(
            src, name, r["spell_f1_mean"], r["macro_f1"], r["n_pred_spells"], r["n_gt_spells"]))
compare(halves, current, nested, "current", "re-tuned (nested)", metric="spell_f1_mean")
compare(halves, current, nested, "current", "re-tuned (nested)", metric="macro_f1", verbose=False)
