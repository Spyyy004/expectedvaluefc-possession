"""Are 'ball invisible' and 'short possessions missed' one problem or two? (measurement only)

For every true possession on the realistic bench, measure how much of it the ball was visible in
the tracking, and whether the current system found it (IoU >= 0.5, official matching).
Metrica is the main answer: its blackouts come from real tracking, so they cluster where real
blackouts do (duels, aerials). Bundesliga blackouts are placed by our own code (shown for contrast).

Uses the current system's leave-one-match-out probabilities cached by 20_hsmm.py
(bench_mixed_probs.npz) and the current referee (D27). Nothing is trained or tuned here.

    python analysis/24_visibility_split.py
"""
import os
import sys
import warnings

import numpy as np

warnings.filterwarnings("ignore", category=RuntimeWarning)
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "evaluation"))
import possession_metrics as pm  # noqa: E402
from analysis.bench import bench, our_mess_profile, BENCH_JITTER  # noqa: E402
from analysis.referee import viterbi, REFEREE_SETTINGS  # noqa: E402

FPS = 25
DERIVED = os.path.join(ROOT, "external", "idsse", "derived")
LENGTHS = [("< 2 s", 0, 2), ("2-5 s", 2, 5), ("5-10 s", 5, 10), ("> 10 s", 10, 1e9)]
VIS = [("ball never seen", -1, 0), ("seen < 50%", 0, 0.5), ("seen 50-99%", 0.5, 0.999), ("seen 100%", 0.999, 2)]

halves, _ = bench(our_mess_profile(), jitter=BENCH_JITTER, sources=("dfl", "metrica"), dead_link=True, wrong_team=True)
z = np.load(os.path.join(DERIVED, "bench_mixed_probs.npz"), allow_pickle=True)
preds = [viterbi(d, h, **REFEREE_SETTINGS) for d, h in zip(z["d"], z["h"])]

rows = {"metrica": [], "dfl": []}   # (length s, visible share, found, swallowed)
for h, p in zip(halves, preds):
    g, q = pm.label_codes(h["label"]), pm.label_codes(p)
    gs, qs = pm.possession_spells(g, [(0, len(g))]), pm.possession_spells(q, [(0, len(q))])
    flags = pm.matched_gt_flags(pm.spell_overlaps(gs, qs), len(gs), 0.5)
    seen = ~np.isnan(h["bx"])
    for (a, b, k), fl in zip(gs, flags):
        # swallowed: a predicted possession of the OTHER team covers >= 80% of it
        other = [(pa, pb) for pa, pb, pk in qs if pk != k and pb > a and pa < b]
        cover = max((min(pb, b) - max(pa, a) for pa, pb in other), default=0)
        rows[h["source"]].append(((b - a) / FPS, seen[a:b].mean(), bool(fl), cover >= 0.8 * (b - a)))

for src in ("metrica", "dfl"):
    r = np.array(rows[src], dtype=float)
    L, V, F, S = r[:, 0], r[:, 1], r[:, 2].astype(bool), r[:, 3].astype(bool)
    vis_band = [(V <= 0) if lo < 0 else ((V > lo) & (V <= hi)) for _, lo, hi in VIS]
    print("\n=== {} ({} true possessions) ===".format(src.upper(), len(L)))
    print("Found rate (IoU 0.5) by length and how much of the possession the ball was visible:")
    print("   {:8s}".format("") + "".join("{:>20s}".format(n) for n, _, _ in VIS) + "{:>12s}".format("all"))
    for name, lo, hi in LENGTHS:
        m = (L >= lo) & (L < hi)
        cells = ["{:>4.0f}% of {:>5d} ".format(100 * F[m & vb].mean() if (m & vb).any() else 0, (m & vb).sum()).rjust(20)
                 for vb in vis_band]
        print("   {:8s}".format(name) + "".join(cells) + "{:>6.0f}% {:>4d}".format(100 * F[m].mean(), m.sum()))
    short = L < 5
    miss = short & ~F
    print("\nShort possessions (< 5 s) we MISS: {} of {} ({:.0f}%)".format(miss.sum(), short.sum(), 100 * miss.mean() / short.mean()))
    for (n, _, _), vb in zip(VIS, vis_band):
        print("   {:18s} {:5.1f}% of misses   (vs {:5.1f}% of short possessions we find)".format(
            n, 100 * (miss & vb).sum() / miss.sum(), 100 * (short & F & vb).sum() / max((short & F).sum(), 1)))
    print("   of the misses, swallowed by the other team's possession: {:.0f}%".format(100 * (miss & S).sum() / miss.sum()))
    mostly_blind = V < 0.5
    print("   split: ball seen < 50% (mostly a blackout problem) {:.0f}% | ball seen >= 50% (evidence problem) {:.0f}%".format(
        100 * (miss & mostly_blind).sum() / miss.sum(), 100 * (miss & ~mostly_blind).sum() / miss.sum()))
