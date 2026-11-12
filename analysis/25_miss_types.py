"""Why do we miss short possessions: timing or wrong label? (measurement only)

For every true possession under 5 s that the current system misses (IoU < 0.5, official matching),
look at what we predicted over its frames:
  swallowed      >= 80% of its frames given to the other team
  wrong team     50-80% given to the other team
  called dead    >= 50% predicted dead
  right team     >= 50% given to the right team, but the overlap still fails -> a TIMING problem:
                   too long  our matching piece is > 2x as long (merged with more of the same team)
                   split     two or more pieces of the right team inside it
                   shifted   one piece, start/end off (offsets reported)
  mixed          none of the above reaches 50%
Same leave-one-match-out predictions as 24_visibility_split.py; nothing is trained or tuned here.

    python analysis/25_miss_types.py
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
TYPES = ["swallowed", "wrong team", "called dead", "timing: too long", "timing: split", "timing: shifted", "mixed"]

halves, _ = bench(our_mess_profile(), jitter=BENCH_JITTER, sources=("dfl", "metrica"), dead_link=True, wrong_team=True)
z = np.load(os.path.join(DERIVED, "bench_mixed_probs.npz"), allow_pickle=True)
preds = [viterbi(d, h, **REFEREE_SETTINGS) for d, h in zip(z["d"], z["h"])]

out = {"metrica": [], "dfl": []}   # (type, length s, start offset s, end offset s, best IoU)
for h, p in zip(halves, preds):
    g, q = pm.label_codes(h["label"]), pm.label_codes(p)
    gs, qs = pm.possession_spells(g, [(0, len(g))]), pm.possession_spells(q, [(0, len(q))])
    flags = pm.matched_gt_flags(pm.spell_overlaps(gs, qs), len(gs), 0.5)
    for (a, b, k), fl in zip(gs, flags):
        if fl or b - a >= 5 * FPS:
            continue
        lab = p[a:b]
        team = h["label"][a]                       # 'H' or 'A'
        other = "A" if team == "H" else "H"
        s_same, s_other, s_dead = (lab == team).mean(), (lab == other).mean(), (lab == "D").mean()
        pieces = [(pa, pb) for pa, pb, pk in qs if pk == k and pb > a and pa < b]
        so = eo = iou = np.nan
        if s_other >= 0.8:
            t = "swallowed"
        elif s_other >= 0.5:
            t = "wrong team"
        elif s_dead >= 0.5:
            t = "called dead"
        elif s_same >= 0.5:
            pa, pb = max(pieces, key=lambda s: min(s[1], b) - max(s[0], a))
            so, eo = (pa - a) / FPS, (pb - b) / FPS
            iou = (min(pb, b) - max(pa, a)) / (max(pb, b) - min(pa, a))
            if (pb - pa) > 2 * (b - a):
                t = "timing: too long"
            elif len(pieces) >= 2:
                t = "timing: split"
            else:
                t = "timing: shifted"
        else:
            t = "mixed"
        out[h["source"]].append((t, (b - a) / FPS, so, eo, iou))

for src in ("metrica", "dfl"):
    r = out[src]
    n = len(r)
    types = np.array([x[0] for x in r])
    L = np.array([x[1] for x in r])
    print("\n=== {}: {} missed possessions under 5 s ===".format(src.upper(), n))
    print("   {:18s} {:>7s} {:>9s} {:>9s}".format("type", "all", "< 2 s", "2-5 s"))
    for t in TYPES:
        m = types == t
        print("   {:18s} {:>6.0f}% {:>8.0f}% {:>8.0f}%".format(
            t, 100 * m.mean(), 100 * (m & (L < 2)).sum() / max((L < 2).sum(), 1),
            100 * (m & (L >= 2)).sum() / max((L >= 2).sum(), 1)))
    timing = np.isin(types, [t for t in TYPES if t.startswith("timing")])
    label = np.isin(types, ["swallowed", "wrong team", "called dead"])
    print("   -> TIMING (right team, bad boundaries) {:.0f}% | LABEL (wrong team or dead) {:.0f}% | mixed {:.0f}%".format(
        100 * timing.mean(), 100 * label.mean(), 100 * (types == "mixed").mean()))
    sh = np.array([x for x in r if x[0] == "timing: shifted"], dtype=object)
    if len(sh):
        so, eo, iou = (np.array(sh[:, i], float) for i in (2, 3, 4))
        print("   shifted: median start offset {:+.2f} s, end offset {:+.2f} s (+ = ours later); median |off| "
              "start {:.2f} s, end {:.2f} s; median IoU {:.2f}".format(
                  np.median(so), np.median(eo), np.median(np.abs(so)), np.median(np.abs(eo)), np.median(iou)))
