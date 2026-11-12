"""The 'called dead' pocket: short possessions (< 5 s) we miss because our dead-ball run covers them.

Where do they sit relative to the TRUE stoppages? (measurement only)
  after restart     the true label just before is D: the possession straight after a restart
  before stoppage   the true label just after is D: the last possession before the ball goes dead
  between           both: a short spell between two stoppages
  open play         neither: we invented a stoppage in live play
For "after restart": how late our dead run ends (we resume play late); for "before stoppage": how
early ours starts. Also: ball visibility in the possession. Same predictions as 24/25.

    python analysis/26_dead_pocket.py
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
WHERE = ["after restart", "before stoppage", "between", "open play"]

halves, _ = bench(our_mess_profile(), jitter=BENCH_JITTER, sources=("dfl", "metrica"), dead_link=True, wrong_team=True)
z = np.load(os.path.join(DERIVED, "bench_mixed_probs.npz"), allow_pickle=True)
preds = [viterbi(d, h, **REFEREE_SETTINGS) for d, h in zip(z["d"], z["h"])]

out = {"metrica": [], "dfl": []}
n_short = {"metrica": 0, "dfl": 0}
for h, p in zip(halves, preds):
    lab = h["label"]
    g, q = pm.label_codes(lab), pm.label_codes(p)
    gs, qs = pm.possession_spells(g, [(0, len(g))]), pm.possession_spells(q, [(0, len(q))])
    flags = pm.matched_gt_flags(pm.spell_overlaps(gs, qs), len(gs), 0.5)
    pdead = p == "D"
    seen = ~np.isnan(h["bx"])
    for (a, b, k), fl in zip(gs, flags):
        if b - a >= 5 * FPS:
            continue
        n_short[h["source"]] += 1
        if fl or pdead[a:b].mean() < 0.5:
            continue
        before = a > 0 and lab[a - 1] == "D"
        after = b < len(lab) and lab[b] == "D"
        where = "between" if before and after else "after restart" if before else "before stoppage" if after else "open play"
        late = early = np.nan
        if before:                                   # our dead run covering the restart: how far past a?
            e = a
            while e < len(p) and pdead[e]:
                e += 1
            late = (e - a) / FPS
        if after:                                    # our dead run covering the stoppage: how far before b?
            s = b - 1
            while s >= 0 and pdead[s]:
                s -= 1
            early = (b - 1 - s) / FPS
        out[h["source"]].append((where, (b - a) / FPS, late, early, seen[a:b].mean()))

for src in ("metrica", "dfl"):
    r = out[src]
    w = np.array([x[0] for x in r])
    L, late, early, vis = (np.array([x[i] for x in r], float) for i in (1, 2, 3, 4))
    print("\n=== {}: {} short possessions called dead ({:.1f}% of all {} short possessions) ===".format(
        src.upper(), len(r), 100 * len(r) / n_short[src], n_short[src]))
    for t in WHERE:
        m = w == t
        extra = ""
        if t in ("after restart", "between") and m.any():
            extra += "  our restart late by median {:.1f} s (p25 {:.1f}, p75 {:.1f})".format(
                np.nanmedian(late[m]), np.nanpercentile(late[m], 25), np.nanpercentile(late[m], 75))
        if t in ("before stoppage", "between") and m.any():
            extra += "  our stoppage early by median {:.1f} s".format(np.nanmedian(early[m]))
        print("   {:16s} {:4.0f}% ({:3d})  length median {:.1f} s  ball fully seen {:3.0f}%{}".format(
            t, 100 * m.mean(), m.sum(), np.median(L[m]) if m.any() else 0,
            100 * (vis[m] >= 0.999).mean() if m.any() else 0, extra))
