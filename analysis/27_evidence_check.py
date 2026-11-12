"""When the ball IS visible, does our team model see the brief possessions we miss? (measurement only)

  1. missed short possessions (< 5 s), ball fully visible, not called dead: mean P(correct team)
  2. found short possessions, same filter: for comparison
  3. false alarms: inside true possessions > 10 s (nothing changed), runs of 1-5 s where the model
     leans to the OTHER team (P > 0.5), per 90 minutes of live play. These look like brief real
     possessions to the referee, which is why it needs a switch cost.
Team-model probabilities: current system, leave-one-match-out (bench_mixed_probs.npz).

    python analysis/27_evidence_check.py
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
from analysis.features import _run_lengths  # noqa: E402
from analysis.referee import viterbi, REFEREE_SETTINGS  # noqa: E402

FPS = 25
DERIVED = os.path.join(ROOT, "external", "idsse", "derived")
halves, _ = bench(our_mess_profile(), jitter=BENCH_JITTER, sources=("dfl", "metrica"), dead_link=True, wrong_team=True)
z = np.load(os.path.join(DERIVED, "bench_mixed_probs.npz"), allow_pickle=True)

res = {s: dict(missed=[], found=[], alarms=0, alarm_len=[], live_frames=0) for s in ("metrica", "dfl")}
for h, pd_, ph in zip(halves, z["d"], z["h"]):
    r = res[h["source"]]
    lab = h["label"]
    p = viterbi(pd_, ph, **REFEREE_SETTINGS)
    g, q = pm.label_codes(lab), pm.label_codes(p)
    gs, qs = pm.possession_spells(g, [(0, len(g))]), pm.possession_spells(q, [(0, len(q))])
    flags = pm.matched_gt_flags(pm.spell_overlaps(gs, qs), len(gs), 0.5)
    seen = ~np.isnan(h["bx"])
    p_true = np.where(lab == "H", ph, 1 - ph)            # P(correct team) while alive
    r["live_frames"] += int((lab != "D").sum())
    for (a, b, k), fl in zip(gs, flags):
        L = b - a
        if L < 5 * FPS and seen[a:b].all() and (p[a:b] == "D").mean() < 0.5:
            r["found" if fl else "missed"].append(p_true[a:b].mean())
        if L > 10 * FPS:                                  # nothing changed in here
            lean_other = p_true[a:b] < 0.5
            length, since, _ = _run_lengths(lean_other)
            starts = lean_other & (since == 0) & (length >= 1 * FPS) & (length <= 5 * FPS)
            r["alarms"] += int(starts.sum())
            r["alarm_len"] += list(length[starts] / FPS)

for src in ("metrica", "dfl"):
    r = res[src]
    per90 = 90 * 60 * FPS / r["live_frames"]
    print("\n=== {} ===".format(src.upper()))
    for name in ("missed", "found"):
        v = np.array(r[name])
        print("   {:6s} short possessions, ball fully seen: {:4d}  mean P(correct team) median {:.2f}  "
              "model leans right (> 0.5) {:3.0f}%  clearly right (> 0.7) {:3.0f}%".format(
                  name, len(v), np.median(v), 100 * (v > 0.5).mean(), 100 * (v > 0.7).mean()))
    m = np.array(r["missed"])
    print("   missed AND model leaned right: {:.0f} per 90 live min".format((m > 0.5).sum() * per90))
    print("   false alarms (1-5 s lean to the wrong team inside long possessions): {:.0f} per 90 live min, "
          "median {:.1f} s".format(r["alarms"] * per90, np.median(r["alarm_len"]) if r["alarm_len"] else 0))
