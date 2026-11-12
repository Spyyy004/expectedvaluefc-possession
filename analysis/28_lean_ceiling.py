"""Ceiling check: what if a PERFECT judge decided which brief leans are real turnovers? (measurement only)

A "brief lean" = a run of frames where the referee says team X has the ball but the team model leans to
the other team (P > 0.5), lasting MIN-5 s. Three versions of the predictions:
  current       the referee as it is (D27)
  accept all    every brief lean becomes a possession of the leaned-to team (no judge at all)
  perfect judge only the leans where the TRUE label is mostly the leaned-to team are accepted
The perfect judge peeks at the true labels, so it is NOT a method; it is the most any real lean judge
could add. Boundaries stay where the model's lean starts/ends (a real judge would have those too).
External bench only (leave-one-match-out probabilities from bench_mixed_probs.npz).

    python analysis/28_lean_ceiling.py
"""
import os
import sys
import warnings

import numpy as np

warnings.filterwarnings("ignore", category=RuntimeWarning)
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from analysis.bench import bench, our_mess_profile, score, BENCH_JITTER  # noqa: E402
from analysis.features import _run_lengths  # noqa: E402
from analysis.referee import viterbi, REFEREE_SETTINGS  # noqa: E402
from analysis.paired import compare  # noqa: E402

FPS = 25
DERIVED = os.path.join(ROOT, "external", "idsse", "derived")
halves, _ = bench(our_mess_profile(), jitter=BENCH_JITTER, sources=("dfl", "metrica"), dead_link=True, wrong_team=True)
z = np.load(os.path.join(DERIVED, "bench_mixed_probs.npz"), allow_pickle=True)
current = [viterbi(d, h, **REFEREE_SETTINGS) for d, h in zip(z["d"], z["h"])]


def apply_leans(p, ph, lab, min_s, judge):
    out = p.copy()
    lean = np.where(ph > 0.5, "H", "A")
    against = (p != "D") & (lean != p)
    length, since, _ = _run_lengths(against)
    starts = np.flatnonzero(against & (since == 0) & (length >= min_s * FPS) & (length <= 5 * FPS))
    n_acc = n_real = 0
    for a in starts:
        b = int(a + length[a])
        team = lean[a]
        real = (lab[a:b] == team).mean() > 0.5
        n_real += real
        if judge == "all" or (judge == "perfect" and real):
            out[a:b] = team
            n_acc += 1
    return out, len(starts), n_real, n_acc


def show(name, preds):
    for src in ("metrica", "dfl"):
        ks = [k for k, h in enumerate(halves) if h["source"] == src]
        r = score([halves[k] for k in ks], [preds[k] for k in ks])
        print("   {:<8} {:<26} spellF1 {:.4f}  macroF1 {:.4f}  poss {}/{}".format(
            src, name, r["spell_f1_mean"], r["macro_f1"], r["n_pred_spells"], r["n_gt_spells"]))


show("current", current)
for min_s in (1.0, 0.4):
    for judge in ("all", "perfect"):
        res = [apply_leans(p, ph, h["label"], min_s, judge) for p, ph, h in zip(current, z["h"], halves)]
        preds = [x[0] for x in res]
        n, real, acc = (sum(x[i] for x in res) for i in (1, 2, 3))
        print("\n-- leans {:.1f}-5 s, judge = {}: {} leans, {} real ({:.0f}%), {} accepted".format(
            min_s, judge, n, real, 100 * real / max(n, 1), acc))
        show("{} ({:.1f}-5 s)".format(judge, min_s), preds)
        if judge == "perfect":
            rows = compare(halves, current, preds, "current", "perfect", verbose=False)
            for g, r in rows.items():
                print("      paired {:<11} mean {:+.4f}  wins {}/{}  90% CI [{:+.4f}, {:+.4f}]".format(
                    g, r["mean"], r["wins"], r["n"], r["lo"], r["hi"]))
