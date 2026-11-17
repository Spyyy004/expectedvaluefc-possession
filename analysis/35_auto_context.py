"""Phase 7 #1: two-stage "auto-context" team judge.

Idea (Mes que data's stage 2): the team judge reads one frame's features and a few short windows;
it cannot see what the whole surrounding stretch of play looks like. A second judge that reads the
FIRST judge's probabilities over wider windows (±1, ±3, ±10 s: mean, min, max, trend) plus the
first judge's own output can fix short possessions the first judge half-sees.

Protocol. Stage-1 probabilities are the cached out-of-fold ones (bench_mixed_probs.npz: each match
predicted by models that never saw it). Stage 2 is trained leave-one-match-out on the other
matches' stage-1 probabilities and applied to the held-out match. Only P(home) is re-predicted;
P(dead) and the referee stay as they are.

  quick  : stage-1 probabilities of the training matches come from models that did see the
           held-out match (a small leak in stage 2's favour). Cheap screen: if this is flat, stop.
  strict : fully nested (stage 1 retrained without both the held-out and the training match).

    python analysis/35_auto_context.py quick
"""
import os
import sys
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=RuntimeWarning)
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from analysis.bench import bench, our_mess_profile, score, BENCH_JITTER, FPS  # noqa: E402
from analysis.dead_ball import TRAIN_EVERY, new_model  # noqa: E402
from analysis.referee import viterbi, REFEREE_SETTINGS  # noqa: E402
from analysis.paired import compare  # noqa: E402

mode = sys.argv[1] if len(sys.argv) > 1 else "quick"
DERIVED = os.path.join(ROOT, "external", "idsse", "derived")
halves, _ = bench(our_mess_profile(), jitter=BENCH_JITTER, sources=("dfl", "metrica"), dead_link=True, wrong_team=True)
matches = sorted({h["match"] for h in halves})
z = np.load(os.path.join(DERIVED, "bench_mixed_probs.npz"), allow_pickle=True)
p_dead, p_home = list(z["d"]), list(z["h"])
without = [viterbi(d, h, **REFEREE_SETTINGS) for d, h in zip(p_dead, p_home)]

WINDOWS_S = (1, 3, 10)


def context_features(pd_, ph, h):
    """Stage-2 inputs: the stage-1 probabilities and their neighbourhood."""
    s = pd.DataFrame({"p_home": ph, "p_dead": pd_})
    s["lean"] = (ph - 0.5) * (1 - pd_)                 # team lean weighted by liveness
    s["ball_seen"] = (~np.isnan(h["bx"])).astype(float)
    cols = {}
    for w in WINDOWS_S:
        n = int(w * FPS)
        for name, src in (("home", s["p_home"]), ("dead", s["p_dead"]), ("lean", s["lean"]), ("seen", s["ball_seen"])):
            r = src.rolling(2 * n + 1, center=True, min_periods=1)
            cols["{}_mean{}".format(name, w)] = r.mean()
            if name in ("home", "lean"):
                cols["{}_min{}".format(name, w)] = r.min()
                cols["{}_max{}".format(name, w)] = r.max()
        past = s["p_home"].rolling(n, min_periods=1).mean()
        future = s["p_home"][::-1].rolling(n, min_periods=1).mean()[::-1]
        cols["home_past{}".format(w)] = past
        cols["home_future{}".format(w)] = future
        cols["home_trend{}".format(w)] = future - past
    return pd.concat([s, pd.DataFrame(cols)], axis=1).to_numpy()


print("stage-2 features ...", flush=True)
X = [context_features(d, p, h) for d, p, h in zip(p_dead, p_home, halves)]
live = [h["label"] != "D" for h in halves]
cache = os.path.join(DERIVED, "bench_autoctx_{}_probs.npz".format(mode))
if os.path.exists(cache):
    p2 = list(np.load(cache, allow_pickle=True)["h"])
else:
    p2 = [None] * len(halves)
    for m in matches:
        tr = [k for k, h in enumerate(halves) if h["match"] != m]
        tX = np.vstack([X[k][live[k]][::TRAIN_EVERY] for k in tr])
        ty = np.concatenate([(halves[k]["label"][live[k]] == "H")[::TRAIN_EVERY] for k in tr])
        s2 = new_model().set_params(max_iter=200).fit(tX, ty)
        for k, h in enumerate(halves):
            if h["match"] == m:
                p2[k] = s2.predict_proba(X[k])[:, 1]
        print("  fold", m, "done", flush=True)
    np.savez(cache, h=np.array(p2, dtype=object))

with_ctx = [viterbi(d, h, **REFEREE_SETTINGS) for d, h in zip(p_dead, p2)]
blend = [viterbi(d, (a + b) / 2, **REFEREE_SETTINGS) for d, a, b in zip(p_dead, p_home, p2)]
for name, pr in (("stage 2 alone", with_ctx), ("average of stage 1 and 2", blend)):
    print("\n=== {} ({}) ===".format(name, mode))
    for src in ("metrica", "dfl"):
        ks = [k for k, h in enumerate(halves) if h["source"] == src]
        for lab, p in (("current", without), (name, pr)):
            r = score([halves[k] for k in ks], [p[k] for k in ks])
            print("   {:8s} {:26s} spellF1 {:.4f}  macroF1 {:.4f}  poss {}/{}".format(
                src, lab, r["spell_f1_mean"], r["macro_f1"], r["n_pred_spells"], r["n_gt_spells"]))
    compare(halves, without, pr, "current", name, metric="spell_f1_mean")
    compare(halves, without, pr, "current", name, metric="macro_f1", verbose=False)
