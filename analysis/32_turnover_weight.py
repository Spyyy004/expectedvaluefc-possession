"""Phase 6 #2: turnover-weighted training of the team model.

Our dominant error is short possessions (2-5 s) read as the other team's. The team model is trained
on every 5th live frame with equal weight, so long possessions (easy, many frames) dominate the
loss and the short ones (hard, few frames) barely count. Here the same model is trained with extra
weight on the hard frames, in two pre-specified shapes:

  short<S>  : every frame of a live possession shorter than S seconds gets weight W
  edge<E>   : the first E seconds of every live possession (just after a turnover/restart) get W

Same leave-one-match-out protocol as the current system (bench_mixed_probs.npz = "without");
only the team model changes, the dead model and the referee are the current ones.

    python analysis/32_turnover_weight.py short5:3 short5:6 edge3:3
"""
import os
import sys
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=RuntimeWarning)
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from analysis.bench import bench, our_mess_profile, score, BENCH_JITTER  # noqa: E402
from analysis.features import team_features, turnover_features  # noqa: E402
from analysis.dead_ball import TRAIN_EVERY, new_model, FPS  # noqa: E402
from analysis.referee import viterbi, REFEREE_SETTINGS  # noqa: E402
from analysis.paired import compare  # noqa: E402

variants = sys.argv[1:] or ["short5:3", "short5:6", "edge3:3"]
DERIVED = os.path.join(ROOT, "external", "idsse", "derived")
halves, _ = bench(our_mess_profile(), jitter=BENCH_JITTER, sources=("dfl", "metrica"), dead_link=True, wrong_team=True)
matches = sorted({h["match"] for h in halves})
z = np.load(os.path.join(DERIVED, "bench_mixed_probs.npz"), allow_pickle=True)
without = [viterbi(d, h, **REFEREE_SETTINGS) for d, h in zip(z["d"], z["h"])]


def runs(label):
    """Per frame: length of the label run it belongs to and its position inside the run (frames)."""
    n = len(label)
    change = np.r_[True, label[1:] != label[:-1]]
    start = np.flatnonzero(change)
    end = np.r_[start[1:], n]
    length = np.repeat(end - start, end - start)
    pos = np.arange(n) - np.repeat(start, end - start)
    return length, pos


def weights(h, scheme, w):
    live = h["label"] != "D"
    length, pos = runs(h["label"])
    if scheme.startswith("short"):
        hard = live & (length < float(scheme[5:]) * FPS)
    elif scheme.startswith("edge"):
        hard = live & (pos < float(scheme[4:]) * FPS)
    else:
        raise ValueError(scheme)
    return np.where(hard, w, 1.0)[live], hard[live].mean()


print("bench: {} halves; variants: {}".format(len(halves), variants), flush=True)
xt = None
for variant in variants:
    scheme, w, *seed = variant.split(":")      # optional third field: random_state (robustness check)
    w, seed = float(w), (int(seed[0]) if seed else 0)
    cache = os.path.join(DERIVED, "bench_tw_{}_{}{}_probs.npz".format(scheme, int(w), "_s{}".format(seed) if seed else ""))
    if os.path.exists(cache):
        p_home = list(np.load(cache, allow_pickle=True)["h"])
    else:
        if xt is None:
            print("features ...", flush=True)
            xt = [pd.concat([team_features(h), turnover_features(h)], axis=1).to_numpy() for h in halves]
        live = [h["label"] != "D" for h in halves]
        wl = [weights(h, scheme, w) for h in halves]
        print("  {}: {:.0%} of live frames weighted x{}".format(variant, np.mean([f for _, f in wl]), int(w)), flush=True)
        p_home = [None] * len(halves)
        for m in matches:
            tr = [k for k, h in enumerate(halves) if h["match"] != m]
            tX = np.vstack([xt[k][live[k]][::TRAIN_EVERY] for k in tr])
            ty = np.concatenate([(halves[k]["label"][live[k]] == "H")[::TRAIN_EVERY] for k in tr])
            tw = np.concatenate([wl[k][0][::TRAIN_EVERY] for k in tr])
            tm = new_model().set_params(random_state=seed).fit(tX, ty, sample_weight=tw)
            for k, h in enumerate(halves):
                if h["match"] == m:
                    p_home[k] = tm.predict_proba(xt[k])[:, 1]
            print("  fold", m, "done", flush=True)
        np.savez(cache, h=np.array(p_home, dtype=object))
    with_w = [viterbi(d, h, **REFEREE_SETTINGS) for d, h in zip(z["d"], p_home)]
    print("\n=== {} ===".format(variant))
    for src in ("metrica", "dfl"):
        ks = [k for k, h in enumerate(halves) if h["source"] == src]
        for name, pr in (("current", without), (variant, with_w)):
            r = score([halves[k] for k in ks], [pr[k] for k in ks])
            print("   {:8s} {:10s} spellF1 {:.4f}  macroF1 {:.4f}  poss {}/{}".format(
                src, name, r["spell_f1_mean"], r["macro_f1"], r["n_pred_spells"], r["n_gt_spells"]))
    compare(halves, without, with_w, "current", variant, metric="spell_f1_mean")
    compare(halves, without, with_w, "current", variant, metric="macro_f1", verbose=False)
