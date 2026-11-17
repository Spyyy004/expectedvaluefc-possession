"""Phase 6 #5: is a partly filled ball worth having? Bench estimate for the broadcast-ball fill.

On our match the broadcast tracking's ball, aligned to the tactical timeline (lag 4,014 frames in
half 1, 26,920 in half 2; median disagreement 0.8 m where both see the ball), is present on 13% of
the frames where the tactical ball is missing. This script simulates exactly that on the
Bundesliga bench, where the hidden ball is known: 13% of the hidden live frames are given back in
contiguous spans (1-10 s) with 0.6 m Gaussian noise per axis. Only the TEAM model sees the filled
ball; the dead model, the goal rule and the referee keep the original ball, so stoppage clues built
on blackouts are untouched. Metrica halves cannot be filled (their gaps are real) and act as a
control: their numbers should not move much.

  variant filled : team model trained on filled bench halves, applied to filled held-out halves
                   (what the submission would do)
  variant mixed  : team model trained on the UNFILLED halves (current training), applied to filled
                   held-out halves (the lazy option)

    python analysis/33_broadcast_fill.py filled mixed
"""
import os
import sys
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=RuntimeWarning)
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from analysis.bench import bench, our_mess_profile, score, BENCH_JITTER, load_dfl_halves, FPS  # noqa: E402
from analysis.features import team_features, turnover_features  # noqa: E402
from analysis.dead_ball import TRAIN_EVERY, new_model  # noqa: E402
from analysis.referee import viterbi, REFEREE_SETTINGS  # noqa: E402
from analysis.paired import compare  # noqa: E402

FILL_SHARE = 0.13          # share of hidden frames the broadcast gives back on our match
FILL_NOISE_M = 0.6         # per-axis Gaussian noise, ~0.8 m median disagreement
SPAN_S = (1.0, 10.0)       # broadcast shots: contiguous spans

variants = sys.argv[1:] or ["filled", "mixed"]
DERIVED = os.path.join(ROOT, "external", "idsse", "derived")
halves, _ = bench(our_mess_profile(), jitter=BENCH_JITTER, sources=("dfl", "metrica"), dead_link=True, wrong_team=True)
matches = sorted({h["match"] for h in halves})
z = np.load(os.path.join(DERIVED, "bench_mixed_probs.npz"), allow_pickle=True)
without = [viterbi(d, h, **REFEREE_SETTINGS) for d, h in zip(z["d"], z["h"])]

clean = {(h["match"], h["half"]): h for h in load_dfl_halves()}


def fill(h, seed):
    """Copy of a degraded DFL half with 13% of its hidden live frames filled from the clean ball."""
    if h.get("source") != "dfl":
        return h
    c = clean[(h["match"], h["half"])]
    rng = np.random.default_rng(seed)
    hidden = np.isnan(h["bx"]) & (h["label"] != "D")
    idx = np.flatnonzero(hidden)
    give = np.zeros(len(h["bx"]), dtype=bool)
    attempts = 0
    while idx.size and give[hidden].mean() < FILL_SHARE and attempts < 100000:
        attempts += 1
        start = int(idx[rng.integers(idx.size)])
        if give[start]:
            continue
        stop = min(len(give), start + int(rng.uniform(*SPAN_S) * FPS))
        run = np.arange(start, stop)
        run = run[hidden[run]]
        give[run] = True
    out = dict(h)
    out["bx"] = np.where(give, c["bx"] + rng.normal(0, FILL_NOISE_M, len(give)), h["bx"])
    out["by"] = np.where(give, c["by"] + rng.normal(0, FILL_NOISE_M, len(give)), h["by"])
    out["filled_share"] = float(give[hidden].mean()) if hidden.any() else 0.0
    return out


filled = [fill(h, 1000 + k) for k, h in enumerate(halves)]
dfl = [h for h in filled if h.get("source") == "dfl"]
print("bench: {} halves; DFL halves filled: {:.1%} of hidden live frames on average; ball missing {:.1%} -> {:.1%}".format(
    len(halves), np.mean([h["filled_share"] for h in dfl]),
    np.mean([np.isnan(h["bx"]).mean() for h in halves if h.get("source") == "dfl"]),
    np.mean([np.isnan(h["bx"]).mean() for h in dfl])), flush=True)


def XT(h):
    return pd.concat([team_features(h), turnover_features(h)], axis=1).to_numpy()


print("features ...", flush=True)
xt_plain = [XT(h) for h in halves]
xt_fill = [XT(h) for h in filled]
live = [h["label"] != "D" for h in halves]
for variant in variants:
    cache = os.path.join(DERIVED, "bench_bfill_{}_probs.npz".format(variant))
    if os.path.exists(cache):
        p_home = list(np.load(cache, allow_pickle=True)["h"])
    else:
        train_x = xt_fill if variant == "filled" else xt_plain
        p_home = [None] * len(halves)
        for m in matches:
            tr = [k for k, h in enumerate(halves) if h["match"] != m]
            tX = np.vstack([train_x[k][live[k]][::TRAIN_EVERY] for k in tr])
            ty = np.concatenate([(halves[k]["label"][live[k]] == "H")[::TRAIN_EVERY] for k in tr])
            tm = new_model().fit(tX, ty)
            for k, h in enumerate(halves):
                if h["match"] == m:
                    p_home[k] = tm.predict_proba(xt_fill[k])[:, 1]
            print("  fold", m, "done", flush=True)
        np.savez(cache, h=np.array(p_home, dtype=object))
    with_fill = [viterbi(d, h, **REFEREE_SETTINGS) for d, h in zip(z["d"], p_home)]
    print("\n=== {} ===".format(variant))
    for src in ("metrica", "dfl"):
        ks = [k for k, h in enumerate(halves) if h["source"] == src]
        for name, pr in (("current", without), (variant, with_fill)):
            r = score([halves[k] for k in ks], [pr[k] for k in ks])
            print("   {:8s} {:8s} spellF1 {:.4f}  macroF1 {:.4f}  poss {}/{}".format(
                src, name, r["spell_f1_mean"], r["macro_f1"], r["n_pred_spells"], r["n_gt_spells"]))
    compare(halves, without, with_fill, "current", "broadcast fill ({})".format(variant), metric="spell_f1_mean")
    compare(halves, without, with_fill, "current", "broadcast fill ({})".format(variant), metric="macro_f1", verbose=False)
