"""Phase 5 #2: does adding SkillCorner's 20 matches to TRAINING help on our 10-match bench?

Same leave-one-match-out protocol as the current system (bench_mixed_probs.npz = "without"):
for each held-out DFL/Metrica match, train on the other 9 PLUS all SkillCorner halves, predict the
held-out match, run the current referee, and compare per match (paired) with the current system.
The scoreboard (the 10 matches) does not change; only the training set does.

  variant team : SkillCorner used for the team model only (its dead-ball labels are automatic)
  variant both : SkillCorner used for both models
  variant team_w : as 'team', but SkillCorner rows down-weighted so their total weight equals the
                   10 matches' (pre-specified follow-up after 'team' lost, 15 Nov)

    python analysis/30_sc_training.py team
    python analysis/30_sc_training.py both
    python analysis/30_sc_training.py team_w
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
from analysis.features import dead_ball_features, team_features, turnover_features  # noqa: E402
from analysis.dead_ball import FROZEN_BALL_ARTEFACTS, TRAIN_EVERY, new_model  # noqa: E402
from analysis.referee import viterbi, REFEREE_SETTINGS  # noqa: E402
from analysis.paired import compare  # noqa: E402

variant = sys.argv[1] if len(sys.argv) > 1 else "team"
DERIVED = os.path.join(ROOT, "external", "idsse", "derived")
halves, prof = bench(our_mess_profile(), jitter=BENCH_JITTER, sources=("dfl", "metrica"), dead_link=True, wrong_team=True)
sc, _ = bench(prof, jitter=BENCH_JITTER, sources=("skillcorner",), dead_link=True, wrong_team=True)
matches = sorted({h["match"] for h in halves})
print("bench: {} halves; SkillCorner: {} halves from {} matches; variant = {}".format(
    len(halves), len(sc), len({h["match"] for h in sc}), variant), flush=True)

z = np.load(os.path.join(DERIVED, "bench_mixed_probs.npz"), allow_pickle=True)
without = [viterbi(d, h, **REFEREE_SETTINGS) for d, h in zip(z["d"], z["h"])]


def XD(h):
    return dead_ball_features(h).drop(columns=FROZEN_BALL_ARTEFACTS)


def XT(h):
    return pd.concat([team_features(h), turnover_features(h)], axis=1)


cache = os.path.join(DERIVED, "bench_sc_{}_probs.npz".format(variant))
if os.path.exists(cache):
    zz = np.load(cache, allow_pickle=True)
    p_dead, p_home = list(zz["d"]), list(zz["h"])
else:
    print("features ...", flush=True)
    xd, xt = [XD(h) for h in halves], [XT(h) for h in halves]
    sxd = [XD(h) for h in sc] if variant == "both" else None
    sxt = [XT(h) for h in sc]
    sc_live = [h["label"] != "D" for h in sc]
    sc_team_X = np.vstack([x.to_numpy()[l][::TRAIN_EVERY] for x, l in zip(sxt, sc_live)])
    sc_team_y = np.concatenate([(h["label"][l] == "H")[::TRAIN_EVERY] for h, l in zip(sc, sc_live)])
    if variant == "both":
        sc_dead_X = np.vstack([x.to_numpy()[::TRAIN_EVERY] for x in sxd])
        sc_dead_y = np.concatenate([(h["label"] == "D")[::TRAIN_EVERY] for h in sc])
    p_dead, p_home = list(z["d"]), [None] * len(halves)          # dead model unchanged in 'team'
    for m in matches:
        tr = [k for k, h in enumerate(halves) if h["match"] != m]
        live = [halves[k]["label"] != "D" for k in tr]
        tX = np.vstack([xt[k].to_numpy()[l][::TRAIN_EVERY] for k, l in zip(tr, live)] + [sc_team_X])
        ty = np.concatenate([(halves[k]["label"][l] == "H")[::TRAIN_EVERY] for k, l in zip(tr, live)] + [sc_team_y])
        w = None
        if variant == "team_w":                   # SkillCorner rows weighted to equal the 10 matches' total
            n10 = len(ty) - len(sc_team_y)
            w = np.concatenate([np.ones(n10), np.full(len(sc_team_y), n10 / len(sc_team_y))])
        tm = new_model().fit(tX, ty, sample_weight=w)
        if variant == "both":
            dX = np.vstack([xd[k].to_numpy()[::TRAIN_EVERY] for k in tr] + [sc_dead_X])
            dy = np.concatenate([(halves[k]["label"] == "D")[::TRAIN_EVERY] for k in tr] + [sc_dead_y])
            dm = new_model().fit(dX, dy)
        for k, h in enumerate(halves):
            if h["match"] == m:
                p_home[k] = tm.predict_proba(xt[k].to_numpy())[:, 1]
                if variant == "both":
                    p_dead[k] = dm.predict_proba(xd[k].to_numpy())[:, 1]
        print("  fold", m, "done ({} training rows)".format(len(ty)), flush=True)
    np.savez(cache, d=np.array(p_dead, dtype=object), h=np.array(p_home, dtype=object))

with_sc = [viterbi(d, h, **REFEREE_SETTINGS) for d, h in zip(p_dead, p_home)]
for src in ("metrica", "dfl"):
    ks = [k for k, h in enumerate(halves) if h["source"] == src]
    for name, pr in (("current (10 matches)", without), ("+ SkillCorner ({})".format(variant), with_sc)):
        r = score([halves[k] for k in ks], [pr[k] for k in ks])
        print("   {:8s} {:22s} spellF1 {:.4f}  macroF1 {:.4f}  poss {}/{}".format(
            src, name, r["spell_f1_mean"], r["macro_f1"], r["n_pred_spells"], r["n_gt_spells"]))
compare(halves, without, with_sc, "current", "+SkillCorner", metric="spell_f1_mean")
compare(halves, without, with_sc, "current", "+SkillCorner", metric="macro_f1", verbose=False)
