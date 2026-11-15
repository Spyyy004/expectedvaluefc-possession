"""Phase 5 #1: does SkillCorner's open data look like our tracking, and do its labels mean what ours mean?

  A  transfer check (no labels): for every feature the models use, compare its distribution on the
     degraded SkillCorner halves with our match, DFL and Metrica. A feature is flagged when
     SkillCorner's median sits further from ours than both DFL's and Metrica's do, by more than
     half of our spread (IQR). Many flags = the data behaves differently and may mislead training.
  B  label check: the CURRENT system (trained on the 10 DFL + Metrica matches, nothing from
     SkillCorner) scored on SkillCorner's halves against SkillCorner's labels. A score in the range
     we see on Metrica means the two label definitions roughly agree; a collapse means they do not.

    python analysis/29_sc_transfer.py
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
from analysis.dead_ball import FROZEN_BALL_ARTEFACTS, baseline_team  # noqa: E402
from analysis.referee import viterbi, REFEREE_SETTINGS  # noqa: E402
from make_submission import our_halves, train_models, dead_X, team_X, apply_goal_rule  # noqa: E402

ten, prof = bench(our_mess_profile(), jitter=BENCH_JITTER, sources=("dfl", "metrica"), dead_link=True, wrong_team=True)
sc, _ = bench(prof, jitter=BENCH_JITTER, sources=("skillcorner",), dead_link=True, wrong_team=True)
ours = our_halves()
print("SkillCorner halves: {}  ({} matches)".format(len(sc), len({h["match"] for h in sc})))
print("ball missing: ours {:.0f}%  SC {:.0f}%  Metrica {:.0f}%  DFL(degraded) {:.0f}%".format(
    *[100 * np.mean([np.isnan(h["bx"]).mean() for h in hs]) for hs in
      (ours, sc, [h for h in ten if h["source"] == "metrica"], [h for h in ten if h["source"] == "dfl"])]))
print("live frames with ball missing: SC {:.0f}%  Metrica {:.0f}%".format(
    *[100 * np.mean([np.isnan(h["bx"][h["label"] != "D"]).mean() for h in hs]) for hs in
      (sc, [h for h in ten if h["source"] == "metrica"])]))
print("dead share: SC {:.0f}%  Metrica {:.0f}%  DFL {:.0f}%".format(
    *[100 * np.mean([(h["label"] == "D").mean() for h in hs]) for hs in
      (sc, [h for h in ten if h["source"] == "metrica"], [h for h in ten if h["source"] == "dfl"])]))


def feats(h):
    return pd.concat([dead_X(h), team_X(h)], axis=1).loc[:, lambda d: ~d.columns.duplicated()]


print("\n=== A. transfer check: feature medians (SC vs ours, with DFL and Metrica for scale) ===")
groups = {"ours": ours, "sc": sc[:8], "dfl": [h for h in ten if h["source"] == "dfl"][:6],
          "metrica": [h for h in ten if h["source"] == "metrica"]}
stats = {}
for g, hs in groups.items():
    X = pd.concat([feats(h) for h in hs])
    stats[g] = pd.DataFrame({"q25": X.quantile(0.25), "med": X.median(), "q75": X.quantile(0.75)})
o = stats["ours"]
iqr = (o.q75 - o.q25).replace(0, np.nan)
dist = {g: (stats[g].med - o.med).abs() for g in ("sc", "dfl", "metrica")}
flag = (dist["sc"] > dist["dfl"]) & (dist["sc"] > dist["metrica"]) & (dist["sc"] > 0.5 * iqr)
print("features: {}  flagged: {}".format(len(o), int(flag.sum())))
rows = [(c, o.med[c], stats["sc"].med[c], stats["dfl"].med[c], stats["metrica"].med[c], dist["sc"][c] / iqr[c])
        for c in o.index[flag]]
for c, a, b, d, m, r in sorted(rows, key=lambda t: -t[-1])[:25]:
    print("   {:40s} ours {:8.3f}  SC {:8.3f}  DFL {:8.3f}  Metrica {:8.3f}   off by {:.1f} IQR".format(c, a, b, d, m, r))

print("\n=== B. label check: current system (trained on DFL + Metrica only) scored on SkillCorner ===")
dead_model, team_model = train_models()
preds, base = [], []
for h in sc:
    pdd, _ = apply_goal_rule(h, dead_model.predict_proba(dead_X(h).to_numpy())[:, 1])
    ph = team_model.predict_proba(team_X(h).to_numpy())[:, 1]
    preds.append(viterbi(pdd, ph, **REFEREE_SETTINGS))
    base.append(baseline_team(h))
for name, pr in (("current system", preds), ("nearest-player baseline", base)):
    r = score(sc, pr)
    print("   {:24s} spellF1 {:.4f}  macroF1 {:.4f}  alive/dead F1 {:.3f}  poss {}/{}".format(
        name, r["spell_f1_mean"], r["macro_f1"], r["alive_dead_f1"], r["n_pred_spells"], r["n_gt_spells"]))
print("   (Metrica, same system, leave-one-out: spellF1 0.595 / macroF1 0.785; Bundesliga 0.483 / 0.855)")
