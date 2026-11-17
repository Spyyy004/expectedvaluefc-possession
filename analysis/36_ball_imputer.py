"""Phase 7 #2: a LEARNED ball imputer: where is the ball during a blackout, given the 22 players?

The heuristic "convergence attractor" (Step 23, D49) was 18.7 m off on Metrica, no better than the
players' centroid. Here a regressor learns the ball position from player geometry AND from the
ball's last-seen / next-seen positions (the ball moves at bounded speed, so interpolation is a
strong prior that player context can correct).

Part A (imputer quality, Bundesliga only: the hidden ball is known there):
  train on the hidden live frames of 6 degraded DFL matches, test on the 7th (leave-one-match-out);
  report median error and share within 5 / 10 m against three baselines:
  linear interpolation last->next seen, players' centroid, the attractor.
  Also: "team nearest the imputed ball" vs the true possession team on blackout frames (the number
  that matters for the team judge), split by possession length.

Part B (downstream, `python analysis/36_ball_imputer.py bench`):
  fill the hidden live frames of every bench half with the imputed ball (imputer trained without
  the held-out match), give the filled ball to the TEAM judge only, leave-one-match-out as always,
  paired comparison with the current system. Metrica's real gaps are filled too, as our match's
  would be.

    python analysis/36_ball_imputer.py          # part A
    python analysis/36_ball_imputer.py bench    # part B
"""
import os
import sys
import warnings

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

warnings.filterwarnings("ignore", category=RuntimeWarning)
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from analysis.bench import bench, our_mess_profile, score, BENCH_JITTER, load_dfl_halves, load_metrica_halves, FPS  # noqa: E402
from analysis.features import inferred_ball, team_features, turnover_features, _run_lengths  # noqa: E402
from analysis.dead_ball import TRAIN_EVERY, new_model  # noqa: E402
from analysis.referee import viterbi, REFEREE_SETTINGS  # noqa: E402
from analysis.paired import compare  # noqa: E402

DERIVED = os.path.join(ROOT, "external", "idsse", "derived")
mode = sys.argv[1] if len(sys.argv) > 1 else "quality"
IMPUTE_EVERY = 3            # training rows: every 3rd hidden live frame


def _ffill(a):
    s = pd.Series(a)
    return s.ffill().to_numpy(), s.bfill().to_numpy()


def _pos_time(mask_seen):
    """Frames since the last seen ball and until the next seen ball (inf where none)."""
    n = len(mask_seen)
    idx = np.arange(n, dtype=float)
    last = pd.Series(np.where(mask_seen, idx, np.nan)).ffill().to_numpy()
    nxt = pd.Series(np.where(mask_seen, idx, np.nan)).bfill().to_numpy()
    return idx - last, nxt - idx


def imputer_features(h):
    """Per-frame inputs for the ball regressor, from degraded players and the degraded ball track."""
    px, py, ps, team = h["px"], h["py"], h["ps"], h["team"]
    n = len(px)
    f = {}
    seen = ~np.isnan(h["bx"])
    lx, nx = _ffill(h["bx"])
    ly, ny = _ffill(h["by"])
    since, until = _pos_time(seen)
    tot = since + until
    w = np.where(np.isfinite(tot) & (tot > 0), until / np.where(tot > 0, tot, 1), 0.5)
    f["last_x"], f["last_y"], f["next_x"], f["next_y"] = lx, ly, nx, ny
    f["since_s"], f["until_s"] = np.minimum(since, 1e4) / FPS, np.minimum(until, 1e4) / FPS
    f["interp_x"] = np.where(np.isnan(nx), lx, np.where(np.isnan(lx), nx, w * lx + (1 - w) * nx))
    f["interp_y"] = np.where(np.isnan(ny), ly, np.where(np.isnan(ly), ny, w * ly + (1 - w) * ny))
    ax, ay = inferred_ball(px, py)
    f["attr_x"], f["attr_y"] = ax, ay
    f["cx"], f["cy"] = np.nanmean(px, axis=1), np.nanmean(py, axis=1)
    f["n_players"] = (~np.isnan(px)).sum(axis=1)
    for t, name in ((0, "home"), (1, "away")):
        sx, sy, ss = px[:, team == t], py[:, team == t], ps[:, team == t]
        f[name + "_cx"], f[name + "_cy"] = np.nanmean(sx, axis=1), np.nanmean(sy, axis=1)
        f[name + "_sdx"], f[name + "_sdy"] = np.nanstd(sx, axis=1), np.nanstd(sy, axis=1)
        f[name + "_minx"], f[name + "_maxx"] = np.nanmin(sx, axis=1), np.nanmax(sx, axis=1)
        f[name + "_miny"], f[name + "_maxy"] = np.nanmin(sy, axis=1), np.nanmax(sy, axis=1)
        f[name + "_speed"] = np.nanmean(ss, axis=1)
        # the team's centroid half a second later: where the block is heading
        f[name + "_vx"] = np.r_[f[name + "_cx"][12:], np.full(12, np.nan)] - f[name + "_cx"]
        f[name + "_vy"] = np.r_[f[name + "_cy"][12:], np.full(12, np.nan)] - f[name + "_cy"]
    f["dcx"], f["dcy"] = f["home_cx"] - f["away_cx"], f["home_cy"] - f["away_cy"]
    # density peak: the player with the most team-mates and opponents within 10 m
    cnt = np.full(px.shape, -1, dtype=np.int32)
    for a in range(0, n, 2000):                       # chunked: the full n x 46 x 46 array is ~1 GB
        b = min(n, a + 2000)
        d = np.hypot(px[a:b, :, None] - px[a:b, None, :], py[a:b, :, None] - py[a:b, None, :])
        c = np.nansum(d < 10, axis=2) - 1
        cnt[a:b] = np.where(np.isnan(px[a:b]), -1, c)
    j = np.argmax(cnt, axis=1)
    f["peak_x"], f["peak_y"] = px[np.arange(n), j], py[np.arange(n), j]
    f["peak_n"] = cnt[np.arange(n), j]
    # fastest player: position and heading (where play is likely moving)
    k = np.argmax(np.where(np.isnan(ps), -1, ps), axis=1)
    f["fast_x"], f["fast_y"], f["fast_s"] = px[np.arange(n), k], py[np.arange(n), k], ps[np.arange(n), k]
    f["fast_vx"] = np.r_[px[12:, :][np.arange(n - 12), k[:n - 12]], np.full(12, np.nan)] - f["fast_x"]
    f["fast_vy"] = np.r_[py[12:, :][np.arange(n - 12), k[:n - 12]], np.full(12, np.nan)] - f["fast_y"]
    return pd.DataFrame(f)


def new_regressor():
    return HistGradientBoostingRegressor(max_iter=300, learning_rate=0.1, random_state=0)


INTERP_COLS = None   # set once the feature frame is known


NEAR_S = 1.0   # within a second of a sighting, straight-line interpolation beats the model (3.0 vs 6.1 m)


class Imputer:
    """Learned ball position from player geometry; plain interpolation next to a sighting."""

    def __init__(self, cols):
        self.ix, self.iy = cols.index("interp_x"), cols.index("interp_y")
        self.since, self.until = cols.index("since_s"), cols.index("until_s")

    def fit(self, rows_X, rows_y):
        X = np.vstack(rows_X)
        y = np.vstack(rows_y)
        self.mx = new_regressor().fit(X, y[:, 0])
        self.my = new_regressor().fit(X, y[:, 1])
        return self

    def predict(self, X):
        px, py = self.mx.predict(X), self.my.predict(X)
        near = (np.minimum(X[:, self.since], X[:, self.until]) < NEAR_S) & ~np.isnan(X[:, self.ix])
        return np.where(near, X[:, self.ix], px), np.where(near, X[:, self.iy], py)


def fit_imputer(rows_X, rows_y):
    return Imputer(INTERP_COLS).fit(rows_X, rows_y)


def nearest_team(px, py, team, x, y):
    d = np.hypot(px - x[:, None], py - y[:, None])
    d = np.where(np.isnan(d), np.inf, d)
    diff = d[:, team == 1].min(axis=1) - d[:, team == 0].min(axis=1)
    return np.where(np.isfinite(diff), np.sign(diff), 0.0)


def possession_length(label):
    out = np.zeros(len(label))
    for side in ("H", "A"):
        L, _, _ = _run_lengths(label == side)
        out = np.where(label == side, L / FPS, out)
    return out


pool = load_dfl_halves() + load_metrica_halves()
halves, _ = bench(our_mess_profile(), jitter=BENCH_JITTER, sources=("dfl", "metrica"), dead_link=True, wrong_team=True)
assert len(pool) == len(halves)
matches = sorted({h["match"] for h in halves})
print("features ...", flush=True)
F = [imputer_features(h) for h in halves]
INTERP_COLS = list(F[0].columns)
hidden =[np.isnan(h["bx"]) & (h["label"] != "D") for h in halves]
truth = [(~np.isnan(c["bx"])) for c in pool]           # true ball known (DFL always; Metrica where seen)
dfl_k = [k for k, h in enumerate(halves) if h["source"] == "dfl"]


def train_rows(ks):
    X = [F[k].to_numpy()[hidden[k] & truth[k]][::IMPUTE_EVERY] for k in ks]
    y = [np.c_[pool[k]["bx"], pool[k]["by"]][hidden[k] & truth[k]][::IMPUTE_EVERY] for k in ks]
    return X, y


if mode == "quality":
    err = {k: [] for k in ("learned", "interp", "centroid", "attractor")}
    team_ok = {k: [] for k in ("learned", "interp", "attractor")}
    tr_truth, lens = [], []
    for m in [mm for mm in matches if mm.startswith("DFL")]:
        tr = [k for k in dfl_k if halves[k]["match"] != m]
        te = [k for k in dfl_k if halves[k]["match"] == m]
        imp = fit_imputer(*train_rows(tr))
        for k in te:
            sel = hidden[k] & truth[k]
            X = F[k]
            pred = imp.predict(X.to_numpy())
            cands = dict(learned=pred, interp=(X["interp_x"].to_numpy(), X["interp_y"].to_numpy()),
                         centroid=(X["cx"].to_numpy(), X["cy"].to_numpy()), attractor=(X["attr_x"].to_numpy(), X["attr_y"].to_numpy()))
            bx, by = pool[k]["bx"], pool[k]["by"]
            for name, (x, y) in cands.items():
                err[name].append(np.hypot(x - bx, y - by)[sel])
                if name in team_ok:
                    team_ok[name].append(nearest_team(halves[k]["px"], halves[k]["py"], halves[k]["team"], x, y)[sel])
            tr_truth.append(np.where(halves[k]["label"] == "H", 1.0, -1.0)[sel])
            lens.append(possession_length(halves[k]["label"])[sel])
        print("  fold", m, "done", flush=True)
    print("\nA. Imputed ball vs true ball on hidden live frames (Bundesliga, leave-one-match-out), {} frames".format(
        sum(len(e) for e in err["learned"])))
    for name in err:
        e = np.concatenate(err[name])
        print("   {:10s} median {:5.1f} m   within 5 m {:5.1%}   within 10 m {:5.1%}".format(name, np.median(e), (e < 5).mean(), (e < 10).mean()))
    t, L = np.concatenate(tr_truth), np.concatenate(lens)
    print("\n   team nearest the imputed ball == true possession team (blackout frames):")
    for name in team_ok:
        a = np.concatenate(team_ok[name])
        print("   {:10s} all {:5.1%}   possessions < 5 s {:5.1%}   < 2 s {:5.1%}".format(name, (a == t).mean(), (a == t)[L < 5].mean(), (a == t)[L < 2].mean()))
    # by blackout length
    e = np.concatenate(err["learned"]); ei = np.concatenate(err["interp"])
    since = np.concatenate([F[k]["since_s"].to_numpy()[hidden[k] & truth[k]] for k in dfl_k])
    until = np.concatenate([F[k]["until_s"].to_numpy()[hidden[k] & truth[k]] for k in dfl_k])
    gap = np.minimum(since, until)
    print("\n   by distance to the nearest seen ball (s): learned / interpolation median error")
    for lo, hi in ((0, 1), (1, 3), (3, 10), (10, 1e9)):
        s = (gap >= lo) & (gap < hi)
        if s.any():
            print("   {:>4.0f}-{:<4.0f} s  n {:7d}   learned {:5.1f} m   interp {:5.1f} m".format(lo, min(hi, 999), s.sum(), np.median(e[s]), np.median(ei[s])))

else:
    z = np.load(os.path.join(DERIVED, "bench_mixed_probs.npz"), allow_pickle=True)
    without = [viterbi(d, h, **REFEREE_SETTINGS) for d, h in zip(z["d"], z["h"])]
    cache = os.path.join(DERIVED, "bench_imputed_probs.npz")
    if os.path.exists(cache):
        p_home = list(np.load(cache, allow_pickle=True)["h"])
    else:
        # imputed ball per half, from an imputer trained without that half's match (DFL rows only)
        filled = []
        for k, h in enumerate(halves):
            tr = [j for j in dfl_k if halves[j]["match"] != h["match"]]
            imp = fit_imputer(*train_rows(tr))
            X = F[k].to_numpy()
            ix, iy = imp.predict(X)
            out = dict(h)
            out["bx"] = np.where(hidden[k], ix, h["bx"])
            out["by"] = np.where(hidden[k], iy, h["by"])
            filled.append(out)
            print("  imputed half", k, flush=True)
        xt = [pd.concat([team_features(f), turnover_features(f)], axis=1).to_numpy() for f in filled]
        live = [h["label"] != "D" for h in halves]
        p_home = [None] * len(halves)
        for m in matches:
            tr = [k for k, h in enumerate(halves) if h["match"] != m]
            tX = np.vstack([xt[k][live[k]][::TRAIN_EVERY] for k in tr])
            ty = np.concatenate([(halves[k]["label"][live[k]] == "H")[::TRAIN_EVERY] for k in tr])
            tm = new_model().fit(tX, ty)
            for k, h in enumerate(halves):
                if h["match"] == m:
                    p_home[k] = tm.predict_proba(xt[k])[:, 1]
            print("  fold", m, "done", flush=True)
        np.savez(cache, h=np.array(p_home, dtype=object))
    with_imp = [viterbi(d, h, **REFEREE_SETTINGS) for d, h in zip(z["d"], p_home)]
    for src in ("metrica", "dfl"):
        ks = [k for k, h in enumerate(halves) if h["source"] == src]
        for name, pr in (("current", without), ("imputed ball", with_imp)):
            r = score([halves[k] for k in ks], [pr[k] for k in ks])
            print("   {:8s} {:14s} spellF1 {:.4f}  macroF1 {:.4f}  poss {}/{}".format(src, name, r["spell_f1_mean"], r["macro_f1"], r["n_pred_spells"], r["n_gt_spells"]))
    compare(halves, without, with_imp, "current", "imputed ball (team judge)", metric="spell_f1_mean")
    compare(halves, without, with_imp, "current", "imputed ball (team judge)", metric="macro_f1", verbose=False)
