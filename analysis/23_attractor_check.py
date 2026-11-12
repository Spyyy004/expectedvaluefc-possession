"""Convergence attractor, step 1: how good is the inferred ball BEFORE we feed it to any model?

Measures only (no training, no decisions on our match's ground truth):
  A. distance inferred ball -> real ball, on live frames where the real ball is known,
     vs a dumb baseline (centroid of all players)
  B. bench only: during ball blackouts in live play, how often "team nearest the inferred ball"
     equals the true possession team, vs the clues we already use (team nearest the ball when it
     was last / next seen). Also for short possessions (< 5 s).
  C. proxy that also works on our match (no labels needed): on ball-visible frames, does the team
     nearest the inferred ball equal the team nearest the real ball? Same number on bench and our
     match tells us whether the attractor behaves alike in both (the trap that sank D47).

Bench = the realistic training bench (dropouts, 1 m jitter, wrong-team episodes). Only summary
numbers are printed.

    python analysis/23_attractor_check.py
"""
import os
import sys
import warnings

import numpy as np

warnings.filterwarnings("ignore", category=RuntimeWarning)
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from analysis.bench import bench, our_mess_profile, load_dfl_halves, load_metrica_halves, BENCH_JITTER  # noqa: E402
from analysis.features import inferred_ball, team_features, _run_lengths, FPS  # noqa: E402
from make_submission import our_halves  # noqa: E402


def nearest_team(px, py, team, x, y):
    """+1 home nearer to (x, y), -1 away nearer, 0 unknown."""
    d = np.hypot(px - x[:, None], py - y[:, None])
    d = np.where(np.isnan(d), np.inf, d)
    diff = d[:, team == 1].min(axis=1) - d[:, team == 0].min(axis=1)
    return np.where(np.isfinite(diff), np.sign(diff), 0.0)


def possession_length(label):
    """Length (s) of the H or A run each frame sits in (0 for D)."""
    out = np.zeros(len(label))
    for side in ("H", "A"):
        L, _, _ = _run_lengths(label == side)
        out = np.where(label == side, L / FPS, out)
    return out


def pct(x):
    return "{:5.1f}%".format(100 * np.mean(x)) if len(x) else "  n/a "


rows = {}   # source -> lists of arrays
pool = load_dfl_halves() + load_metrica_halves()
halves, _ = bench(our_mess_profile(), jitter=BENCH_JITTER, sources=("dfl", "metrica"), dead_link=True,
                  wrong_team=True)
assert len(pool) == len(halves)
for clean, h in zip(pool, halves):
    src = h["source"]
    r = rows.setdefault(src, {k: [] for k in ("err", "err_c", "vis_agree", "bo_truth", "bo_inf", "bo_last",
                                               "bo_next", "bo_len")})
    ibx, iby = inferred_ball(h["px"], h["py"])
    cx, cy = np.nanmean(h["px"], axis=1), np.nanmean(h["py"], axis=1)
    alive = h["label"] != "D"
    truth_ball = ~np.isnan(clean["bx"]) & alive          # real ball known (Metrica: own gaps excluded)
    r["err"].append(np.hypot(ibx - clean["bx"], iby - clean["by"])[truth_ball])
    r["err_c"].append(np.hypot(cx - clean["bx"], cy - clean["by"])[truth_ball])
    inf_team = nearest_team(h["px"], h["py"], h["team"], ibx, iby)
    vis = ~np.isnan(h["bx"])                               # dead included, to match our match
    ball_team = nearest_team(h["px"], h["py"], h["team"], h["bx"], h["by"])
    r["vis_agree"].append((inf_team == ball_team)[vis & (ball_team != 0)])
    # B: blackouts in live play (only DFL has the true ball under its simulated blackouts,
    # but the possession label is known for both sources)
    blk = np.isnan(h["bx"]) & alive
    tf = team_features(h)
    truth = np.where(h["label"] == "H", 1.0, -1.0)
    r["bo_truth"].append(truth[blk])
    r["bo_inf"].append(inf_team[blk])
    r["bo_last"].append(np.sign(np.nan_to_num(tf["near_diff_last_seen"].to_numpy()))[blk])
    r["bo_next"].append(np.sign(np.nan_to_num(tf["near_diff_next_seen"].to_numpy()))[blk])
    r["bo_len"].append(possession_length(h["label"])[blk])

print("A. Distance inferred ball -> real ball (live frames, real ball known)")
print("   {:10s} {:>8s} {:>10s} {:>10s} | centroid baseline: {:>8s} {:>10s}".format(
    "source", "median", "within 5m", "within 10m", "median", "within 10m"))
for src, r in rows.items():
    e, c = np.concatenate(r["err"]), np.concatenate(r["err_c"])
    print("   {:10s} {:7.1f}m {:>10s} {:>10s} | {:>27.1f}m {:>10s}".format(
        src, np.median(e), pct(e < 5), pct(e < 10), np.median(c), pct(c < 10)))

print("\nB. Team call during live-play ball blackouts vs true possession (bench)")
print("   {:10s} {:>10s} {:>9s} {:>9s} {:>9s} {:>16s}".format(
    "source", "frames", "inferred", "last-seen", "next-seen", "inferred, <5 s"))
for src, r in rows.items():
    t, i, l, n, L = (np.concatenate(r[k]) for k in ("bo_truth", "bo_inf", "bo_last", "bo_next", "bo_len"))
    short = L < 5
    print("   {:10s} {:>10d} {:>9s} {:>9s} {:>9s} {:>16s}".format(
        src, len(t), pct(i == t), pct(l == t), pct(n == t), pct((i == t)[short])))
    print("   {:10s} {:>10s} {:>9s} {:>9s} {:>9s}   (short possessions only, {} frames)".format(
        "", "<5 s:", pct((i == t)[short]), pct((l == t)[short]), pct((n == t)[short]), short.sum()))
    # When last-seen and next-seen disagree (a change probably happened inside the blackout),
    # who is right more often?
    dis = (l != n) & (l != 0) & (n != 0)
    print("   {:10s} {:>10s} {:>9s} {:>9s} {:>9s}   (last vs next disagree, {} frames)".format(
        "", "disagree:", pct((i == t)[dis]), pct((l == t)[dis]), pct((n == t)[dis]), dis.sum()))

print("\nC. Proxy (no labels): team nearest inferred ball == team nearest real ball, ball visible")
for src, r in rows.items():
    print("   {:10s} {}".format(src, pct(np.concatenate(r["vis_agree"]))))
ours, err_ours = [], []
for h in our_halves():
    ibx, iby = inferred_ball(h["px"], h["py"])
    vis = ~np.isnan(h["bx"])
    a = nearest_team(h["px"], h["py"], h["team"], ibx, iby)
    b = nearest_team(h["px"], h["py"], h["team"], h["bx"], h["by"])
    ours.append((a == b)[vis & (b != 0)])
    err_ours.append(np.hypot(ibx - h["bx"], iby - h["by"])[vis])
e = np.concatenate(err_ours)
print("   {:10s} {}   (all visible frames, dead balls included)".format("our match", pct(np.concatenate(ours))))
print("   {:10s} distance median {:.1f}m, within 10m {}".format("", np.median(e), pct(e < 10)))
