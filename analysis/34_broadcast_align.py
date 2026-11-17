"""Phase 6 #5: align the broadcast tracking's ball to the tactical timeline and count what it could fill.

The broadcast recording is not frame-aligned to the tactical one (DATA.md). A ball moves the same
way in both, so the offset is found by sliding one ball track over the other: for each candidate
lag (broadcast index = tactical index - lag) the share of shared frames where the two balls are
within 2 m. One lag per half (the broadcast has cuts between halves). Replays and close-ups carry a
ball from another moment, so only broadcast frames with a valid homography count as fillable.

Result on our match (16 Nov): lags 4,014 and 26,920 frames; median disagreement 0.8 m; the
broadcast has a usable ball on 13% of the frames where the tactical ball is missing (40.7% -> 35.6%
missing). The bench simulation of that fill (33_broadcast_fill.py) gained +0.0015 spellF1, inside
seed noise, so the fill is not used (D56).

    python analysis/34_broadcast_align.py
"""
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from analysis.atd_arrays import load_arrays  # noqa: E402
from games import GAME, asset_paths, half_video_blocks  # noqa: E402

t = load_arrays(GAME)
b = load_arrays(GAME, source="broadcast")
L, W = t["pitch"]
nb = len(b["bx"])
H = pd.read_csv(asset_paths(GAME, source="broadcast")["homography"])
valid = np.zeros(nb, dtype=bool)
fr = H["frame"].to_numpy().astype(int) - 1
valid[fr[fr < nb]] = H["valid"].to_numpy()[fr < nb].astype(bool)
print("broadcast: {} frames, ball seen {:.0%}, homography valid {:.0%}".format(nb, np.mean(~np.isnan(b["bx"])), valid.mean()))


def agreement(f0, f1, lag_lo, lag_hi, step, every=25):
    it = np.arange(f0, f1 + 1, every)
    it = it[~np.isnan(t["bx"][it])]
    out = []
    for lag in range(lag_lo, lag_hi, step):
        ib = it - lag
        ok = (ib >= 0) & (ib < nb)
        if ok.sum() < 200:
            continue
        d = np.hypot(t["bx"][it[ok]] - b["bx"][ib[ok]], t["by"][it[ok]] - b["by"][ib[ok]])
        d = d[~np.isnan(d)]
        if len(d) >= 200:
            out.append(((d < 2).mean(), lag, float(np.median(d))))
    return sorted(out, reverse=True)


total_missing = total_fill = total_frames = 0
for half, f0, f1 in half_video_blocks(GAME):
    coarse = agreement(f0, f1, -5000, 120000, 5)
    lag = agreement(f0, f1, coarse[0][1] - 12, coarse[0][1] + 13, 1)[0][1]
    it = np.arange(f0, f1 + 1)
    ib = it - lag
    ok = (ib >= 0) & (ib < nb)
    it, ib = it[ok], ib[ok]
    tb, bb = np.isnan(t["bx"][it]), np.isnan(b["bx"][ib])
    both = ~tb & ~bb
    d = np.hypot(t["bx"][it][both] - b["bx"][ib][both], t["by"][it][both] - b["by"][ib][both])
    fillable = tb & ~bb & valid[ib]
    print("half {}: lag {} frames; both balls seen on {:.0%} of frames, median disagreement {:.2f} m, within 2 m {:.0%}; "
          "tactical ball missing {:.1%}, fillable {:.1%} of the half ({:.0%} of the missing frames)".format(
              half, lag, both.mean(), np.median(d), (d < 2).mean(), tb.mean(), fillable.mean(), fillable.sum() / tb.sum()))
    total_missing += tb.sum()
    total_fill += fillable.sum()
    total_frames += len(it)
print("overall: ball missing {:.1%} -> {:.1%} after filling".format(total_missing / total_frames, (total_missing - total_fill) / total_frames))
