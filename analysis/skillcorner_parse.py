"""Parse SkillCorner's open data (A-League 2024/25, 20 matches, MIT licence; credit SkillCorner) into
the same compact arrays as the IDSSE and Metrica matches, with control-style labels.

Source: github.com/SkillCorner/opendata, downloaded into external/skillcorner/data/<match id>/ (git-ignored).
Broadcast tracking at 10 fps with the ball, plus SkillCorner's automatic "phases of play"
(not human-corrected; "other games", so allowed by RULES.md).

Labels per frame (checked on match 1874553, analysis Phase 5 #1):
  team  = team_in_possession of the phase of play covering the frame. Consecutive phases of the
          same team form one possession; a phase change with a team change is a turnover.
          Phase teams agree with SkillCorner's own on-ball events 98.5% of the time (the per-frame
          possession.group field only 87%), so phases are used.
  dead  = frames covered by no phase, when that gap lasts >= DEAD_GAP_S (the tracking says
          "nobody in possession" on 74-95% of such frames and the ball is often outside the lines);
          shorter gaps (loose balls, 4 cases per match) keep the previous team. Before the first and
          after the last phase of a period: dead.
Tracking: positions in metres from the pitch centre -> metres from the corner, scaled to 105 x 68;
          10 fps -> 25 fps by linear interpolation (gaps are not bridged); ball kept only where
          SkillCorner detected it (is_detected), so the blackouts are real (60% visible, like ours);
          players include SkillCorner's extrapolated positions for those outside the camera view
          (KEEP_EXTRAPOLATED), since our tactical camera sees ~20.5 of 22 players.

    python analysis/skillcorner_parse.py            # all matches in external/skillcorner/data
"""
import glob
import json
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(ROOT, "external", "skillcorner", "data")
CACHE = os.path.join(ROOT, "external", "skillcorner", "cache")
LENGTH, WIDTH, FPS, SRC_FPS = 105.0, 68.0, 25, 10
DEAD_GAP_S = 2.0
KEEP_EXTRAPOLATED = True
TICK = 50                                # 1/50 s: common grid of 10 fps (5 ticks) and 25 fps (2 ticks)


def speeds(px, py):
    """m/s from positions over a centred 5-frame span (as for Metrica)."""
    sp = np.full(px.shape, np.nan)
    sp[2:-2] = np.hypot(px[4:] - px[:-4], py[4:] - py[:-4]) / (4 / FPS)
    return sp


def upsample(a10):
    """(n10, k) array at 10 fps -> 25 fps by linear interpolation; a missing 10 fps sample stays a gap."""
    n10 = a10.shape[0]
    i10 = np.arange(n10) * (TICK // SRC_FPS)
    i25 = np.arange(0, i10[-1] + 1, TICK // FPS)
    df = pd.DataFrame(a10, index=i10)
    df = df.reindex(df.index.union(i25)).interpolate(method="index", limit=2, limit_area="inside")
    return df.reindex(i25).to_numpy(), i25


def labels_from_phases(ph, start, n10, home):
    team = np.full(n10, "", dtype="<U1")
    covered = np.zeros(n10, dtype=bool)
    for _, r in ph.sort_values("frame_start").iterrows():
        s, e = int(r.frame_start) - start, int(r.frame_end) - start
        s, e = max(s, 0), min(max(e, s + 1), n10)
        team[s:e] = "H" if r.team_in_possession_id == home else "A"
        covered[s:e] = True
    lab = np.where(covered, team, "D")
    # short uncovered gaps between two phases: loose ball, keep the previous team
    edges = np.diff(np.concatenate(([1], covered.astype(np.int8), [1])))
    for a, b in zip(np.flatnonzero(edges == -1), np.flatnonzero(edges == 1)):
        if a > 0 and b < n10 and (b - a) < DEAD_GAP_S * SRC_FPS:
            lab[a:b] = lab[a - 1]
    return lab


def parse_match(mid):
    d = os.path.join(SRC, mid)
    m = json.load(open(os.path.join(d, "{}_match.json".format(mid))))
    home = m["home_team"]["id"]
    L, W = float(m.get("pitch_length") or LENGTH), float(m.get("pitch_width") or WIDTH)
    players = m["players"]
    col = {p["id"]: j for j, p in enumerate(players)}
    team = np.array([0 if p["team_id"] == home else 1 for p in players], dtype=np.int8)
    periods = {p["period"]: (int(p["start_frame"]), int(p["end_frame"])) for p in m["match_periods"]}
    ph = pd.read_csv(os.path.join(d, "{}_phases_of_play.csv".format(mid)))

    P = len(players)
    px10, py10, bx10, by10 = {}, {}, {}, {}
    for p, (a, b) in periods.items():
        n = b - a + 1
        px10[p], py10[p] = np.full((n, P), np.nan), np.full((n, P), np.nan)
        bx10[p], by10[p] = np.full(n, np.nan), np.full(n, np.nan)
    unknown = 0
    with open(os.path.join(d, "{}_tracking_extrapolated.jsonl".format(mid))) as f:
        for line in f:
            fr = json.loads(line)
            p = fr["period"]
            if p not in periods:
                continue
            i = fr["frame"] - periods[p][0]
            if not 0 <= i < len(bx10[p]):
                continue
            for q in fr["player_data"]:
                j = col.get(q["player_id"])
                if j is None:
                    unknown += 1
                    continue
                if q["x"] is None or (not KEEP_EXTRAPOLATED and not q.get("is_detected")):
                    continue
                px10[p][i, j] = (q["x"] + L / 2) * LENGTH / L
                py10[p][i, j] = (q["y"] + W / 2) * WIDTH / W
            bd = fr["ball_data"]
            if bd.get("is_detected") and bd["x"] is not None:
                bx10[p][i] = (bd["x"] + L / 2) * LENGTH / L
                by10[p][i] = (bd["y"] + W / 2) * WIDTH / W

    halves = []
    for p in sorted(periods):
        a, b = periods[p]
        n10 = b - a + 1
        lab10 = labels_from_phases(ph[ph.period == p], a, n10, home)
        px, i25 = upsample(px10[p])
        py, _ = upsample(py10[p])
        ball, _ = upsample(np.column_stack([bx10[p], by10[p]]))
        lab = lab10[i25 // (TICK // SRC_FPS)]
        halves.append(dict(label=lab, bx=ball[:, 0], by=ball[:, 1], px=px, py=py, ps=speeds(px, py), team=team))
    title = "{} vs {} ({})".format(m["home_team"]["short_name"], m["away_team"]["short_name"], m["date_time"][:10])
    return title, halves, unknown


def save(name, title, halves):
    out = {"title": title}
    for k, h in enumerate(halves, 1):
        for key, val in h.items():
            out["h{}_{}".format(k, key)] = val
    os.makedirs(CACHE, exist_ok=True)
    np.savez_compressed(os.path.join(CACHE, name + ".npz"), **out)


if __name__ == "__main__":
    ids = sys.argv[1:] or sorted(os.path.basename(p) for p in glob.glob(os.path.join(SRC, "[0-9]*")))
    for mid in ids:
        trk = os.path.join(SRC, mid, "{}_tracking_extrapolated.jsonl".format(mid))
        if not os.path.exists(trk) or os.path.getsize(trk) < 10000:
            print("skip", mid, "(tracking not downloaded)")
            continue
        title, halves, unknown = parse_match(mid)
        save("SC-" + mid, title, halves)
        for k, h in enumerate(halves, 1):
            lab = h["label"]
            print("saved SC-{} h{}: {} frames, dead {:.0f}%, H {:.0f}%, ball seen {:.0f}% (live {:.0f}%, dead {:.0f}%), "
                  "players/frame {:.1f}, unknown ids {}".format(
                      mid, k, len(lab), 100 * (lab == "D").mean(), 100 * (lab == "H").mean(),
                      100 * np.mean(~np.isnan(h["bx"])), 100 * np.mean(~np.isnan(h["bx"][lab != "D"])),
                      100 * np.mean(~np.isnan(h["bx"][lab == "D"])), np.mean(~np.isnan(h["px"])) * h["px"].shape[1],
                      unknown))
