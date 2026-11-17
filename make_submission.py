"""Produce our possession submission for BAR-ATH, end to end, from the provided data.

    python make_submission.py                       # writes submissions/ExpectedValueFC.csv
    python make_submission.py --out other.csv

Pipeline (see docs/possession_rules.md and docs/process-log.html):
  1. Train on 10 external matches (D38): 7 Bundesliga (IDSSE/DFL, CC-BY 4.0) made messy like our
     tracking (ball blackouts linked to stoppages at Metrica's rates (D39), player dropouts,
     unfrozen ball, 1 m position jitter, wrong-team tracking episodes (D45)) and 3 Metrica
     sample games (organisers confirmed use; player dropouts and 1 m jitter added):
       - dead-ball detector  (gradient-boosted trees on dead-ball clues)        Step 7
       - team judge          (bidirectional GRU over 20 s windows of the team + turnover clues,
                              three seeds averaged; Phase 7 #3, D60. Until 16 Nov this was a
                              gradient-boosted model on single frames, Steps 8a, 9)
  2. Our match: compute the same clues per half, predict, then decode with the match-flow referee
     (4-state HMM + Viterbi, settings chosen on the bench)                       Step 8b
  3. Goal rule: a long ball blackout ending with the ball on the centre spot is a goal stoppage.
  4. Kick-off rule: frames before the organisers' kick-off anchor are dead (D55).
  5. Write one label per tactical-video frame; X outside the two halves.

Nothing here reads the ground truth. All settings were chosen on external data only.
Deterministic: fixed seeds, CPU only; running it twice gives the identical file. The GRU weights
are trained on the CPU with deterministic algorithms (about 12 minutes) and saved in models/, so a
later run or a fresh clone loads them instead of retraining. --retrain-gru trains them again.
"""
import argparse
import os
import sys
import time

import numpy as np
import pandas as pd
import torch
from sklearn.ensemble import HistGradientBoostingClassifier

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
from games import GAMES, anchor_video_frame, half_video_blocks, source_frames  # noqa: E402
from analysis.atd_arrays import load_arrays  # noqa: E402
from analysis.bench import bench, our_mess_profile, BENCH_JITTER  # noqa: E402
from analysis.features import (dead_ball_features, team_features, turnover_features,  # noqa: E402
                               LENGTH, WIDTH)
from analysis.dead_ball import FROZEN_BALL_ARTEFACTS, TRAIN_EVERY  # noqa: E402
from analysis.referee import viterbi, REFEREE_SETTINGS  # noqa: E402
from analysis import sequence_judge as sj  # noqa: E402

FPS = 25
GAME = "BAR-ATH"
MODELS = os.path.join(ROOT, "models")
CPU = torch.device("cpu")
torch.use_deterministic_algorithms(True)
# Goal rule (Step 7 follow-up): long blackout ending on the centre spot = goal celebration + kick-off.
GOAL_BLACKOUT_MIN_S = 15.0
CENTRE_SPOT_RADIUS_M = 3.0


def model():
    return HistGradientBoostingClassifier(max_iter=300, learning_rate=0.1, random_state=0)


def dead_X(h):
    return dead_ball_features(h).drop(columns=FROZEN_BALL_ARTEFACTS)


def team_X(h):
    return pd.concat([team_features(h), turnover_features(h)], axis=1)


def train_models(retrain_gru=False):
    halves, _ = bench(our_mess_profile(), jitter=BENCH_JITTER, sources=("dfl", "metrica"), dead_link=True,
                      wrong_team=True)
    Xd = [dead_X(h) for h in halves]
    yd = [h["label"] == "D" for h in halves]
    dead_model = model().fit(np.vstack([x.to_numpy()[::TRAIN_EVERY] for x in Xd]),
                             np.concatenate([y[::TRAIN_EVERY] for y in yd]))
    Xt = [team_X(h).to_numpy(dtype=np.float32) for h in halves]
    mu, sd = sj.stats(Xt)
    paths = [os.path.join(MODELS, "gru_seed{}.pt".format(s)) for s in range(sj.SEEDS)]
    stats_path = os.path.join(MODELS, "gru_stats.npz")
    if not retrain_gru and all(os.path.exists(p) for p in paths) and os.path.exists(stats_path):
        z = np.load(stats_path)
        if not (np.array_equal(z["mu"], mu) and np.array_equal(z["sd"], sd)):
            raise RuntimeError("models/gru_stats.npz does not match the training clues; run with --retrain-gru")
        nets = []
        for p in paths:
            net = sj.Net(Xt[0].shape[1])
            net.load_state_dict(torch.load(p, map_location=CPU))
            nets.append(net)
        print("  GRU team judge: loaded {} saved networks from models/".format(len(nets)), flush=True)
    else:
        E = [sj.encode(x, mu, sd) for x in Xt]
        Y = [(h["label"] == "H") for h in halves]
        LIVE = [(h["label"] != "D") for h in halves]
        os.makedirs(MODELS, exist_ok=True)
        nets = []
        for s in range(sj.SEEDS):
            print("  GRU team judge: training seed {} on the CPU ...".format(s), flush=True)
            net = sj.train_one(E, Y, LIVE, s, CPU, log=lambda m: print(m, flush=True))
            torch.save(net.state_dict(), paths[s])
            nets.append(net)
        np.savez(stats_path, mu=mu, sd=sd)
    return dead_model, (nets, mu, sd)


def our_halves():
    """Our match, one dict per graded half, scaled to the 105 x 68 m pitch the models learned on."""
    t = load_arrays(GAME)
    sx = LENGTH / t["pitch"][0]
    sy = WIDTH / t["pitch"][1]
    out = []
    for half, first, last in half_video_blocks(GAME):
        s = slice(first, last + 1)
        out.append(dict(half=half, first=first, last=last, team=t["team"],
                        px=t["px"][s] * sx, py=t["py"][s] * sy, ps=t["ps"][s],
                        bx=t["bx"][s] * sx, by=t["by"][s] * sy))
    return out


def apply_goal_rule(h, p_dead):
    """Force dead through long blackouts that end with the ball on the centre spot."""
    p = p_dead.copy()
    missing = np.isnan(h["bx"])
    edges = np.diff(np.concatenate(([0], missing.astype(np.int8), [0])))
    n_goals = 0
    for a, b in zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)):
        if b >= len(missing) or (b - a) < GOAL_BLACKOUT_MIN_S * FPS:
            continue
        if np.hypot(h["bx"][b] - LENGTH / 2, h["by"][b] - WIDTH / 2) < CENTRE_SPOT_RADIUS_M:
            p[a:b] = 0.999
            n_goals += 1
    return p, n_goals


def apply_kickoff_rule(h, p_dead):
    """No possession before the kick-off (D55).

    The graded block of a half can start before the ball is struck: the second half's block begins
    117 s before the organisers' kick-off anchor in games.py, while the players warm up and line up.
    The anchor is organiser metadata ("the ball is struck on this frame"), and the tracking agrees
    (both sides balanced and the ball out of view just before it, play after it). Frames from the
    block start up to the anchor become dead, exactly as the goal rule forces dead through a goal.
    """
    anchor = GAMES[GAME]["kickoff" if h["half"] == 1 else "kickoff_2nd"]
    n = max(0, min(anchor_video_frame(anchor) - h["first"], len(p_dead)))
    p = p_dead.copy()
    p[:n] = 0.999
    return p, n


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    # SUBMISSION.md: name the file after the team.
    parser.add_argument("--out", default=os.path.join(ROOT, "submissions", "ExpectedValueFC.csv"))
    parser.add_argument("--retrain-gru", action="store_true", help="train the GRU team judge again instead of loading models/")
    args = parser.parse_args()
    t0 = time.time()

    print("training on 10 external matches (7 Bundesliga + 3 Metrica) ...", flush=True)
    dead_model, (nets, mu, sd) = train_models(args.retrain_gru)
    print("  done in {:.0f} s".format(time.time() - t0), flush=True)

    labels = np.full(source_frames("tactical", GAME), "X", dtype="<U1")
    for h in our_halves():
        p_dead = dead_model.predict_proba(dead_X(h).to_numpy())[:, 1]
        p_dead, n_goals = apply_goal_rule(h, p_dead)
        p_dead, n_pre = apply_kickoff_rule(h, p_dead)
        p_home = sj.predict_ensemble(nets, sj.encode(team_X(h).to_numpy(dtype=np.float32), mu, sd), CPU)
        labels[h["first"]:h["last"] + 1] = viterbi(p_dead, p_home, **REFEREE_SETTINGS)
        print("  half {}: goal-rule blackouts {}, pre-kick-off frames forced dead {}".format(h["half"], n_goals, n_pre), flush=True)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        f.write("frame,label\n")
        f.writelines("{},{}\n".format(i, lab) for i, lab in enumerate(labels))
    print("wrote {} rows to {} in {:.0f} s".format(len(labels), os.path.relpath(args.out, ROOT), time.time() - t0))


if __name__ == "__main__":
    main()
