# Expected Value FC - Technical Document

## More than a Hack 2027, Phase 1: Ball Possession

Team: **Expected Value FC** | Match: FC Barcelona vs Athletic Club, LaLiga 2024-25 | Code: **github.com/Spyyy004/expectedvaluefc-possession** | Submission: `ExpectedValueFC.csv` (SHA-1 `7fe033d5ae8253c0c07ad736b95d0c4d1edb12e9`)

## 1. Summary

We label every frame of the tactical video as `H`, `A` or `D` with a three-layer system built only from the provided automatic tracking: football features per frame, two gradient-boosted tree classifiers (dead ball; home vs away), and a hidden Markov "match-flow referee" decoded with Viterbi. Because the rules forbid training or tuning on the provided game, **every model and setting was trained and chosen on external data**: 7 Bundesliga matches (IDSSE/DFL) and 3 Metrica sample games, deliberately degraded to look like the provided automatic tracking. Every change was judged per match with the official scoring code.

On our external test bench (each match predicted by models that never saw it), the final system scores **spellF1 0.595 on the Metrica games** (baseline 0.262) and **0.483 on the Bundesliga matches** (baseline 0.203), beating the organisers' baseline on all 10 matches. On the provided match it predicts 249 possessions (mean 14.2 s), with Barcelona holding the ball 62% of live time.

## 2. Approach and why

The ranking metric (spellF1) scores whole possessions, so the central problem is **flicker**: a frame-level classifier with 87% team accuracy still scores poorly because one wrong frame splits a possession into three. We therefore separate "what does each frame look like" from "what is the most believable sequence of possessions".

| Layer | What it does | Why this choice |
|---|---|---|
| 1. Features | ~75 football features per frame from tracking, using past and future windows (offline labelling) | Interpretable, and each one can be checked for transfer to our match |
| 2. Frame judges | `HistGradientBoostingClassifier` (300 iterations, learning rate 0.1): P(dead) on all frames; P(home) on live frames | Strong on tabular data, fast, deterministic, works with ~10 matches; deep models would need far more data |
| 3. Referee | 4-state HMM (home, away, dead-after-home, dead-after-away), Viterbi; switch cost 10, stop cost 160, restart costs 5, team-evidence weight 2 | Turns noisy per-frame evidence into stable possessions; stoppages and restarts are explicit states |
| Rule | A ball blackout longer than 15 s ending on the centre spot is a goal stoppage | Goal celebrations make the tracker lose the ball; all 4 such blackouts coincide with tagged kick-offs |

**Most useful features.** Dead ball: players' mean speed (now, 1/3/6 s windows, next 3 s), share of players walking, ball just went out, ball blackout length, gap between team centroids (kick-offs). Team: team width difference over 3 s, block moving forward, nearest home vs away player over the last 3 s, team speed difference, close control (player within 1.5 m) before and after, receptions, controller changes, shape flips. The attacking direction of each team is inferred per half from team centroids; no constant is tied to a frame of the match.

## 3. Assets used and dropped

| Asset | Use |
|---|---|
| Tactical automatic tracking (ATD) | **Core input.** All features. Positions scaled to a 105 x 68 m pitch. |
| Smart Tagging | Used for understanding only (it confirmed the goal rule: all 4 long centre-spot blackouts match kick-off tags). Not a model input, because the external matches have no equivalent and its effect cannot be measured. |
| Homography, tactical video | Not used. The video was watched only to understand situations, never to check output. |
| Broadcast video, tracking, homography | Dropped: not synchronised to the tactical timeline; syncing would cost days. |
| Public 5-minute ground truth | Format and frame-alignment check only, plus understanding what counts as possession. Never used to train, tune or choose between versions. |

## 4. Training data

**No label of any kind from the provided game was used for training or validation.**

| Dataset | Matches | Labels and who produced them | Licence |
|---|---|---|---|
| IDSSE / DFL open data (Bassek et al. 2025, Scientific Data) | 7: Koln-Bayern, Bochum-Leverkusen (Bundesliga); Dusseldorf vs Regensburg, Rostock, Nurnberg, St. Pauli, Kaiserslautern (2. Bundesliga), 2022-23 | Per-frame `BallPossession` and `BallStatus` from operator-corrected TRACAB tracking (DFL / Sportec Solutions) | CC-BY 4.0 |
| Metrica Sports open sample data | 3 anonymised games | Metrica's own event data; **we derived per-frame labels algorithmically**: possession = team of the latest controlled action (pass, carry, recovery, set piece, shot); dead from ball-out, foul or goal until the next set piece | Open; use confirmed by the organisers |

About 950 minutes of labelled football in total. **Making external data look like our tracking** (statistics taken only from the provided tracking or from the external data, as the organisers confirmed is allowed):

- **Ball blackouts.** Real tracking loses the ball far more during stoppages: on Metrica's tracking it is missing on 79% of dead frames vs 16% of live frames. Bundesliga balls are removed at these rates, with gap lengths drawn from our match's tracking (blackout length quartiles 6.3/12.1/22.5 s vs ours 6.1/10.7/21.1 s).
- **Un-frozen ball.** The DFL data parks the ball during stoppages (still on 99% of dead frames, vs 2% in real tracking). We replace it with "nearest player carries it, then it moves to the restart spot".
- **Player dropouts** to ~20.5 visible players and **1 m smooth position jitter**, matching our tracking's close-control statistics.
- **Wrong-team episodes.** Our tracking sometimes lists a player under the wrong team (one team shows 12, the other 10): about once per minute, median 2.6 s, 10.8% of frames. We copy this into the training data.

## 5. Possession rules implemented

1. **Three states, always exactly one.** A loose ball belongs to the last team in control.
2. **Dead ball** when the ball leaves the pitch, a goal is scored or play is stopped; for fouls, from the moment of the foul. Detected by combining evidence (players slowing and walking, ball outside the lines, long blackouts). Stoppages are never brief, so dead "blips" are removed by the referee.
3. **Restarts**: set pieces end stoppages; the referee has explicit dead-after-home and dead-after-away states.
4. **Open-play changes need control** (the challenge's definition and the Opta convention): a touch, deflection, 50/50 or aerial duel without control does not change possession; a brief spell counts only if it contains a deliberate action.
5. **Missing ball** (41% of graded frames, mostly blackouts over 10 s): bridged with who last and next had close control, team shape and players' movement.
6. **Bad tracking is distrusted**: team-level features only, ball speed from positions (the provided speed field is missing 92% of the time), robustness to wrong-team episodes.
7. **Always guess on graded frames.** We compared "write X when unsure" with "always guess": because the scorer fills X with the previous label, X inside a possession leaves spellF1 unchanged and costs macroF1.

## 6. Missing and noisy tracking

The provided tracking is the main difficulty: the ball is missing on 41% of graded frames, players swap track identities about 11 times a minute, positions wobble by about a metre, and about 11% of frames have a player under the wrong team. We handle it in three ways: (a) features that do not need the ball (team width, block, speeds, walking share) and features that bridge blackouts (last/next seen values); (b) training on external data degraded with exactly these defects; (c) a **transfer check** before trusting any feature: its overall distribution must match between the external bench and our match (no labels needed). This check caught several traps: the DFL frozen ball (ball-speed features dropped), the ball sitting outside the lines during DFL stoppages (fixed by un-freezing), touch counts inflated by jitter (debounced), and team-movement signals that occur twice as often in every external match as in ours (not adopted).

## 7. Validation and results

**Protocol.** Leave-one-match-out over the 10 external matches; settings chosen by nested leave-one-match-out; every change judged by **per-match paired deltas** (wins, sign test, bootstrap 90% interval) with the official `possession_metrics.evaluate`. Metrica is the primary scoreboard: same provider as the ground truth, and its label style is closest to the public 5 minutes (about 15-17 possessions per 5 minutes, median 7 s).

**Per-match results** of the final system on the realistic bench (with wrong-team episodes):

| Match | Baseline spellF1 | **Our spellF1** | Baseline macroF1 | Our macroF1 | Possessions ours / true |
|---|---|---|---|---|---|
| Metrica game 1 | 0.280 | **0.600** | 0.675 | 0.799 | 264 / 277 |
| Metrica game 2 | 0.278 | **0.590** | 0.692 | 0.769 | 243 / 276 |
| Metrica game 3 | 0.234 | **0.595** | 0.656 | 0.781 | 278 / 318 |
| Koln - Bayern | 0.196 | **0.427** | 0.662 | 0.850 | 273 / 511 |
| Bochum - Leverkusen | 0.211 | **0.458** | 0.647 | 0.832 | 315 / 500 |
| Dusseldorf - Regensburg | 0.198 | **0.506** | 0.643 | 0.846 | 284 / 402 |
| Dusseldorf - Rostock | 0.197 | **0.489** | 0.638 | 0.851 | 300 / 456 |
| Dusseldorf - Nurnberg | 0.209 | **0.488** | 0.662 | 0.869 | 270 / 421 |
| Dusseldorf - St. Pauli | 0.218 | **0.529** | 0.664 | 0.865 | 307 / 401 |
| Dusseldorf - Kaiserslautern | 0.194 | **0.494** | 0.657 | 0.849 | 273 / 370 |
| **Metrica (pooled)** | 0.262 | **0.595** | 0.676 | 0.785 | |
| **Bundesliga (pooled)** | 0.203 | **0.483** | 0.657 | 0.855 | |

Bundesliga labels count many brief "touch" possessions (30% under 2 s) that a control-based definition does not, which is why scores are lower there and our possession counts are below the truth.

**What each step contributed** (spellF1): dead-ball detector +0.07 and team judge +0.14 over the baseline on the Bundesliga bench; HMM referee +0.02; turnover features +0.02 under realistic noise; Metrica in training +0.04 on Metrica; stoppage-linked blackouts +0.024 on Metrica; wrong-team robustness +0.010 on Metrica (3/3 matches).

**Tested and not adopted** (all with nested leave-one-match-out): location-of-blackout features; per-frame change-moment costs; touch-event features; larger models; time shifts; training on control-style converted labels (lost on 10/10 matches); X abstention; a duration-aware semi-Markov decoder (no gain on Metrica); team-velocity crossing features (+0.027 on Bundesliga but nothing on Metrica, and twice as frequent there as in our match); a last-touch-before-out rule (lost on 10/10); an inferred ball from where the players are running ("convergence attractor": 18.7 m median error on Metrica, no better than the players' centroid, and worse than the last-seen clue for the team call).

**Diagnostic ceilings.** Perfect per-frame evidence through our referee gives 0.998, so the decoder is not the bottleneck. Our evidence with the true boundaries fixed gives 0.744 on Metrica: about 0.14 of the remaining gap is boundary timing and about 0.26 is wrong labels on short possessions.

## 8. Limitations

- **Short possessions and interceptions** are the dominant error: we find 82% of possessions over 10 s but only a third of those of 2-5 s. On Metrica's real tracking this is **not** a blackout problem: the ball is fully visible in 84% of the short possessions we miss (the same share as in those we find). Of the misses, 71% are given to the other team, 22% fall inside a stoppage we called (mostly 1-2 s timing at the edges of real stoppages) and 6% are boundary timing. A ceiling check shows where the room is: a perfect judge of the team model's brief leans would add only +0.001 spellF1 (our referee already follows clear leans), while fixing every missed short possession would add +0.17 on Metrica. Almost all of that sits where the **team model itself reads the wrong team** although the ball is tracked. The next step would be a sharper short-window team judge, not a different referee.
- **Definition uncertainty.** Our training labels are touch-based (DFL) or derived by us (Metrica); the ground truth comes from corrected tracking. Evidence suggests it is control-based in open play but last-touch just before the ball goes out; we could not exploit this reliably. On the Bundesliga bench we send 43% of restarts to the other team where the truth says 70%.
- **The bench is an approximation.** Synthetic blackouts are placed by game state (dead/live) but not at duels and aerial balls, where real ones cluster. Our match's team movement is calmer than every external match.
- **Small external pool** (10 matches); the learning curve was still rising.
- The public 5-minute alignment check (spellF1 0.58, macroF1 0.90; best-tracked stretch of the match) hints the test labels may be finer than Metrica's. We did not react to it.

## 9. Reproduction

```
git clone https://github.com/Spyyy004/expectedvaluefc-possession && cd expectedvaluefc-possession
python3 -m venv .venv && source .venv/bin/activate      # Python 3.14.3
pip install -r requirements.txt                           # numpy 2.5.4, pandas 3.1.1, scikit-learn 1.9.1
# put the provided match data in data/ (or set MTAH_DATA_DIR)
python fetch_external.py                                  # downloads, verifies and parses external data
python make_submission.py                                 # -> submissions/ExpectedValueFC.csv (~1-2 min)
python evaluation/validate_submission.py submissions/ExpectedValueFC.csv
```

The output is deterministic and byte-identical across runs and processor thread counts. Every experiment behind this document is in `analysis/` (scripts 01-22), and every decision with its evidence is in `docs/process-log.html` (D1-D48).

**Data credits.** IDSSE: Bassek, M., Rein, R., Weber, H. & Memmert, D. (2025), An integrated dataset of spatiotemporal and event data in elite soccer, Scientific Data, figshare DOI 10.6084/m9.figshare.28196177, CC-BY 4.0, with authorisation of the DFL. Metrica Sports open sample data, github.com/metrica-sports/sample-data.
