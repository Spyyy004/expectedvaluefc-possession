# Expected Value FC - Technical Document

## More than a Hack 2027, Phase 1: Ball Possession

Team: **Expected Value FC** | Match: FC Barcelona vs Athletic Club, LaLiga 2024-25 | Code: **github.com/Spyyy004/expectedvaluefc-possession** | Submission: `ExpectedValueFC.csv` (SHA-1 `ce1d5c2499ee7d7224d11aa3184374d20d1d09f6`)

## 1. Summary

We label every frame of the tactical video as `H`, `A` or `D` with a three-layer system built only from the provided automatic tracking: football features per frame, two frame judges (a gradient-boosted tree classifier for dead ball; a small bidirectional GRU reading 20-second windows for home vs away), and a hidden Markov "match-flow referee" decoded with Viterbi. Because the rules forbid training or tuning on the provided game, **every model and setting was trained and chosen on external data**: 7 Bundesliga matches (IDSSE/DFL) and 3 Metrica sample games, deliberately degraded to look like the provided automatic tracking. Every change was judged per match with the official scoring code.

On our external test bench (each match predicted by models that never saw it), the final system scores **spellF1 0.639 on the Metrica games** (baseline 0.262) and **0.533 on the Bundesliga matches** (baseline 0.203), beating the organisers' baseline on all 10 matches. On the provided match it predicts 230 possessions (mean 15.0 s, median 9.4 s), with Barcelona holding the ball 62% of live time.

## 2. Approach and why

The ranking metric (spellF1) scores whole possessions, so the central problem is **flicker**: a frame-level classifier with 87% team accuracy still scores poorly because one wrong frame splits a possession into three. We therefore separate "what does each frame look like" from "what is the most believable sequence of possessions".

| Layer | What it does | Why this choice |
|---|---|---|
| 1. Features | ~75 football features per frame from tracking, using past and future windows (offline labelling) | Interpretable, and each one can be checked for transfer to our match |
| 2a. Dead-ball judge | `HistGradientBoostingClassifier` (300 iterations, learning rate 0.1): P(dead) on all frames | Strong on tabular data, fast, deterministic, works with ~10 matches |
| 2b. Team judge | Bidirectional GRU (2 layers of 64 units, 132k parameters) reading the same 58 team and turnover features, standardised with a missing-flag per feature, over 512-frame (20.5 s) windows; P(home) on live frames; three seeds averaged; trained on the CPU with fixed seeds (`analysis/sequence_judge.py`) | A turnover is a sequence, not a frame. The GRU replaced a gradient-boosted frame judge on 16 Nov after it won on all 10 bench matches (+0.039 spellF1, 3 seeds, 12 epochs), our largest single gain; averaging it with the boosted judge was weaker (+0.031) |
| 3. Referee | 4-state HMM (home, away, dead-after-home, dead-after-away), Viterbi; switch cost 20, stop cost 80, restart cost 2 (other team) / 5 (same team), team-evidence weight 3 | Turns noisy per-frame evidence into stable possessions; stoppages and restarts are explicit states |
| Rule | A ball blackout longer than 15 s ending on the centre spot is a goal stoppage | Goal celebrations make the tracker lose the ball; all 4 such blackouts coincide with tagged kick-offs |
| Rule | Graded frames before a half's kick-off are dead ball | The second half's graded block starts 117 s before the organisers' published kick-off anchor (`games.py`), during warm-up and line-up; the tracking shows balanced sides and the ball out of view just before the anchor, and the first Smart Tagging phase of the half starts 81 frames before it. Removes 8 warm-up "possessions" |

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
| Metrica game 1 | 0.280 | **0.657** | 0.675 | 0.808 | 256 / 277 |
| Metrica game 2 | 0.278 | **0.621** | 0.692 | 0.767 | 232 / 276 |
| Metrica game 3 | 0.234 | **0.638** | 0.656 | 0.791 | 254 / 318 |
| Koln - Bayern | 0.196 | **0.471** | 0.662 | 0.865 | 264 / 511 |
| Bochum - Leverkusen | 0.211 | **0.517** | 0.647 | 0.846 | 301 / 500 |
| Dusseldorf - Regensburg | 0.198 | **0.570** | 0.643 | 0.864 | 268 / 402 |
| Dusseldorf - Rostock | 0.197 | **0.542** | 0.638 | 0.864 | 301 / 456 |
| Dusseldorf - Nurnberg | 0.209 | **0.529** | 0.662 | 0.871 | 275 / 421 |
| Dusseldorf - St. Pauli | 0.218 | **0.581** | 0.664 | 0.878 | 298 / 401 |
| Dusseldorf - Kaiserslautern | 0.194 | **0.532** | 0.657 | 0.849 | 266 / 370 |
| **Metrica (pooled)** | 0.262 | **0.639** | 0.676 | 0.790 | |
| **Bundesliga (pooled)** | 0.203 | **0.533** | 0.657 | 0.865 | |

**The GRU team judge** (16 Nov) was compared with the gradient-boosted frame judge it replaced on exactly this protocol: +0.039 spellF1 over the 10 matches (10 of 10 better, sign test p < 0.01, bootstrap 90% interval +0.032 to +0.047; Metrica +0.030, Bundesliga +0.044), with the smallest per-match gain +0.014 and the largest +0.069, and macroF1 up by +0.008 to +0.010. A first run with one seed and 8 epochs had shown +0.015; averaging three seeds and training for 12 epochs raised it. Possession counts hardly change (742 vs 737 predicted on Metrica), so the gain comes from better boundaries and fewer wrong-team calls, not from calling more possessions.

Referee settings were re-tuned on 15 Nov by nested leave-one-match-out on this bench (the original settings came from the 7 Bundesliga matches alone): 6 of 10 folds chose the same setting in two grid rounds, the nested gain is +0.006 spellF1 (9 of 10 matches, sign test p = 0.02), and it holds for team models trained with four different random seeds. Re-tuning them again on the GRU's probabilities (three nested grids, 675 and 193 settings) found nothing better: the current setting is within 0.003 of the best pooled one and every nested alternative lost (4 of 10 matches at best), so the settings stayed. The table applies the chosen setting to every match, so it is slightly optimistic by about 0.003 versus the nested estimate. Training-seed noise alone moves a single match by up to ±0.01 and the 10-match mean by about ±0.003, which is the resolution of this bench.

Bundesliga labels count many brief "touch" possessions (30% under 2 s) that a control-based definition does not, which is why scores are lower there and our possession counts are below the truth.

**What each step contributed** (spellF1): dead-ball detector +0.07 and team judge +0.14 over the baseline on the Bundesliga bench; HMM referee +0.02; turnover features +0.02 under realistic noise; Metrica in training +0.04 on Metrica; stoppage-linked blackouts +0.024 on Metrica; wrong-team robustness +0.010 on Metrica (3/3 matches); referee re-tuned on the full bench +0.006 (9/10 matches); GRU sequence judge in place of the boosted frame judge +0.039 (10/10 matches).

**Tested and not adopted** (all with nested leave-one-match-out): location-of-blackout features; per-frame change-moment costs; touch-event features; larger models; time shifts; training on control-style converted labels (lost on 10/10 matches); X abstention; a duration-aware semi-Markov decoder (no gain on Metrica); team-velocity crossing features (+0.027 on Bundesliga but nothing on Metrica, and twice as frequent there as in our match); a last-touch-before-out rule (lost on 10/10); an inferred ball from where the players are running ("convergence attractor": 18.7 m median error on Metrica, no better than the players' centroid, and worse than the last-seen clue for the team call); **20 more training matches from SkillCorner's open data** (A-League 2024-25, MIT licence, broadcast tracking at 10 fps upsampled to 25, labels derived by us from SkillCorner's automatic phases of play). Its label style matches Metrica's (24 vs 23 possessions per 5 live minutes) and every feature's distribution matches our match, yet adding it to training lost on 9 of 10 bench matches (Metrica −0.008 to −0.017, Bundesliga −0.021 to −0.027 spellF1, three weighting variants). Broadcast tracking, with about ten of the 22 players' positions estimated rather than seen, teaches the team model to hesitate at turnovers. More data of a different kind of tracking did not help; data volume is not our bottleneck. **Turnover-weighted training** of the team model (extra weight on short possessions or on the first seconds after each change of possession, five variants) moved the mean by −0.001 to +0.006, inside the seed noise of the bench, with single matches swinging ±0.02 to ±0.04 between variants; **averaging several team models** (two to four seeds, with or without the weighting) gained +0.005 to +0.016 on Metrica but nothing on Bundesliga, failed a pre-set adoption rule, and lost under the re-tuned referee. **The broadcast tracking's ball**, aligned to the tactical timeline by sliding the two ball tracks over each other (lags of 4,014 and 26,920 frames per half; median disagreement 0.8 m where both see the ball; `analysis/34_broadcast_align.py`), can fill 13% of our missing-ball frames (40.7% to 35.6% missing). A bench simulation of exactly that fill, with only the team judge seeing the filled ball, gained +0.0015 spellF1 over the 10 matches, inside the seed noise: our misses are not where the ball is missing. A **two-stage judge** that re-reads the first judge's probabilities over 1, 3 and 10 s windows lost on 9 of 10 matches (−0.007): wider context smooths over the short possessions instead of recovering them. A **learned ball imputer** (gradient-boosted regressor from player geometry and the ball's last and next sightings; `analysis/36_ball_imputer.py`) guesses the hidden ball to 8.3 m median on Bundesliga blackouts, against 13.6 m for interpolation and 19.6 m for the players' centroid, yet feeding that guess to the team judge lost 0.012 spellF1 (8 of 10 matches worse): a wrong ball presented as real is worse than an honest "missing", which the judge already handles with last-seen and next-seen clues. **Averaging the GRU with the boosted judge** (+0.031, against +0.039 for the GRU alone) and **re-tuning the referee for the GRU** (above) were tested and not adopted.

**Diagnostic ceilings.** Perfect per-frame evidence through our referee gives 0.998, so the decoder is not the bottleneck. Our evidence with the true boundaries fixed gives 0.744 on Metrica: about 0.14 of the remaining gap is boundary timing and about 0.26 is wrong labels on short possessions.

## 8. Limitations

- **Short possessions and interceptions** are the dominant error: we find 82% of possessions over 10 s but only a third of those of 2-5 s. On Metrica's real tracking this is **not** a blackout problem: the ball is fully visible in 84% of the short possessions we miss (the same share as in those we find). Of the misses, 71% are given to the other team, 22% fall inside a stoppage we called (mostly 1-2 s timing at the edges of real stoppages) and 6% are boundary timing. A ceiling check shows where the room is: a perfect judge of the team model's brief leans would add only +0.001 spellF1 (our referee already follows clear leans), while fixing every missed short possession would add +0.17 on Metrica. Almost all of that sits where the **team model itself reads the wrong team** although the ball is tracked. The GRU team judge was the answer to exactly this diagnosis (a sharper judge, not a different referee); the remaining room is still there, and larger or longer-trained sequence models are the obvious next step.
- **Definition uncertainty.** Our training labels are touch-based (DFL) or derived by us (Metrica); the ground truth comes from corrected tracking. Evidence suggests it is control-based in open play but last-touch just before the ball goes out; we could not exploit this reliably. On the Bundesliga bench we send 43% of restarts to the other team where the truth says 70%.
- **The bench is an approximation.** Synthetic blackouts are placed by game state (dead/live) but not at duels and aerial balls, where real ones cluster. Our match's team movement is calmer than every external match.
- **Video not used.** Video ball detection was allowed as an input but not used: it could not be validated on any external data available to us (our external matches have no video), and 84% of our short-possession misses occur with the ball tracked.
- **Small external pool** (10 matches); the learning curve was still rising.
- The public 5-minute alignment check (spellF1 0.61, macroF1 0.82 for the final file; 0.55 / 0.79 with the boosted team judge; best-tracked stretch of the match) hints the test labels may be finer than Metrica's. We did not react to it: every choice was made on the 10 external matches.

## 9. Reproduction

```
git clone https://github.com/Spyyy004/expectedvaluefc-possession && cd expectedvaluefc-possession
python3 -m venv .venv && source .venv/bin/activate      # Python 3.14.3
pip install -r requirements.txt                           # numpy 2.5.4, pandas 3.1.1, scikit-learn 1.9.1, torch 2.15.0
# put the provided match data in data/ (or set MTAH_DATA_DIR)
python fetch_external.py                                  # downloads, verifies and parses external data
python make_submission.py                                 # -> submissions/ExpectedValueFC.csv (~2 min with the saved GRU weights)
python evaluation/validate_submission.py submissions/ExpectedValueFC.csv
python make_submission.py --retrain-gru                   # optional: retrain the three GRU seeds on the CPU (~12 min) and rebuild
```

The output is deterministic and byte-identical across runs and processor thread counts. The three GRU networks are trained on the CPU only (PyTorch's deterministic algorithms, fixed seeds 0-2) and their weights are committed in `models/` (1.6 MB) together with the feature statistics; `make_submission.py` checks that the statistics match the training data it rebuilt and loads the weights, so a fresh clone reproduces the file without retraining. `--retrain-gru` regenerated identical weights on our machine; on a different CPU the last bits of floating-point arithmetic may differ, which is why the weights ship with the code. Every experiment behind this document is in `analysis/` (scripts 01-37), and every decision with its evidence is in `docs/process-log.html` (D1-D60).

**Data credits.** IDSSE: Bassek, M., Rein, R., Weber, H. & Memmert, D. (2025), An integrated dataset of spatiotemporal and event data in elite soccer, Scientific Data, figshare DOI 10.6084/m9.figshare.28196177, CC-BY 4.0, with authorisation of the DFL. Metrica Sports open sample data, github.com/metrica-sports/sample-data.
