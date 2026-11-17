# Expected Value FC — More than a Hack 2027, Phase 1: Ball Possession

Frame-level ball possession for **FC Barcelona vs Athletic Club** (LaLiga 2024-25): one label per
tactical-video frame — `H` (home in possession), `A` (away in possession), `D` (dead ball) — from
the provided automatic tracking only. Team **Expected Value FC**.

## Reproduce the submission

```bash
python3 -m venv .venv && source .venv/bin/activate     # tested with Python 3.14.3
pip install -r requirements.txt                          # pinned versions

# 1. Provided match data (gated; not in this repository): unpack the organisers' bundle into data/
#    (data/tactical/BAR-ATH_tactical_atd.xml, ..._atd.txt, ...) or point MTAH_DATA_DIR at it.
# 2. External training data (open; downloaded and checksum-verified, ~2.8 GB):
python fetch_external.py
# 3. Build the submission (~2 minutes with the saved GRU weights in models/, deterministic):
python make_submission.py                                # -> submissions/ExpectedValueFC.csv
#    optional: retrain the three GRU seeds on the CPU (~12 minutes) and rebuild
python make_submission.py --retrain-gru
python evaluation/validate_submission.py submissions/ExpectedValueFC.csv
```

Expected output: `submissions/ExpectedValueFC.csv`, 199,319 rows,
SHA-1 `ce1d5c2499ee7d7224d11aa3184374d20d1d09f6`. The result is byte-identical across runs and
across processor thread counts. The GRU is trained on the CPU only (PyTorch deterministic
algorithms, fixed seeds); the weights ship in `models/` because another CPU may differ in the last
bits of floating-point arithmetic, and `make_submission.py` checks they match the training data.

## Method in one screen

```
Layer 1  Football features per frame from tracking only (past and future windows):
           players' speeds and walking share, ball outside the lines, ball blackouts,
           nearest home vs away player (sustained), team width / block / movement,
           close control, receptions, controller changes, shape flips
Layer 2  Two frame judges: P(dead ball) from a gradient-boosted tree model (scikit-learn
           HistGradientBoosting); P(home | ball alive) from a bidirectional GRU (PyTorch) that
           reads 20-second windows of the same features, three seeds averaged
Layer 3  Match-flow referee: 4-state hidden Markov model (H, A, dead-after-H, dead-after-A)
           decoded with Viterbi -- switching team is costly, stoppages last seconds
Rule     Goal rule: a ball blackout > 15 s ending on the centre spot is a goal stoppage
Rule     Kick-off rule: graded frames before the organisers' kick-off anchor are dead ball
```

**Trained only on external data** (competition rules forbid training or tuning on the provided game):
7 Bundesliga matches (IDSSE/DFL, per-frame possession labels) and 3 Metrica sample games (labels
derived from Metrica's events). The clean external tracking is degraded to look like the provided
automatic tracking: ball blackouts linked to stoppages at real-tracking rates, an "un-frozen" ball
during stoppages, player dropouts, 1 m position jitter and wrong-team tracking episodes — all
calibrated from tracking statistics or external data only.

**Validation:** leave-one-match-out over the 10 external matches, settings chosen by nested
leave-one-match-out, every change judged per match with the official scoring code
(`evaluation/possession_metrics.py`). Current system: spellF1 **0.639** on the Metrica games
(organisers' baseline 0.265) and **0.533** on the Bundesliga matches (baseline 0.155), on the
realistic bench with wrong-team episodes. The GRU team judge (16 Nov) beat the gradient-boosted
frame judge it replaced on all 10 matches (+0.039 spellF1).

Full details: `docs/possession_rules.md` (the football rules), `docs/process-log.html` (every step,
decision D1–D60 and result), `docs/GLOSSARY.md`, and the technical document.

## Repository layout

```
make_submission.py     the one command that builds the submission
fetch_external.py      downloads + verifies + parses the external training data
analysis/              features, models, referee, bench, parsers; numbered experiment scripts 01-37
models/                the three trained GRU team judges and their feature statistics (1.6 MB)
docs/                  process log, possession rules, glossary, expert review brief and feedback
games.py, evaluation/, starter/baseline.py   from the organisers' repository (MIT, see LICENSE-organisers)
data/                  put the provided match data here (git-ignored)
external/              external datasets land here (git-ignored)
submissions/           generated files land here (git-ignored)
```

## Compliance with the competition rules

- No hand labels or manual corrections of the provided game; every label comes from the pipeline.
- No training, fitting, calibration or tuning on the ground truth or the provided video. The public
  5-minute ground truth was used only to check frame alignment and label meaning.
- The provided video was not used as input.
- No match data, derived frame-level data or submission files are in this repository.

## Data credits and licences

- **IDSSE / DFL open data**, CC-BY 4.0: Bassek, M., Rein, R., Weber, H. & Memmert, D. (2025).
  *An integrated dataset of spatiotemporal and event data in elite soccer.* Scientific Data.
  figshare DOI 10.6084/m9.figshare.28196177. Data provided with authorisation of the Deutsche
  Fußball Liga (DFL).
- **Metrica Sports open sample data**, github.com/metrica-sports/sample-data — source acknowledged
  as requested; use confirmed by the organisers.
- Files from the organisers' repository (`games.py`, `evaluation/`, `starter/baseline.py`) are MIT
  licensed; see `LICENSE-organisers`. The provided match data is not covered by any licence here.
- Our code: MIT, see `LICENSE`.
