"""Phase 7 #3: a sequence network as the team judge.

The boosted team judge reads one frame's 58 features (which already contain short windows). A
bidirectional GRU reads the whole stretch of play around each frame and can learn what a turnover
looks like as a sequence, which is where 71% of our misses are. Same inputs as the boosted judge,
standardised on the training folds; P(home) on live frames; dead model and referee unchanged.

Protocol: leave-one-match-out over the 10 bench matches (18 training halves per fold), windows of
WIN frames sampled at random each epoch, loss on live frames only. Inference slides windows over
the held-out halves and averages overlapping predictions. SEEDS networks per fold are averaged
(the usual remedy for a net's seed noise). Compared with the current judge via the current referee;
also the average of net and boosted judge.

    python analysis/37_sequence_judge.py            # default settings
    python analysis/37_sequence_judge.py tag=big hidden=128 epochs=20
"""
import os
import sys
import time
import warnings

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

warnings.filterwarnings("ignore", category=RuntimeWarning)
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from analysis.bench import bench, our_mess_profile, score, BENCH_JITTER  # noqa: E402
from analysis.features import team_features, turnover_features  # noqa: E402
from analysis.referee import viterbi, REFEREE_SETTINGS  # noqa: E402
from analysis.paired import compare  # noqa: E402

opts = dict(kv.split("=") for kv in sys.argv[1:])
TAG = opts.get("tag", "base")
HIDDEN = int(opts.get("hidden", 64))
LAYERS = int(opts.get("layers", 2))
WIN = int(opts.get("win", 512))            # 20.5 s windows
EPOCHS = int(opts.get("epochs", 12))
SEEDS = int(opts.get("seeds", 3))
BATCH = int(opts.get("batch", 32))
LR = float(opts.get("lr", 2e-3))
DROPOUT = float(opts.get("dropout", 0.2))
WINDOWS_PER_EPOCH = int(opts.get("wpe", 3000))
DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

DERIVED = os.path.join(ROOT, "external", "idsse", "derived")
halves, _ = bench(our_mess_profile(), jitter=BENCH_JITTER, sources=("dfl", "metrica"), dead_link=True, wrong_team=True)
matches = sorted({h["match"] for h in halves})
z = np.load(os.path.join(DERIVED, "bench_mixed_probs.npz"), allow_pickle=True)
p_dead, p_home = list(z["d"]), list(z["h"])
without = [viterbi(d, h, **REFEREE_SETTINGS) for d, h in zip(p_dead, p_home)]

print("features ...", flush=True)
X = [pd.concat([team_features(h), turnover_features(h)], axis=1).to_numpy(dtype=np.float32) for h in halves]
Y = [(h["label"] == "H").astype(np.float32) for h in halves]
LIVE = [(h["label"] != "D").astype(np.float32) for h in halves]
n_feat = X[0].shape[1]
print("device {}; {} halves; {} features; tag {}: hidden {} layers {} win {} epochs {} seeds {}".format(
    DEVICE, len(halves), n_feat, TAG, HIDDEN, LAYERS, WIN, EPOCHS, SEEDS), flush=True)


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.inp = nn.Sequential(nn.Linear(n_feat * 2, HIDDEN), nn.ReLU(), nn.Dropout(DROPOUT))
        self.gru = nn.GRU(HIDDEN, HIDDEN, LAYERS, batch_first=True, bidirectional=True, dropout=DROPOUT if LAYERS > 1 else 0)
        self.out = nn.Linear(2 * HIDDEN, 1)

    def forward(self, x):                       # x: (B, T, F) with NaN already encoded
        h, _ = self.gru(self.inp(x))
        return self.out(h).squeeze(-1)


def encode(x, mu, sd):
    """Standardise, fill NaN with 0 and append a missing-flag per feature."""
    miss = np.isnan(x)
    v = np.where(miss, 0.0, (x - mu) / sd)
    return np.concatenate([v, miss.astype(np.float32)], axis=1).astype(np.float32)


def train_one(tr, seed):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    net = Net().to(DEVICE)
    opt = torch.optim.AdamW(net.parameters(), lr=LR, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=LR, total_steps=EPOCHS * (WINDOWS_PER_EPOCH // BATCH))
    xs = [torch.from_numpy(E[k]) for k in tr]
    ys = [torch.from_numpy(Y[k]) for k in tr]
    ls = [torch.from_numpy(LIVE[k]) for k in tr]
    lengths = np.array([len(Y[k]) for k in tr], dtype=float)
    pick = lengths / lengths.sum()
    bce = nn.BCEWithLogitsLoss(reduction="none")
    for ep in range(EPOCHS):
        net.train()
        tot = 0.0
        for _ in range(WINDOWS_PER_EPOCH // BATCH):
            ks = rng.choice(len(tr), BATCH, p=pick)
            starts = [int(rng.integers(0, len(ys[k]) - WIN)) for k in ks]
            xb = torch.stack([xs[k][s:s + WIN] for k, s in zip(ks, starts)]).to(DEVICE)
            yb = torch.stack([ys[k][s:s + WIN] for k, s in zip(ks, starts)]).to(DEVICE)
            lb = torch.stack([ls[k][s:s + WIN] for k, s in zip(ks, starts)]).to(DEVICE)
            loss = (bce(net(xb), yb) * lb).sum() / lb.sum().clamp(min=1)
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step()
            sched.step()
            tot += float(loss)
        print("      epoch {:2d} loss {:.4f}".format(ep + 1, tot / (WINDOWS_PER_EPOCH // BATCH)), flush=True)
    return net


@torch.no_grad()
def predict(net, e):
    """Slide windows (stride WIN/2) over a half, average the overlapping logits."""
    net.eval()
    n = len(e)
    acc, cnt = np.zeros(n, dtype=np.float64), np.zeros(n, dtype=np.float64)
    starts = list(range(0, max(1, n - WIN + 1), WIN // 2))
    if starts[-1] + WIN < n:
        starts.append(n - WIN)
    for i in range(0, len(starts), 64):
        ss = starts[i:i + 64]
        xb = torch.stack([torch.from_numpy(e[s:s + WIN]) for s in ss]).to(DEVICE)
        out = net(xb).cpu().numpy()
        for s, o in zip(ss, out):
            acc[s:s + WIN] += o
            cnt[s:s + WIN] += 1
    return 1 / (1 + np.exp(-acc / np.maximum(cnt, 1)))


cache = os.path.join(DERIVED, "bench_seq_{}_probs.npz".format(TAG))
if os.path.exists(cache):
    p_net = list(np.load(cache, allow_pickle=True)["h"])
else:
    p_net = [None] * len(halves)
    t0 = time.time()
    for m in matches:
        tr = [k for k, h in enumerate(halves) if h["match"] != m]
        te = [k for k, h in enumerate(halves) if h["match"] == m]
        stack = np.vstack([X[k] for k in tr])
        mu, sd = np.nanmean(stack, axis=0), np.nanstd(stack, axis=0) + 1e-6
        E = {k: encode(X[k], mu, sd) for k in tr + te}
        preds = {k: np.zeros(len(Y[k])) for k in te}
        for seed in range(SEEDS):
            print("  fold {} seed {}".format(m, seed), flush=True)
            net = train_one(tr, seed)
            for k in te:
                preds[k] += predict(net, E[k]) / SEEDS
        for k in te:
            p_net[k] = preds[k]
        print("  fold {} done ({:.0f} s so far)".format(m, time.time() - t0), flush=True)
    np.savez(cache, h=np.array(p_net, dtype=object))

for name, ph in (("GRU judge", p_net), ("average of GRU and boosted", [(a + b) / 2 for a, b in zip(p_home, p_net)])):
    pr = [viterbi(d, h, **REFEREE_SETTINGS) for d, h in zip(p_dead, ph)]
    print("\n=== {} ({}) ===".format(name, TAG))
    for src in ("metrica", "dfl"):
        ks = [k for k, h in enumerate(halves) if h["source"] == src]
        for lab, p in (("current", without), (name, pr)):
            r = score([halves[k] for k in ks], [p[k] for k in ks])
            print("   {:8s} {:28s} spellF1 {:.4f}  macroF1 {:.4f}  poss {}/{}".format(
                src, lab, r["spell_f1_mean"], r["macro_f1"], r["n_pred_spells"], r["n_gt_spells"]))
    compare(halves, without, pr, "current", name, metric="spell_f1_mean")
    compare(halves, without, pr, "current", name, metric="macro_f1", verbose=False)
