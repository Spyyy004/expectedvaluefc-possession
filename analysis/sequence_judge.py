"""The GRU sequence judge: P(home | alive) from a whole stretch of play (Phase 7 #3, D60).

Same 58 team + turnover clues as the boosted judge, standardised on the training halves with a
missing-flag per clue, read by a bidirectional GRU over 512-frame (20.5 s) windows. Trained on the
live frames of the 10 external matches; SEEDS networks are averaged at inference. Settings are the
ones of the confirmation run on the bench (analysis/37_sequence_judge.py tag=base): +0.039 spellF1
over the boosted judge, 10 of 10 matches.

Reproducibility: training and inference run on the CPU with fixed seeds and deterministic
algorithms, so the saved weights and the submission rebuild identically (see make_submission.py).
"""
import numpy as np
import torch
import torch.nn as nn

HIDDEN = 64
LAYERS = 2
WIN = 512
EPOCHS = 12
SEEDS = 3
BATCH = 32
LR = 2e-3
DROPOUT = 0.2
WINDOWS_PER_EPOCH = 3000
WEIGHT_DECAY = 1e-4


class Net(nn.Module):
    def __init__(self, n_feat, hidden=HIDDEN, layers=LAYERS, dropout=DROPOUT):
        super().__init__()
        self.inp = nn.Sequential(nn.Linear(n_feat * 2, hidden), nn.ReLU(), nn.Dropout(dropout))
        self.gru = nn.GRU(hidden, hidden, layers, batch_first=True, bidirectional=True,
                          dropout=dropout if layers > 1 else 0)
        self.out = nn.Linear(2 * hidden, 1)

    def forward(self, x):                       # x: (B, T, 2F), NaN already encoded
        h, _ = self.gru(self.inp(x))
        return self.out(h).squeeze(-1)


def stats(X):
    """Per-clue mean and standard deviation over the training halves (NaN ignored)."""
    stack = np.vstack(X)
    return np.nanmean(stack, axis=0), np.nanstd(stack, axis=0) + 1e-6


def encode(x, mu, sd):
    """Standardise, fill NaN with 0 and append a missing-flag per clue."""
    miss = np.isnan(x)
    v = np.where(miss, 0.0, (x - mu) / sd)
    return np.concatenate([v, miss.astype(np.float32)], axis=1).astype(np.float32)


def train_one(E, Y, LIVE, seed, device=torch.device("cpu"), epochs=EPOCHS, log=None):
    """Train one network. E: encoded halves; Y: home labels; LIVE: live-frame masks (lists of arrays)."""
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    net = Net(E[0].shape[1] // 2).to(device)
    opt = torch.optim.AdamW(net.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    steps = WINDOWS_PER_EPOCH // BATCH
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=LR, total_steps=epochs * steps)
    xs = [torch.from_numpy(e) for e in E]
    ys = [torch.from_numpy(np.asarray(y, dtype=np.float32)) for y in Y]
    ls = [torch.from_numpy(np.asarray(l, dtype=np.float32)) for l in LIVE]
    lengths = np.array([len(y) for y in ys], dtype=float)
    pick = lengths / lengths.sum()
    bce = nn.BCEWithLogitsLoss(reduction="none")
    for ep in range(epochs):
        net.train()
        tot = 0.0
        for _ in range(steps):
            ks = rng.choice(len(xs), BATCH, p=pick)
            starts = [int(rng.integers(0, len(ys[k]) - WIN)) for k in ks]
            xb = torch.stack([xs[k][s:s + WIN] for k, s in zip(ks, starts)]).to(device)
            yb = torch.stack([ys[k][s:s + WIN] for k, s in zip(ks, starts)]).to(device)
            lb = torch.stack([ls[k][s:s + WIN] for k, s in zip(ks, starts)]).to(device)
            loss = (bce(net(xb), yb) * lb).sum() / lb.sum().clamp(min=1)
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step()
            sched.step()
            tot += loss.item()
        if log:
            log("      seed {} epoch {:2d} loss {:.4f}".format(seed, ep + 1, tot / steps))
    return net


@torch.no_grad()
def predict(net, e, device=torch.device("cpu")):
    """Slide windows (stride WIN/2) over one encoded half; average the overlapping logits."""
    net.eval()
    n = len(e)
    acc, cnt = np.zeros(n, dtype=np.float64), np.zeros(n, dtype=np.float64)
    starts = list(range(0, max(1, n - WIN + 1), WIN // 2))
    if starts[-1] + WIN < n:
        starts.append(n - WIN)
    for i in range(0, len(starts), 64):
        ss = starts[i:i + 64]
        xb = torch.stack([torch.from_numpy(e[s:s + WIN]) for s in ss]).to(device)
        out = net(xb).cpu().numpy()
        for s, o in zip(ss, out):
            acc[s:s + WIN] += o
            cnt[s:s + WIN] += 1
    return 1 / (1 + np.exp(-acc / np.maximum(cnt, 1)))


def predict_ensemble(nets, e, device=torch.device("cpu")):
    return sum(predict(net, e, device) for net in nets) / len(nets)
