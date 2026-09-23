"""M17G stronger target-free AIS trajectory encoders.

M17G keeps the frozen M17A retrieval contract fixed (6 h geo+kin sequence,
canonical-destination gate, K=9, similarity-weighted median ETA) and changes
only the self-supervised trajectory representation.  Two predeclared encoders
are benchmarked: a MoCo-style TCN encoder and a compact masked-autoencoder
Transformer.  All standardisation and pretraining are outer-train-only.
"""
from __future__ import annotations

from dataclasses import dataclass
import copy
import math
import random

import numpy as np

from ais_eta.m17a import (
    M17A_RETRIEVAL_CONFIG,
    l2_normalize,
    standardize_apply,
    standardize_fit,
)

M17G_VERSION = "m17g-stronger-ssl-v1-20260922"
M17G_SEED = 1717
M17G_EMBED_DIM = 64
M17G_HIDDEN_DIM = 64
M17G_EPOCHS = 10
M17G_BATCH_SIZE = 128
M17G_LR = 1.5e-3
M17G_MOCO_QUEUE = 512
M17G_MOCO_MOMENTUM = 0.995
M17G_TEMPERATURE = 0.15
M17G_MAE_MASK_RATIO = 0.50
M17G_PROMOTION_MIN_GAIN_H = 1.0
M17G_PROMOTION_MIN_FOLD_WINS = 3
M17G_PROMOTION_MAX_P90_RATIO = 1.03
M17G_RETRIEVAL_CONFIG = M17A_RETRIEVAL_CONFIG
M17G_CANDIDATES = ("moco_tcn64", "masked_transformer_mae64")


@dataclass(frozen=True)
class EncoderTrainResult:
    embeddings: np.ndarray
    loss_start: float
    loss_end: float
    windows: int
    feature_mean: np.ndarray
    feature_std: np.ndarray
    objective: str


def _set_deterministic(seed: int) -> None:
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(1)
    try:
        torch.use_deterministic_algorithms(True)
    except Exception:
        pass


def _augment(x, *, generator, noise_scale: float = 0.025, point_mask_ratio: float = 0.15):
    import torch

    noise = torch.randn(x.shape, dtype=x.dtype, device=x.device, generator=generator) * noise_scale
    out = x + noise
    # Mask complete AIS timesteps so the representation cannot rely on one exact sample.
    mask = torch.rand((x.shape[0], x.shape[1], 1), dtype=x.dtype, device=x.device, generator=generator) < point_mask_ratio
    return torch.where(mask, torch.zeros_like(out), out)


def train_moco_tcn(
    train_windows: np.ndarray,
    query_sequences: np.ndarray,
    *,
    seed: int,
    epochs: int = M17G_EPOCHS,
) -> EncoderTrainResult:
    """Train a compact MoCo-style TCN trajectory encoder without ETA labels."""
    import torch
    from torch import nn
    import torch.nn.functional as F

    _set_deterministic(seed)
    mean, std = standardize_fit(train_windows)
    train_x = standardize_apply(train_windows, mean, std)
    query_x = standardize_apply(query_sequences, mean, std)
    n_features = train_x.shape[2]

    class Encoder(nn.Module):
        def __init__(self):
            super().__init__()
            self.net = nn.Sequential(
                nn.Conv1d(n_features, M17G_HIDDEN_DIM, 3, padding=1), nn.GELU(),
                nn.Conv1d(M17G_HIDDEN_DIM, M17G_HIDDEN_DIM, 3, padding=1), nn.GELU(),
                nn.Conv1d(M17G_HIDDEN_DIM, M17G_HIDDEN_DIM, 3, padding=1), nn.GELU(),
            )
            self.embed = nn.Linear(M17G_HIDDEN_DIM, M17G_EMBED_DIM)
        def forward(self, x):
            h = self.net(x.transpose(1, 2)).mean(dim=2)
            return self.embed(h)

    class Projector(nn.Module):
        def __init__(self):
            super().__init__()
            self.net = nn.Sequential(nn.Linear(M17G_EMBED_DIM, M17G_EMBED_DIM), nn.GELU(), nn.Linear(M17G_EMBED_DIM, M17G_EMBED_DIM))
        def forward(self, x): return self.net(x)

    qenc = Encoder(); kenc = copy.deepcopy(qenc)
    qproj = Projector(); kproj = copy.deepcopy(qproj)
    for p in list(kenc.parameters()) + list(kproj.parameters()): p.requires_grad_(False)
    opt = torch.optim.AdamW(list(qenc.parameters()) + list(qproj.parameters()), lr=M17G_LR, weight_decay=1e-4)
    tensor = torch.from_numpy(train_x)
    n = len(tensor); batch = min(M17G_BATCH_SIZE, n)
    qgen = torch.Generator(device="cpu"); qgen.manual_seed(seed + 404)
    queue = F.normalize(torch.randn(M17G_MOCO_QUEUE, M17G_EMBED_DIM, generator=qgen), dim=1)
    ptr = 0
    losses: list[float] = []

    @torch.no_grad()
    def momentum_update():
        for pq, pk in zip(qenc.parameters(), kenc.parameters()):
            pk.data.mul_(M17G_MOCO_MOMENTUM).add_(pq.data, alpha=1.0 - M17G_MOCO_MOMENTUM)
        for pq, pk in zip(qproj.parameters(), kproj.parameters()):
            pk.data.mul_(M17G_MOCO_MOMENTUM).add_(pq.data, alpha=1.0 - M17G_MOCO_MOMENTUM)

    for epoch in range(int(epochs)):
        pg = torch.Generator(device="cpu"); pg.manual_seed(seed * 1000 + epoch)
        perm = torch.randperm(n, generator=pg)
        total = 0.0; seen = 0
        for start in range(0, n, batch):
            idx = perm[start:start+batch]; clean = tensor[idx]
            g1 = torch.Generator(device="cpu"); g1.manual_seed(seed*100000 + epoch*2000 + start)
            g2 = torch.Generator(device="cpu"); g2.manual_seed(seed*200000 + epoch*3000 + start)
            xq = _augment(clean, generator=g1); xk = _augment(clean, generator=g2)
            q = F.normalize(qproj(qenc(xq)), dim=1)
            with torch.no_grad():
                momentum_update()
                k = F.normalize(kproj(kenc(xk)), dim=1)
            pos = torch.sum(q * k, dim=1, keepdim=True)
            neg = q @ queue.T
            logits = torch.cat([pos, neg], dim=1) / M17G_TEMPERATURE
            labels = torch.zeros(len(idx), dtype=torch.long)
            loss = F.cross_entropy(logits, labels)
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
            with torch.no_grad():
                m = len(k); end = ptr + m
                if end <= len(queue): queue[ptr:end] = k
                else:
                    first = len(queue)-ptr; queue[ptr:] = k[:first]; queue[:m-first] = k[first:]
                ptr = end % len(queue)
            total += float(loss.detach()) * len(idx); seen += len(idx)
        losses.append(total / max(seen, 1))

    qenc.eval(); embs=[]
    with torch.no_grad():
        qx = torch.from_numpy(query_x)
        for start in range(0, len(qx), 256):
            embs.append(F.normalize(qenc(qx[start:start+256]), dim=1).numpy())
    return EncoderTrainResult(
        embeddings=np.concatenate(embs).astype(np.float64),
        loss_start=float(losses[0]), loss_end=float(losses[-1]), windows=int(n),
        feature_mean=mean.astype(np.float64), feature_std=std.astype(np.float64),
        objective="MoCo momentum-contrast InfoNCE",
    )


def _sinusoidal_position(length: int, dim: int):
    import torch
    pos = torch.arange(length, dtype=torch.float32).unsqueeze(1)
    div = torch.exp(torch.arange(0, dim, 2, dtype=torch.float32) * (-math.log(10000.0) / dim))
    pe = torch.zeros(length, dim, dtype=torch.float32)
    pe[:, 0::2] = torch.sin(pos * div)
    pe[:, 1::2] = torch.cos(pos * div)
    return pe


def train_masked_transformer(
    train_windows: np.ndarray,
    query_sequences: np.ndarray,
    *,
    seed: int,
    epochs: int = M17G_EPOCHS,
) -> EncoderTrainResult:
    """Train a compact masked-autoencoder Transformer on unlabeled AIS windows."""
    import torch
    from torch import nn
    import torch.nn.functional as F

    _set_deterministic(seed)
    mean, std = standardize_fit(train_windows)
    train_x = standardize_apply(train_windows, mean, std)
    query_x = standardize_apply(query_sequences, mean, std)
    n_points, n_features = train_x.shape[1:]

    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.input = nn.Linear(n_features, M17G_HIDDEN_DIM)
            self.mask_token = nn.Parameter(torch.zeros(1, 1, M17G_HIDDEN_DIM))
            self.register_buffer("pos", _sinusoidal_position(n_points, M17G_HIDDEN_DIM).unsqueeze(0), persistent=False)
            layer = nn.TransformerEncoderLayer(
                d_model=M17G_HIDDEN_DIM, nhead=4, dim_feedforward=2*M17G_HIDDEN_DIM,
                dropout=0.0, activation="gelu", batch_first=True, norm_first=False,
            )
            self.encoder = nn.TransformerEncoder(layer, num_layers=2)
            self.norm = nn.LayerNorm(M17G_HIDDEN_DIM)
            self.embed = nn.Linear(M17G_HIDDEN_DIM, M17G_EMBED_DIM)
            self.decode = nn.Linear(M17G_HIDDEN_DIM, n_features)
        def hidden(self, x, point_mask=None):
            h = self.input(x)
            if point_mask is not None:
                h = torch.where(point_mask.unsqueeze(-1), self.mask_token.expand_as(h), h)
            h = h + self.pos
            return self.norm(self.encoder(h))
        def encode(self, x):
            h = self.hidden(x, None).mean(dim=1)
            return self.embed(h)
        def reconstruct(self, x, point_mask):
            return self.decode(self.hidden(x, point_mask))

    model = Model(); opt = torch.optim.AdamW(model.parameters(), lr=M17G_LR, weight_decay=1e-4)
    tensor = torch.from_numpy(train_x); n=len(tensor); batch=min(M17G_BATCH_SIZE,n); losses=[]
    for epoch in range(int(epochs)):
        pg=torch.Generator(device="cpu"); pg.manual_seed(seed*1000+epoch)
        perm=torch.randperm(n,generator=pg); total=0.0; seen=0
        for start in range(0,n,batch):
            idx=perm[start:start+batch]; clean=tensor[idx]
            mg=torch.Generator(device="cpu"); mg.manual_seed(seed*100000+epoch*2000+start)
            point_mask=torch.rand((len(idx),n_points),generator=mg)<M17G_MAE_MASK_RATIO
            # Guarantee at least one visible and one masked point per sequence.
            point_mask[:,0]=False; point_mask[:,-1]=True
            recon=model.reconstruct(clean,point_mask)
            mask3=point_mask.unsqueeze(-1).expand_as(clean)
            loss=F.smooth_l1_loss(recon[mask3],clean[mask3],beta=1.0)
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
            total+=float(loss.detach())*len(idx); seen+=len(idx)
        losses.append(total/max(seen,1))

    model.eval(); embs=[]
    with torch.no_grad():
        q=torch.from_numpy(query_x)
        for start in range(0,len(q),256): embs.append(F.normalize(model.encode(q[start:start+256]),dim=1).numpy())
    return EncoderTrainResult(
        embeddings=np.concatenate(embs).astype(np.float64),
        loss_start=float(losses[0]), loss_end=float(losses[-1]), windows=int(n),
        feature_mean=mean.astype(np.float64), feature_std=std.astype(np.float64),
        objective="masked Transformer reconstruction (SmoothL1)",
    )


def train_candidate(candidate: str, train_windows: np.ndarray, query_sequences: np.ndarray, *, seed: int, epochs: int=M17G_EPOCHS) -> EncoderTrainResult:
    if candidate == "moco_tcn64":
        return train_moco_tcn(train_windows, query_sequences, seed=seed, epochs=epochs)
    if candidate == "masked_transformer_mae64":
        return train_masked_transformer(train_windows, query_sequences, seed=seed, epochs=epochs)
    raise ValueError(candidate)


def promotion_pass(*, gain_h: float, fold_wins: int, p90_ratio: float) -> bool:
    return bool(
        gain_h >= M17G_PROMOTION_MIN_GAIN_H
        and fold_wins >= M17G_PROMOTION_MIN_FOLD_WINS
        and p90_ratio <= M17G_PROMOTION_MAX_P90_RATIO
    )
