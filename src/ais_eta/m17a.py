"""M17A target-free self-supervised trajectory retrieval utilities.

M17A changes only the trajectory representation / similarity metric relative to
M16D.  ETA aggregation remains the frozen M16D destination-gated KNN weighted
median.  Every learned encoder is fit strictly on causal trajectories belonging
to the applicable outer-train MMSIs; old M10 final MMSIs and outer-valid
trajectories are excluded from representation learning.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
import random
from typing import Iterable

import numpy as np
import pandas as pd

from ais_eta.m16d import M16D_POINTS, RouteConfig, resample_causal_trajectory

M17A_VERSION = "m17a-ssl-retrieval-v1-20260922"
M17A_SEED = 1701
M17A_WINDOW_H = 6
M17A_POINTS = M16D_POINTS
M17A_MAX_WINDOWS_PER_MMSI = 8
M17A_EMBED_DIM = 32
M17A_HIDDEN_DIM = 32
M17A_EPOCHS = 16
M17A_BATCH_SIZE = 128
M17A_LR = 2e-3
M17A_MASK_RATIO = 0.30
M17A_CONTRASTIVE_WEIGHT = 0.10
M17A_TEMPERATURE = 0.15
M17A_RETRIEVAL_CONFIG = RouteConfig("geo_kin_6h", "destination", 9)
M17A_PROMOTION_MIN_FOLD_WINS = 3
M17A_PROMOTION_MAX_P90_RATIO = 1.05


@dataclass(frozen=True)
class SSLTrainResult:
    embeddings: np.ndarray
    train_loss_start: float
    train_loss_end: float
    windows: int
    owners: int
    feature_mean: np.ndarray
    feature_std: np.ndarray


def build_causal_pretraining_windows(
    states: pd.DataFrame,
    dev: pd.DataFrame,
    *,
    max_windows_per_mmsi: int = M17A_MAX_WINDOWS_PER_MMSI,
    hours: int = M17A_WINDOW_H,
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """Create deterministic target-free causal windows for SSL pretraining.

    Each development MMSI contributes at most ``max_windows_per_mmsi`` windows,
    chosen uniformly over its available causal history ending no later than its
    frozen ``history_last_at`` cutoff.  No target/reference ETA field is read.
    """
    s = states.copy()
    s["recorded_at"] = pd.to_datetime(s["recorded_at"])
    cutoffs = dict(zip(dev["mmsi"].astype(int), pd.to_datetime(dev["history_last_at"])))
    grouped = {int(k): g.sort_values("recorded_at", kind="mergesort").copy() for k, g in s.groupby("mmsi", sort=False)}
    windows: list[np.ndarray] = []
    owners: list[int] = []
    audit: list[dict] = []

    for mmsi in sorted(cutoffs):
        g = grouped.get(int(mmsi))
        if g is None:
            continue
        cutoff = pd.Timestamp(cutoffs[mmsi])
        mask = g["recorded_at"].le(cutoff)
        if "position_valid" in g.columns:
            mask &= g["position_valid"].fillna(False).astype(bool)
        gg = g.loc[mask].dropna(subset=["lat_clean", "lon_clean"]).copy()
        if len(gg) < 2:
            continue
        times = gg["recorded_at"].drop_duplicates().sort_values().to_numpy()
        if len(times) < 2:
            continue
        # Uniform positions in the *causal history*, not row-count-weighted dense polling.
        n = min(int(max_windows_per_mmsi), max(1, len(times) - 1))
        idx = np.unique(np.linspace(1, len(times) - 1, n, dtype=int))
        for j in idx:
            end_at = pd.Timestamp(times[int(j)])
            try:
                seq, meta = resample_causal_trajectory(
                    gg,
                    end_at=end_at,
                    hours=hours,
                    n_points=M17A_POINTS,
                    include_kinematics=True,
                )
            except ValueError:
                continue
            if pd.Timestamp(meta["last_at"]) > cutoff:
                raise AssertionError("M17A causal pretraining window crossed history cutoff")
            windows.append(seq.astype(np.float32))
            owners.append(int(mmsi))
            audit.append({
                "mmsi": int(mmsi),
                "window_end_at": end_at.isoformat(),
                "history_cutoff_at": cutoff.isoformat(),
                "raw_points": int(meta["raw_points"]),
                "observed_span_h": float(meta["observed_span_h"]),
            })
    if not windows:
        raise AssertionError("M17A produced no self-supervised windows")
    return np.stack(windows), np.asarray(owners, dtype=np.int64), pd.DataFrame(audit)


def standardize_fit(windows: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    x = np.asarray(windows, dtype=np.float32)
    mean = x.reshape(-1, x.shape[-1]).mean(axis=0)
    std = x.reshape(-1, x.shape[-1]).std(axis=0)
    std = np.where(std < 1e-6, 1.0, std)
    return mean.astype(np.float32), std.astype(np.float32)


def standardize_apply(x: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    return ((np.asarray(x, dtype=np.float32) - mean[None, None, :]) / std[None, None, :]).astype(np.float32)


def l2_normalize(x: np.ndarray) -> np.ndarray:
    a = np.asarray(x, dtype=np.float64)
    n = np.linalg.norm(a, axis=1, keepdims=True)
    n = np.where(n < 1e-12, 1.0, n)
    return (a / n).astype(np.float64)


def cosine_distance_matrix(embeddings: np.ndarray) -> np.ndarray:
    e = l2_normalize(embeddings)
    d = 1.0 - e @ e.T
    np.fill_diagonal(d, 0.0)
    return np.clip(d, 0.0, 2.0)


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


def _augment(x, *, generator, noise_scale: float = 0.025, mask_ratio: float = 0.15):
    import torch

    noise = torch.randn(x.shape, dtype=x.dtype, device=x.device, generator=generator) * noise_scale
    out = x + noise
    mask = torch.rand((x.shape[0], x.shape[1], 1), dtype=x.dtype, device=x.device, generator=generator) < mask_ratio
    out = torch.where(mask, torch.zeros_like(out), out)
    return out


def _nt_xent(z1, z2, temperature: float):
    import torch
    import torch.nn.functional as F

    z1 = F.normalize(z1, dim=1)
    z2 = F.normalize(z2, dim=1)
    z = torch.cat([z1, z2], dim=0)
    sim = z @ z.T / float(temperature)
    n = z1.shape[0]
    eye = torch.eye(2 * n, dtype=torch.bool, device=z.device)
    sim = sim.masked_fill(eye, -1e9)
    targets = torch.arange(2 * n, device=z.device)
    targets = (targets + n) % (2 * n)
    return F.cross_entropy(sim, targets)


def train_ssl_encoder(
    train_windows: np.ndarray,
    query_sequences: np.ndarray,
    *,
    seed: int,
    epochs: int = M17A_EPOCHS,
) -> SSLTrainResult:
    """Train a compact NaviSight/MoCo-inspired masked+contrastive GRU encoder."""
    import torch
    from torch import nn
    import torch.nn.functional as F

    _set_deterministic(seed)
    device = torch.device("cpu")
    mean, std = standardize_fit(train_windows)
    train_x = standardize_apply(train_windows, mean, std)
    query_x = standardize_apply(query_sequences, mean, std)
    n_points, n_features = train_x.shape[1], train_x.shape[2]

    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            # Small TCN-style encoder: preserves temporal locality but is much cheaper
            # than a recurrent encoder for repeated outer-fold self-supervised fits.
            self.temporal = nn.Sequential(
                nn.Conv1d(n_features, M17A_HIDDEN_DIM, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.Conv1d(M17A_HIDDEN_DIM, M17A_HIDDEN_DIM, kernel_size=3, padding=1),
                nn.ReLU(),
            )
            self.embed = nn.Sequential(nn.Linear(M17A_HIDDEN_DIM, M17A_EMBED_DIM), nn.Tanh())
            self.project = nn.Linear(M17A_EMBED_DIM, M17A_EMBED_DIM)
            self.decode = nn.Sequential(
                nn.Linear(M17A_EMBED_DIM, 2 * M17A_HIDDEN_DIM),
                nn.ReLU(),
                nn.Linear(2 * M17A_HIDDEN_DIM, n_points * n_features),
            )

        def encode(self, x):
            h = self.temporal(x.transpose(1, 2))
            pooled = h.mean(dim=2)
            return self.embed(pooled)

        def forward(self, x):
            e = self.encode(x)
            return e, self.decode(e).reshape(-1, n_points, n_features)

    model = Model().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=M17A_LR, weight_decay=1e-4)
    tensor = torch.from_numpy(train_x).to(device)
    n = len(tensor)
    batch = min(M17A_BATCH_SIZE, n)
    gen = torch.Generator(device="cpu"); gen.manual_seed(seed + 17)
    losses: list[float] = []

    # Fixed deterministic order by seeded randperm each epoch.
    for epoch in range(int(epochs)):
        perm_gen = torch.Generator(device="cpu"); perm_gen.manual_seed(seed * 1000 + epoch)
        perm = torch.randperm(n, generator=perm_gen)
        epoch_loss = 0.0; seen = 0
        for start in range(0, n, batch):
            idx = perm[start:start + batch]
            clean = tensor[idx]
            mask_gen = torch.Generator(device="cpu"); mask_gen.manual_seed(seed * 100000 + epoch * 1000 + start)
            mask = torch.rand(clean.shape, generator=mask_gen, dtype=clean.dtype) < M17A_MASK_RATIO
            masked = torch.where(mask, torch.zeros_like(clean), clean)
            emb, recon = model(masked)
            if bool(mask.any()):
                recon_loss = F.mse_loss(recon[mask], clean[mask])
            else:
                recon_loss = F.mse_loss(recon, clean)
            aug_gen1 = torch.Generator(device="cpu"); aug_gen1.manual_seed(seed * 200000 + epoch * 2000 + start)
            aug_gen2 = torch.Generator(device="cpu"); aug_gen2.manual_seed(seed * 300000 + epoch * 3000 + start)
            a1 = _augment(clean, generator=aug_gen1)
            a2 = _augment(clean, generator=aug_gen2)
            z1 = model.project(model.encode(a1)); z2 = model.project(model.encode(a2))
            contrastive = _nt_xent(z1, z2, M17A_TEMPERATURE)
            loss = recon_loss + M17A_CONTRASTIVE_WEIGHT * contrastive
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
            epoch_loss += float(loss.detach()) * len(idx); seen += len(idx)
        losses.append(epoch_loss / max(seen, 1))

    model.eval()
    with torch.no_grad():
        q = torch.from_numpy(query_x).to(device)
        embs = []
        for start in range(0, len(q), 256):
            e = model.encode(q[start:start+256])
            embs.append(F.normalize(e, dim=1).cpu().numpy())
    embeddings = np.concatenate(embs, axis=0).astype(np.float64)
    return SSLTrainResult(
        embeddings=embeddings,
        train_loss_start=float(losses[0]),
        train_loss_end=float(losses[-1]),
        windows=int(len(train_windows)),
        owners=0,
        feature_mean=mean.astype(np.float64),
        feature_std=std.astype(np.float64),
    )
