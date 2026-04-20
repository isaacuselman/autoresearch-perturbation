"""
pipeline_la.py — PerturBench-style "Latent Additive" baseline.

Reproduces the Latent Additive (LA) model from PerturBench (Wu et al.,
2024, Section 3, "Latent Additive"):

  z_ctrl = f_ctrl(x)
  z_pert = f_pert(p_one_hot)
  x'     = f_dec(z_ctrl + z_pert)

f_ctrl, f_pert, f_dec are MLPs with dropout + layer normalization. The
perturbation encoder takes a multi-hot encoding of which genes are
perturbed (size = number of training target genes), so dual
perturbations are handled by flipping two entries in the input.

This file exposes a `Pipeline` class matching the interface in
pipeline.py so either harness (`harness.py`, `harness_perturbench.py`)
can run it.
"""
from __future__ import annotations

import re
from typing import Optional

import numpy as np
import torch
import torch.nn as nn


def _resolve_target_indices(pert: str) -> list[str]:
    """Parse a perturbation label into a list of target gene symbols.
    Mirrors the resolver in pipeline.py but keeps symbols, not indices
    — LA's one-hot is over the set of training target symbols, not the
    gene expression matrix columns."""
    if pert == "control" or pert == "":
        return []
    parts = pert.split("+") if "+" in pert else [pert]
    out: list[str] = []
    for part in parts:
        token = part.strip()
        if token.lower() in {"ctrl", "control", ""}:
            continue
        for prefix in ("pert_gene_", "pert_", "guide_", "sg"):
            if token.startswith(prefix):
                token = token[len(prefix):]
        for suffix in ("_ctrl",):
            if token.endswith(suffix):
                token = token[: -len(suffix)]
        if token:
            out.append(token)
    return out


class _MLP(nn.Module):
    def __init__(self, in_dim: int, hidden: int, out_dim: int, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.LayerNorm(hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
            nn.LayerNorm(hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class _LatentAdditive(nn.Module):
    def __init__(
        self,
        n_genes: int,
        n_pert_tokens: int,
        latent_dim: int = 128,
        hidden: int = 512,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.f_ctrl = _MLP(n_genes, hidden, latent_dim, dropout)
        self.f_pert = _MLP(n_pert_tokens, hidden, latent_dim, dropout)
        self.f_dec = _MLP(latent_dim, hidden, n_genes, dropout)

    def forward(self, x_ctrl: torch.Tensor, p_onehot: torch.Tensor) -> torch.Tensor:
        z = self.f_ctrl(x_ctrl) + self.f_pert(p_onehot)
        return self.f_dec(z)


def _to_dense_mean(adata_slice) -> np.ndarray:
    x = adata_slice.X
    if hasattr(x, "toarray"):
        return np.asarray(x.mean(axis=0)).flatten()
    return np.asarray(x).mean(axis=0).flatten()


class Pipeline:
    """Latent Additive pipeline. Predicts per-pert mean expression.

    Training targets: per-pert mean expression (one target vector per
    train pert). This matches what our harnesses score on.
    """

    def __init__(
        self,
        latent_dim: int = 128,
        hidden: int = 512,
        dropout: float = 0.1,
        lr: float = 1e-3,
        epochs: int = 2000,
        seed: int = 0,
        n_ensemble: int = 5,
        verbose: bool = True,
    ):
        self.latent_dim = latent_dim
        self.hidden = hidden
        self.dropout = dropout
        self.lr = lr
        self.epochs = epochs
        self.seed = seed
        self.n_ensemble = n_ensemble
        self.verbose = verbose
        self.models: list[_LatentAdditive] = []
        self._token_to_idx: dict[str, int] = {}
        self._control_mean: Optional[np.ndarray] = None
        self._device = torch.device("cpu")

    def _pert_to_multihot(self, pert: str) -> np.ndarray:
        vec = np.zeros(len(self._token_to_idx), dtype=np.float32)
        for sym in _resolve_target_indices(pert):
            idx = self._token_to_idx.get(sym)
            if idx is not None:
                vec[idx] = 1.0
        return vec

    def fit(self, train_adata) -> None:
        # Build the pert-token vocabulary from training perts (shared across ensemble).
        train_perts = sorted(
            set(train_adata.obs["perturbation"].unique()) - {"control"}
        )
        tokens: set[str] = set()
        for p in train_perts:
            tokens.update(_resolve_target_indices(p))
        self._token_to_idx = {t: i for i, t in enumerate(sorted(tokens))}
        n_tokens = max(1, len(self._token_to_idx))

        control_mask = train_adata.obs["perturbation"] == "control"
        if control_mask.sum() == 0:
            raise ValueError("No control cells in train split.")
        control_mean = _to_dense_mean(train_adata[control_mask])
        n_genes = control_mean.size
        self._control_mean = control_mean

        X_pert: list[np.ndarray] = []
        Y_target: list[np.ndarray] = []
        for p in train_perts:
            mask = train_adata.obs["perturbation"] == p
            if mask.sum() == 0:
                continue
            X_pert.append(self._pert_to_multihot(p))
            Y_target.append(_to_dense_mean(train_adata[mask]))

        if not X_pert:
            raise ValueError("No resolvable training perturbations.")

        X = torch.tensor(np.stack(X_pert), dtype=torch.float32, device=self._device)
        Y = torch.tensor(np.stack(Y_target), dtype=torch.float32, device=self._device)
        C = torch.tensor(control_mean, dtype=torch.float32, device=self._device)
        C_broadcast = C.unsqueeze(0).expand(X.shape[0], -1)

        loss_fn = nn.MSELoss()
        self.models = []
        for k in range(self.n_ensemble):
            seed_k = self.seed + k
            torch.manual_seed(seed_k)
            np.random.seed(seed_k)
            model = _LatentAdditive(
                n_genes=n_genes,
                n_pert_tokens=n_tokens,
                latent_dim=self.latent_dim,
                hidden=self.hidden,
                dropout=self.dropout,
            ).to(self._device)
            opt = torch.optim.Adam(model.parameters(), lr=self.lr)
            model.train()
            for epoch in range(self.epochs):
                opt.zero_grad()
                pred = model(C_broadcast, X)
                loss = loss_fn(pred, Y)
                loss.backward()
                opt.step()
            self.models.append(model)
            print(f"la_ens: model {k+1}/{self.n_ensemble} seed={seed_k} final_loss={loss.item():.5f}", flush=True)

        print(
            f"la: trained {self.n_ensemble}-way ensemble on {X.shape[0]} perts, "
            f"{n_tokens} pert-tokens, {n_genes} genes, latent={self.latent_dim}",
            flush=True,
        )

    def predict(
        self,
        test_perts: list[str],
        control_mean: np.ndarray,
        train_adata,
    ) -> dict[str, np.ndarray]:
        if not self.models:
            raise RuntimeError("Pipeline not fit.")
        for m in self.models:
            m.eval()
        C = torch.tensor(control_mean, dtype=torch.float32, device=self._device)
        out: dict[str, np.ndarray] = {}
        with torch.no_grad():
            for p in test_perts:
                onehot = torch.tensor(
                    self._pert_to_multihot(p), dtype=torch.float32, device=self._device
                )
                preds = [
                    m(C.unsqueeze(0), onehot.unsqueeze(0)).squeeze(0)
                    for m in self.models
                ]
                pred = torch.stack(preds).mean(dim=0)
                out[p] = pred.cpu().numpy()
        return out
