"""
pipeline_la_ablation.py — Latent Additive with a `training_mode`
switch for ablating per-pert-mean vs per-cell training. Everything
else (architecture, ensemble, residual, target override, seeds,
dropout, lr) is held constant.

Two modes:
  - 'per_pert_mean' (default, matches pipeline.py on pb-apr20): one
    training example per train perturbation. Target = mean
    expression of cells with that pert. Full-batch, N_epochs grad
    steps total.
  - 'per_cell': one training example per train cell. Target = that
    cell's own expression. Mini-batched with `batch_size`. Same
    grad-step count as per_pert_mean unless `steps` is specified.

This file is not imported by the default harness. It's driven by
`ablation_training.py` which runs both modes with the same seeds
and reports the delta — the cleanest answer we can produce for
"how much of the 0.87 advantage comes from training procedure
alone?"
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import torch
import torch.nn as nn


def _resolve_target_indices(pert: str) -> list[str]:
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
    def __init__(self, in_dim: int, hidden: int, out_dim: int, dropout: float = 0.0):
        super().__init__()
        layers: list[nn.Module] = [nn.Linear(in_dim, hidden)]
        layers += [
            nn.LayerNorm(hidden),
            nn.ReLU(),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
            nn.Linear(hidden, hidden),
            nn.LayerNorm(hidden),
            nn.ReLU(),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
            nn.Linear(hidden, out_dim),
        ]
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class _LatentAdditive(nn.Module):
    def __init__(self, n_genes: int, n_pert_tokens: int, latent_dim: int,
                 hidden: int, dropout: float):
        super().__init__()
        self.f_ctrl = _MLP(n_genes, hidden, latent_dim, dropout)
        self.f_pert = _MLP(n_pert_tokens, hidden, latent_dim, dropout)
        self.f_dec = _MLP(latent_dim, hidden, n_genes, dropout)

    def forward(self, x_ctrl: torch.Tensor, p: torch.Tensor) -> torch.Tensor:
        z = self.f_ctrl(x_ctrl) + self.f_pert(p)
        return x_ctrl + self.f_dec(z)  # output-space residual


def _to_dense_mean(adata_slice) -> np.ndarray:
    x = adata_slice.X
    if hasattr(x, "toarray"):
        return np.asarray(x.mean(axis=0)).flatten()
    return np.asarray(x).mean(axis=0).flatten()


def _densify(x):
    return np.asarray(x.toarray() if hasattr(x, "toarray") else x, dtype=np.float32)


class Pipeline:
    """LA pipeline with a training-procedure ablation switch.

    Everything except `training_mode` matches pipeline.py on
    autoresearch/pb-apr20. This is the ablation of claim 3.
    """

    def __init__(
        self,
        training_mode: str = "per_pert_mean",
        latent_dim: int = 128,
        hidden: int = 512,
        dropout: float = 0.0,
        lr: float = 1e-3,
        epochs: int = 4000,
        seed: int = 0,
        n_ensemble: int = 5,
        batch_size: int = 256,   # per_cell mode only
        verbose: bool = True,
    ):
        assert training_mode in ("per_pert_mean", "per_cell"), training_mode
        self.training_mode = training_mode
        self.latent_dim = latent_dim
        self.hidden = hidden
        self.dropout = dropout
        self.lr = lr
        self.epochs = epochs
        self.seed = seed
        self.n_ensemble = n_ensemble
        self.batch_size = batch_size
        self.verbose = verbose
        self.models: list[_LatentAdditive] = []
        self._token_to_idx: dict[str, int] = {}
        self._sym_to_hvg: dict[str, int] = {}
        self._per_gene_delta: dict[int, float] = {}
        self._control_mean: Optional[np.ndarray] = None
        self._device = torch.device("cpu")

    def _pert_to_multihot(self, pert: str, dtype=np.float32) -> np.ndarray:
        vec = np.zeros(len(self._token_to_idx), dtype=dtype)
        for sym in _resolve_target_indices(pert):
            idx = self._token_to_idx.get(sym)
            if idx is not None:
                vec[idx] = 1.0
        return vec

    def fit(self, train_adata) -> None:
        train_perts = sorted(
            set(train_adata.obs["perturbation"].unique()) - {"control"}
        )
        tokens: set[str] = set()
        for p in train_perts:
            tokens.update(_resolve_target_indices(p))
        self._token_to_idx = {t: i for i, t in enumerate(sorted(tokens))}
        n_tokens = max(1, len(self._token_to_idx))

        ctrl_mask = train_adata.obs["perturbation"] == "control"
        control_mean = _to_dense_mean(train_adata[ctrl_mask])
        n_genes = control_mean.size
        self._control_mean = control_mean

        var_names = list(train_adata.var_names)
        gene_names = (
            list(train_adata.var["gene_name"].astype(str))
            if "gene_name" in train_adata.var.columns else var_names
        )
        self._sym_to_hvg = {n: i for i, n in enumerate(var_names)}
        for i, gn in enumerate(gene_names):
            self._sym_to_hvg.setdefault(gn, i)

        per_gene_drops: dict[int, list[float]] = {}

        # Prepare examples depending on training_mode.
        if self.training_mode == "per_pert_mean":
            X_list, Y_list = [], []
            for p in train_perts:
                mask = train_adata.obs["perturbation"] == p
                if mask.sum() == 0:
                    continue
                mean_p = _to_dense_mean(train_adata[mask])
                X_list.append(self._pert_to_multihot(p))
                Y_list.append(mean_p)
                for sym in _resolve_target_indices(p):
                    hvg = self._sym_to_hvg.get(sym)
                    if hvg is not None:
                        per_gene_drops.setdefault(hvg, []).append(
                            float(mean_p[hvg] - control_mean[hvg])
                        )
            X = torch.tensor(np.stack(X_list), dtype=torch.float32)
            Y = torch.tensor(np.stack(Y_list), dtype=torch.float32)
        else:  # per_cell
            # Filter out control cells — training predicts perturbed profiles.
            pert_mask = train_adata.obs["perturbation"] != "control"
            pert_adata = train_adata[pert_mask]
            cell_pert_labels = list(pert_adata.obs["perturbation"])
            Y_all = _densify(pert_adata.X)
            X_all = np.stack([self._pert_to_multihot(p) for p in cell_pert_labels])
            X = torch.tensor(X_all, dtype=torch.float32)
            Y = torch.tensor(Y_all, dtype=torch.float32)
            # Per-gene deltas still computed from per-pert means (identical
            # to per_pert_mean mode — this isn't part of the ablation).
            for p in train_perts:
                mask = train_adata.obs["perturbation"] == p
                if mask.sum() == 0:
                    continue
                mean_p = _to_dense_mean(train_adata[mask])
                for sym in _resolve_target_indices(p):
                    hvg = self._sym_to_hvg.get(sym)
                    if hvg is not None:
                        per_gene_drops.setdefault(hvg, []).append(
                            float(mean_p[hvg] - control_mean[hvg])
                        )

        self._per_gene_delta = {
            g: float(np.median(ds)) for g, ds in per_gene_drops.items()
        }

        C = torch.tensor(control_mean, dtype=torch.float32)
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
            )
            opt = torch.optim.Adam(model.parameters(), lr=self.lr)
            model.train()

            if self.training_mode == "per_pert_mean":
                C_b = C.unsqueeze(0).expand(X.shape[0], -1)
                for epoch in range(self.epochs):
                    opt.zero_grad()
                    pred = model(C_b, X)
                    loss = loss_fn(pred, Y)
                    loss.backward()
                    opt.step()
                final_loss = loss.item()
            else:  # per_cell
                # Aim for the same number of gradient steps as per-pert-mean
                # (= self.epochs). Each step is a minibatch of self.batch_size
                # randomly-sampled cells.
                n_cells = X.shape[0]
                rng = np.random.default_rng(seed_k)
                running_loss = 0.0
                for step in range(self.epochs):
                    idx = rng.integers(0, n_cells, size=self.batch_size)
                    xb = X[idx]
                    yb = Y[idx]
                    cb = C.unsqueeze(0).expand(self.batch_size, -1)
                    opt.zero_grad()
                    pred = model(cb, xb)
                    loss = loss_fn(pred, yb)
                    loss.backward()
                    opt.step()
                    running_loss += loss.item()
                final_loss = running_loss / max(1, self.epochs)

            self.models.append(model)
            if self.verbose:
                print(
                    f"ablation[{self.training_mode}]: model {k+1}/{self.n_ensemble} "
                    f"seed={seed_k} final_loss={final_loss:.5f}",
                    flush=True,
                )

        print(
            f"ablation[{self.training_mode}]: trained {self.n_ensemble}-way "
            f"ensemble; n_tokens={n_tokens} n_genes={n_genes}",
            flush=True,
        )

    def predict(
        self,
        test_perts: list[str],
        control_mean: np.ndarray,
        train_adata,
    ) -> dict[str, np.ndarray]:
        if not self.models:
            raise RuntimeError("not fit")
        for m in self.models:
            m.eval()
        C = torch.tensor(control_mean, dtype=torch.float32)
        out: dict[str, np.ndarray] = {}
        with torch.no_grad():
            for p in test_perts:
                onehot = torch.tensor(
                    self._pert_to_multihot(p), dtype=torch.float32
                )
                preds = [
                    m(C.unsqueeze(0), onehot.unsqueeze(0)).squeeze(0)
                    for m in self.models
                ]
                pred = torch.stack(preds).mean(dim=0).cpu().numpy()
                for sym in _resolve_target_indices(p):
                    hvg = self._sym_to_hvg.get(sym)
                    if hvg is None:
                        continue
                    delta = self._per_gene_delta.get(hvg)
                    if delta is None:
                        continue
                    pred[hvg] = float(control_mean[hvg]) + delta
                out[p] = pred
        return out
