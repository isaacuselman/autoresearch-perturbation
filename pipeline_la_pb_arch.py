"""
pipeline_la_pb_arch.py — Latent Additive with PerturBench's best
Norman19 hyperparameters, running under our training procedure.

Their config
(`perturbench/configs/experiment/neurips2025/norman19/latent_best_params_norman19.yaml`):
  encoder_width: 4352
  latent_dim:    512
  n_layers:      1
  lr:            9.26e-5
  wd:            2.18e-8
  dropout:       0.1
  softplus_output: true
  max_epochs:    500

Architectural difference from our best LA pipeline:
  - Their MLP structure: Linear(in, h) → [Linear(h, h) → LayerNorm →
    ReLU → Dropout] × n_layers → Linear(h, out). With n_layers=1 this
    is: Linear(in, 4352) → Linear(4352, 4352) → LN → ReLU → Dropout
    → Linear(4352, out). Total encoder/pert/decoder each ≈ 19M + 23M +
    19M params ≈ ~4M × 3 modules. Ours at latent=128, hidden=512 is
    ~0.8M.
  - Their forward applies softplus at the output (non-negativity).
    Our best pipeline uses an output-space residual instead.

Training procedure: per-pert-mean, 5-seed ensemble, per-target-gene
override on top — same as the pipeline.py on this branch. Only
architecture and its matched hyperparameters (lr, wd, dropout,
epochs, softplus) change.

This answers the question: does PerturBench's larger architecture,
under our training, do better, worse, or the same as our small one?
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


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


class _PBMLPLayer(nn.Sequential):
    """Matches perturbench.modelcore.nn.mlp.MLP exactly:
        Linear(in, h)
        [Linear(h, h), LayerNorm(h, elementwise_affine=False), ReLU,
         Dropout] × n_layers
        Linear(h, out)
    """

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        out_dim: int,
        n_layers: int,
        dropout: float,
    ):
        layers: list[nn.Module] = [nn.Linear(in_dim, hidden_dim)]
        for _ in range(n_layers):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.LayerNorm(hidden_dim, elementwise_affine=False))
            layers.append(nn.ReLU())
            if dropout is not None and dropout > 0:
                layers.append(nn.Dropout(dropout))
        layers.append(nn.Linear(hidden_dim, out_dim))
        super().__init__(*layers)


class _LatentAdditivePB(nn.Module):
    """PerturBench's architecture; forward matches their
    LatentAdditive.forward when covariates are empty and embeddings
    is None."""

    def __init__(
        self,
        n_genes: int,
        n_pert_tokens: int,
        latent_dim: int,
        encoder_width: int,
        n_layers: int,
        dropout: float,
        softplus_output: bool,
    ):
        super().__init__()
        self.gene_encoder = _PBMLPLayer(n_genes, encoder_width, latent_dim,
                                        n_layers, dropout)
        self.pert_encoder = _PBMLPLayer(n_pert_tokens, encoder_width, latent_dim,
                                        n_layers, dropout)
        self.decoder = _PBMLPLayer(latent_dim, encoder_width, n_genes,
                                   n_layers, dropout)
        self.softplus_output = softplus_output

    def forward(self, x_ctrl: torch.Tensor, p_onehot: torch.Tensor) -> torch.Tensor:
        z = self.gene_encoder(x_ctrl) + self.pert_encoder(p_onehot)
        y = self.decoder(z)
        if self.softplus_output:
            y = F.softplus(y)
        return y


def _to_dense_mean(adata_slice) -> np.ndarray:
    x = adata_slice.X
    if hasattr(x, "toarray"):
        return np.asarray(x.mean(axis=0)).flatten()
    return np.asarray(x).mean(axis=0).flatten()


class Pipeline:
    """PerturBench LA architecture + their hyperparams, under our
    training procedure (per-pert-mean, 5-seed ensemble, per-target
    override)."""

    def __init__(
        self,
        latent_dim: int = 512,
        encoder_width: int = 4352,
        n_layers: int = 1,
        dropout: float = 0.1,
        lr: float = 9.26e-5,
        wd: float = 2.18e-8,
        epochs: int = 500,
        softplus_output: bool = True,
        seed: int = 0,
        n_ensemble: int = 5,
        verbose: bool = True,
    ):
        self.latent_dim = latent_dim
        self.encoder_width = encoder_width
        self.n_layers = n_layers
        self.dropout = dropout
        self.lr = lr
        self.wd = wd
        self.epochs = epochs
        self.softplus_output = softplus_output
        self.seed = seed
        self.n_ensemble = n_ensemble
        self.verbose = verbose
        self.models: list[_LatentAdditivePB] = []
        self._token_to_idx: dict[str, int] = {}
        self._sym_to_hvg: dict[str, int] = {}
        self._per_gene_delta: dict[int, float] = {}
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
        self._per_gene_delta = {
            g: float(np.median(ds)) for g, ds in per_gene_drops.items()
        }

        X = torch.tensor(np.stack(X_list), dtype=torch.float32)
        Y = torch.tensor(np.stack(Y_list), dtype=torch.float32)
        C = torch.tensor(control_mean, dtype=torch.float32)
        C_b = C.unsqueeze(0).expand(X.shape[0], -1)

        loss_fn = nn.MSELoss()
        self.models = []
        for k in range(self.n_ensemble):
            seed_k = self.seed + k
            torch.manual_seed(seed_k)
            np.random.seed(seed_k)
            model = _LatentAdditivePB(
                n_genes=n_genes,
                n_pert_tokens=n_tokens,
                latent_dim=self.latent_dim,
                encoder_width=self.encoder_width,
                n_layers=self.n_layers,
                dropout=self.dropout,
                softplus_output=self.softplus_output,
            )
            opt = torch.optim.Adam(
                model.parameters(), lr=self.lr, weight_decay=self.wd
            )
            model.train()
            for epoch in range(self.epochs):
                opt.zero_grad()
                pred = model(C_b, X)
                loss = loss_fn(pred, Y)
                loss.backward()
                opt.step()
            self.models.append(model)
            if self.verbose:
                print(
                    f"pb_arch: model {k+1}/{self.n_ensemble} seed={seed_k} "
                    f"final_loss={loss.item():.5f}",
                    flush=True,
                )
        n_params = sum(p.numel() for p in self.models[0].parameters()) / 1e6
        print(
            f"pb_arch: trained {self.n_ensemble}-way ensemble; "
            f"{n_params:.1f}M params per model; "
            f"latent={self.latent_dim} encoder_width={self.encoder_width} "
            f"lr={self.lr} wd={self.wd} epochs={self.epochs} "
            f"softplus_output={self.softplus_output}",
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
