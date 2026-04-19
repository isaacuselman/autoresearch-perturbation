"""
pipeline.py — AGENT-MUTABLE. Edit freely.

Contract:
- class Pipeline with methods:
    - fit(train_adata: AnnData) -> None
    - predict(test_perts: list[str], control_mean: np.ndarray, train_adata: AnnData) -> dict[str, np.ndarray]
      Returns one predicted mean expression vector per test perturbation,
      shape (n_genes,), in the same gene order as train_adata.var.

The starter baseline is deliberately weak: it predicts the SAME delta (the
mean of all training deltas) for every test perturbation. This gives you
obvious low-hanging fruit on experiment #1 — use perturbation identity.

The real floor to beat is ridge regression on perturbation identity or
features. A 2026 preprint reported that this linear baseline matches or
beats foundation models (scGPT, Geneformer) on held-out perturbations.
"""
from __future__ import annotations

import numpy as np


class Pipeline:
    def __init__(self):
        self.mean_delta: np.ndarray | None = None

    def fit(self, train_adata) -> None:
        control_mask = train_adata.obs["perturbation"] == "control"
        if control_mask.sum() == 0:
            raise ValueError("No control cells in train split.")

        ctrl = train_adata[control_mask].X
        control_mean = (
            np.asarray(ctrl.mean(axis=0)).flatten()
            if hasattr(ctrl, "toarray")
            else np.asarray(ctrl).mean(axis=0).flatten()
        )

        train_perts = sorted(
            set(train_adata.obs["perturbation"].unique()) - {"control"}
        )
        deltas = []
        for p in train_perts:
            mask = train_adata.obs["perturbation"] == p
            x = train_adata[mask].X
            mean_p = (
                np.asarray(x.mean(axis=0)).flatten()
                if hasattr(x, "toarray")
                else np.asarray(x).mean(axis=0).flatten()
            )
            deltas.append(mean_p - control_mean)

        self.mean_delta = np.mean(deltas, axis=0) if deltas else np.zeros_like(control_mean)

    def predict(
        self,
        test_perts: list[str],
        control_mean: np.ndarray,
        train_adata,
    ) -> dict[str, np.ndarray]:
        if self.mean_delta is None:
            raise RuntimeError("Pipeline not fit.")
        # Naive: same delta for every test perturbation.
        return {p: control_mean + self.mean_delta for p in test_perts}
