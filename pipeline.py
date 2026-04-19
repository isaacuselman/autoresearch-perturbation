"""
pipeline.py — AGENT-MUTABLE.

Experiment 1: pert-identity-aware target-gene knockdown.
For each test perturbation, parse the target gene from the label, predict
mean_delta for non-target genes, and override the target gene's predicted
value with the learned average post-knockdown expression of training
target genes. Falls back to baseline behavior for perts whose target gene
can't be resolved.
"""
from __future__ import annotations

import re
from typing import Optional

import numpy as np
from scipy import stats as _stats


def _resolve_target_index(pert: str, var_names: list[str], gene_names: list[str]) -> Optional[int]:
    """Try to map a perturbation label to a single gene column index.

    Strategies, in order:
      1. Direct match against var_names.
      2. Direct match against var.gene_name.
      3. Stripping common prefixes/suffixes ('pert_gene_', '+ctrl', etc.).
      4. Extracting a trailing integer (handles synthetic 'pert_gene_<n>').
    """
    if pert in var_names:
        return var_names.index(pert)
    if pert in gene_names:
        return gene_names.index(pert)

    stripped = pert
    for prefix in ("pert_gene_", "pert_", "guide_", "sg"):
        if stripped.startswith(prefix):
            stripped = stripped[len(prefix):]
    for suffix in ("+ctrl", "_ctrl", "+control"):
        if stripped.endswith(suffix):
            stripped = stripped[: -len(suffix)]
    if stripped in var_names:
        return var_names.index(stripped)
    if stripped in gene_names:
        return gene_names.index(stripped)

    m = re.search(r"(\d+)$", stripped)
    if m:
        idx = int(m.group(1))
        if 0 <= idx < len(var_names):
            return idx
    return None


def _to_dense_mean(adata_slice) -> np.ndarray:
    x = adata_slice.X
    if hasattr(x, "toarray"):
        return np.asarray(x.mean(axis=0)).flatten()
    return np.asarray(x).mean(axis=0).flatten()


class Pipeline:
    def __init__(self):
        self.mean_delta: np.ndarray | None = None
        self.avg_target_delta: float | None = None
        self.var_names: list[str] | None = None
        self.gene_names: list[str] | None = None

    def fit(self, train_adata) -> None:
        control_mask = train_adata.obs["perturbation"] == "control"
        if control_mask.sum() == 0:
            raise ValueError("No control cells in train split.")

        control_mean = _to_dense_mean(train_adata[control_mask])

        train_perts = sorted(
            set(train_adata.obs["perturbation"].unique()) - {"control"}
        )
        self.var_names = list(train_adata.var_names)
        if "gene_name" in train_adata.var.columns:
            self.gene_names = list(train_adata.var["gene_name"].astype(str))
        else:
            self.gene_names = list(self.var_names)

        deltas = []
        target_drops = []  # post[target] - control_mean[target] per train pert
        for p in train_perts:
            mask = train_adata.obs["perturbation"] == p
            mean_p = _to_dense_mean(train_adata[mask])
            deltas.append(mean_p - control_mean)
            tgt = _resolve_target_index(p, self.var_names, self.gene_names)
            if tgt is not None:
                target_drops.append(float(mean_p[tgt] - control_mean[tgt]))

        self.mean_delta = (
            _stats.trim_mean(np.asarray(deltas), proportiontocut=0.1, axis=0)
            if deltas
            else np.zeros_like(control_mean)
        )
        self.avg_target_delta = (
            float(np.mean(target_drops)) if target_drops else None
        )
        self._control_mean = control_mean

    def predict(
        self,
        test_perts: list[str],
        control_mean: np.ndarray,
        train_adata,
    ) -> dict[str, np.ndarray]:
        if self.mean_delta is None:
            raise RuntimeError("Pipeline not fit.")

        out: dict[str, np.ndarray] = {}
        for p in test_perts:
            pred = control_mean + self.mean_delta
            if self.avg_target_delta is not None:
                tgt = _resolve_target_index(p, self.var_names, self.gene_names)
                if tgt is not None:
                    pred = pred.copy()
                    pred[tgt] = control_mean[tgt] + self.avg_target_delta
            out[p] = pred
        return out
