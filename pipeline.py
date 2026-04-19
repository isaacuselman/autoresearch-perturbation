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


def _resolve_target_indices(
    pert: str,
    name_to_idx: dict[str, int],
) -> list[int]:
    """Map a perturbation label to one or more gene column indices.

    Handles:
      - single gene: 'FOXA1' or 'pert_gene_42' or 'guide_FOXA1'
      - dual perturbations: 'GENE1+GENE2' (each side resolved independently)
      - control suffixes: 'GENE+ctrl' / 'GENE_ctrl' / 'GENE+control'
      - synthetic 'pert_gene_<n>': trailing integer is the gene index
    """
    if pert == "control" or pert == "":
        return []

    parts = pert.split("+") if "+" in pert else [pert]
    out: list[int] = []
    seen: set[int] = set()
    for part in parts:
        token = part.strip()
        if token in {"ctrl", "control", ""}:
            continue
        for prefix in ("pert_gene_", "pert_", "guide_", "sg"):
            if token.startswith(prefix):
                token = token[len(prefix):]
        for suffix in ("_ctrl",):
            if token.endswith(suffix):
                token = token[: -len(suffix)]
        idx = name_to_idx.get(token)
        # synthetic fallback: 'pert_gene_<n>' tokens are pure integers post-strip
        if idx is None and token.isdigit():
            i = int(token)
            n = max(name_to_idx.values()) + 1 if name_to_idx else 0
            if 0 <= i < n:
                idx = i
        if idx is not None and idx not in seen:
            out.append(idx)
            seen.add(idx)
    return out


def _resolve_target_index(
    pert: str, var_names: list[str], gene_names: list[str]
) -> Optional[int]:
    """Backward-compatible single-target resolver. Returns first match or None."""
    name_to_idx: dict[str, int] = {n: i for i, n in enumerate(var_names)}
    for i, gn in enumerate(gene_names):
        name_to_idx.setdefault(gn, i)
    idxs = _resolve_target_indices(pert, name_to_idx)
    return idxs[0] if idxs else None


def _to_dense_mean(adata_slice) -> np.ndarray:
    x = adata_slice.X
    if hasattr(x, "toarray"):
        return np.asarray(x.mean(axis=0)).flatten()
    return np.asarray(x).mean(axis=0).flatten()


def _trim_mean_excl(deltas_arr: np.ndarray, exclude_idx: int, prop: float = 0.1) -> np.ndarray:
    mask = np.ones(deltas_arr.shape[0], dtype=bool)
    mask[exclude_idx] = False
    return _stats.trim_mean(deltas_arr[mask], proportiontocut=prop, axis=0)


def _pearson_top_de(pred_delta: np.ndarray, actual_delta: np.ndarray, k: int = 200) -> float:
    k = min(k, actual_delta.size)
    de_idx = np.argsort(np.abs(actual_delta))[-k:]
    p = pred_delta[de_idx]
    a = actual_delta[de_idx]
    if p.std() < 1e-10 or a.std() < 1e-10:
        return 0.0
    p = (p - p.mean()) / p.std()
    a = (a - a.mean()) / a.std()
    return float((p * a).mean())


class Pipeline:
    def __init__(self):
        self.mean_delta: np.ndarray | None = None
        self.avg_target_delta: float | None = None
        self.var_names: list[str] | None = None
        self.gene_names: list[str] | None = None
        self.alpha: float = 3.0

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

        # Combined name → index map (var_names overrides gene_names if both present).
        self._name_to_idx = {n: i for i, n in enumerate(self.var_names)}
        for i, gn in enumerate(self.gene_names):
            self._name_to_idx.setdefault(gn, i)

        deltas = []
        target_drops = []  # post[target] - control_mean[target] per (train pert, target gene)
        train_target_idxs: list[list[int]] = []
        for p in train_perts:
            mask = train_adata.obs["perturbation"] == p
            mean_p = _to_dense_mean(train_adata[mask])
            deltas.append(mean_p - control_mean)
            idxs = _resolve_target_indices(p, self._name_to_idx)
            train_target_idxs.append(idxs)
            for tgt in idxs:
                target_drops.append(float(mean_p[tgt] - control_mean[tgt]))

        n_resolved = sum(1 for x in train_target_idxs if x)
        print(
            f"resolver: {n_resolved}/{len(train_perts)} train perts have ≥1 resolved target; "
            f"{sum(len(x) for x in train_target_idxs)} target-gene observations",
            flush=True,
        )

        if deltas:
            deltas_arr = np.asarray(deltas)
            self.mean_delta = _stats.trim_mean(
                deltas_arr, proportiontocut=0.1, axis=0
            )
        else:
            deltas_arr = np.zeros((0, len(control_mean)))
            self.mean_delta = np.zeros_like(control_mean)
        self.avg_target_delta = (
            float(np.median(target_drops)) if target_drops else None
        )
        self._control_mean = control_mean

        # LOO-CV alpha sweep, with multi-target override.
        candidate_alphas = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]
        best_alpha, best_score = 1.0, -np.inf
        if deltas and self.avg_target_delta is not None:
            for alpha in candidate_alphas:
                rs = []
                for i in range(len(train_perts)):
                    md_loo = _trim_mean_excl(deltas_arr, i, prop=0.1)
                    pred_delta = alpha * md_loo
                    if train_target_idxs[i]:
                        pred_delta = pred_delta.copy()
                        for tgt in train_target_idxs[i]:
                            pred_post = max(0.0, control_mean[tgt] + self.avg_target_delta)
                            pred_delta[tgt] = pred_post - control_mean[tgt]
                    rs.append(_pearson_top_de(pred_delta, deltas_arr[i]))
                mean_r = float(np.mean(rs)) if rs else -np.inf
                if mean_r > best_score:
                    best_score, best_alpha = mean_r, alpha
        self.alpha = best_alpha
        print(f"loo_alpha: {self.alpha:.2f}  loo_pearson: {best_score:.4f}", flush=True)

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
            pred = control_mean + self.alpha * self.mean_delta
            if self.avg_target_delta is not None:
                idxs = _resolve_target_indices(p, self._name_to_idx)
                if idxs:
                    pred = pred.copy()
                    for tgt in idxs:
                        pred[tgt] = max(0.0, control_mean[tgt] + self.avg_target_delta)
            out[p] = pred
        return out
