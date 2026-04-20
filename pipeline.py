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
from pathlib import Path
from typing import Optional

import numpy as np
from scipy import stats as _stats

_SCGPT_EMB_PATH = (
    Path.home() / ".cache" / "autoresearch-perturbation" / "embeddings" / "scgpt_hvg_emb.npy"
)


def _load_scgpt_emb(n_genes: int) -> np.ndarray | None:
    """Per-gene scGPT pretrained embedding (n_genes, d). None if cache missing."""
    if not _SCGPT_EMB_PATH.exists():
        return None
    emb = np.load(_SCGPT_EMB_PATH).astype(np.float32)
    if emb.shape[0] != n_genes:
        return None
    return emb


def _load_scgpt_kernel(n_genes: int) -> np.ndarray | None:
    """Cosine-similarity matrix between scGPT pretrained gene embeddings.
    Returns (n_genes, n_genes) float32 or None if cache missing.
    Diagonal zeroed; rows for genes missing from scGPT vocab become all-zero.
    """
    if not _SCGPT_EMB_PATH.exists():
        return None
    emb = np.load(_SCGPT_EMB_PATH).astype(np.float32)
    if emb.shape[0] != n_genes:
        return None
    norms = np.linalg.norm(emb, axis=1)
    norms_safe = np.where(norms < 1e-10, 1.0, norms)
    normed = emb / norms_safe[:, None]
    sim = (normed @ normed.T).astype(np.float32)
    sim = np.nan_to_num(sim, nan=0.0)
    np.fill_diagonal(sim, 0.0)
    missing = norms < 1e-10
    if missing.any():
        sim[missing, :] = 0.0
        sim[:, missing] = 0.0
    return sim


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

        ctrl_x = train_adata[control_mask].X
        ctrl_dense = (
            np.asarray(ctrl_x.toarray()) if hasattr(ctrl_x, "toarray") else np.asarray(ctrl_x)
        ).astype(np.float64)
        control_mean = ctrl_dense.mean(axis=0)
        # Pearson gene-gene correlation matrix from controls (used for propagation).
        centered = ctrl_dense - control_mean
        std = centered.std(axis=0)
        std_safe = np.where(std < 1e-10, 1.0, std)
        normed = centered / std_safe
        n_ctrl = ctrl_dense.shape[0]
        gene_corr = (normed.T @ normed) / max(1, n_ctrl - 1)
        gene_corr = np.nan_to_num(gene_corr, nan=0.0)
        np.fill_diagonal(gene_corr, 0.0)  # diag handled by override; no self-propagation
        self._gene_corr = gene_corr

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

        # scGPT pretrained gene-embedding cosine kernel + raw embeddings.
        scgpt_kernel = _load_scgpt_kernel(len(control_mean))
        scgpt_emb = _load_scgpt_emb(len(control_mean))
        if scgpt_kernel is not None:
            self._scgpt_kernel = scgpt_kernel
            self._scgpt_emb = scgpt_emb
            n_hits = int((np.linalg.norm(scgpt_kernel, axis=1) > 0).sum())
            print(f"scgpt: loaded ({n_hits}/{len(control_mean)} genes covered)", flush=True)
        else:
            self._scgpt_kernel = None
            self._scgpt_emb = None
            print("scgpt: no precomputed embeddings (skipping)", flush=True)

        # Conditional expectation kernel: beta_jg = cov(delta[:,g], delta[:,j])
        #                                          / var(delta[:,g])
        # E[delta[j] | delta[g] = x] ≈ mean_delta[j] + beta_jg * (x - mean_delta[g])
        if len(deltas_arr) >= 3:
            d_centered = deltas_arr - deltas_arr.mean(axis=0, keepdims=True)
            n_d = deltas_arr.shape[0]
            delta_cov = (d_centered.T @ d_centered) / max(1, n_d - 1)
            delta_cov = np.nan_to_num(delta_cov, nan=0.0)
            d_var = np.diag(delta_cov).copy()
            d_var_safe = np.where(d_var < 1e-10, 1.0, d_var)
            # beta[g, j] = delta_cov[g, j] / var(g)  → shape (n_genes, n_genes)
            self._beta_kernel = delta_cov / d_var_safe[:, None]
            np.fill_diagonal(self._beta_kernel, 0.0)
            # Keep delta_corr around in case downstream experiments need it.
            d_std = np.sqrt(d_var_safe)
            self._delta_corr = (delta_cov / d_std[:, None]) / d_std[None, :]
            self._delta_corr = np.nan_to_num(self._delta_corr, nan=0.0)
            np.fill_diagonal(self._delta_corr, 0.0)
        else:
            self._beta_kernel = np.zeros_like(self._gene_corr)
            self._delta_corr = np.zeros_like(self._gene_corr)

        # scGPT-as-features ridge regression: per-output-gene linear model
        # with target gene's scGPT embedding as input feature.
        if self._scgpt_emb is not None:
            X_rows = []
            d_rows = []
            kept_idx = []  # train pert indices with at least 1 resolved + scGPT-known target
            for i, idxs in enumerate(train_target_idxs):
                feats = [self._scgpt_emb[t] for t in idxs
                         if np.linalg.norm(self._scgpt_emb[t]) > 0]
                if not feats:
                    continue
                X_rows.append(np.mean(feats, axis=0))
                d_rows.append(deltas_arr[i])
                kept_idx.append(i)
            if len(X_rows) >= 32:
                X = np.stack(X_rows).astype(np.float64)  # (m, 512)
                D = np.stack(d_rows).astype(np.float64)  # (m, n_genes)
                # Ridge: B = (X^T X + λ I)^{-1} X^T D
                lam = 1.0 * X.shape[0]
                XtX = X.T @ X + lam * np.eye(X.shape[1])
                self._scgpt_ridge_B = np.linalg.solve(XtX, X.T @ D).astype(np.float32)
                print(f"scgpt-ridge: fit on {len(kept_idx)} perts, lambda={lam:.0f}", flush=True)
            else:
                self._scgpt_ridge_B = None
                print(f"scgpt-ridge: insufficient training perts ({len(X_rows)})", flush=True)
        else:
            self._scgpt_ridge_B = None

        # LOO-CV joint sweep: alpha (mean_delta scale), gamma (delta_corr kernel
        # weight, matching predict()), delta_w (scGPT cosine kernel weight).
        # beta (control-corr) was consistently dominated by delta-space kernels;
        # dropped from the sweep.
        candidate_alphas = [0.0, 1.0]
        candidate_gammas = [0.5, 1.0]
        candidate_deltas = [0.0, 0.5, 1.0]  # scGPT cosine-kernel weight
        candidate_etas = [10.0, 15.0, 20.0, 30.0, 50.0]    # scGPT-ridge prediction weight
        best_alpha, best_gamma, best_delta_w, best_eta, best_score = (
            1.0, 1.0, 0.0, 0.0, -np.inf
        )
        if deltas and self.avg_target_delta is not None:
            for alpha in candidate_alphas:
                for gamma in candidate_gammas:
                    for delta_w in candidate_deltas:
                        for eta in candidate_etas:
                            if eta > 0 and self._scgpt_ridge_B is None:
                                continue
                            rs = []
                            for i in range(len(train_perts)):
                                md_loo = _trim_mean_excl(deltas_arr, i, prop=0.1)
                                pred_delta = alpha * md_loo
                                if train_target_idxs[i]:
                                    prop_d = np.zeros_like(pred_delta)
                                    prop_s = np.zeros_like(pred_delta)
                                    prop_r = np.zeros_like(pred_delta)
                                    feats = []
                                    for tgt in train_target_idxs[i]:
                                        prop_d += self._delta_corr[tgt] * self.avg_target_delta
                                        if self._scgpt_kernel is not None:
                                            prop_s += self._scgpt_kernel[tgt] * self.avg_target_delta
                                        if self._scgpt_emb is not None and np.linalg.norm(self._scgpt_emb[tgt]) > 0:
                                            feats.append(self._scgpt_emb[tgt])
                                    if eta > 0 and feats and self._scgpt_ridge_B is not None:
                                        x_test = np.mean(feats, axis=0).astype(np.float32)
                                        prop_r = (x_test @ self._scgpt_ridge_B).astype(np.float64)
                                    pred_delta = (pred_delta + gamma * prop_d
                                                  + delta_w * prop_s + eta * prop_r)
                                    pred_delta = pred_delta.copy()
                                    for tgt in train_target_idxs[i]:
                                        pred_post = max(0.0, control_mean[tgt] + self.avg_target_delta)
                                        pred_delta[tgt] = pred_post - control_mean[tgt]
                                rs.append(_pearson_top_de(pred_delta, deltas_arr[i]))
                            mean_r = float(np.mean(rs)) if rs else -np.inf
                            if mean_r > best_score:
                                best_score = mean_r
                                best_alpha, best_gamma = alpha, gamma
                                best_delta_w, best_eta = delta_w, eta
        self.alpha = best_alpha
        self.beta = 0.0
        self.gamma = best_gamma
        self.delta_w = best_delta_w
        self.eta = best_eta
        print(
            f"loo_alpha: {self.alpha:.2f}  loo_gamma: {self.gamma:.2f}  "
            f"loo_delta_w: {self.delta_w:.2f}  loo_eta: {self.eta:.2f}  "
            f"loo_pearson: {best_score:.4f}",
            flush=True,
        )

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
                    if self.gamma > 0 or self.delta_w > 0 or self.eta > 0:
                        prop_d = np.zeros_like(pred)
                        prop_s = np.zeros_like(pred)
                        prop_r = np.zeros_like(pred)
                        feats = []
                        for tgt in idxs:
                            prop_d += self._delta_corr[tgt] * self.avg_target_delta
                            if self._scgpt_kernel is not None:
                                prop_s += self._scgpt_kernel[tgt] * self.avg_target_delta
                            if self._scgpt_emb is not None and np.linalg.norm(self._scgpt_emb[tgt]) > 0:
                                feats.append(self._scgpt_emb[tgt])
                        if self.eta > 0 and feats and self._scgpt_ridge_B is not None:
                            x_test = np.mean(feats, axis=0).astype(np.float32)
                            prop_r = (x_test @ self._scgpt_ridge_B).astype(np.float64)
                        pred = pred + self.gamma * prop_d + self.delta_w * prop_s + self.eta * prop_r
                    pred = pred.copy()
                    for tgt in idxs:
                        pred[tgt] = max(0.0, control_mean[tgt] + self.avg_target_delta)
            out[p] = pred
        return out
