"""
pipeline_cim.py — "CRISPR-informed mean" baseline from
Wenteler et al. 2025 ("Simple controls exceed best deep learning
algorithms and reveal foundation model effectiveness for predicting
genetic perturbations", Bioinformatics).

Algorithm:
  - For non-target genes: predict mean expression across all
    perturbed cells in training.
  - For the target gene(s) of a CRISPRa perturbation (Norman 2019):
    predict 2 × mean expression of that gene in training.
  - (CRISPRi case would predict 0 at target genes; not used here
    since Norman is CRISPRa.)

This is a deliberately tiny baseline. The Wenteler paper claims it
matches or beats published deep learning methods on Norman 2019.
Reproducing it in our harness lets us put a number on it for
the BENCHMARK.md head-to-head.

Reference: https://github.com/pfizer-opensource/perturb_seq
"""
from __future__ import annotations

from typing import Optional

import numpy as np


def _resolve_target_indices(pert: str) -> list[str]:
    if pert == "control" or pert == "":
        return []
    parts = pert.split("+") if "+" in pert else [pert]
    out: list[str] = []
    for part in parts:
        token = part.strip()
        if token.lower() in {"ctrl", "control", ""}:
            continue
        out.append(token)
    return out


def _to_dense_mean(adata_slice) -> np.ndarray:
    x = adata_slice.X
    if hasattr(x, "toarray"):
        return np.asarray(x.mean(axis=0)).flatten()
    return np.asarray(x).mean(axis=0).flatten()


class Pipeline:
    """CRISPR-informed mean baseline.

    Norman 2019 is CRISPRa, so the target-gene rule is
    `predict[T] = 2 × mean expression of T across all training
    perturbed cells`. For non-targets, every test pert gets the same
    prediction = mean of all training perturbed cells.
    """

    def __init__(self, perturbation_type: str = "CRISPRa"):
        self.perturbation_type = perturbation_type
        self._mean_perturbed: Optional[np.ndarray] = None
        self._sym_to_hvg: dict[str, int] = {}

    def fit(self, train_adata) -> None:
        # Mean expression across ALL perturbed cells (excluding controls).
        pert_mask = train_adata.obs["perturbation"] != "control"
        if pert_mask.sum() == 0:
            raise ValueError("No perturbed cells in train split.")
        self._mean_perturbed = _to_dense_mean(train_adata[pert_mask])

        # Symbol → HVG index map for target overrides.
        var_names = list(train_adata.var_names)
        gene_names = (
            list(train_adata.var["gene_name"].astype(str))
            if "gene_name" in train_adata.var.columns else var_names
        )
        self._sym_to_hvg = {n: i for i, n in enumerate(var_names)}
        for i, gn in enumerate(gene_names):
            self._sym_to_hvg.setdefault(gn, i)

        n_perts = train_adata.obs["perturbation"].nunique() - 1
        print(
            f"cim: trained on {pert_mask.sum()} perturbed cells across "
            f"{n_perts} train perts; ptype={self.perturbation_type}",
            flush=True,
        )

    def predict(
        self,
        test_perts: list[str],
        control_mean: np.ndarray,
        train_adata,
    ) -> dict[str, np.ndarray]:
        if self._mean_perturbed is None:
            raise RuntimeError("Pipeline not fit.")
        out: dict[str, np.ndarray] = {}
        for p in test_perts:
            pred = self._mean_perturbed.copy()
            for sym in _resolve_target_indices(p):
                hvg = self._sym_to_hvg.get(sym)
                if hvg is None:
                    continue
                if self.perturbation_type == "CRISPRa":
                    pred[hvg] = 2.0 * float(self._mean_perturbed[hvg])
                else:  # CRISPRi
                    pred[hvg] = 0.0
            out[p] = pred
        return out
