"""
prepare_data.py — one-time data setup. Run once before kicking off the loop.

Tries to download Norman 2019 perturb-seq data via pertpy. If that fails
(network, API drift, missing dep), generates a synthetic dataset with
structure realistic enough to validate the evaluation loop.

The agent's first real task, if real data is unavailable, is to fix the
loader — see the TODOs below.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import scanpy as sc

CACHE_DIR = Path.home() / ".cache" / "autoresearch-perturbation"
CACHE_DIR.mkdir(parents=True, exist_ok=True)
DATA_PATH = CACHE_DIR / "norman_2019.h5ad"
SYNTHETIC_PATH = CACHE_DIR / "synthetic.h5ad"


def try_real_data() -> bool:
    """Attempt to download Norman 2019 via pertpy. Return True on success."""
    if DATA_PATH.exists():
        print(f"[real] already cached: {DATA_PATH}")
        return True
    try:
        import pertpy as pt
    except ImportError:
        print("[real] pertpy not installed — skipping real data attempt")
        return False

    try:
        # TODO(agent): if this API has drifted, find the current loader.
        # Alternative routes: direct h5ad download from scperturb.org;
        # scanpy.datasets; CZ CELLxGENE Census with perturbation annotation.
        print("[real] downloading Norman 2019 via pertpy — this may take a few minutes")
        adata = pt.data.norman_2019()
    except Exception as e:
        print(f"[real] pertpy load failed: {e}")
        return False

    # Normalize column names to what harness.py expects.
    # TODO(agent): Norman 2019's perturbation column may be named
    # 'perturbation_name', 'guide_identity', 'gene', etc. Verify and rename.
    if "perturbation" not in adata.obs.columns:
        for candidate in ["perturbation_name", "guide_identity", "gene", "condition"]:
            if candidate in adata.obs.columns:
                adata.obs["perturbation"] = adata.obs[candidate].astype(str)
                break
    if "perturbation" not in adata.obs.columns:
        print("[real] could not identify perturbation column — see adata.obs.columns")
        print(list(adata.obs.columns))
        return False

    # Normalize the control label.
    pert_values = adata.obs["perturbation"].astype(str)
    ctrl_variants = {"control", "ctrl", "NT", "non-targeting", "Non-Targeting", "NTC"}
    hit = None
    for v in ctrl_variants:
        if (pert_values == v).any():
            hit = v
            break
    if hit is None:
        # Best-effort: the most common label is often the control
        mode = pert_values.value_counts().idxmax()
        print(f"[real] no canonical control label found; using modal value `{mode}` as control")
        hit = mode
    adata.obs["perturbation"] = pert_values.replace({hit: "control"})

    # Basic preprocessing: library-size normalization, log1p, keep top 5000 HVGs.
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    try:
        sc.pp.highly_variable_genes(adata, n_top_genes=5000, flavor="seurat")
        adata = adata[:, adata.var["highly_variable"]].copy()
    except Exception as e:
        print(f"[real] HVG selection failed, keeping all genes: {e}")

    adata.write_h5ad(DATA_PATH)
    print(f"[real] saved: {DATA_PATH}  shape: {adata.shape}  "
          f"perts: {adata.obs['perturbation'].nunique()}")
    return True


def make_synthetic(n_cells_per_pert: int = 50, n_perts: int = 40,
                   n_genes: int = 1000, seed: int = 0) -> None:
    """Generate synthetic perturbation data with structured deltas.

    Each perturbation targets one gene. The targeted gene has a strong
    negative delta (knockdown). A handful of downstream genes (random per
    perturbation) have smaller correlated deltas. Cells have gaussian noise.
    """
    print(f"[synthetic] generating: {n_perts} perts × {n_cells_per_pert} cells × {n_genes} genes")
    rng = np.random.default_rng(seed)

    base = rng.normal(loc=5.0, scale=1.5, size=n_genes).astype(np.float32)
    base = np.clip(base, 0.1, None)

    # Pick target gene for each perturbation (first n_perts genes)
    n_ctrl = n_cells_per_pert * 4
    all_cells = []
    all_labels = []

    # Controls
    ctrl_x = base + rng.normal(0, 0.5, size=(n_ctrl, n_genes)).astype(np.float32)
    all_cells.append(ctrl_x)
    all_labels.extend(["control"] * n_ctrl)

    # Perturbations
    for p in range(n_perts):
        target = p  # perturbation p knocks down gene p
        delta = np.zeros(n_genes, dtype=np.float32)
        delta[target] = -3.0  # strong knockdown
        # 5-15 downstream genes affected
        n_downstream = rng.integers(5, 16)
        downstream = rng.choice(n_genes, size=n_downstream, replace=False)
        delta[downstream] = rng.normal(0, 1.0, size=n_downstream).astype(np.float32)

        pert_x = base + delta + rng.normal(0, 0.5, size=(n_cells_per_pert, n_genes)).astype(np.float32)
        all_cells.append(pert_x)
        all_labels.extend([f"pert_gene_{p}"] * n_cells_per_pert)

    import anndata as ad
    X = np.vstack(all_cells)
    X = np.clip(X, 0, None)  # expression non-negative

    adata = ad.AnnData(
        X=X,
        obs={"perturbation": all_labels},
        var={"gene_name": [f"gene_{i}" for i in range(n_genes)]},
    )
    adata.write_h5ad(SYNTHETIC_PATH)
    print(f"[synthetic] saved: {SYNTHETIC_PATH}  shape: {adata.shape}  "
          f"perts: {adata.obs['perturbation'].nunique()}")


def main():
    if try_real_data():
        print("\nReal data ready. Run: uv run harness.py")
        return
    print("\n[real data unavailable — falling back to synthetic]")
    make_synthetic()
    print("\nSynthetic data ready. Run: uv run harness.py")
    print(
        "NOTE: synthetic data is for loop validation only. "
        "The agent's first real task is to get real data working — see TODOs in this file."
    )


if __name__ == "__main__":
    main()
