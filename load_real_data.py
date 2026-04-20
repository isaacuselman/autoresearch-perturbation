"""
load_real_data.py — one-time helper to populate ~/.cache/autoresearch-perturbation/norman_2019.h5ad
from a pre-downloaded raw file. Bypasses pertpy's __init__ chain (broken
under current jax/numpyro versions in this venv).

Mirrors the preprocessing in prepare_data.py:try_real_data():
  - normalize_total → log1p → top-5000 HVGs
  - rename perturbation column candidates to "perturbation"
  - rename a control-variant label to "control"

Run after curling https://exampledata.scverse.org/pertpy/norman_2019.h5ad to
norman_2019.raw.h5ad in the cache dir.
"""
from __future__ import annotations

from pathlib import Path

import scanpy as sc

CACHE_DIR = Path.home() / ".cache" / "autoresearch-perturbation"
RAW_PATH = CACHE_DIR / "norman_2019.raw.h5ad"
OUT_PATH = CACHE_DIR / "norman_2019.h5ad"


def main() -> None:
    if not RAW_PATH.exists():
        raise SystemExit(
            f"raw file missing: {RAW_PATH}\n"
            f"download with: curl -fL -o {RAW_PATH} "
            f"https://exampledata.scverse.org/pertpy/norman_2019.h5ad"
        )

    print(f"[real] loading raw: {RAW_PATH}")
    adata = sc.read_h5ad(RAW_PATH)
    print(f"[real] raw shape: {adata.shape}")
    print(f"[real] obs columns: {list(adata.obs.columns)}")

    if "perturbation" not in adata.obs.columns:
        for candidate in [
            "perturbation_name",
            "guide_identity",
            "gene",
            "condition",
            "perturbation_label",
        ]:
            if candidate in adata.obs.columns:
                adata.obs["perturbation"] = adata.obs[candidate].astype(str)
                print(f"[real] using `{candidate}` as perturbation column")
                break
    if "perturbation" not in adata.obs.columns:
        raise SystemExit(
            f"could not identify perturbation column in obs: "
            f"{list(adata.obs.columns)}"
        )

    pert_values = adata.obs["perturbation"].astype(str)
    ctrl_variants = {"control", "ctrl", "NT", "non-targeting", "Non-Targeting", "NTC"}
    hit = None
    for v in ctrl_variants:
        if (pert_values == v).any():
            hit = v
            break
    if hit is None:
        mode = pert_values.value_counts().idxmax()
        print(f"[real] no canonical control label; using modal value `{mode}` as control")
        hit = mode
    else:
        print(f"[real] using `{hit}` as control label")
    adata.obs["perturbation"] = pert_values.replace({hit: "control"})

    print(
        f"[real] n_perts (incl control): {adata.obs['perturbation'].nunique()} | "
        f"n_control_cells: {(adata.obs['perturbation'] == 'control').sum()}"
    )

    print("[real] normalize_total / log1p / HVG-5000")
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    try:
        sc.pp.highly_variable_genes(adata, n_top_genes=5000, flavor="seurat")
        adata = adata[:, adata.var["highly_variable"]].copy()
    except Exception as e:
        print(f"[real] HVG selection failed, keeping all genes: {e}")

    adata.write_h5ad(OUT_PATH)
    print(
        f"[real] saved: {OUT_PATH}  shape: {adata.shape}  "
        f"perts: {adata.obs['perturbation'].nunique()}"
    )


if __name__ == "__main__":
    main()
