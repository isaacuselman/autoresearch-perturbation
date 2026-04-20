"""
precompute_scgpt_emb.py — extract scGPT pretrained gene embeddings for the
HVGs of our processed Norman 2019 file. One-time precompute that pipeline.py
loads at fit() time. Caches at:

  ~/.cache/autoresearch-perturbation/embeddings/scgpt_hvg_emb.npy

Outputs a (n_genes, 512) float32 array aligned with adata.var_names.
Genes missing from scGPT's vocab get zeros; pipeline.py treats those rows
as having no scGPT signal.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import scanpy as sc
import torch

CACHE = Path.home() / ".cache" / "autoresearch-perturbation"
SCGPT_DIR = CACHE / "embeddings" / "scgpt"
DATA_PATH = CACHE / "norman_2019.h5ad"
OUT_PATH = CACHE / "embeddings" / "scgpt_hvg_emb.npy"
HIT_PATH = CACHE / "embeddings" / "scgpt_hvg_hits.json"


def main() -> None:
    if not DATA_PATH.exists():
        raise SystemExit(f"data missing: {DATA_PATH}")
    if not (SCGPT_DIR / "best_model.pt").exists():
        raise SystemExit(f"scGPT checkpoint missing: {SCGPT_DIR}")

    print(f"loading vocab: {SCGPT_DIR/'vocab.json'}")
    vocab: dict[str, int] = json.loads((SCGPT_DIR / "vocab.json").read_text())
    print(f"  vocab size: {len(vocab)}")

    print(f"loading checkpoint: {SCGPT_DIR/'best_model.pt'}")
    sd = torch.load(SCGPT_DIR / "best_model.pt", map_location="cpu", weights_only=False)
    emb_w = sd["encoder.embedding.weight"]
    print(f"  embedding shape: {tuple(emb_w.shape)}")
    emb = emb_w.detach().numpy().astype(np.float32)

    print(f"loading adata: {DATA_PATH}")
    ad = sc.read_h5ad(DATA_PATH)
    var_names = list(ad.var_names)
    gene_names = (
        list(ad.var["gene_name"].astype(str))
        if "gene_name" in ad.var.columns else var_names
    )
    n = len(var_names)
    out = np.zeros((n, emb.shape[1]), dtype=np.float32)
    hits: list[str] = []
    misses: list[str] = []
    for i, sym in enumerate(var_names):
        idx = vocab.get(sym)
        if idx is None:
            idx = vocab.get(gene_names[i])
        if idx is not None and 0 <= idx < emb.shape[0]:
            out[i] = emb[idx]
            hits.append(sym)
        else:
            misses.append(sym)

    print(f"hits: {len(hits)}/{n} HVGs found in scGPT vocab")
    print(f"first 5 misses: {misses[:5]}")
    np.save(OUT_PATH, out)
    HIT_PATH.write_text(json.dumps({"n_hits": len(hits), "n_total": n, "first_5_misses": misses[:5]}))
    print(f"saved: {OUT_PATH}  shape: {out.shape}")


if __name__ == "__main__":
    main()
