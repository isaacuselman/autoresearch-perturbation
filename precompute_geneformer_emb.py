"""
precompute_geneformer_emb.py — extract Geneformer V1-10M pretrained gene
token embeddings for our HVGs. Caches at:

  ~/.cache/autoresearch-perturbation/embeddings/geneformer_hvg_emb.npy

Geneformer's vocab is keyed by Ensembl gene IDs, not symbols, so we go
symbol → Ensembl → token → embedding via the dictionaries shipped with
the model repo.
"""
from __future__ import annotations

import json
import pickle
from pathlib import Path

import numpy as np
import scanpy as sc
from safetensors import safe_open

CACHE = Path.home() / ".cache" / "autoresearch-perturbation"
GF_DIR = CACHE / "embeddings" / "geneformer"
DATA_PATH = CACHE / "norman_2019.h5ad"
OUT_PATH = CACHE / "embeddings" / "geneformer_hvg_emb.npy"


def main() -> None:
    print(f"loading dictionaries from {GF_DIR}")
    sym2ens = pickle.load(
        open(GF_DIR / "geneformer/gene_dictionaries_30m/gene_name_id_dict_gc30M.pkl", "rb")
    )
    tok = pickle.load(
        open(GF_DIR / "geneformer/gene_dictionaries_30m/token_dictionary_gc30M.pkl", "rb")
    )
    print(f"  symbol->ensembl: {len(sym2ens)}, token dict: {len(tok)}")

    print(f"loading model: Geneformer-V1-10M/model.safetensors")
    with safe_open(GF_DIR / "Geneformer-V1-10M/model.safetensors", framework="pt") as f:
        emb_t = f.get_tensor("bert.embeddings.word_embeddings.weight")
    emb = emb_t.detach().numpy().astype(np.float32)
    print(f"  embedding shape: {emb.shape}")

    print(f"loading adata: {DATA_PATH}")
    ad = sc.read_h5ad(DATA_PATH)
    var_names = list(ad.var_names)
    gene_names = (
        list(ad.var["gene_name"].astype(str))
        if "gene_name" in ad.var.columns else var_names
    )
    n = len(var_names)
    out = np.zeros((n, emb.shape[1]), dtype=np.float32)
    hits, misses = [], []
    for i, sym in enumerate(var_names):
        ens = sym2ens.get(sym) or sym2ens.get(gene_names[i])
        if ens is None:
            misses.append(sym)
            continue
        token_idx = tok.get(ens)
        if token_idx is None:
            misses.append(sym)
            continue
        out[i] = emb[int(token_idx)]
        hits.append(sym)

    print(f"hits: {len(hits)}/{n}, first 5 misses: {misses[:5]}")
    np.save(OUT_PATH, out)
    print(f"saved: {OUT_PATH}  shape: {out.shape}")


if __name__ == "__main__":
    main()
