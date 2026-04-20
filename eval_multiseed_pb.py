"""
eval_multiseed_pb.py — multi-seed evaluation of pipeline.py through
harness_perturbench.py.

Each base seed produces a fresh 5-model ensemble (seeds [base, base+1,
base+2, base+3, base+4]). Reports mean ± std over the base seeds —
the right error bar for a defensible SOTA claim.

Run:
  perl -e 'alarm shift; exec @ARGV' 7200 \
    uv run --no-sync python eval_multiseed_pb.py
"""
from __future__ import annotations

import time

import numpy as np

import harness as h_default
import harness_perturbench as h_pb
from pipeline import Pipeline


BASE_SEEDS = [0, 100, 200]  # 3 independent ensemble runs


def main() -> None:
    print("loading data...", flush=True)
    adata, source = h_default.load_adata()
    print(f"source: {source}  shape: {adata.shape}", flush=True)

    train_adata, test_adata, test_perts = h_pb.make_split(adata)
    ctrl_mean = h_default.to_dense_mean(
        train_adata[train_adata.obs["perturbation"] == "control"]
    )
    print(
        f"split: combo  train_perts: "
        f"{len(set(train_adata.obs['perturbation'].unique()) - {'control'})}  "
        f"test_perts: {len(test_perts)}",
        flush=True,
    )

    rows: list[tuple[int, float, float, float]] = []
    for s in BASE_SEEDS:
        t0 = time.time()
        pipe = Pipeline(seed=s, verbose=False)
        pipe.fit(train_adata)
        preds = pipe.predict(test_perts, control_mean=ctrl_mean, train_adata=train_adata)
        r = h_pb.score_predictions(preds, test_adata, ctrl_mean)
        elapsed = time.time() - t0
        rows.append((s, r["raw_cosine_logFC"], r["cosine_logFC_rank"], elapsed))
        print(
            f"base_seed {s}: cosine={r['raw_cosine_logFC']:.4f}  "
            f"rank={r['cosine_logFC_rank']:.4f}  ({elapsed:.0f}s)",
            flush=True,
        )

    scores = np.array([row[1] for row in rows])
    ranks = np.array([row[2] for row in rows])

    print("\n=== summary ===", flush=True)
    print(f"n_base_seeds:       {len(scores)}")
    print(f"cosine_logFC mean:  {scores.mean():.4f}")
    print(f"cosine_logFC std:   {scores.std():.4f}")
    print(f"cosine_logFC range: [{scores.min():.4f}, {scores.max():.4f}]")
    print(f"rank mean:          {ranks.mean():.4f}")
    print(f"rank std:           {ranks.std():.4f}")


if __name__ == "__main__":
    main()
