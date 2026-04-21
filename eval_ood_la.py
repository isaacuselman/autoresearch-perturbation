"""
eval_ood_la.py — run the Latent Additive pipeline (pipeline_la.Pipeline,
the run-2 baseline) through the OOD harness. Tests LA's expected
failure mode: when component target genes weren't seen as singles,
LA's per-token latent `z_pert` is zero for those genes and the model
should degrade sharply compared to the ridge+FM pipeline which can
consume embeddings for unseen genes.
"""
from __future__ import annotations

import os
import time

import harness_perturbench as h_pb
import harness_perturbench_ood as h_ood
from pipeline_la import Pipeline as LA


def main() -> None:
    ood_mode = os.environ.get("OOD_MODE", "combo_seen1")
    t0 = time.time()
    adata, source = h_pb.load_adata()
    print(f"source: {source}  shape: {adata.shape}  ood_mode: {ood_mode}", flush=True)

    train_adata, test_adata, test_perts = h_ood.make_split_ood(adata, ood_mode=ood_mode)
    ctrl_mean = h_pb.to_dense_mean(
        train_adata[train_adata.obs["perturbation"] == "control"]
    )
    pipe = LA(verbose=False)
    pipe.fit(train_adata)
    preds = pipe.predict(test_perts, control_mean=ctrl_mean, train_adata=train_adata)
    r = h_pb.score_predictions(preds, test_adata, ctrl_mean)
    elapsed = time.time() - t0
    print(
        f"\nLA on ood_mode={ood_mode}: "
        f"cosine_logFC={r['raw_cosine_logFC']:.4f}  "
        f"rank={r['cosine_logFC_rank']:.4f}  "
        f"coverage={r['coverage']:.4f}  "
        f"wall={elapsed:.0f}s",
        flush=True,
    )


if __name__ == "__main__":
    main()
