"""
eval_pb_arch.py — run pipeline_la_pb_arch.Pipeline (PerturBench's
4M-parameter LA architecture with their best Norman19
hyperparameters, wrapped in our per-pert-mean training + ensemble
+ target override) through harness_perturbench.py.

This answers the question: does their larger architecture, under
our training, score higher than either pipeline alone?

Single base seed by default — this is the fastest result; variance
is separately bounded by the ablation numbers.
"""
from __future__ import annotations

import sys
import time

import harness as h_default
import harness_perturbench as h_pb
from pipeline_la_pb_arch import Pipeline


def main():
    base_seed = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    t0 = time.time()
    adata, _ = h_default.load_adata()
    train_adata, test_adata, test_perts = h_pb.make_split(adata)
    ctrl_mean = h_default.to_dense_mean(
        train_adata[train_adata.obs["perturbation"] == "control"]
    )
    pipe = Pipeline(seed=base_seed, verbose=True)
    pipe.fit(train_adata)
    preds = pipe.predict(test_perts, control_mean=ctrl_mean, train_adata=train_adata)
    r = h_pb.score_predictions(preds, test_adata, ctrl_mean)
    elapsed = time.time() - t0
    print(
        f"\npb_arch base_seed {base_seed}: cosine_logFC={r['raw_cosine_logFC']:.4f}  "
        f"rank={r['cosine_logFC_rank']:.4f}  coverage={r['coverage']:.4f}  "
        f"wall={elapsed:.0f}s",
        flush=True,
    )


if __name__ == "__main__":
    main()
