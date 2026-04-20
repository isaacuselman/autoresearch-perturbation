"""
eval_la.py — run Latent Additive (pipeline_la.Pipeline) through both
harnesses and print the 2×2 head-to-head numbers vs the existing
ridge/FM-features pipeline.

Does not modify harness.py or harness_perturbench.py — reuses their
module-level functions. This is a human-driven evaluation script; the
agent-facing harness/pipeline contract is unchanged.
"""
from __future__ import annotations

import json
import time

import numpy as np

import harness as h_default
import harness_perturbench as h_pb
from pipeline_la import Pipeline as LaPipeline


def _run(make_split_fn, score_fn, tag: str, pipe_cls=LaPipeline, pipe_kwargs=None):
    t0 = time.time()
    adata, source = h_default.load_adata()
    train_adata, test_adata, test_perts = make_split_fn(adata)

    ctrl_mask = train_adata.obs["perturbation"] == "control"
    ctrl_mean = h_default.to_dense_mean(train_adata[ctrl_mask])

    pipe = pipe_cls(**(pipe_kwargs or {}))
    pipe.fit(train_adata)
    preds = pipe.predict(test_perts, control_mean=ctrl_mean, train_adata=train_adata)
    result = score_fn(preds, test_adata, ctrl_mean)
    result["wallclock_sec"] = time.time() - t0
    result["data_source"] = source
    result["tag"] = tag

    print(f"\n=== {tag} ===", flush=True)
    for k in (
        "score",
        "raw_score",
        "raw_cosine_logFC",
        "cosine_logFC_rank",
        "coverage",
        "n_covered",
        "n_test",
        "wallclock_sec",
    ):
        if k in result:
            v = result[k]
            print(f"{k}: {v:.6f}" if isinstance(v, float) else f"{k}: {v}")
    return result


def main():
    print("=" * 60)
    print("LA (pipeline_la.Pipeline) vs the current ridge/FM pipeline")
    print("through both harnesses (random split + PerturBench split)")
    print("=" * 60)

    r1 = _run(
        make_split_fn=h_default.make_split,
        score_fn=h_default.score_predictions,
        tag="LA on harness.py (random 20% pert holdout, Pearson top-200 DE)",
    )
    r2 = _run(
        make_split_fn=h_pb.make_split,
        score_fn=h_pb.score_predictions,
        tag="LA on harness_perturbench.py (combo split, cosine logFC)",
    )

    print("\n" + "=" * 60)
    print(json.dumps({"random_split": r1, "combo_split": r2}, indent=2))


if __name__ == "__main__":
    main()
