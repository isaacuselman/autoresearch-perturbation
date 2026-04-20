"""
eval_baselines.py — run alternative pipelines through both harnesses.

Currently supports the CRISPR-informed mean baseline (pipeline_cim).
Add more methods here as they are reimplemented.
"""
from __future__ import annotations

import json
import time

import harness as h_default
import harness_perturbench as h_pb
from pipeline_cim import Pipeline as CIMPipeline


def _run(make_split_fn, score_fn, tag: str, pipe_cls, pipe_kwargs=None):
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
    print("=" * 70)
    print("Baseline pipelines through both harnesses")
    print("=" * 70)

    rows = {}
    rows["cim_random"] = _run(
        make_split_fn=h_default.make_split,
        score_fn=h_default.score_predictions,
        tag="CIM on harness.py (random pert holdout, Pearson top-200 DE)",
        pipe_cls=CIMPipeline,
    )
    rows["cim_combo"] = _run(
        make_split_fn=h_pb.make_split,
        score_fn=h_pb.score_predictions,
        tag="CIM on harness_perturbench.py (combo split, cosine logFC)",
        pipe_cls=CIMPipeline,
    )

    print("\n" + "=" * 70)
    print(json.dumps(rows, indent=2, default=str))


if __name__ == "__main__":
    main()
