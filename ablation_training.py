"""
ablation_training.py — run `pipeline_la_ablation.Pipeline` in both
training modes ('per_pert_mean', 'per_cell') with all else held
constant and compare scores on harness_perturbench.py.

This is the cleanest answer we can produce locally to the question:
'how much of the 0.871 advantage comes from per-pert-mean training
vs per-cell training, architecture and everything else fixed?'
"""
from __future__ import annotations

import time

import numpy as np

import harness as h_default
import harness_perturbench as h_pb
from pipeline_la_ablation import Pipeline


BASE_SEEDS = [0, 100, 200]


def run(mode: str, base_seed: int) -> dict:
    t0 = time.time()
    adata, _ = h_default.load_adata()
    train_adata, test_adata, test_perts = h_pb.make_split(adata)
    ctrl_mean = h_default.to_dense_mean(
        train_adata[train_adata.obs["perturbation"] == "control"]
    )
    pipe = Pipeline(training_mode=mode, seed=base_seed, verbose=False)
    pipe.fit(train_adata)
    preds = pipe.predict(test_perts, control_mean=ctrl_mean, train_adata=train_adata)
    r = h_pb.score_predictions(preds, test_adata, ctrl_mean)
    r["wallclock_sec"] = time.time() - t0
    r["mode"] = mode
    r["base_seed"] = base_seed
    return r


def main() -> None:
    print("loading data (once for sanity)...", flush=True)
    h_default.load_adata()
    results: dict[str, list[float]] = {"per_pert_mean": [], "per_cell": []}
    ranks: dict[str, list[float]] = {"per_pert_mean": [], "per_cell": []}
    waves: list[dict] = []

    for seed in BASE_SEEDS:
        for mode in ("per_pert_mean", "per_cell"):
            r = run(mode, seed)
            print(
                f"seed {seed}  mode {mode}: cosine_logFC={r['raw_cosine_logFC']:.4f}  "
                f"rank={r['cosine_logFC_rank']:.4f}  wall={r['wallclock_sec']:.0f}s",
                flush=True,
            )
            results[mode].append(r["raw_cosine_logFC"])
            ranks[mode].append(r["cosine_logFC_rank"])
            waves.append(r)

    def ms(vals):
        a = np.array(vals)
        return a.mean(), a.std(), a.min(), a.max()

    print("\n=== ablation: training procedure, all else held constant ===")
    for mode in ("per_pert_mean", "per_cell"):
        m, s, lo, hi = ms(results[mode])
        rm, rs, *_ = ms(ranks[mode])
        print(
            f"  {mode:15s} cosine={m:.4f} ± {s:.4f}  range=[{lo:.4f}, {hi:.4f}]  "
            f"rank={rm:.4f} ± {rs:.4f}",
            flush=True,
        )
    d = np.mean(results["per_pert_mean"]) - np.mean(results["per_cell"])
    print(f"\ndelta (per_pert_mean - per_cell): {d:+.4f}", flush=True)


if __name__ == "__main__":
    main()
