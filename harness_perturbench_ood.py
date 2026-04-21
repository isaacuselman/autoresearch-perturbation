"""
harness_perturbench_ood.py — PerturBench-compatible evaluator with
GEARS-style out-of-distribution splits on Norman 2019. Specifically:

  combo_seen0  Test duals where NEITHER component gene appears
               as a single perturbation in training.
  combo_seen1  Test duals where exactly ONE component gene was
               seen as a single.
  combo_seen2  Test duals where BOTH component genes were seen
               as singles. Equivalent to PerturBench's combo split;
               this is the setup harness_perturbench.py already
               uses.

Why this exists: the cosine-logFC claim in `BENCHMARK.md` holds for
combo_seen2, the easiest setting. LA-style additive methods are known
to degrade significantly on combo_seen1 and especially combo_seen0
because the z_pert latent for an unseen target gene is zero (or has
to be inferred from nothing). Measuring *how much* they degrade is
useful and was explicitly flagged open in `PROCESS_PB.md`.

Mirror's `harness_perturbench.py` otherwise: same data load, same
metrics, same coverage floor. Only the split function differs.
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import numpy as np

from harness_perturbench import (
    CACHE_DIR,
    DATA_PATH,
    SYNTHETIC_PATH,
    SPLIT_SEED,
    DUAL_TRAIN_FRAC,
    COVERAGE_FLOOR,
    WALLCLOCK_CAP,
    load_adata,
    is_dual,
    to_dense_mean,
    score_predictions,
)

# Which OOD setting to run. Override via the OOD_MODE env var at launch.
OOD_MODE = os.environ.get("OOD_MODE", "combo_seen1")
assert OOD_MODE in ("combo_seen0", "combo_seen1", "combo_seen2")


def _pert_components(pert: str) -> list[str]:
    parts = [x.strip() for x in pert.split("+")]
    return [x for x in parts if x.lower() not in {"ctrl", "control", ""}]


def _canonical_single(pert: str) -> str:
    """Handle labels like 'FOXA1+ctrl' or 'FOXA1'; return 'FOXA1'."""
    comps = _pert_components(pert)
    return comps[0] if comps else pert


def make_split_ood(adata, control_label="control", ood_mode: str = OOD_MODE):
    """GEARS-style combinatorial OOD splits.

    Norman 2019 by experimental design has every dual's component
    genes also present as singles, so naively filtering by
    seen-component count yields 131/131 combo_seen2 duals and zero
    of the others. To synthesize combo_seen0 / combo_seen1 tests,
    we ABLATE some singles from training:

      combo_seen2  (easy, baseline): train all singles + 30% of
                   duals, test on the remaining 70% of duals.
                   Matches harness_perturbench.py.
      combo_seen1  pick test duals, then remove ONE component
                   single per test dual from training. Model must
                   predict a dual where one component gene was
                   never seen as a single.
      combo_seen0  pick test duals, then remove BOTH component
                   singles per test dual from training. Model must
                   predict a dual with no prior on either gene.

    Test duals are a fixed 70% subset drawn from the same pool as
    harness_perturbench.py so numbers across modes are comparable.
    """
    pert_col = "perturbation"
    assert pert_col in adata.obs.columns
    assert control_label in adata.obs[pert_col].values

    all_perts = [p for p in adata.obs[pert_col].unique() if p != control_label]
    singles = sorted(p for p in all_perts if not is_dual(p))
    duals = sorted(p for p in all_perts if is_dual(p))

    rng = np.random.default_rng(SPLIT_SEED)

    # Fixed test-duals pool: same 30/70 split on duals as harness_perturbench.
    duals_shuf = list(duals)
    rng.shuffle(duals_shuf)
    n_train_dual = max(1, int(round(len(duals_shuf) * DUAL_TRAIN_FRAC)))
    train_duals = set(duals_shuf[:n_train_dual])
    test_duals = set(duals_shuf[n_train_dual:])

    # Build a gene-symbol -> canonical-single-pert-label index so we can
    # ablate singles.
    single_label_by_gene: dict[str, str] = {}
    for s in singles:
        g = _canonical_single(s)
        if g:
            single_label_by_gene[g] = s

    # For each test dual, decide which component singles to ablate.
    ablate_singles: set[str] = set()
    for d in test_duals:
        comps = _pert_components(d)
        if len(comps) < 2:
            continue
        if ood_mode == "combo_seen2":
            pass  # no ablation
        elif ood_mode == "combo_seen1":
            # Remove one component single (the first). Use a deterministic
            # pick so the split is reproducible: alphabetically first.
            c = sorted(comps)[0]
            if c in single_label_by_gene:
                ablate_singles.add(single_label_by_gene[c])
        elif ood_mode == "combo_seen0":
            for c in comps:
                if c in single_label_by_gene:
                    ablate_singles.add(single_label_by_gene[c])
        else:
            raise ValueError(ood_mode)

    train_perts = (set(singles) | train_duals) - ablate_singles
    test_perts = test_duals  # same test set across modes for comparability

    train_mask = (
        adata.obs[pert_col].isin(train_perts) | (adata.obs[pert_col] == control_label)
    )
    test_mask = adata.obs[pert_col].isin(test_perts)

    train_adata = adata[train_mask].copy()
    test_adata = adata[test_mask].copy()

    train_pert_set = set(train_adata.obs[pert_col].unique()) - {control_label}
    assert not (train_pert_set & test_perts), (
        "TEST LEAKAGE DETECTED — train/test perturbations overlap."
    )
    print(
        f"ood_mode={ood_mode}  ablated_singles={len(ablate_singles)}  "
        f"train_perts={len(train_pert_set)}  test_perts={len(test_perts)}",
        flush=True,
    )
    return train_adata, test_adata, sorted(test_perts)


def main():
    t0 = time.time()
    print(f"Loading data...  OOD_MODE={OOD_MODE}", flush=True)
    adata, source = load_adata()
    print(f"source: {source}  shape: {adata.shape}", flush=True)

    train_adata, test_adata, test_perts = make_split_ood(adata, ood_mode=OOD_MODE)
    n_train_perts = len(set(train_adata.obs["perturbation"].unique()) - {"control"})
    print(
        f"split: {OOD_MODE}  train_perts: {n_train_perts}  "
        f"test_perts: {len(test_perts)}  "
        f"train_cells: {train_adata.n_obs}  test_cells: {test_adata.n_obs}",
        flush=True,
    )
    if len(test_perts) == 0:
        print(
            f"ERROR: no test perturbations under OOD_MODE={OOD_MODE}. "
            f"This dataset may not contain combo_seen{OOD_MODE[-1]} duals.",
            flush=True,
        )
        sys.exit(1)

    control_mask = train_adata.obs["perturbation"] == "control"
    if control_mask.sum() == 0:
        print("ERROR: no control cells in train split", flush=True)
        sys.exit(1)
    control_mean = to_dense_mean(train_adata[control_mask])

    print("Running pipeline...", flush=True)
    from pipeline import Pipeline

    t_fit = time.time()
    pipe = Pipeline()
    pipe.fit(train_adata)
    print(f"fit_sec: {time.time() - t_fit:.1f}", flush=True)

    t_pred = time.time()
    predictions = pipe.predict(
        test_perts,
        control_mean=control_mean,
        train_adata=train_adata,
    )
    print(f"predict_sec: {time.time() - t_pred:.1f}", flush=True)

    result = score_predictions(predictions, test_adata, control_mean)
    result["wallclock_sec"] = time.time() - t0
    result["data_source"] = source
    result["ood_mode"] = OOD_MODE

    print(f"score: {result['score']:.6f}")
    print(f"raw_cosine_logFC: {result['raw_cosine_logFC']:.6f}")
    print(f"cosine_logFC_rank: {result['cosine_logFC_rank']:.6f}")
    print(f"coverage: {result['coverage']:.4f}")
    print(f"n_covered: {result['n_covered']}")
    print(f"n_test: {result['n_test']}")
    print(f"wallclock_sec: {result['wallclock_sec']:.1f}")
    print(f"ood_mode: {OOD_MODE}")
    print(f"full_json: {json.dumps(result)}")


if __name__ == "__main__":
    main()
