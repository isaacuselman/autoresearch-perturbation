"""
harness_perturbench.py — parallel evaluator that mirrors PerturBench
(Wu et al., 2024, arXiv:2408.10609) for honest head-to-head comparison.

Differences from harness.py:

  - Split: Norman 2019 combo-prediction split. Train on all single
    perturbations plus 30% of dual perturbations; test on the remaining
    70% of duals. Every single-target gene is seen at training time, so
    additive-style models (LA) can compose dual targets cleanly. This
    is the split used in PerturBench Table 3.
  - Metric: cosine similarity of predicted vs actual log fold change
    (logFC), computed per test perturbation across ALL genes. Our
    preprocessed data is already log1p-transformed, so logFC ≈ mean
    expression difference against the control mean. Reported scores are
    the mean cosine across test perturbations.
  - Secondary metric: cosine logFC *rank* — the rank of the correct
    test perturbation's predicted logFC among all test perturbations'
    actual logFCs, normalized to [0, 1]. Lower is better. Catches
    mode-collapse where a model predicts the same vector for every
    pert.

Immutable in the same sense as harness.py: pipelines may not edit this
file.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

CACHE_DIR = Path.home() / ".cache" / "autoresearch-perturbation"
DATA_PATH = CACHE_DIR / "norman_2019.h5ad"
SYNTHETIC_PATH = CACHE_DIR / "synthetic.h5ad"

SPLIT_SEED = 42          # independent of harness.py's SPLIT_SEED
DUAL_TRAIN_FRAC = 0.30   # PerturBench Table 3: 30% of duals in train
COVERAGE_FLOOR = 0.95
WALLCLOCK_CAP = 1200


def load_adata():
    import scanpy as sc
    if DATA_PATH.exists():
        return sc.read_h5ad(DATA_PATH), "real"
    if SYNTHETIC_PATH.exists():
        return sc.read_h5ad(SYNTHETIC_PATH), "synthetic"
    raise FileNotFoundError(
        f"No data found. Run `uv run prepare_data.py` first. "
        f"Expected: {DATA_PATH} or {SYNTHETIC_PATH}"
    )


def is_dual(pert: str) -> bool:
    """Dual perturbation label: 'GENE1+GENE2' where neither side is a
    control sentinel. Single-with-control labels ('GENE+ctrl') count as
    single perturbations — they appear in train."""
    if "+" not in pert:
        return False
    parts = [p.strip() for p in pert.split("+")]
    non_ctrl = [p for p in parts if p.lower() not in {"ctrl", "control", ""}]
    return len(non_ctrl) >= 2


def make_split(adata, control_label="control"):
    """PerturBench combo-prediction split.
    Train: all singles + DUAL_TRAIN_FRAC of duals (+ all control cells).
    Test: the other (1 - DUAL_TRAIN_FRAC) of duals.
    """
    pert_col = "perturbation"
    assert pert_col in adata.obs.columns, f"Expected `{pert_col}` in adata.obs"
    assert control_label in adata.obs[pert_col].values, (
        f"Expected `{control_label}` label in {pert_col}"
    )

    perts = [p for p in adata.obs[pert_col].unique() if p != control_label]
    singles = sorted(p for p in perts if not is_dual(p))
    duals = sorted(p for p in perts if is_dual(p))

    rng = np.random.default_rng(SPLIT_SEED)
    dual_shuf = list(duals)
    rng.shuffle(dual_shuf)
    n_train_dual = max(1, int(round(len(dual_shuf) * DUAL_TRAIN_FRAC)))
    train_duals = set(dual_shuf[:n_train_dual])
    test_duals = set(dual_shuf[n_train_dual:])

    train_perts = set(singles) | train_duals
    test_perts = test_duals

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
    return train_adata, test_adata, sorted(test_perts)


def to_dense_mean(adata_slice):
    x = adata_slice.X
    if hasattr(x, "toarray"):
        return np.asarray(x.mean(axis=0)).flatten()
    return np.asarray(x).mean(axis=0).flatten()


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na < 1e-12 or nb < 1e-12:
        return 0.0
    v = float(np.dot(a, b) / (na * nb))
    if np.isnan(v):
        return 0.0
    return v


def score_predictions(predictions, test_adata, control_mean, control_label="control"):
    """
    predictions: dict[str, np.ndarray] — pert_name -> predicted mean expression.
    test_adata: held-out cells.
    control_mean: train control mean expression.

    Metrics:
      cosine_logFC      — mean cosine(predicted_delta, actual_delta) across
                           test perts. This is PerturBench's headline
                           'Cosine logFC'.
      cosine_logFC_rank — average normalized rank of the correct
                           prediction's cosine similarity when compared
                           against every test pert's actual delta.
                           Lower is better. PerturBench's 'Cosine logFC rank'.
      score             — cosine_logFC, floored to 0 if coverage < 0.95.
                           This is the scalar pipelines ratchet on.
    """
    pert_col = "perturbation"
    test_perts = sorted(test_adata.obs[pert_col].unique())
    n_test = len(test_perts)

    actual_deltas: dict[str, np.ndarray] = {}
    predicted_deltas: dict[str, np.ndarray] = {}
    n_covered = 0
    for pert in test_perts:
        cells = test_adata[test_adata.obs[pert_col] == pert]
        if cells.n_obs == 0:
            continue
        actual = to_dense_mean(cells)
        actual_deltas[pert] = actual - control_mean
        if pert not in predictions:
            continue
        pred = np.asarray(predictions[pert]).flatten()
        if pred.shape != actual.shape:
            continue
        predicted_deltas[pert] = pred - control_mean
        n_covered += 1

    coverage = n_covered / n_test if n_test > 0 else 0.0
    cosines: list[float] = []
    ranks: list[float] = []

    # Stack actuals once for rank computation
    actual_matrix = np.stack(
        [actual_deltas[p] for p in test_perts if p in actual_deltas]
    ) if actual_deltas else np.zeros((0, 0))

    for pert in test_perts:
        if pert not in predicted_deltas or pert not in actual_deltas:
            continue
        own = _cosine(predicted_deltas[pert], actual_deltas[pert])
        cosines.append(own)
        # How many test perts' actual logFC does this prediction "match"
        # at least as well as the correct one? Normalized rank in [0, 1].
        others = np.array([
            _cosine(predicted_deltas[pert], actual_matrix[i])
            for i in range(actual_matrix.shape[0])
        ])
        better_or_equal = int((others >= own).sum())
        # own counts itself; normalize so 0 = best rank, 1 = worst.
        ranks.append(max(0.0, (better_or_equal - 1) / max(1, len(test_perts) - 1)))

    mean_cosine = float(np.mean(cosines)) if cosines else 0.0
    mean_rank = float(np.mean(ranks)) if ranks else 1.0
    final_score = mean_cosine if coverage >= COVERAGE_FLOOR else 0.0
    return {
        "score": final_score,
        "raw_cosine_logFC": mean_cosine,
        "cosine_logFC_rank": mean_rank,
        "coverage": float(coverage),
        "n_covered": int(n_covered),
        "n_test": int(n_test),
    }


def main():
    t0 = time.time()
    print("Loading data...", flush=True)
    adata, source = load_adata()
    print(f"source: {source}  shape: {adata.shape}", flush=True)

    train_adata, test_adata, test_perts = make_split(adata)
    n_train_perts = len(set(train_adata.obs["perturbation"].unique()) - {"control"})
    print(
        f"split: PerturBench combo  train_perts: {n_train_perts}  "
        f"test_perts: {len(test_perts)}  "
        f"train_cells: {train_adata.n_obs}  test_cells: {test_adata.n_obs}",
        flush=True,
    )

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

    print(f"score: {result['score']:.6f}")
    print(f"raw_cosine_logFC: {result['raw_cosine_logFC']:.6f}")
    print(f"cosine_logFC_rank: {result['cosine_logFC_rank']:.6f}")
    print(f"coverage: {result['coverage']:.4f}")
    print(f"n_covered: {result['n_covered']}")
    print(f"n_test: {result['n_test']}")
    print(f"wallclock_sec: {result['wallclock_sec']:.1f}")
    print(f"full_json: {json.dumps(result)}")


if __name__ == "__main__":
    main()
