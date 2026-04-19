"""
harness.py — IMMUTABLE. The agent must not edit this file.

Loads the pinned train/test split, runs pipeline.Pipeline, scores predictions,
prints grep-able scalars. Coverage floor enforced at 95% to prevent
silent-dropout gaming.

Metric: mean Pearson r across held-out perturbations on the top-200 most
differentially-expressed genes per perturbation (DE measured on held-out
actuals vs. train control mean). This is a reasonable proxy for the kind of
metric used in Open Problems / Polaris perturbation tasks, but is not
identical to any specific leaderboard's scoring. When submitting externally,
adapt separately.
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
from scipy.stats import pearsonr

# scanpy imported lazily in load_adata() — it's only needed for h5ad I/O,
# not for scoring. This lets the scoring functions be tested in isolation.

CACHE_DIR = Path.home() / ".cache" / "autoresearch-perturbation"
DATA_PATH = CACHE_DIR / "norman_2019.h5ad"
SYNTHETIC_PATH = CACHE_DIR / "synthetic.h5ad"

SPLIT_SEED = 42
TEST_FRAC = 0.2
COVERAGE_FLOOR = 0.95
TOP_DE_GENES = 200
WALLCLOCK_CAP = 1200  # 20 min, matches program.md


def load_adata():
    """Prefer real data. Fall back to synthetic if real data not present."""
    import scanpy as sc  # deferred import
    if DATA_PATH.exists():
        return sc.read_h5ad(DATA_PATH), "real"
    if SYNTHETIC_PATH.exists():
        return sc.read_h5ad(SYNTHETIC_PATH), "synthetic"
    raise FileNotFoundError(
        f"No data found. Run `uv run prepare_data.py` first. "
        f"Expected: {DATA_PATH} or {SYNTHETIC_PATH}"
    )


def make_split(adata, control_label="control"):
    """Split perturbations (not cells) into train/test. Returns (train_adata, test_adata, test_perts)."""
    pert_col = "perturbation"
    assert pert_col in adata.obs.columns, f"Expected `{pert_col}` in adata.obs"
    assert control_label in adata.obs[pert_col].values, (
        f"Expected `{control_label}` label in {pert_col}"
    )

    perts = sorted([p for p in adata.obs[pert_col].unique() if p != control_label])
    rng = np.random.default_rng(SPLIT_SEED)
    perts_shuffled = list(perts)
    rng.shuffle(perts_shuffled)
    n_test = max(1, int(len(perts_shuffled) * TEST_FRAC))
    test_perts = set(perts_shuffled[:n_test])
    train_perts = set(perts_shuffled[n_test:])

    train_mask = (
        adata.obs[pert_col].isin(train_perts) | (adata.obs[pert_col] == control_label)
    )
    test_mask = adata.obs[pert_col].isin(test_perts)

    train_adata = adata[train_mask].copy()
    test_adata = adata[test_mask].copy()

    # Anti-leakage assertion
    train_pert_set = set(train_adata.obs[pert_col].unique()) - {control_label}
    assert not (train_pert_set & test_perts), (
        "TEST LEAKAGE DETECTED — train/test perturbations overlap. "
        "Did the pipeline modify the split? This is a hard failure."
    )
    return train_adata, test_adata, sorted(test_perts)


def to_dense_mean(adata_slice):
    """Mean expression across cells in a slice, returns 1D ndarray (n_genes,)."""
    x = adata_slice.X
    if hasattr(x, "toarray"):
        return np.asarray(x.mean(axis=0)).flatten()
    return np.asarray(x).mean(axis=0).flatten()


def score_predictions(predictions, test_adata, control_mean, control_label="control"):
    """
    predictions: dict[str, np.ndarray] — pert_name -> predicted mean expression vector.
    test_adata: held-out cells, with obs['perturbation'] labels.
    control_mean: train control mean expression.
    Returns dict with 'score', 'coverage', 'n_covered', 'n_test', 'per_pert'.
    """
    pert_col = "perturbation"
    test_perts = sorted(test_adata.obs[pert_col].unique())
    n_test = len(test_perts)

    pearsons = []
    per_pert = {}
    n_covered = 0

    for pert in test_perts:
        if pert not in predictions:
            per_pert[pert] = None
            continue
        cells = test_adata[test_adata.obs[pert_col] == pert]
        if cells.n_obs == 0:
            per_pert[pert] = None
            continue

        actual = to_dense_mean(cells)
        actual_delta = actual - control_mean
        pred = np.asarray(predictions[pert]).flatten()
        if pred.shape != actual.shape:
            per_pert[pert] = None
            continue
        pred_delta = pred - control_mean

        # Score on top-K DE genes (by actual |delta|) — rewards getting
        # perturbation-specific signal right, not matching noise on stable genes.
        k = min(TOP_DE_GENES, actual_delta.size)
        de_idx = np.argsort(np.abs(actual_delta))[-k:]
        if np.std(pred_delta[de_idx]) < 1e-10 or np.std(actual_delta[de_idx]) < 1e-10:
            r = 0.0
        else:
            r, _ = pearsonr(actual_delta[de_idx], pred_delta[de_idx])
            if np.isnan(r):
                r = 0.0
        pearsons.append(r)
        per_pert[pert] = float(r)
        n_covered += 1

    coverage = n_covered / n_test if n_test > 0 else 0.0
    raw_score = float(np.mean(pearsons)) if pearsons else 0.0
    final_score = raw_score if coverage >= COVERAGE_FLOOR else 0.0

    return {
        "score": final_score,
        "raw_score": raw_score,
        "coverage": float(coverage),
        "n_covered": int(n_covered),
        "n_test": int(n_test),
        "per_pert": per_pert,
    }


def main():
    t0 = time.time()
    print("Loading data...", flush=True)
    adata, source = load_adata()
    print(f"source: {source}  shape: {adata.shape}", flush=True)

    train_adata, test_adata, test_perts = make_split(adata)
    print(
        f"train_cells: {train_adata.n_obs}  test_cells: {test_adata.n_obs}  "
        f"test_perts: {len(test_perts)}",
        flush=True,
    )

    control_mask = train_adata.obs["perturbation"] == "control"
    if control_mask.sum() == 0:
        print("ERROR: no control cells in train split", flush=True)
        sys.exit(1)
    control_mean = to_dense_mean(train_adata[control_mask])

    # Run the pipeline
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

    # Grep-able scalars — these are what program.md tells the agent to parse
    print(f"score: {result['score']:.6f}")
    print(f"raw_score: {result['raw_score']:.6f}")
    print(f"coverage: {result['coverage']:.4f}")
    print(f"n_covered: {result['n_covered']}")
    print(f"n_test: {result['n_test']}")
    print(f"wallclock_sec: {result['wallclock_sec']:.1f}")
    print(f"full_json: {json.dumps({k: v for k, v in result.items() if k != 'per_pert'})}")


if __name__ == "__main__":
    main()
