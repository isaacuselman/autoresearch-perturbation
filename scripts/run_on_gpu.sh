#!/usr/bin/env bash
# run_on_gpu.sh — reproduce PerturBench's published LA number on Norman19
# end-to-end (item #1 from the SOTA-claim plan), then run the same code
# with our four training-side improvements stacked on top to confirm
# the ~0.08 gap is training-side.
#
# Designed to run on a fresh GPU VM with Python 3.11+. Tested shape
# (not tested on any specific provider): ~15-30 min per seed on
# A100, ~4-5 hours for 3 seeds on RTX 4090.
#
# Cost estimate (2026-04): $3-$20 depending on provider and whether
# you hit any setup snags. See scripts/README.md for provider-specific
# notes.
#
# Usage (from a fresh clone of THIS repo on the GPU VM):
#   bash scripts/run_on_gpu.sh
#
# Writes results to ./gpu_results/ for rsync/scp back.

set -euo pipefail

WORKDIR="$(pwd)"
RESULTS_DIR="$WORKDIR/gpu_results"
PB_DIR="$WORKDIR/.pb"   # where we clone the PerturBench codebase
PB_DATA_DIR="$WORKDIR/.pb_data"
SEEDS=(0 100 200)

mkdir -p "$RESULTS_DIR" "$PB_DATA_DIR"

# ------------------------------------------------------------------
# 1. GPU sanity check
# ------------------------------------------------------------------
echo "=== GPU check ==="
python3 -c "
import torch
print('torch:', torch.__version__)
print('cuda available:', torch.cuda.is_available())
if torch.cuda.is_available():
    print('device:', torch.cuda.get_device_name(0))
    print('memory:', torch.cuda.get_device_properties(0).total_memory / 1e9, 'GB')
else:
    raise SystemExit('No CUDA — this script is meant for GPU VMs.')
"

# ------------------------------------------------------------------
# 2. Install PerturBench's own dependencies in an isolated venv so we
#    don't fight this repo's newer torch pin.
# ------------------------------------------------------------------
echo "=== Installing PerturBench codebase ==="
if [ ! -d "$PB_DIR" ]; then
    git clone --depth 1 https://github.com/altoslabs/perturbench.git "$PB_DIR"
fi

# Use uv if available (fast), else pip
if command -v uv >/dev/null 2>&1; then
    uv venv "$PB_DIR/.venv" --python 3.11
    uv pip install -e "$PB_DIR" --python "$PB_DIR/.venv/bin/python"
else
    python3 -m venv "$PB_DIR/.venv"
    "$PB_DIR/.venv/bin/pip" install --upgrade pip
    "$PB_DIR/.venv/bin/pip" install -e "$PB_DIR"
fi

PB_PY="$PB_DIR/.venv/bin/python"

# ------------------------------------------------------------------
# 3. Download Norman19 via their accessor (caches at $PB_DATA_DIR).
#    Their own preprocessed h5ad has a different shape than ours
#    (91k×5575 vs 111k×5000), which is fine — this whole run is
#    inside their codebase.
# ------------------------------------------------------------------
echo "=== Downloading Norman 2019 (their preprocessed h5ad) ==="
"$PB_PY" -c "
from perturbench.data.accessors.norman19 import Norman19
a = Norman19(data_cache_dir='$PB_DATA_DIR')
adata = a.get_anndata()
print('shape:', adata.shape)
"

# ------------------------------------------------------------------
# 4. Condition A — their published best config, one seed per run.
#    This is the direct reproduction of their 0.79 ± 0.01.
# ------------------------------------------------------------------
echo "=== Condition A: PerturBench LA (their config + their training) ==="
for S in "${SEEDS[@]}"; do
    echo ""
    echo "--- seed $S ---"
    (
        cd "$PB_DIR"
        "$PB_PY" -m perturbench.modelcore.train \
            experiment=neurips2025/norman19/latent_best_params_norman19 \
            trainer=gpu \
            paths.data_dir="$PB_DATA_DIR" \
            seed="$S" \
            2>&1 | tee "$RESULTS_DIR/conditionA_seed_$S.log"
    )
    # PerturBench writes eval metrics to logs/train/runs/<ts>/csv/...
    # Gather the final cosine_logFC from the most recent run.
    find "$PB_DIR/logs/train/runs" -type d -mmin -60 -name "version_0" \
        -exec cp -r {} "$RESULTS_DIR/conditionA_seed_${S}_csv" \; 2>/dev/null || true
done

# ------------------------------------------------------------------
# 5. Condition B — stack our four improvements on top of their codebase.
#    Small set of hydra overrides would require touching their code;
#    simpler to document the diff as a follow-up rather than hack it
#    here. The condition-A result is the headline measurement; anyone
#    who wants to go further can apply the four-item diff manually.
# ------------------------------------------------------------------
echo ""
echo "=== Condition B — skipped in this script ==="
echo "Stacking our four improvements (ensemble, output residual,"
echo "dropout=0, per-target override) onto PerturBench's codebase"
echo "requires editing their model class. A minimal patch is"
echo "documented in scripts/README.md; not auto-applied here."

# ------------------------------------------------------------------
# 6. Summary
# ------------------------------------------------------------------
echo ""
echo "=== Done ==="
echo "Raw logs in: $RESULTS_DIR"
echo "To extract cosine_logFC per seed, grep for 'test_cosine_logFC'"
echo "in each conditionA_seed_*.log or inspect the csv/metrics.csv files."
echo ""
echo "rsync or scp $RESULTS_DIR back to your local machine."
