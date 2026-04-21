"""
scripts/condition_b_wrapper.py — run PerturBench's own Latent Additive
through their training (Condition A), then stack three of our four
improvements on top post-hoc (ensembling, output residual, per-target
override). Reports cosine_logFC for both conditions so the delta
attribution is clean.

What this script does NOT change: per-pert-mean training. The training
ablation already showed that's worth ~+0.008, and implementing it
inside PerturBench's Lightning training_step requires a code patch
(see `condition_b_training_patch.md`). Skipping it here keeps this
wrapper portable across minor PerturBench version drift.

Designed to be run on a GPU VM after `scripts/run_on_gpu.sh`'s
Condition A has produced N trained checkpoints.

Typical flow on the GPU VM:
    bash scripts/run_on_gpu.sh        # produces checkpoints per seed
    python scripts/condition_b_wrapper.py \\
        --checkpoints gpu_results/conditionA_seed_*_csv/checkpoints/*.ckpt \\
        --data /tmp/perturbench/.pb_data/norman19_processed.h5ad \\
        --output gpu_results/condition_b.json

The script assumes the PerturBench venv is active (or accessible via
an interpreter path) so it can import their data module for the
split + scoring. See scripts/README.md for environment setup.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoints",
        nargs="+",
        required=True,
        help="Paths to N trained-LA checkpoints from Condition A runs "
             "(one per ensemble member, one per seed).",
    )
    parser.add_argument(
        "--data",
        type=Path,
        required=True,
        help="Path to norman19_processed.h5ad (PerturBench's preprocessing).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Where to write the results JSON.",
    )
    parser.add_argument(
        "--no-residual",
        action="store_true",
        help="Disable the output-space residual on top of their predictions.",
    )
    parser.add_argument(
        "--no-override",
        action="store_true",
        help="Disable the per-target-gene override.",
    )
    args = parser.parse_args()

    # Local imports — assumes PerturBench env is active.
    try:
        import torch
        import numpy as np
        import scanpy as sc
        from perturbench.modelcore.models.latent_additive import LatentAdditive
    except ImportError as e:
        print(f"ERROR: run this from inside PerturBench's venv. {e}")
        sys.exit(1)

    print(f"loading N={len(args.checkpoints)} checkpoints", flush=True)
    models = [LatentAdditive.load_from_checkpoint(p) for p in args.checkpoints]
    for m in models:
        m.eval()

    print(f"loading data: {args.data}", flush=True)
    adata = sc.read_h5ad(args.data)

    # PerturBench's data module applies their own split + preprocessing;
    # here we use their split-csv directly from the training run directory
    # (the checkpoints should be under logs/train/runs/<ts>/).
    # For simplicity, run their evaluation function on the ensemble mean.

    # NOTE: full implementation left as an exercise — the fiddly bit is
    # matching their DataModule's batch collate so you can get pred_delta
    # and actual_delta per test pert. Roughly:
    #
    #   for test_pert in test_perts:
    #       for ensemble_member in models:
    #           preds.append(ensemble_member(ctrl_input, pert_onehot,
    #                                        covariates))
    #       pred = mean(preds)
    #       if not args.no_residual:
    #           pred = ctrl_input + pred   # residual override
    #       if not args.no_override:
    #           for tgt in resolved_target_genes(test_pert):
    #               pred[tgt] = ctrl[tgt] + per_gene_delta[tgt]
    #       score += cosine(pred - ctrl, actual - ctrl)
    #
    # Their DataModule has `test_dataloader()` and the cosine metric
    # is in perturbench.analysis.cosine_delta. Wire it up on-GPU after
    # the checkpoints are available.

    result = {
        "note": (
            "Condition B ensemble-of-LA wrapper — implementation stub. "
            "Needs to be completed on the GPU VM once checkpoints exist "
            "and the PerturBench DataModule is loaded. The three hooks "
            "(ensemble mean, output residual, per-target override) are "
            "documented inline; each is a few lines of code against the "
            "loaded models + their DataModule."
        ),
        "n_checkpoints": len(args.checkpoints),
        "data": str(args.data),
        "residual": not args.no_residual,
        "override": not args.no_override,
    }
    args.output.write_text(json.dumps(result, indent=2))
    print(f"wrote stub result to {args.output}")
    print("NOTE: script stub — complete it on the GPU VM when checkpoints are ready.")


if __name__ == "__main__":
    main()
