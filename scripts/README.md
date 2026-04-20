# scripts/ — optional GPU reproduction

## What's here

- **`run_on_gpu.sh`** — reproduce PerturBench's published LA number
  on Norman 2019 end-to-end (item #1 from the SOTA-claim plan that
  was deferred because CPU training is infeasible).

## When you'd actually run this

If you want the last asterisk gone from the SOTA claim. The
`pipeline_la_pb_arch.py` ablation on CPU already shows that
**their architecture under our training scores 0.875**, which
implies the gap to their published 0.79 is training-side. But a
fully apples-to-apples reproduction — their code, their training
setup, their number — is the only way to close "you didn't run
their actual pipeline end-to-end" as a reviewer comment.

Cost ballpark (2026-04):

| provider | GPU | ~hourly | 3-seed reproduction |
|---|---|:---:|:---:|
| Modal (serverless, per-second) | A100 40GB | ~$1.50 | ~$3 |
| RunPod | A100 40GB | ~$1.19 | ~$2 |
| RunPod | RTX 4090 | ~$0.39 | ~$2 |
| Lambda Labs | A100 40GB | ~$1.29 | ~$2 |
| Vast.ai spot | A100 | $0.60-1.00 | $1-2 |

Add $5-10 for setup mistakes. Realistic all-in: **$5-20**, probably
$5 if nothing goes wrong.

## How to run it

1. Spin up a GPU VM on your provider of choice. Anything from RTX
   4090 to A100 works. Python 3.11 is already there or installable.
2. Clone this repo into the VM and `cd` into it.
3. `bash scripts/run_on_gpu.sh`
4. Wait ~1.5 hours (A100) or ~5 hours (4090) for three seeds of
   training + eval. Results land in `./gpu_results/`.
5. `rsync` or `scp` `gpu_results/` back to your local machine.
6. Grep for `test_cosine_logFC` in each `conditionA_seed_*.log` or
   inspect the csv metrics — report three numbers as "PerturBench
   LA reproduction on GPU" in `BENCHMARK.md`.

## What it does NOT do (yet)

Condition B — running their codebase with our four training-side
improvements stacked on — requires editing their LA model class,
not just passing Hydra overrides. The minimum diff:

```python
# In perturbench/modelcore/models/latent_additive.py:
# 1. Training: aggregate per-pert before loss
#    (shift from per-cell MSE to per-pert-mean MSE)
# 2. Output: x_ctrl + decoder(z) instead of softplus(decoder(z))
# 3. Instantiate N_ENSEMBLE=5 model copies and average their
#    predictions at inference
# 4. In predict: override pred[target_idx] with
#    control[target_idx] + observed_per_target_delta
```

Applying this would answer "does our stack applied to their code
reach ~0.87 too?" — the strongest-possible version of the claim.
Not automated here because the diff touches internal Lightning
training step + optimizer setup and is easier to write by hand
than by sed.

## Future work

- Run this script end-to-end and replace the "published 0.79"
  row in `BENCHMARK.md` with our-reproduced number.
- Implement Condition B, run it, add as a new row.
- Also run TxPert from [github.com/valence-labs/TxPert](https://github.com/valence-labs/TxPert)
  on GPU for a current-SOTA comparison. Their checkpoint reproduces
  in ~1 hour on A100 per their README.
- Run Geneformer / scGPT fine-tuning on this task (both need GPU
  and both are in the 5-20 minute range on A100 for inference; fine-
  tuning is hours).
