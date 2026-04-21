# program.md — PerturBench-parity loop (LA-seeded)

You are running an autoresearch loop on the Norman 2019 combinatorial
gene-overexpression prediction task using the PerturBench evaluation
conventions (Wu et al. 2024, arXiv:2408.10609). Your job is to edit
`pipeline.py` to improve the scalar score printed by
`harness_perturbench.py`. Run experiments indefinitely until the human
interrupts.

## What changed from the previous run

This is a fresh loop. A prior run used `harness.py` (random 20% pert
holdout, Pearson on top-200 DE genes) with a ridge + foundation-model
pipeline; that run plateaued near 0.718. This run uses a **different
evaluator, split, and seed pipeline**:

- **Evaluator:** `harness_perturbench.py`, not `harness.py`. You should
  ratchet on the `score:` line that `harness_perturbench.py` prints.
  `harness.py` still exists for cross-evaluation but it is *not* the
  primary score.
- **Split:** train on all single perturbations plus 30% of dual
  perturbations; test on the remaining 70% of duals. Every target gene
  is seen at training time as a single.
- **Metric:** mean cosine similarity of predicted vs actual logFC
  across held-out perts, across *all* genes. Secondary: cosine-logFC
  rank in [0, 1] (lower is better). Scale is 0-1 with 1 = perfect.
- **Starting pipeline:** Latent Additive (LA). `pipeline.py` starts
  with an MLP autoencoder that encodes the control mean + pert
  multi-hot into a shared latent space and decodes back to expression.
  This is PerturBench's best published baseline.
- **Baseline to beat:** LA already scores ~0.825 cosine logFC on this
  harness (reproduced across 4 seeds, std 0.006). PerturBench's paper
  reports 0.79 for the same model; this implementation's extra ~3
  points come from training on per-pert means rather than per-cell
  examples. Your floor is 0.825.

## Files in scope

- **You CAN edit:** `pipeline.py` only.
- **You CANNOT edit:** `harness.py`, `harness_perturbench.py`,
  `prepare_data.py`, `program.md`, `pyproject.toml`, anything under
  `~/.cache/autoresearch-perturbation/`, `pipeline_la.py` (the
  reference LA implementation on `main`).
- **You must not touch:** the pinned data split (`SPLIT_SEED` in
  `harness_perturbench.py`), the train/test perturbation assignment,
  the scoring function, or the coverage floor. Write concerns in
  `NOTES_FOR_HUMAN.md` instead.

## Setup (one-time)

1. Confirm the branch is `autoresearch/pb-apr20`.
2. Read in order: this file, `harness_perturbench.py`, `pipeline.py`.
   The priors from the prior run are already in this file under
   "Domain priors carried over"; you do not need to consult any
   other documentation.
3. Verify the data exists: `ls ~/.cache/autoresearch-perturbation/`.
   The real Norman 2019 h5ad is expected at `norman_2019.h5ad`. If
   missing, ask the human — do not re-run `prepare_data.py` blindly.
4. Run the baseline once:
   `perl -e 'alarm shift; exec @ARGV' 1200 uv run --no-sync harness_perturbench.py > run.log 2>&1`
   Confirm the `score:` line appears and is near 0.825. Record as
   your baseline.
5. Initialize `results.tsv` with header:
   ```
   commit\tscore\tcosine_logFC_rank\tcoverage\twallclock_sec\tstatus\tdescription
   ```
6. Begin the loop.

## Experiment loop

LOOP FOREVER:

1. Look at the git state (`git log --oneline -5`) and the tail of
   `results.tsv`.
2. Propose ONE experimental change. Write a one-line hypothesis before
   editing.
3. Edit `pipeline.py`.
4. `git add pipeline.py && git commit -m "<one-line hypothesis>"`
5. Run:
   `perl -e 'alarm shift; exec @ARGV' 1200 uv run --no-sync harness_perturbench.py > run.log 2>&1`
   (20-minute cap — do NOT use `tee`; do NOT pipe harness output to
   your context.)
6. Parse: `grep "^score:\|^cosine_logFC_rank:\|^coverage:\|^wallclock_sec:" run.log`.
7. If grep is empty, the run crashed. `tail -n 50 run.log`. Up to 2
   in-place fixes. If still broken, status=crash, `git reset --hard HEAD~1`.
8. Append a row to `results.tsv`. **Do NOT commit `results.tsv`.**
9. Decision:
   - `score > current_best` AND `coverage >= 0.95` → status=keep.
   - otherwise → status=discard, `git reset --hard HEAD~1`.
10. Go to step 1.

## Domain priors carried over from the prior run

These priors were earned on `harness.py` + ridge/FM pipeline. They may
or may not apply to LA + combo split. Test before assuming.

- **Foundation-model embeddings (scGPT, Geneformer) help as REGRESSION
  FEATURES, not as similarity kernels.** Earned in 44 experiments.
  scGPT gene embeddings can be loaded via bare PyTorch from
  `~/.cache/autoresearch-perturbation/embeddings/scgpt_hvg_emb.npy`
  (shape (5000, 512)); Geneformer at `geneformer_hvg_emb.npy`
  (shape (5000, 256)). For LA, a natural try is concatenating the
  target gene's embedding(s) onto `z_pert` or passing embeddings as
  auxiliary input to `f_pert`.
- **Per-pert mean training beats per-cell training on this scoring
  function.** Directly optimizes what the harness scores. LA in
  `pipeline.py` already uses this; don't regress.
- **Delta-space coexpression (gene-gene correlation across training
  delta vectors) carries real signal;** control-cell coexpression
  does not. For LA, this might show up as a useful decoder
  regularizer or as a per-gene output-space prior.
- **Watch for kernel-mismatch bugs.** The prior run had a bug where
  LOO scored one kernel and predict used another; it produced a
  phantom +0.01 jump that survived ~10 experiments. Whenever training
  and inference use different code paths, they must use the SAME
  transforms.

## Ideas worth trying on LA (not exhaustive, not ordered)

- **Scale capacity.** Wider latent (256, 512), deeper MLPs (3-4
  hidden layers), bigger hidden (1024). LA's ~0.8M params is tiny.
- **Add foundation-model features.** Replace or augment the
  multi-hot input to `f_pert` with the pretrained embedding of the
  target gene(s). This is the LA + scGPT combination PerturBench
  found worth trying.
- **Ensemble seeds.** Train N independent LAs with different seeds,
  average predictions. Standard ML trick; low-risk +0.005-0.01.
- **Per-cell training revisited.** The 3-point gap vs PerturBench's
  reported number came from per-pert-mean training. Maybe a hybrid
  (per-cell for gradient noise, per-pert for calibration) is even
  better.
- **Learned control encoder.** Currently `f_ctrl(control_mean)` is
  deterministic. Sampling control cells per training step would add
  useful noise and could capture cell-state heterogeneity.
- **Multiplicative composition for duals.** z_ctrl + z_pert1 + z_pert2
  is *additive*. Biologically, some dual perturbations interact
  non-linearly. An interaction term z_{pert1,pert2} learned only for
  duals might help on a subset of the combinations.
- **L2 regularization on z_pert.** Keeps rare training perts from
  dominating latent space. Gentle ridge, LOO-tuned.
- **Output-space residual connection.** Predict `control_mean +
  f_dec(z)` instead of `f_dec(z)`. Stabilizes training when the
  pert effect is small.
- **Explicit target-gene drop.** Override the predicted expression
  of the perturbed gene with a learned constant, as in the prior
  run's pipeline.py. LA may already approximate this; explicit
  override sometimes wins.

## Ideas NOT worth trying early

- **Fine-tuning full scGPT or Geneformer.** Too slow for 20-min
  wallclock; also largely redundant with the existing per-gene
  embeddings.
- **Adversarial training (CPA-style).** PerturBench's results say
  it underperforms LA and training is unstable. Do not.
- **Diffusion / VAE / exotic architectures** before LA is
  convincingly improved.
- **Anything requiring >1 GPU or >32 GB VRAM.**
- **Using scGPT/Geneformer as a similarity kernel for propagation.**
  Earned in 44 experiments: weight collapses to 0 or hurts. Use
  embeddings as features instead.
- **Modifying `harness_perturbench.py` or `harness.py`.**

## Simplicity criterion

All else equal, simpler is better. LA in its current form is ~150
lines and scores 0.825. A 0.01 gain that adds 200 lines of code and
two dependencies is probably not worth it. A 0.01 gain from deleting
or generalizing code is a clear win.

## Results table format

Columns in `results.tsv` (tab-separated):
- `commit` — short hash from `git rev-parse --short HEAD`
- `score` — cosine logFC from harness output (float, 6 decimals)
- `cosine_logFC_rank` — secondary metric (float, 4 decimals)
- `coverage` — fraction of test perts predicted (float, 4 decimals)
- `wallclock_sec` — total run time
- `status` — `keep`, `discard`, or `crash`
- `description` — short human-readable hypothesis

Crashes are logged as `0.000000` score, `1.0000` rank, `0.0000`
coverage, status=`crash`.

## NEVER STOP

Once the loop has begun, do NOT pause to ask the human if you should
continue. Do NOT ask "should I keep going?" or "is this a good
stopping point?". The human may be asleep, may have stepped away, and
expects you to continue working indefinitely until manually stopped.

If you run out of ideas: re-read this file, re-read `POST.md` and
`PROCESS.md` for priors earned on the prior run, combine previous
near-misses, or try a more radical architectural change. The loop
runs until the human interrupts it, period.
