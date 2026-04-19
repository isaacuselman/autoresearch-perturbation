# program.md — Perturbation Response Prediction Autoresearch

You are running an autoresearch loop on single-cell perturbation response prediction. Your job is to edit `pipeline.py` to improve the held-out score reported by `harness.py`. Run experiments indefinitely until the human interrupts.

## Files in scope

- **You CAN edit:** `pipeline.py` only.
- **You CANNOT edit:** `harness.py`, `prepare_data.py`, `program.md`, `pyproject.toml`, anything under `data/` or `~/.cache/autoresearch-perturbation/`.
- **You must not touch:** the pinned data split (`SPLIT_SEED` in harness.py), the test perturbation set, the scoring function, or the coverage floor. If you believe the metric is flawed, write a note in `NOTES_FOR_HUMAN.md` instead of changing the harness.

## Setup (one-time, run these before starting the loop)

1. Agree on a run tag with the human (e.g., `nov1`). The branch `autoresearch/<tag>` must not already exist.
2. `git checkout -b autoresearch/<tag>`
3. Read in order: `README.md`, `program.md`, `harness.py`, `pipeline.py`.
4. Verify data exists: `ls ~/.cache/autoresearch-perturbation/`. If the `.h5ad` file is missing, run `uv run prepare_data.py`. If that fails, the synthetic fallback will be used automatically — note this in `NOTES_FOR_HUMAN.md`.
5. Run the baseline once: `uv run harness.py > run.log 2>&1`. Confirm it prints a `score:` line. Record the score as your baseline.
6. Initialize `results.tsv` with header:
   ```
   commit	score	coverage	wallclock_sec	status	description
   ```
7. Confirm with the human, then begin the loop.

## Experiment loop

LOOP FOREVER:

1. Look at the git state (`git log --oneline -5`) and the tail of `results.tsv`.
2. Propose ONE experimental change to `pipeline.py`. Write a one-line hypothesis before editing.
3. Edit `pipeline.py`.
4. `git add pipeline.py && git commit -m "<one-line hypothesis>"`
5. Run: `timeout 1200 uv run harness.py > run.log 2>&1` (20-minute cap — do NOT use `tee`, do NOT pipe to your context).
6. Parse results: `grep "^score:\|^coverage:\|^wallclock_sec:" run.log`
7. If grep is empty, the run crashed. `tail -n 50 run.log`. Attempt up to 2 fixes in-place. If still broken, status=crash, `git reset --hard HEAD~1`.
8. Append a row to `results.tsv`. **Do NOT commit results.tsv** — it stays untracked.
9. Decision:
   - `score > current_best` AND `coverage >= 0.95` → status=keep, branch advances.
   - otherwise → status=discard, `git reset --hard HEAD~1`.
10. Go to step 1.

## Domain priors (read before proposing any experiments)

- **The baseline is deliberately weak.** It predicts the same delta for every test perturbation. Your first improvement should use perturbation identity as a feature. Expect a large jump on experiment #1.
- **The real floor you need to beat is ridge regression on perturbation one-hot → gene delta.** A 2026 preprint reported that this linear baseline matches or beats scGPT/Geneformer on held-out perturbations. If you hit that floor and plateau, the agent is working correctly — try combining features (identity + gene embeddings + coexpression priors), not just bigger models.
- **Data leakage is the #1 failure mode in this field.** The harness asserts no train/test perturbation overlap. Do not undermine this. If you need a validation set, split INSIDE the training perturbations — never touch the test set.
- **The metric is mean Pearson r across test perturbations on the top-200 differentially-expressed genes per perturbation.** The coverage floor (95%) zeroes the score if you fail to predict enough test perturbations. Predicting `nan` or `0` is worse than predicting the mean delta.
- **Wallclock matters.** 20-minute hard cap. A run that would take longer is a crash. If you want to try foundation models, cache embeddings across runs — put cache files under `~/.cache/autoresearch-perturbation/embeddings/`.

## Ideas worth trying (not exhaustive, not in order)

- Ridge regression on perturbation one-hot → delta expression
- Bilinear: `delta[g,p] = sum_k U[g,k] * V[p,k]` with learned factors
- Perturbation features from external sources: GO terms, STRING edges, protein family, gene ontology depth
- Nearest-neighbor in perturbation-feature space (if pert q is similar to trained pert p, predict p's delta)
- Graph propagation on a gene-gene coexpression graph from the control cells
- scGPT / Geneformer gene embeddings as features for the *perturbed* gene
- Ensembling prior runs' predictions (weighted average of the top-3 commits)
- Per-gene calibration: scale predicted deltas to match training delta magnitudes
- Outlier-robust losses or Huber regression
- Explicit modeling of the perturbation target gene's own expression drop

## Ideas NOT worth trying early

- Full fine-tuning of scGPT or Geneformer (too slow for 20-min wallclock)
- Diffusion models, VAEs, or exotic architectures before ridge is beaten
- Anything requiring >1 GPU or >32 GB VRAM
- Meta-learning, multi-task setups, or anything that needs additional datasets loaded at runtime
- Modifications that require changing `harness.py`

## Simplicity criterion

All else equal, simpler is better. A 0.01 score gain that adds 200 lines of code and two new dependencies is probably not worth it. A 0.01 gain from deleting code is a clear win and should be kept. Complexity is a tax on the human reader — and the human is going to write this up.

## Results table format

Columns in `results.tsv` (tab-separated):
- `commit` — short hash from `git rev-parse --short HEAD`
- `score` — the scalar from harness output (float, 6 decimals)
- `coverage` — fraction of test perturbations predicted (float, 4 decimals)
- `wallclock_sec` — total run time
- `status` — `keep`, `discard`, or `crash`
- `description` — short human-readable hypothesis (same as commit message)

Crashes are logged as `0.000000` score, `0.0000` coverage, status=`crash`, so the file always parses.

## NEVER STOP

Once the loop has begun, do NOT pause to ask the human if you should continue. Do NOT ask "should I keep going?" or "is this a good stopping point?". The human may be asleep, may have stepped away, and expects you to continue working indefinitely until manually stopped. You are autonomous.

If you run out of ideas: re-read this file, read `harness.py` for new angles on the metric, combine previous near-misses, or try a more radical architectural change. The loop runs until the human interrupts it, period.
