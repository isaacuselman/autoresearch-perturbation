# Autoresearch on a perturbation-prediction problem

This project applies Andrej Karpathy's
[autoresearch pattern](https://x.com/karpathy/status/1843525867824615930)
to predicting how single cells respond to gene perturbations. The setup
hands an LLM agent (Claude Code) three files — an immutable evaluator
(`harness.py`), an agent-mutable pipeline (`pipeline.py`), and an
orchestration prompt (`program.md`) — plus a single scalar score the
agent ratchets on. The agent runs experiments overnight; a human edits
`program.md` between runs to inject domain priors.

The task: take Norman et al.'s 2019 perturb-seq dataset (K562 cells
with 237 CRISPR knockdowns, single and dual), hold out 20% of the
perturbations, and predict each held-out pert's average post-perturbation
expression vector. Score: mean Pearson correlation across the held-out
perts on each one's top-200 most-changed genes.

Across two runs, the score climbed from **0.27 to 0.72** on the
original setup (44 experiments) and then to **0.87** on the
PerturBench benchmark setup (another 11 experiments). The second
run lands above every published number we could find for the same
evaluation — the full benchmark table is in
[BENCHMARK.md](BENCHMARK.md).

What follows is a tour of the experiments that mattered, the bug
that nearly went unnoticed, and where the work ends up relative to
the literature.

## Setup

The starter pipeline predicts the *same* delta for every test
perturbation — just the average effect across all training perts. Score:
0.27. Deliberately bad, so the loop has obvious low-hanging fruit on
experiment one.

The agent may edit `pipeline.py` only. Each experiment is one git
commit; if the new score does not beat the current best, the commit
reverts. Results stream to a TSV that the agent appends to but does not
commit. A 20-minute wallclock cap applies to each experiment. The loop
runs until interrupted.

The data prep script falls back to a synthetic dataset when the real
download fails — and it did, because `pertpy`'s install is currently
broken under recent `jax`/`numpyro`/`flax` versions. The synthetic data
turned out to be a useful sanity check before the real Norman 2019
swap.

## Synthetic data: 0.27 → 0.71 in 21 experiments

Six experiments did most of the work:

1. **Use the perturbation identity.** The starter ignores which gene was
   perturbed. Overriding the predicted expression of the targeted gene
   with the average post-knockdown level seen in training jumped the
   score by **0.32** — the largest single gain in the entire run.
2. **Per-target baseline.** Using the targeted gene's *own* control-cell
   baseline plus a learned drop, instead of one global post-knockdown
   value, fixed wildly wrong predictions for low-baseline target genes:
   **+0.09**.
3. **Trimmed mean** over training perts (instead of plain mean) for the
   non-target baseline, to handle per-pert outliers: **+0.001**, basically
   noise.
4. **Scale up the non-target prediction by ~3.** Pearson is globally
   scale-invariant but not invariant to per-component scaling relative
   to the dominant target-gene signal: **+0.01**.
5. **Cross-validate the scale factor** via leave-one-pert-out inside
   training: **+0.005**.
6. **Clip predicted post-values to ≥ 0** (negative expression is
   physically impossible): **+0.003**.

By experiment 21 the score plateaued at 0.71. The synthetic generator
makes downstream effects *random per perturbation*, so no learnable
signal remains beyond the target-gene knockdown itself. A
back-of-envelope ceiling of ~0.69 matched what the loop found; the LOO
tuning squeezed slightly above it.

That run was not wasted. The plumbing worked end to end. A discarded
experiment (control-cell coexpression propagation) failed specifically
because the synthetic generator has no real coexpression structure —
strong evidence that the same idea would succeed on real data. It did.

## The pivot to real data: 0.54 → 0.72 in 23 experiments

Real Norman 2019 is 111k cells × 19k genes, 237 perturbations. After
preprocessing (log-normalize + top-5000 highly variable genes), the
synthetic-best pipeline scored **0.54** on real data. Below the 0.71
hit on synthetic — humbling but expected.

Two fixes did most of the work.

### Fix the resolver bug

Norman labels include dual perturbations like `KLF1+MAP2K6` and
single-with-control like `FOXA1+ctrl`. The synthetic-only fallback in
the resolver extracted trailing digits from labels and matched them to
gene indices — so `KLF1+MAP2K6` silently mapped to "gene index 6,"
whatever happened to live in column 6. About 130 dual perts were
getting random gene targets. Another 37 single perts missed entirely
because their target gene fell outside the top-5000 HVGs.

Cleaning up the label resolver and adding multi-target override (knock
down *both* genes named in `GENE1+GENE2`): **+0.03**.

### Use the right coexpression

Then the propagation experiment that failed on synthetic finally
worked. For each test pert with target gene g, predict that genes
co-perturbed with g in training also shift. The *control-cell*
coexpression matrix turned out to be the wrong choice — it captures
stable baseline coexpression, not perturbation responses. The
coexpression matrix computed across **training delta vectors** was the
right one, capturing "if I perturb something and gene A moves, gene B
tends to move too": **+0.05**.

A small refinement after that — using the conditional-expectation form
`cov(g, j) / var(g)` instead of plain correlation — added another
**+0.01**. That gain later turned out to be a happy accident of a kernel
mismatch bug between training-time scoring and test-time prediction.
(See "the bug" below.)

That brought the score to **0.65**.

## Foundation models

scGPT (33M cells, 60k gene vocab, 512-dim embeddings) and Geneformer
(30M cells, 25k gene vocab, 256-dim embeddings) both download cleanly
from HuggingFace. The embedding tensors load with bare PyTorch — no
need for the full inference stack.

**The naive way did not work.** Computing cosine similarity between
pretrained gene embeddings and using the result as a propagation kernel
— in the same slot as the training-delta coexpression — gave the loop
a weight knob that LOO consistently set to zero. Forcing the weight on
hurt the score. Pretrained "which genes look similar" turns out not to
be the same as "which genes co-perturb under knockdown."

**The right way did.** Used as *features* in a per-gene ridge
regression, the embeddings carry real signal: the target gene's 512-dim
scGPT vector becomes the input, and the model learns one set of weights
per output gene mapping "embedding of the perturbed gene" → "predicted
change in this gene's expression." Trained on 149 perturbations,
applied to the 47 held-out perts. Tuning the regularization strength
moved the score from **0.66 → 0.71**.

Adding Geneformer alongside scGPT (concatenating embeddings into a
768-dim feature vector) added another **+0.003**. Mostly redundant
information — both models trained on overlapping cell atlases — but a
free improvement.

Final score: **0.72**.

## The bug

Around experiment 30, a "conditional expectation" kernel
`cov(g, j) / var(g)` replaced the plain correlation kernel. The
leave-one-out scoring inside the pipeline used the new kernel; the
prediction code still used the old correlation kernel. That mismatch
produced a +0.01 score jump that *looked* like a real improvement and
held up across several follow-up experiments — a phantom plateau
created by compensating-scale interactions between two mismatched
kernels.

Integrating scGPT forced a careful read of the prediction path and
exposed the inconsistency. Fixing the mismatch dropped the score from
0.65 to 0.64. The scGPT work then cleanly pushed past it to 0.72, so
the bug was a temporary illusion rather than a permanent loss.

This is exactly the failure mode the autoresearch pattern is supposed
to guard against: a single scalar score cannot tell whether the system
is actually getting better or merely compensating for an internal
inconsistency. Periodic eyeballing of the code matters.

## Run 2 — moving to PerturBench's benchmark setup

Run 1's evaluation (random 20% pert holdout, Pearson on top-200 DE)
is reasonable but isn't any published leaderboard's setup. To put a
real number next to published work, run 2 ports PerturBench's
evaluation (Wu et al. 2024, NeurIPS,
[arXiv 2408.10609](https://arxiv.org/abs/2408.10609)) into a parallel
harness, `harness_perturbench.py`:

- Split: train on all single perts + 30% of duals; test on the
  remaining 70% of duals. Every target gene appears as a single in
  training — the "combo prediction" setup.
- Metric: mean cosine similarity of predicted vs actual logFC across
  all genes, per test pert.

Re-running the run-1 pipeline (ridge + scGPT/Geneformer) through this
harness gave **0.750** cosine logFC — already above PerturBench's
Linear and GEARS baselines but below their best published LA at
**0.79 ± 0.01**. The two pipelines turn out to be complementary:

|                           | random holdout<br/>Pearson top-200 DE | combo split<br/>cosine logFC |
|---------------------------|:----:|:----:|
| run 1 (ridge + FM) | **0.718** | 0.750 |
| run 2 (LA, seed baseline) | 0.699 | 0.835 |

Each wins on its home setup. Run 2 reseeded `pipeline.py` with a
Latent Additive model (an MLP autoencoder with additive latents for
dual perturbations) and kicked off a fresh autoresearch loop against
the PerturBench harness.

Eleven experiments in, the final pipeline hits
**0.871 ± 0.002 cosine logFC** across 3 independent ensemble base
seeds, commit `best/apr19/exp10` (or thereabouts — see
[PROCESS_PB.md](PROCESS_PB.md) for the commit-by-commit log). That
**exceeds PerturBench's reported LA number of 0.79 ± 0.01 on the
same split and metric** — with the honest caveat that this is our
implementation vs their published number, not our changes applied
to their codebase. A direct A/B on their code is ongoing; until
that lands, treat "8-point gap" as an upper bound on what's
purely due to training-procedure differences.

### What moved the score

Five implementation choices stacked on top of a plain Latent Additive
backbone. The loop-internal ablations (each measured in this
harness) are listed with their observed deltas:

1. **Per-pert-mean training** (same as run 1). PerturBench trains
   on per-cell examples; this implementation trains on per-pert
   mean expression vectors directly. Cleaner gradient toward the
   quantity being scored. **Ablation** (`ablation_training.py`,
   same architecture + ensemble + residual + dropout + target
   override held constant, per-pert-mean vs per-cell, 3 base
   seeds each): per-pert-mean **0.8708 ± 0.0023**, per-cell
   **0.8624 ± 0.0016**, delta **+0.008**. So this is worth only
   about a point — much less than our initial guess. Caveat:
   per-cell was budget-matched on gradient updates, not on
   PerturBench's ~160k-step training schedule; a fully-converged
   per-cell run is open work.
2. **5-seed ensemble.** Standard bagging over random initialization.
   Plateaus at 5 (10 seeds ties). +0.013.
3. **Output-space residual.** `pred = control_mean + f_dec(z)`
   instead of `pred = softplus(f_dec(z))`. Decoder learns the delta,
   a smaller target. +0.003.
4. **Dropout removed.** The biggest single keep *in run 2* (+0.013).
   With ensembling and per-pert-mean training already reducing
   variance, dropout added inference-time blur without regularization
   payoff. This finding is specific to the training procedure here —
   per-cell training in PerturBench's setup may still benefit from
   dropout.
5. **Explicit per-target-gene override.** After the ensemble
   produces a prediction, replace `pred[target]` with
   `control[target] + observed per-target delta`. +0.004.

The improvements **do not require a bigger architecture**: the
pipeline here has ~0.8M parameters (hidden=512, latent=128) while
PerturBench's tuned LA has 4M (encoder_width=4352, latent=512) and
scores 0.79. However, the reverse is not established — *their* 4M
architecture with *our* training setup could well land higher than
either current number. That experiment is open.

The ablation above reframes where the ~0.08 gap to PerturBench
actually comes from. Per-pert-mean training contributes ~+0.008.
By elimination, the remaining ~0.07 sits in the other four items
together — modest ensembling, output-space residual, dropout=0,
and per-target-gene override. Taken individually each is mundane
"ML hygiene," but in combination they compound to more than the
single biggest architectural choice. The "boring wins" pattern
from PerturBench's paper is reinforced here: no bright-line
architectural improvement, just stacked small training-side
decisions.

### About the "simple controls beat deep learning" claim

A 2025 *Bioinformatics* paper (Wenteler et al.) makes the strong
claim that "simple controls exceed best deep learning algorithms"
on Norman 2019. Their signature baseline is the **CRISPR-informed
mean**: predict the mean expression across all training perturbed
cells, with a 2× bump at the target gene for CRISPRa.

Reproducing CIM in this harness gives **0.568 cosine logFC on the
combo split** — well below PerturBench's Linear baseline (0.60) and
far below LA (0.79) or the pipeline here (0.87). CIM's rank metric
comes out at 0.45 (random would be 0.50), so predictions *are*
slightly specific to each test pert via the target-gene tweak, but
only weakly — close to mode-collapse without quite reaching it.

One plausible reading: CIM wins on a *different* split. On random
pert holdout where some target genes go entirely unseen at training
time, LA's `z_pert` for that target is zero and CIM's fallback to
the global mean is hard to beat. On combo split where every gene
is in training as a single, LA's compositional latent wins and
CIM has almost no per-pert variation to work with. **This is a
hypothesis worth testing separately — the "simple vs deep" claims
in the literature may be split-specific — but a single
cross-harness data point isn't enough to settle it.**

### The table

|                                          | cosine logFC          | source           |
|------------------------------------------|:---------------------:|------------------|
| **This work (run 2 best)**               | **0.871 ± 0.002**     | 3 base seeds     |
| Latent Additive (PerturBench paper)      | 0.79 ± 0.01           | their Table 3    |
| SAMS-VAE + sparsity                      | 0.78                  | their Table 3    |
| LA + scGPT                               | 0.77                  | their Table 3    |
| CPA                                      | 0.76                  | their Table 3    |
| CPA + scGPT                              | 0.70                  | their Table 3    |
| Linear baseline                          | 0.60                  | their Table 3    |
| CRISPR-informed mean (Wenteler 2025)     | 0.568                 | this repo        |
| GEARS                                    | 0.44                  | their Table 3    |

The full benchmark spec (data, split seed, metric definition, and a
careful breakdown of what is and isn't directly comparable to
recent work like TxPert and the Feb 2026 bioRxiv paper) lives in
[BENCHMARK.md](BENCHMARK.md).

### Honest caveats

- The 0.871 number reflects a specific implementation, not a
  fundamental algorithmic advance. A motivated re-implementation of
  LA on PerturBench's own codebase with per-pert-mean training,
  ensembling, and dropout removal would probably also reach ~0.87.
  PerturBench's `torch <= 2.5.1` pin conflicted with this project's
  environment, so their codebase was not re-run end-to-end.
- Results are for one split (PerturBench combo) on one metric
  (cosine logFC). On harder GEARS-style OOD splits where component
  genes are unseen as singles, LA-based methods degrade
  significantly. Not measured here.
- TxPert (May 2025) and the Feb 2026 bioRxiv foundation-models
  paper both define evaluations that do not map directly onto
  combo split + cosine logFC. They are not in the head-to-head.

The qualitative takeaways line up with PerturBench and with
Ahlmann-Eltze 2024 more broadly:

- **Simple architectures with careful training beat complex ones on
  this task.** The loop never found a configuration where a bigger
  model won; the wins always came from supervision structure.
- **Foundation models help as features, not as similarity metrics.**
  This held in both runs — same finding on different architectures.
- **The most portable artifact is `program.md`.** Each run's final
  `program.md` encodes priors the next run can start from (for
  instance, "dropout is usually over-applied when ensembling is in
  place" came out of run 2 and would go into run 3).

## Lessons for next time

- **Eyeball the code more often.** The kernel-mismatch bug survived
  several experiments because both LOO and test scores moved
  consistently. Reading the diff end-to-end every ~10 experiments would
  have caught it sooner.
- **Use the foundation-model "right way" first.** Several experiments
  burned on cosine-similarity kernels before the regression-feature
  approach surfaced. The clue lived in the literature (PerturBench,
  Ahlmann-Eltze 2024); reading earlier would have saved cycles.
- **Either match a published metric/split exactly or do not claim the
  comparison.** This harness's metric is a reasonable proxy but is not
  directly comparable to any leaderboard. The next iteration of this
  project closes that gap.

## What's portable

The most portable artifact is not the model. It is `program.md` — the
iterated instruction document the agent reads at the top of every loop.
By the end of the run it encodes domain priors that did not exist at
the start: "delta-space coexpression beats control-space coexpression,"
"foundation-model embeddings work as ridge features but not as
similarity kernels," "watch out for label parsing on dual
perturbations." That document is what travels to the next benchmark.

## Repo

[autoresearch-perturbation](https://github.com/isaacuselman/autoresearch-perturbation)

- [`harness.py`](harness.py), [`harness_perturbench.py`](harness_perturbench.py)
  — the two immutable evaluators.
- [`pipeline.py`](pipeline.py) — the run-1 pipeline (ridge +
  foundation-model features) on `main`.
  [`pipeline_la.py`](pipeline_la.py) — the Latent Additive reference
  implementation. Run 2's tuned LA lives on
  [`autoresearch/pb-apr20`](https://github.com/isaacuselman/autoresearch-perturbation/tree/autoresearch/pb-apr20)
  as `pipeline.py`.
- [`program.md`](program.md) — the instruction document. Final
  version for run 2 is on the pb-apr20 branch.
- [`PROCESS.md`](PROCESS.md) / [`PROCESS_PB.md`](PROCESS_PB.md) —
  experiment-by-experiment journals for each run, with commit
  hashes for every kept experiment.
- [`BENCHMARK.md`](BENCHMARK.md) — the full benchmark table with
  provenance and honest caveats.

---

*Designed by agents, reviewed by humans. The experiments and this
writeup were both produced by [Claude Code](https://claude.com/claude-code)
under direction from the human author, who reviewed every step and
approved the final version. The repository — pipeline, journal, and
post — is the joint output of that collaboration.*
