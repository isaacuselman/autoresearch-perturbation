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

The score climbed from **0.27 to 0.72** across 44 experiments over a
couple of evenings. What follows is a tour of the experiments that
mattered, the bug that nearly went unnoticed, and where this lands
relative to published work.

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

## Where this lands relative to published work

PerturBench (NeurIPS 2024,
[arXiv 2408.10609](https://arxiv.org/abs/2408.10609)) evaluates
published perturbation-prediction methods on Norman 2019. Their best
result is a simple MLP autoencoder ("Latent Additive") at 0.79 *cosine
logFC* on a different split (combo prediction: train on all single
perts + 30% of duals, test on remaining duals).

Rather than leave that comparison at "different metric, different
split," this project ports the PerturBench evaluation into a parallel
harness (`harness_perturbench.py`) and re-implements Latent Additive
(`pipeline_la.py`) to produce a real 2×2 head-to-head.

The numbers:

|                         | random 20% pert holdout<br/>Pearson top-200 DE | combo split<br/>cosine logFC |
|-------------------------|:----:|:----:|
| ridge + scGPT/Geneformer (this work's pipeline) | **0.718** | 0.750 |
| Latent Additive (reproduction)                   | 0.699 | **0.825 ± 0.006**¹ |
| Latent Additive (PerturBench paper)              | —     | 0.79  ± 0.01 |

¹ Mean ± std over four random seeds.

Two observations stand out:

- **Each method wins on the split it was designed for.** LA's additive
  latent makes combo prediction almost free: every target gene has been
  seen as a single, and a dual pert is just the sum of two latent
  shifts. The ridge + foundation-model pipeline generalizes better to
  the random holdout, where some test perts have target genes never
  seen as singles — a case where LA's `z_pert` for that gene is zero.
- **The reproduction of LA lands ~3 points above the published number.**
  The one departure from the paper: this implementation trains on
  per-pert *mean* expression rather than per-cell examples. Directly
  optimizing the quantity the scorer measures gives a cleaner gradient
  signal from the ~150 training perts and appears to matter more than
  architectural tweaks.

The qualitative findings otherwise line up with PerturBench:

- **Simple methods are surprisingly hard to beat on this kind of task.**
  PerturBench's MLP baseline beats every other method they tested,
  including scGPT-augmented variants and a published GNN method
  (GEARS at 0.44 cosine logFC). The story here matched: gains came
  from getting the structure right (target-gene override, multi-target
  labels, delta-space coexpression), not from increasingly clever
  models.
- **Foundation models help when used as features, not as similarity
  metrics.** PerturBench shows scGPT *augmentation* of CPA and Latent
  Additive moves scores by a few percent — the same ballpark as the
  scGPT-ridge contribution here.
- **Holding out *perturbations* is much harder than holding out cells
  from seen perts.** Most published wins for foundation models are on
  the easier setting.

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
contains the pipeline, the harness, the experiment log, and the final
`program.md`. `PROCESS.md` has the full experiment-by-experiment
journal with commit hashes for every run.

---

*Designed by agents, reviewed by humans. The experiments and this
writeup were both produced by [Claude Code](https://claude.com/claude-code)
under direction from the human author, who reviewed every step and
approved the final version. The repository — pipeline, journal, and
post — is the joint output of that collaboration.*
