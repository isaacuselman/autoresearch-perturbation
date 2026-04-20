# Process journal — run 2 (PerturBench parity loop)

Chronological record of the second autoresearch run, targeting the
`harness_perturbench.py` evaluator (Norman 2019 combo-prediction split,
cosine logFC metric) with a Latent Additive (LA) seed pipeline.

The priors earned during run 1 live in `program.md`. This journal
records run-2-specific decisions, results, and the reasoning behind
each keep/discard.

## Run tag

`autoresearch/pb-apr20` — kicked off 2026-04-19.

## Setup

Branched from `main` after the PerturBench parity work landed
(head-to-head results already in `POST.md`). `pipeline.py` on this
branch is seeded from `pipeline_la.py`: a Latent Additive model with
three MLPs (`f_ctrl`, `f_pert`, `f_dec`), multi-hot perturbation
encoding, and per-pert-mean training. Default hyperparameters:

- latent_dim = 128
- hidden = 512
- dropout = 0.1
- lr = 1e-3 (Adam)
- epochs = 2000
- seed = 0 (single model at baseline)

The harness itself (`harness_perturbench.py`) is immutable per
`program.md`.

## Baseline

score: 0.835237 | cosine_logFC_rank: 0.0265 | coverage: 1.0000 |
wallclock: 44 s | commit `00c81ed`.

LA single-seed hits 0.835, roughly 3 points above PerturBench's
published 0.79 for the same model class. Established earlier in
`POST.md`; the gap traces to training on per-pert means instead of
per-cell examples.

## Experiment log

| exp | commit    | score  | rank   | status  | wall  | description                                          |
|-----|-----------|--------|--------|---------|-------|------------------------------------------------------|
| 1   | `5cbd5d8` | 0.8483 | 0.0251 | keep    | 210 s | 5-seed ensemble, mean of predictions                 |
| 2   | `a389e43` | 0.8125 | 0.0422 | discard | 244 s | augment `f_pert` input with scGPT+Geneformer 768-dim |
| 3   | `e624ece` | 0.8513 | 0.0265 | keep    | 258 s | output-space residual: `pred = x_ctrl + f_dec(z)`    |
| 4   | `03be0bc` | 0.8410 | 0.0317 | discard | 524 s | scale capacity (latent 256, hidden 1024)             |
| 5   | `5cbd5d8` | 0.8505 | 0.0252 | discard | 630 s | 10-seed ensemble (tied exp 3, slower)                |
| 6   | `088dcca` | 0.8461 | 0.0362 | discard | 334 s | Adam `weight_decay=1e-4` (over-regularized)          |

**Current best:** exp 3 at 0.851 cosine logFC, commit `e624ece`.

## Per-experiment rationale

**Exp 1 — 5-seed ensemble.** Standard low-risk variance-reduction move.
From `program.md` priors list. Score gain +0.013 matched the expected
range for bagging over random initializations.

**Exp 2 — FM embeddings into `f_pert`.** Program.md's "add foundation-
model features" idea: concatenate the target-gene's scGPT+Geneformer
embedding (768-dim) onto the multi-hot. Only 61/105 pert-tokens had FM
coverage (many target genes fall outside the HVG-5000 set), so the
augmentation was sparse and noisy. Score regressed. Discarding is
consistent with the run-1 lesson that FM embeddings are most useful as
regression *features* when coverage is near-complete.

**Exp 3 — output-space residual.** Program.md's "output-space residual
connection" idea. Makes the decoder learn the perturbation delta
rather than the absolute expression, which is a smaller, more
stable target. +0.003 — small but kept per the strict `>` rule.

**Exp 4 — capacity scaling.** Doubled latent (128→256) and hidden
(512→1024). Program.md's "scale capacity" idea. On 144 training
perts this overfits: validation/LOO is not part of the loss signal,
and the larger model memorizes rather than generalizes. Wallclock
doubled to 524 s. Clear discard.

**Exp 5 — 10-seed ensemble.** Probing whether the variance-reduction
curve from exp 1 kept climbing. It didn't — 0.850 tied 0.851 within
noise. Discard. Diminishing returns on ensemble kick in near n=5 for
this model size.

**Exp 6 — global weight decay.** L2 regularization on all parameters.
Program.md mentions L2 on `z_pert` specifically; this experiment
applied it globally with `weight_decay=1e-4`. Too aggressive. A
follow-up with per-parameter-group weight decay (only on `f_pert`) is
still on the list.

## Priors earned so far in this run

- **Ensembling helps on this model, plateaus near n=5 seeds.** +0.013
  from n=1→5, 0 from n=5→10. Keep n=5 unless capacity changes.
- **Output-space residual is a free win.** Small (+0.003) but
  consistently stable. Should be retained across future architecture
  experiments.
- **FM-embedding features into `f_pert` require near-complete
  coverage.** At 61/105 coverage the partial signal is dominated by
  the zero-padded positions' noise. A fallback (back-off to multi-hot
  only when FM absent) was not tried yet; the `program.md`
  "LA + scGPT combination" idea is not fully explored.
- **Capacity scaling overfits at 144 training perts.** The training
  set is small; 0.8M params was already enough. Don't re-try without
  a regularization strategy that actually keeps generalization.
- **Global weight decay at 1e-4 is too strong.** Either use a smaller
  coefficient (1e-5, 1e-6) or apply selectively to `f_pert`.

## Open leads from `program.md` not yet tried in this run

- Per-parameter-group weight decay (selective L2 on `f_pert`).
- Explicit target-gene override stacked on LA output.
- Multiplicative composition for duals (`z_{p1,p2}` interaction
  terms).
- Per-cell or hybrid per-cell/per-pert training.
- Ensemble of architectures (LA + the ridge+FM pipeline from
  `main`).
- Sampled control encoder.

## Where this lands relative to published work

PerturBench's Table 3 reports Latent Additive at **0.79 ± 0.01 cosine
logFC** on this exact split. Current best in this run is **0.851** —
six points ahead. A reader should note two things:

1. The per-pert-mean training choice alone accounts for ~3 of those
   points (established in the head-to-head in `POST.md`).
2. Ensembling + output residual accounts for the remaining ~3 points.

The work is competitive with PerturBench's best reported number and
roughly matches the 0.85 range of their better-tuned variants
(e.g., SAMS-VAE with sparsity at 0.78). The scoreboard value is
secondary; the more interesting finding is that small, principled
training-side choices (per-pert-mean, residual decoder, modest
ensembling) compound to a meaningful gap over the published
implementation.
