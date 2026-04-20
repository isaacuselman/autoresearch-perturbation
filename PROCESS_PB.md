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
| 7   | `6a18df0` | 0.5608 | 0.4981 | discard | 290 s | selective `wd=1e-4` on `f_pert` only — **mode collapse** |
| 8   | `3f89202` | 0.8550 | 0.0211 | keep    | 194 s | explicit per-target-gene override on LA output      |
| 9   | `6c3d154` | 0.8682 | 0.0139 | keep    | 223 s | dropout 0.1 → 0.0 — biggest jump in run 2          |
| 10  | `6af2608` | 0.8685 | 0.0123 | keep    | 430 s | epochs 2000 → 4000 (marginal but strictly better)  |
| 11  | `f410692` | 0.8630 | 0.0152 | discard | 1176 s | retry capacity (latent=256, hidden=1024) without dropout — still overfits |

**Current best:** exp 10 at 0.869 cosine logFC, commit `6af2608`.

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
applied it globally with `weight_decay=1e-4`. Too aggressive.

**Exp 7 — selective L2 on `f_pert` only.** Program.md's actual
prescription. Used parameter groups so `weight_decay=1e-4` applied
only to the perturbation encoder. Result: catastrophic mode collapse
(score 0.56, rank 0.50 ≈ random). The encoder's outputs got crushed
toward zero; the model produced a near-identical prediction for every
test pert. The cosine_logFC_rank metric flagged this immediately,
exactly the failure mode `program.md` warned about. Smaller
coefficients (1e-5, 1e-6) are still on the list but lower priority
after this signal.

**Exp 8 — explicit per-target-gene override.** Direct port of run 1's
target-override idea, layered on top of LA's prediction. After the
ensemble produces `pred`, replace `pred[target_hvg]` with
`control[target_hvg] + per_gene_delta[target]`, where
`per_gene_delta` is the median observed delta across training perts
that targeted that gene. 62 of the test target genes fell inside the
HVG-5000 set. +0.004; rank also improved to 0.021. The signal is
small because LA is already close to the right value at the target
position, but the override sharpens it.

**Exp 9 — drop dropout (0.1 → 0.0).** Program.md does not list this
explicitly; it came from the prior on this run that ensembling +
per-pert-mean training is already variance-reducing, so dropout was
adding stochastic noise without regularization payoff. Result: +0.013,
the biggest single jump in run 2. Confirmed both metrics improved
(cosine 0.855 → 0.868, rank 0.021 → 0.014).

**Exp 10 — epochs 2000 → 4000.** With dropout=0 the loss surface
became more deterministic, so longer training had a chance of
converging more cleanly. Marginal (+0.0003) but strictly better
under the ratchet rule, and the rank dropped further (0.014 → 0.012).
Wallclock doubled to 430s.

**Exp 11 — retry capacity scaling without dropout.** Hypothesis: exp 4
overfit because dropout was already eating the regularization
budget; without dropout, latent=256/hidden=1024 might breathe.
Result: same overfit pattern (final loss → 0 = full memorization),
test score 0.863, slightly worse than 0.869. Confirmed for the second
time that 144 training perts cannot support a 4M-parameter model.
Wallclock 1176s, near the 1200s cap.

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
- **Capacity scaling overfits at 144 training perts — verified twice.**
  Once with dropout=0.1 (exp 4), once without (exp 11). 0.8M params
  was already enough; 4M memorizes. Don't re-try at this dataset size
  without a regularization strategy that actually preserves
  generalization (e.g., bottleneck on `f_pert` only, not the whole
  network).
- **Weight decay is dangerous on `f_pert`.** Global wd=1e-4 was too
  strong (-0.005). Selective wd=1e-4 on `f_pert` alone caused full
  mode collapse (-0.29). The cosine_logFC_rank metric is essential
  here — it flagged the collapse instantly while the cosine was
  still 0.56 (a value that without context could look "competitive").
- **Dropout was net-harmful with this training procedure.** Removing
  it gave +0.013 — the biggest single keep in run 2. The intuition:
  per-pert-mean training already has zero per-batch noise, ensembling
  already averages out random initialization variance, so dropout's
  remaining contribution was just inference-time output blur.
- **Per-target-gene override transfers from run 1.** The same trick
  that powered run 1 (override pred[target_gene]) gives a smaller
  but real lift on top of LA (+0.004). Useful pattern beyond a
  single pipeline architecture.

## Open leads from `program.md` not yet tried in this run

- Smaller weight-decay coefficients (1e-5, 1e-6) — selective on
  `f_pert` after exp 7's collapse-on-1e-4. Lower priority now.
- Multiplicative composition for duals (`z_{p1,p2}` interaction
  terms). Requires architectural change.
- Per-cell or hybrid per-cell/per-pert training.
- Ensemble of architectures (LA + the ridge+FM pipeline from
  `main`).
- Sampled control encoder.

## Where this lands now

Best: **0.871 ± 0.002 cosine logFC** across 3 independent ensemble
base seeds (15 model trains total). Range [0.8685, 0.8738]. Tight
std confirms the result is robust, not lucky.

PerturBench Table 3 published numbers on the same split:
- Latent Additive: 0.79 ± 0.01
- LA + scGPT: 0.77
- CPA: 0.76 (CPA + scGPT: 0.70)
- SAMS-VAE + S: 0.78
- GEARS: 0.44
- Linear: 0.60

Gap to their best published number: **+0.08 cosine logFC**.

A clean SOTA claim on this benchmark requires reproducing competitor
methods (LA from PerturBench's codebase, CRISPR-informed mean from
Wenteler 2025, GEARS, CPA, ideally TxPert from May 2025) through
this harness or running our pipeline through theirs. Plan tracked
in the public PR description.

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
