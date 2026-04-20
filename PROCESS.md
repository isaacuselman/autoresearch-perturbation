# Process journal

A running record of the autoresearch session, organized chronologically
so a reader can trace the decisions and dead ends alongside the final
score. All commit hashes and tag references are real and navigable via
`git show <hash>` or `git log`.

## Run tag

`autoresearch/apr19` — kicked off 2026-04-19.

## Initial data setup

The `pertpy` dependency group was not installed at kickoff (it is an
optional extra), so `prepare_data.py` fell back to the synthetic
generator at `~/.cache/autoresearch-perturbation/synthetic.h5ad` (shape
(2200, 1000), 41 perts, 8 held out). The synthetic fallback produces a
validation-only dataset: it exercises the loop end to end but scores on
it have no relation to Norman 2019 baselines until the real data
arrives.

## Setup caveats

- The working directory was not a git repository at kickoff. The
  starter files were `git init`-ed and committed on `main` before
  branching to `autoresearch/apr19`. A cleaner setup would have
  initialized earlier.
- macOS does not ship `timeout(1)`. The loop uses
  `perl -e 'alarm shift; exec @ARGV'` as a portable wallclock cap. If
  `gtimeout` is preferred, installing `coreutils` via Homebrew enables
  it.

## Baseline

score: 0.270669 | coverage: 1.0000 | wallclock: 0.8 s | commit before
any pipeline edits.

## Synthetic-data ceiling observation (after exp 5)

Synthetic downstream effects are i.i.d. random per perturbation,
meaning they are unpredictable in principle. The achievable Pearson
reduces to matching the target-gene knockdown plus weak structure in
the median training delta. A back-of-envelope ceiling lands at ~0.69;
exp 4 reached 0.688.

One consequence: experiments that depend on real coexpression
structure (e.g., exp 5: control-cell coexpression propagation) *hurt*
on synthetic but should *help* on real data. When the real Norman 2019
swap lands, re-running exp 5 (commit `606cdb2`) is worth it — the
design generalizes, and the synthetic-time discard was a measurement
artifact of the generator, not a failure of the idea.

## Plateau confirmation (after exp 18)

By exp 18 the synthetic score settled at 0.704 from exp 15 (LOO-CV
alpha tuning). Three subsequent experiments (16: joint alpha+trim
sweep; 17: 1.5x LOO bias correction; 18: nearest-pert residual in
target-baseline space) all tied or regressed. The plateau is real on
synthetic.

The best-synthetic pipeline (commit `c9bb825`) contains:

- target-gene drop override using the median of training drops
- non-target = trimmed-mean (10%) of training deltas, scaled by
  LOO-tuned alpha
- LOO picks alpha=2 with internal Pearson 0.67

Two discarded experiments stand out as worth resurrecting on real
data:

- exp 5 (commit `606cdb2`): control-cell coexpression propagation.
  Should help when controls have biological coexpression structure
  (real data does; synthetic does not).
- exp 18 (commit `e6511a9`): nearest-training-pert residual in
  target-baseline space. LOO picked epsilon=0 on synthetic; real data
  may pick epsilon > 0 if cell-state matters.

## Pivot to real Norman 2019 (after exp 21)

Approved by user. The `pertpy` install hit a `jax`/`numpyro`/`flax`
version conflict in the current venv (jax 0.10 vs numpyro 0.20 →
`xla_pmap_p` import error; downgrading numpyro shifted the error to
flax). The fix bypassed pertpy by curl-ing the same h5ad pertpy uses
internally:

  https://exampledata.scverse.org/pertpy/norman_2019.h5ad  (1.7 GB)

- raw: `~/.cache/autoresearch-perturbation/norman_2019.raw.h5ad`
- preprocessed (normalize_total / log1p / HVG-5000) by
  `load_real_data.py`
- saved: `~/.cache/autoresearch-perturbation/norman_2019.h5ad`

`load_real_data.py` is a one-time helper outside the program.md
"immutable" list. If pertpy's deps get unpinned upstream, it can go
away and `prepare_data.py` takes over.

Real data shape: (111,255 cells × 5,000 HVGs), 237 perturbations (190
train, 47 held-out). Perturbation labels are gene names — including
dual perturbations like `FOXA1+CEBPE` and single+ctrl like
`FOXA1+ctrl`. Control label is already canonical.

**Real-data baseline (tag `real-baseline/apr19`):** score 0.5407,
coverage 1.0, wallclock 13.3 s. LOO picked alpha=1 on real (alpha=2 on
synthetic). The synthetic-best pipeline transfers reasonably but loses
~0.17 of score. The loop continues from here.

## scGPT integration (after exp 43)

Downloaded `wanglab/scGPT-human` from HuggingFace (pretrained on
cellxgene). Extracted the (60697, 512) gene-token embedding layer via
bare PyTorch — no need for the full `scgpt` package and no
`flash_attn` dependency. Saved the per-HVG slice to
`scgpt_hvg_emb.npy` (4556/5000 covered). Helper:
`precompute_scgpt_emb.py`.

Two integrations were attempted:

- **Cosine-similarity kernel** (analogous to delta_corr propagation):
  LOO consistently picked weight=0; forcing it > 0 hurt the score.
  Pretrained similarity is not the same signal as perturbation-response
  similarity.
- **Per-gene ridge regression** with the target's scGPT embedding as a
  512-dim feature: substantial signal. LOO-tuned `eta` was always > 0;
  loosening ridge regularization unlocked more gain (lambda 1.0·n →
  0.001·n moved the score 0.686 → 0.712).

The scGPT integration also uncovered a kernel-mismatch bug from exp
30: LOO was scoring `_beta_kernel` (`cov/var(g)`) while predict() was
applying the weight to `_delta_corr` (`cov/std(g)/std(j)`). The
reported 0.652 was a happy accident from compensating scales. The
honest pipeline at exp 34 scored 0.639. The scGPT-ridge work pushed
cleanly past that to 0.715.

## Geneformer add-on (after exp 44)

Same pattern as scGPT: pulled the BERT-style Geneformer V1-10M (256-dim
embeddings) from HuggingFace, mapped HVG symbols → Ensembl IDs → token
IDs (3557/5000 hit), concatenated with scGPT into a 768-dim ridge
feature vector. Gain: +0.003 (0.715 → 0.718). Mostly redundant with
scGPT — both models trained on overlapping cell atlases — but a real,
free improvement. Helper: `precompute_geneformer_emb.py`.

## Real-data progression

Best so far: **0.7177** (commit `best/apr19/exp44`).

| exp | score | what changed                                                  |
|-----|-------|---------------------------------------------------------------|
| baseline | 0.541 | synthetic-best pipeline on real data                    |
| 22  | 0.574 | dropped synthetic regex hack; multi-target override           |
| 23  | 0.586 | control-cell coexpression propagation (beta=1)                |
| 24  | 0.588 | extended LOO beta range, picked beta=1.5                      |
| 26  | 0.639 | **delta-space corr propagation** (LOO chose beta=0, gamma=1)  |
| 30  | (0.652) | conditional-expectation kernel — **bug**: LOO/predict mismatch |
| 34  | 0.639 | bug-fix: consistent kernels across LOO and predict             |
| 36  | 0.663 | **scGPT-as-features ridge regression**                        |
| 37-39 | 0.676–0.686 | extended eta sweep                                  |
| 40-42 | 0.697–0.712 | reduced ridge lambda                                |
| **43** | **0.715** | extended eta range, picked eta=5                      |
| 44  | 0.718 | combined scGPT + Geneformer ridge features (768-dim)          |

Discarded but informative experiments on real data:

- per-gene drop magnitude (overfit on LOO, exp 25)
- sparsified delta_corr (top-50, exp 28, hurt)
- squared correlation (exp 29, lost signal)
- ridge regularization on the kernel (exp 32-33, over-shrank)

Two decisive unlocks on real data: dropping the synthetic-only regex
resolver (it was silently matching dual perts to random gene indices)
and moving propagation from control-space to delta-space coexpression.
The conditional-expectation kernel `cov[g,j]/var[g]` cleanly extends
delta-space correlation by accounting for per-target-gene variance.

The resolver still misses 37/189 train perts (genes not in HVG-5000).
Re-running preprocessing with a different HVG selection (or no HVG
cap) might give more lift, at the cost of a much larger
gene_corr/beta_kernel matrix. Adding gene-symbol aliases is another
path.

## Where this lands

Best: **0.7177** on the held-out 47 perts, coverage 1.0, wallclock
~140 s per harness run. For calibration against published work, see
the "Where this lands relative to published work" section in
[`POST.md`](POST.md).

## Next steps after this journal closes

1. Port PerturBench's evaluation into this harness (cosine-logFC
   scorer + combo-prediction split + Latent Additive baseline) for an
   honest head-to-head.
2. Re-run the autoresearch loop against the ported setup.
3. Decide whether GPU + full scGPT/Geneformer inference is worth the
   setup cost, based on where (2) plateaus.
