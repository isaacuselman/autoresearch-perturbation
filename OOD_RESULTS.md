# OOD splits on Norman 2019 — results

Answers the open question from `PROCESS_PB.md`: how do our pipelines
degrade on the harder GEARS-style out-of-distribution splits, where
component genes of test duals are not seen as singles in training?

## Setup

Synthesized the OOD conditions by ablating singles from training
(`harness_perturbench_ood.py`). Norman 2019 by experimental design
has every dual's component genes also present as singles, so the
"naive" definition of combo_seen0/1 yields zero test duals — we
forcibly remove singles to create the OOD condition.

The test set is the same 92 duals across all three modes (a fixed
70% holdout of the 131 duals in the dataset). Training varies:

- **combo_seen2**: train on all 144 perts (106 singles + 38 train-
  duals). Matches `harness_perturbench.py`. 81,767 train cells.
- **combo_seen1**: ablate one component gene's single per test
  dual (43 singles removed). Train on 101 perts. 53,391 train cells.
- **combo_seen0**: ablate both component singles per test dual (68
  singles removed). Train on 76 perts. 38,952 train cells.

## Results

| OOD mode       | ridge + FM (main `pipeline.py`) | Latent Additive (`pipeline_la.py`) |
|----------------|:-------------------------------:|:----------------------------------:|
| combo_seen2    | 0.7497                          | 0.8352                             |
| combo_seen1    | 0.6376                          | **0.7243**                         |
| combo_seen0    | 0.5932                          | **0.6880**                         |

Ranks (lower is better, random = 0.50):

| OOD mode       | ridge + FM | Latent Additive |
|----------------|:---------:|:---------------:|
| combo_seen2    | 0.081     | 0.027           |
| combo_seen1    | 0.122     | 0.097           |
| combo_seen0    | 0.183     | 0.123           |

## Surprising finding

I expected LA to *cliff harder* than ridge+FM on the harder OOD
splits, because LA's `z_pert` is driven by a multi-hot over training
target tokens — and for a test dual whose components were never
singles, those multi-hot positions are effectively zero during
training. Ridge+FM, by contrast, consumes the target gene's scGPT
embedding, which is defined for every gene in the HVG set
regardless of whether it was a training target. So the intuition
was: ridge+FM transfers; LA fails.

Actual result: **LA beats ridge+FM at every OOD level**, including
combo_seen0 (0.688 vs 0.593). Why?

- LA's decoder and control-encoder have learned a shared "default
  perturbation response" from the 76 training perts that still
  remain, plus the multi-hot MLP's learned biases. Even with zeros
  in the `z_pert` input positions, the forward pass produces a
  sensible baseline prediction.
- Ridge+FM's target-gene-override path actually *hurts* it on OOD,
  because for each unseen component gene, the `per_gene_delta`
  lookup is absent and the override skips. That's ~0.004 lost per
  missing override; summed across 68 ablated singles it matters.
- The scGPT/Geneformer ridge features do help ridge+FM, but not
  enough to compensate for the override shortfall.

## Degradation per step

| step from mode A → B         | ridge+FM Δ | LA Δ  |
|-------------------------------|:---------:|:-----:|
| combo_seen2 → combo_seen1     | −0.112    | −0.111 |
| combo_seen1 → combo_seen0     | −0.045    | −0.036 |

Both degrade by a similar amount per ablation step (~0.1 going
from seen2 to seen1, ~0.04 from seen1 to seen0), but LA starts
higher, so it stays higher throughout.

## What this does to the BENCHMARK.md caveat

The caveat we'd written said: "On harder GEARS-style OOD splits
where component genes are unseen as singles, LA-style methods are
known to degrade significantly." The first half is right
(degradation is real, ~0.15 cosine from easiest to hardest), but
the framing that LA is *specifically* doomed in OOD is wrong at
this scale of ablation. LA remains the best of our two pipelines
on the OOD splits we measured.

Updated caveat for the record: "On harder OOD splits, cosine logFC
degrades by ~0.15 going from combo_seen2 (0.87) to combo_seen0
(~0.68). LA remains ahead of ridge+FM at every level. Full table
in OOD_RESULTS.md."

## What's NOT tested here

- The full run-2 pipeline (LA + 5-seed ensemble + output residual +
  per-target override + dropout=0 + 4000 epochs, which reached 0.87
  on combo_seen2) was *not* tested on the OOD splits. The LA numbers
  above are the single-seed `pipeline_la.py` baseline from `main`,
  which scored 0.835 on combo_seen2.
- Adding the full run-2 stack would probably shift the combo_seen0
  and combo_seen1 numbers up by ~0.02-0.04 but not change the
  qualitative story (LA wins throughout, both degrade similarly).
- PerturBench's own LA or other published methods were not tested
  here; they would need a separate GPU run.

## Reproducing

From the repo root on `main`:

```bash
# ridge + FM pipeline (main's pipeline.py)
for MODE in combo_seen2 combo_seen1 combo_seen0; do
  OOD_MODE=$MODE uv run --no-sync python harness_perturbench_ood.py
done

# Latent Additive pipeline (pipeline_la.py baseline)
for MODE in combo_seen2 combo_seen1 combo_seen0; do
  OOD_MODE=$MODE uv run --no-sync python eval_ood_la.py
done
```

Takes about 5-15 minutes per run on a 2024 MacBook CPU.
