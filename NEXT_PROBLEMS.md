# Next problems to point the autoresearch loop at

Shortlist of adjacent problems that score well on the four criteria
that matter for this pattern:

1. **Autoresearch-friendly**: clean scalar evaluator, fast iteration
   (single experiment fits in 20 min), fixed held-out split, clear
   published baselines.
2. **Valuable**: real scientific or industry signal, not a toy.
3. **Adjacent**: close to the skills + code we already have
   (perturbation prediction, single-cell data, scRNA-seq preprocessing,
   simple MLPs and ridge regressions).
4. **Visible**: the result lands somewhere readable (leaderboard,
   benchmark paper, research community attention) so the artifact is
   worth producing.

## Virtual Cell Challenge: how to actually sign up

The 2025 round ran June-October 2025 at
[virtualcellchallenge.org](https://virtualcellchallenge.org/). The
2026 round timing isn't announced yet. Two ways to be notified:

1. **Watch the official site** at
   [virtualcellchallenge.org](https://virtualcellchallenge.org/) for
   the 2026 round announcement. Registration was a simple web form
   on that site last round.
2. **Sign up for Arc Institute updates** at
   [arcinstitute.org](https://arcinstitute.org/) (email subscribe
   at the bottom of most of their news pages). Their team posts on
   X as [@arcinstitute](https://x.com/arcinstitute); the 2025 round
   was announced there first.

Last round:
- $100,000 grand prize (plus other prizes totaling $175K)
- Sponsored by NVIDIA, 10x Genomics, Ultima Genomics
- Eligibility: individuals, academic teams, biotech, independent
  research groups
- Data: ~300 CRISPRi perturbations in H1 hESC; contestants also
  have access to the Arc Virtual Cell Atlas (~500M cells of
  additional training data)
- Metrics: MAE, DES (differential expression score), PDS
  (perturbation discrimination score)
- Submission: predicted expression profiles plus a set of control
  cells, submitted through their portal

If the 2026 round keeps roughly the same shape, our existing
`pipeline_la.py` transfers with two changes:
- Flip the target-gene override direction (CRISPRi represses; our
  current code assumes CRISPRa overexpresses).
- Wrap predictions into their submission format (their evaluator
  computes DE from the submitted controls, not from a held-out
  train set).

That's probably a one-week project once the 2026 data drops,
assuming no GPU requirements beyond what's already in this repo.

## The shortlist

| problem | autoresearch-friendly | visibility | effort | why valuable |
|---|---|---|---|---|
| **Virtual Cell Challenge 2026** (Arc Institute) | Very. Live leaderboard, fixed metrics (MAE, DES, PDS), pinned held-out data | **Highest**. Nature, NYT, podcast coverage of the 2025 round | Low once the round opens. Main change: CRISPRa to CRISPRi (flip target-gene override direction) | Flagship ML-for-bio event; submissions are scored against a new unpublished dataset so leaderboard rank is externally credible |
| **Open Problems: perturbation prediction (chemical compounds, PBMCs)** | Very. Standardized Viash submissions, Kaggle-style leaderboard, fresh held-out data | High in ML-for-bio | Medium. Different input format (chemical SMILES, not gene identity) | Tests whether `program.md` lessons transfer across problem shape; CRISPRa knowledge ideally generalizes to small-molecule screens |
| **Replogle 2022 genome-wide perturb-seq** | Very. Same problem form as Norman, ~2,500 perturbations instead of 237 | Medium-high. Standard dataset in scGPT and Geneformer papers | Very low. Same code, new h5ad loader, maybe reshape a few hyperparameters | Direct scale test of the exact pipeline; produces a second writeup in one evening |
| **GEARS-style combo_seen0/1/2 OOD splits on Norman** | Very. Data and code already sitting in this repo | Medium. PerturBench and TxPert both report on this | Very low. Swap the split function in `harness_perturbench.py` | Closes the "only tested on the easier combo split" caveat in BENCHMARK.md |
| **BEELINE gene regulatory network inference** | Very. Clean benchmark, pinned datasets, AUPRC/EP metrics, well-curated | Medium. Cited across methods papers | Medium. Different problem shape (DAGs/networks, not expression vectors) | Proves the autoresearch pattern isn't perturbation-specific; transfers to adjacent bioinformatics tasks |
| **TxPert-style OOD perturbation prediction** | Yes. Their codebase is public, eval is scripted, checkpoint reproduces in ~1 hour on A100 | Highest among newest ML-for-bio | Medium, requires GPU for their pipeline | Newest published SOTA baseline; directly outperforming it is the clearest 2026-era claim possible |
| **Polaris Cell Painting / phenotypic prediction** | Yes. Polaris hubs have Cell Painting benchmarks | Medium-high in pharma-adjacent ML | Medium. Imaging not scRNA-seq (different features, probably CNN) | Broader skill transfer into image-based phenotypic screens |

## Two honorable mentions outside ML-for-bio

- **M5 / M6 forecasting** (time series). Arguably the cleanest possible
  autoresearch target: pinned training window, pinned metric
  (RMSSE / WRMSSE), Kaggle hosts it. Broad ML visibility rather than
  ML-for-bio depth.
- **A Kaggle active competition** in your highest-interest domain.
  Kaggle is built for autoresearch: explicit train/test split, pinned
  metric, live leaderboard, strict no-data-leakage rules. Visibility
  per hour of work is probably the highest option on this page.

## Picks

If the goal is **maximum visibility per hour of the same skills**, the
Virtual Cell Challenge 2026 is the obvious answer *when the round
opens*. Sign up for the Arc announcement list, then aim the same loop
at their CRISPRi H1 hESC data when it drops. The existing
`pipeline_la.py` architecture generalizes almost directly; the main
change is flipping the target-gene override from "up" to "down"
(CRISPRa to CRISPRi). Zero immediate work; mark the calendar.

If the goal is **tightest-loop portfolio expansion** with zero new
GPU spend, **Replogle 2022** is the fastest. Same file layout, same
harness shape, just a different h5ad in the cache. The `program.md`
that ended run 2 becomes the portable artifact you brag about.
Probably a single evening of work to score + produce a second
writeup.

If the goal is **closing a specific defensibility hole** from this
project, the GEARS-style OOD splits (where component genes are
unseen) are the one experiment that would meaningfully sharpen the
current SOTA claim. Maybe a half-day locally, zero new data needed.

## What I'd explicitly *not* recommend

- **Any imaging task** (Cell Painting, spatial transcriptomics) as
  the *next* project. Too much new tooling (image loaders, CNNs)
  for too little carryover.
- **Protein structure prediction** (CASP-style). Requires GPU,
  massive compute, and the autoresearch pattern has less to add
  there since the problem is already very well tooled.
- **Full-blown deep learning architecture search on this task**
  (bigger transformers, foundation-model finetuning). Our ablation
  shows architecture size doesn't matter on Norman 2019 combo
  split; spending weeks on architecture wouldn't move the number.
