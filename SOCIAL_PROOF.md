# Social-proof paths for this work

Tier 4 from the SOTA-follow-up plan: get the work in front of a
benchmark / leaderboard / peer-review audience so the SOTA claim
has external credibility, not just internal rigor.

## What I looked at

| venue | uses Norman 2019? | uses cosine logFC? | submission path | fit for this work |
|---|---|---|---|---|
| [Virtual Cell Challenge](https://virtualcellchallenge.org/) (Arc Institute) | **no** — H1 hESC CRISPRi in 2025 round | uses MAE + DES + PDS | via their portal when challenge opens | poor fit — different cell line, different perturbation type (CRISPRi, not CRISPRa), requires retraining |
| PerturBench | yes (paper uses it) | **yes** | GitHub PR against `task_perturbation_prediction` | best fit but it's not a live leaderboard — the paper is the table |
| [Open Problems — perturbation prediction task](https://openproblems.bio/results/perturbation_prediction/) | **no** — PBMC + small-molecule compounds | logFC-based but different (clipped sign log10 p-value) | Viash component PR | different task entirely; would be a new project |
| [scPerturBench](https://bm2-lab.github.io/scPerturBench-reproducibility/) (Nature Methods 2026) | probably yes (29 datasets) | unclear — benchmarks multiple metrics | no explicit submission docs | unclear — would need to contact authors |
| Polaris ([polarishub.io](https://polarishub.io/)) | no Norman benchmark found | — | — | no match — Polaris' leaderboards are drug-discovery-focused |
| arXiv / bioRxiv | — | — | direct preprint | **viable** — no gatekeeping, timestamps the claim, lets other researchers respond |

## Realistic recommendation

The work targets a specific evaluation (PerturBench combo split,
cosine logFC on Norman 2019). No live public leaderboard uses
*exactly* that evaluation. That's not a problem — it's the state
of the field. Recent benchmark papers each define their own
(dataset, split, metric) tuple. The appropriate venues are:

**Short-term (no GPU needed, no peer-review gating):**

1. **Publish the Substack / blog post.** Distribution path.
   Hacker News + a few ML Twitter accounts will surface it if the
   framing is tight.
2. **Post a self-contained thread to BlueSky / X summarizing the
   key finding** — our four-item training-side stack on their
   architecture matches their 4M-parameter model, so the gap is
   training-procedure work, not architectural work. This is the
   interesting-in-itself claim even without a leaderboard.

**Medium-term (requires GPU, $5-20):**

3. **Run `scripts/run_on_gpu.sh`** to produce our own reproduction
   of PerturBench's published 0.79 on their code, then replace
   the "published 0.79" line in `BENCHMARK.md` with "ours,
   reproduced 0.78x on their code, 0.87 with our training stack."
   This is the strongest version of the claim.
4. **Watch for the Virtual Cell Challenge 2026 round opening.** Sign
   up for their announcement list. When they post the data, the
   loop we built here can be repointed at their task in probably
   a day of work. Their `MAE / DES / PDS` metric set is different
   from ours, but the architectural lessons port.

**Long-term (real commitment, GPU, write-up time):**

5. **Write it up as a ~6-page workshop paper** targeting a venue
   like the NeurIPS workshop on ML for Biology or the ICLR
   workshop on generative biology. These have lighter review
   bars than main-conference and welcome reproductions +
   ablations, which is what this is.
6. **Submit a PR to the PerturBench repo** adding the four-item
   training-side-improvements family to their baselines. They
   maintain the benchmark and would probably accept a
   well-argued addition; that would bake the finding into the
   canonical reference table for the field.

## What I'm NOT recommending right now

- **Full benchmark submission** (Virtual Cell Challenge, Polaris)
  on the current pipeline: the target tasks differ in dataset or
  metric enough that it would be a new project, not a submission
  of existing work. If the 2026 VCC round uses Norman 2019, that
  changes; until then, don't.
- **arXiv preprint** from where things are: the writeup is Substack-
  class, not paper-class. Going to paper-class is real work and
  probably needs either Condition A reproduced on GPU or one more
  dataset applied, ideally both. Don't preprint what you haven't
  made rigorous enough to survive a careful reviewer.

## One specific "minimize for a bit" action

**Publish the Substack post and tweet the interesting finding.** Cost:
30 min of editing, 0 GPU, 0 dollars. Upside: the finding is
unambiguous and falsifiable, the repo is public and clean, and the
distribution surface for "I applied Karpathy's autoresearch pattern
to a real benchmark and it worked" is probably larger than
arXiv-level rigor deserves right now.

After that: wait for 2026 VCC to open, or wait for GPU, whichever
comes first.
