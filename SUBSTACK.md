# I let Claude run ML experiments unsupervised for two evenings and it found state-of-the-art

*Draft. Written for a general-technical audience. Plain-English framing
up top; the benchmark-grade numbers are further down.*

---

## The setup

Andrej Karpathy keeps tweeting about an idea he calls **autoresearch**.
You give a language model agent a three-file ML project. One file is an
**evaluator** you promise never to change. One is a **pipeline** the
agent is allowed to edit. One is an **instruction document** that
describes the game. Then you let it loop. For every experiment it
proposes a one-line hypothesis, edits the pipeline, runs the evaluator,
looks at a single scalar score, and either keeps the change or reverts
the commit. Over time you accumulate a git history of everything that
worked and everything that didn't. The instruction document, not the
model, becomes the portable artifact, because you as the human rewrite
it between runs to inject what you've learned.

I tried it. Two evenings, mostly unattended. The problem I gave it was
**predicting how single cells respond to genetic perturbations**.
Specifically, the Norman et al. 2019 CRISPRa dataset, where scientists
overexpressed different genes in K562 cells and measured the resulting
expression profile on every other gene. You hold out 20% of the
perturbations at random, train on the rest, and see how well you can
predict the held-out ones.

The score went from **0.27 to 0.72** in 44 experiments. Then I pivoted
to a harder evaluation used by a published 2024 benchmark (PerturBench,
NeurIPS) and got to **0.87** in another 11 experiments. That's eight
points above PerturBench's reported best-in-class number on the same
split and metric.

This post is the tour. It's about what autoresearch is actually like in
practice, the bug I almost shipped, and why (on this task at least)
"simple wins" is split-specific.

---

## Round 1: 0.27 → 0.72

The starter pipeline predicts the **same** response for every held-out
perturbation: just the average response across all training
perturbations. It scores 0.27. That's deliberately bad so the agent has
obvious low-hanging fruit on experiment one.

The agent is allowed to edit one file (`pipeline.py`). Every experiment
is a single git commit. If the score doesn't strictly beat the current
best, the commit gets reverted. There's a 20-minute wall-clock cap per
experiment so it can't just throw more compute at a bad idea.

The data arrives with a catch. The official loader (`pertpy`, the
standard Python library for this kind of data) is currently broken
against recent versions of the surrounding ML stack, so everything
falls through to a synthetic dataset the prep script ships for
validation. That turned out to be useful. We got to bang out 21
experiments on synthetic data first, confirm the loop worked, and
then swap in the real thing.

Six experiments did most of the work on synthetic:

1. **Use the perturbation identity.** The starter doesn't even know
   which gene was perturbed. It predicts the same vector for every
   held-out pert. Step one was to parse the target gene from the
   label and force the prediction's entry for *that gene* to match
   the average post-perturbation level seen in training. Single
   largest jump in the entire project: **+0.32 points**.
2. Use the target gene's **own** baseline expression instead of a
   global average. **+0.09.**
3. Use the median instead of the mean for the "default" response.
   **Trivial**, about 0.001.
4. Scale the default response by roughly 3×. **+0.01.**
5. Let the loop pick the scale factor via leave-one-out
   cross-validation. **+0.005.**
6. Clip predicted expression levels to be ≥ 0. (Negative expression
   is impossible.) **+0.003.**

By experiment 21 on synthetic data the score plateaued around 0.71.
This matches what you'd expect from the generator's math. The
downstream gene effects in synthetic data are random per-perturbation
by design, so there's literally no signal left to learn beyond the
target-gene drop. A back-of-envelope calculation gave a ceiling of
~0.69. We were slightly above it, which is squeezing-test-set-noise
territory.

Then the pivot to real Norman 2019.

## The bug that survived ten experiments

When I swapped in the real dataset, the synthetic-tuned pipeline scored
0.54. That's a lot worse than 0.72 on synthetic, which is confusing
until you read what the synthetic-era pipeline was actually doing.

Norman 2019 includes **dual perturbations** (labels like `KLF1+MAP2K6`
where two genes are perturbed simultaneously). The synthetic generator
had labels like `pert_gene_42`. My code's "parse the target gene from
the label" step had a fallback: if the obvious match didn't work,
extract the trailing integer from the label and use it as a gene index.

For `KLF1+MAP2K6`, the trailing integer is "6." The code silently
decided the target gene of that perturbation was *whichever gene
happened to be in column 6 of the expression matrix*. About 130
different perturbations were getting random gene targets with no
error, no warning, no log line.

Fixing that one bug (treating multi-gene labels properly):
**+0.03 points**.

Then the idea that had *failed* on synthetic data finally worked.

The idea: for each held-out target gene, predict that genes with a
similar pattern of change across training perturbations will shift
too. You compute a (gene × gene) matrix of correlations in the
training data, and use it to propagate the target-gene drop outward.
On synthetic data, the control cells had no real biological structure,
just random noise, so the correlation matrix was noise and the
experiment failed. On real data, those correlations capture
something real about how genes move together under perturbation.

**+0.05 points.**

And then a third improvement that I later realized was an accident.

I switched the correlation formula from a standard Pearson-style form
to a conditional-expectation form (mathematically cleaner, same
intent). The score went up by 0.01. I was pleased. I kept going. I ran
five more experiments on top of it.

## The kernel-mismatch bug

Around experiment 30 I was integrating foundation-model embeddings
(I'll get to these) and had to re-read the prediction code carefully.
I noticed something: the **training-time scoring loop** was using the
new conditional-expectation formula, but the **test-time prediction
code** was still using the old Pearson formula. The two have different
scales. They had just happened to *compensate for each other* in a
way that increased the single scalar I was watching.

Fixing the inconsistency dropped the score back to where it was. No
net loss over the long run. Subsequent experiments pushed well past
the illusion. But here's the thing: **for ten experiments, the score
was lying to me**. A single scalar tells you nothing about whether
you're actually improving or drifting. This is exactly the failure
mode autoresearch is supposed to defend against, and I walked right
into it. I hadn't read the diff between the two pipelines end-to-end
in two days.

Lesson: do that. Even when the scalar is going up.

## Foundation models, used wrong and then right

scGPT and Geneformer are pretrained neural networks trained on
roughly 30 million single cells each. Each gene gets a learned
embedding (512-dimensional for scGPT, 256 for Geneformer). You can
download the weights from HuggingFace and use them without running
the full inference stack.

My first instinct was to treat these embeddings the same way I'd
treated the training-delta correlation matrix: compute cosine
similarity between gene embeddings and use it as a propagation
kernel. Two genes with similar pretrained embeddings should shift
together under perturbation, right?

The loop's cross-validation consistently set that weight to zero. I
manually forced it on. The score dropped. Turns out "which genes
look similar in a pretrained embedding" is a very different signal
from "which genes co-perturb under knockdown." One is about baseline
similarity; the other is about response similarity. Pretrained
embeddings are trained on the first and don't transfer to the
second.

What *did* work was using them as **features in a regression**.
Specifically: for each gene in our expression matrix, learn a linear
function that maps "the embedding of the perturbed gene" → "the
predicted change in this gene's expression." Trained on 149
perturbations, applied to the 47 held out. Tuning the regularization
strength walked the score from 0.66 to 0.71.

Adding Geneformer alongside scGPT (concatenate both into one bigger
feature vector): **+0.003**. Mostly redundant, since both models were
trained on overlapping cell atlases, but a free improvement.

Final score for round one: **0.72**.

## Round 2: now do it against a real benchmark

"0.72 on my home-brewed evaluation" is a comfortable claim but not a
comparable one. Nobody else uses my exact metric on my exact split.
PerturBench (Wu et al. 2024, NeurIPS) is the closest widely-cited
benchmark for Norman 2019. They use a specific split (train on every
single perturbation + 30% of the dual combinations; test on the
remaining 70% of duals) and a specific metric (cosine similarity
between predicted and actual log fold-change vectors across all genes).

I ported their evaluation into a parallel harness, ran my round-one
pipeline through it: **0.750**. PerturBench's best baseline, a
"Latent Additive" model (a small MLP autoencoder), scores 0.79 in
their paper. I was 4 points behind.

Round two reseeded the pipeline with a clean Latent Additive
implementation and kicked off a fresh autoresearch loop. Eleven
experiments later:

- The single baseline (LA alone) scored **0.835**, already above
  PerturBench's published 0.79.
- Ensembling 5 models with different random seeds: **+0.013**.
- Adding a simple output residual (predict the *delta* from baseline
  instead of the absolute expression): **+0.003**.
- Removing dropout entirely: **+0.013**. Biggest single jump in round
  two. Turns out the ensembling and training procedure were already
  reducing variance enough; dropout was just adding inference-time
  noise.
- A few attempted wins that actively hurt and got reverted (weight
  decay on the perturbation encoder caused mode collapse, and the
  model started predicting the same vector for every test
  perturbation).

Final round-two score: **0.871 ± 0.002** across three independent
multi-seed runs.

That's **eight points above PerturBench's published best on the same
evaluation**.

## The obvious question

Is 8 points above their best reflecting an actual algorithmic gap, or
did they just not tune their model well? To answer, I ran two
ablations.

**Their architecture under my training.** PerturBench's best Latent
Additive uses 107 million parameters (a wide, tuned MLP). Mine uses
0.8 million. I took their exact hyperparameters, plugged them into
my training loop, ran three seeds: **0.8748**. Essentially tied with
my small pipeline. Architecture size contributes maybe 0.004 to the
gap. So it's not that they're under-parameterized.

**Training procedure alone.** With architecture, ensembling, residual
connection, dropout, and target override all held constant, I
switched only the training mode: from "train on per-perturbation
mean expression vectors" to "train on per-cell examples" (what
PerturBench does). Result: **0.8708 vs 0.8624, delta +0.008**. About
a point. So training-procedure-as-such is also a small piece.

By subtraction, the remaining **~0.07** (the bulk of the gap) sits in
the four-item stack of ensembling + output residual + dropout removal
+ target-gene override. Each one alone is mundane. Stacked, they
compound to more than any single architectural improvement matters.

This lines up with what PerturBench's own results show. Their
"fancy" methods (CPA, SAMS-VAE, GEARS) all underperform their plain
Latent Additive baseline. The lesson isn't "foundation models are
useless" or "simple always wins." It's that this particular
evaluation rewards careful training-side work much more than it
rewards architecture or inductive bias.

## Things I'd do differently

**Read the code end-to-end more often.** The kernel-mismatch bug
should have been obvious. It wasn't, because I was skimming one
function at a time and trusting the score.

**Pay attention to CI.** At one point during round two, I seeded the
pipeline with a new model that imports PyTorch, but PyTorch lived in
my repo's optional dependencies group. Continuous integration was
red for two hours and nine commits before I noticed. I caught a
similar "single scalar isn't telling you the whole story" bug in
round one, built CI specifically to defend against it in round two,
then ignored CI and made the exact same class of mistake.

**Ablate aggressively.** The story I first told myself about round
two ("per-pert-mean training is worth three points") turned out to be
wrong once I actually ran the clean ablation. The real contributors
were smaller and more plural than the executive summary suggested.

## What autoresearch is and isn't

It's not a robot scientist. The most important edits to the
instruction document ("try foundation models as features, not as
kernels"; "watch for parsing bugs on multi-gene labels"; "output
residual is free") came from me reading experiment logs, spotting
patterns, and writing them into the program file between runs. The
agent was fast, tireless, good at specific local experiments, and
bad at noticing weeks-old priors that would save it an hour. I was
bad at those same things, but in a different way.

It *is* extraordinarily useful for the kind of grunt work ML
actually is: "try fifteen permutations of a learning rate, a
regularization weight, and an ensemble size; keep what wins." The
agent did twenty such experiments in a single evening. I would have
done three over a weekend.

## Where this lands

A full benchmark table with error bars, dependencies, and honest
caveats is in the repo's
[`BENCHMARK.md`](https://github.com/isaacuselman/autoresearch-perturbation/blob/main/BENCHMARK.md).
The session-by-session journals are in `PROCESS.md` (round 1) and
`PROCESS_PB.md` (round 2), with commit hashes for every experiment.

The repo itself is at
[github.com/isaacuselman/autoresearch-perturbation](https://github.com/isaacuselman/autoresearch-perturbation).
Two pipelines live there. The round-one ridge + foundation-model
hybrid is on `main`, and the round-two Latent Additive on
`autoresearch/pb-apr20`. Each wins on its home evaluation.

The portable artifact isn't either of them. It's
[`program.md`](https://github.com/isaacuselman/autoresearch-perturbation/blob/main/program.md),
the instruction document that went from "be a good ML researcher,
try interesting things" in round zero to something like a domain
field guide by the end. That's the thing I'd take to the next
benchmark, not the model weights.

---

*Written collaboratively by [Claude Code](https://claude.com/claude-code)
(which ran every experiment, parsed every log, and drafted this post)
and the human author (who directed the project, course-corrected in
real time, and edited the final version). The agent is fast and
patient; the human is responsible for what stays.*
