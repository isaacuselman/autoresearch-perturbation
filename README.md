# autoresearch-perturbation

Karpathy's autoresearch pattern, adapted to single-cell perturbation response prediction. Three-file split: an immutable evaluator (`harness.py`), an agent-mutable pipeline (`pipeline.py`), and a markdown orchestration prompt (`program.md`). You hand `program.md` to Claude Code and it runs experiments overnight, ratcheting on a single scalar score.

The goal is not to win a benchmark. It is to produce (a) a tuned `program.md` that encodes your domain taste, (b) a `results.tsv` log showing what the loop found, and (c) a portfolio post describing what transferred and what didn't. Ship the writeup. That is the artifact.

## What's here

- `program.md` — orchestration. The human edits this; the agent reads and follows it.
- `harness.py` — the evaluator. **Immutable.** Loads pinned data split, scores `pipeline.py`, prints a grep-able scalar. The agent must not modify it.
- `pipeline.py` — the thing being optimized. Agent-mutable. Starts with a deliberately weak baseline.
- `prepare_data.py` — downloads Norman 2019 perturb-seq data via pertpy. Falls back to synthetic data if the download fails (so you can validate the loop immediately).
- `claude-code-kickoff.md` — the literal text to paste into Claude Code to start a run.
- `pyproject.toml` — uv-compatible dep list.

## Prerequisites

- Python 3.10+ and `uv` installed.
- Claude Code configured with file-edit permissions in this directory.
- ~8 GB RAM for Norman 2019. CPU is fine for the baseline; GPU matters only when the agent introduces foundation-model features.
- Disk: ~2 GB for the cached dataset.

## Quickstart

```bash
cd autoresearch-perturbation
uv sync
uv run prepare_data.py         # downloads Norman 2019, falls back to synthetic
uv run harness.py              # runs the baseline end-to-end, prints score
```

Expected output includes lines like `score: 0.0XX` and `coverage: 1.0000`. If you see those, the loop is ready.

Then open Claude Code in this directory and paste the contents of `claude-code-kickoff.md`.

## The 7-day plan

**Day 0 (today)** — Drop this folder on disk, run the quickstart, confirm baseline runs. Read `program.md` and edit it once: add any domain priors you care about (e.g., "prefer interpretable models," "reject any change that makes wallclock exceed 15 min"). The program.md is *your* instruction file; the default is a starting point.

**Day 1** — Kick off Claude Code with the `kickoff.md` text. Watch the first 3-5 experiments to confirm the agent is editing the right file, committing correctly, and honoring the git reset on discards. Fix anything obviously wrong in `program.md`. Let it run overnight.

**Day 2** — Read `results.tsv` and the git log. Tag the best commit. Identify what kind of improvements the agent found (data preprocessing tweaks? feature engineering? ensembling?). Update `program.md` to push it toward the remaining unexplored axes. Run again overnight.

**Day 3** — Introduce a foundation-model feature source. scGPT embeddings are the obvious first choice. Add a loader to `pipeline.py` (or let the agent do it, but front-load the HF download). Run overnight.

**Day 4** — Diminishing returns will start. Read the log. Note the plateau point — this is the creativity ceiling Karpathy warned about. It's a real data point for your writeup.

**Day 5** — Start the portfolio writeup. Structure: (i) why autoresearch for this problem, (ii) what the loop found, (iii) where it plateaued, (iv) what the final `program.md` looks like — this is the most portable artifact.

**Day 6** — If the best score beats an obvious public baseline (the linear ridge-on-one-hot that the 2026 preprint discussed), format a submission to Polaris or Open Problems. If not, skip — the writeup is still the deliverable.

**Day 7** — Publish the post. Push the repo. Ship.

If you're still tuning on day 10, you are overcomplicating. The writeup is the ship condition, not the leaderboard position.

## Honest caveats

- **The pertpy API may have drifted.** `prepare_data.py` uses what I believe is the current surface (`pertpy.data.norman_2019()`). If that function has moved, your agent's first fix is to find the right loader. The synthetic fallback lets you validate everything else while fixing this.
- **The starter baseline is deliberately terrible.** It predicts the same delta for every test perturbation. This is so the agent has obvious low-hanging fruit on experiment #1. The *real* floor you need to beat is ridge regression on perturbation identity — the known-hard-to-beat linear baseline. Hit that within the first 10 experiments or the loop isn't working.
- **Local evaluation ≠ competition evaluation.** The scored metric here is a reasonable proxy for the real Open Problems / Polaris metrics, but the exact scoring differs. Your submission pipeline (if you get that far) will need adaptation. That's fine — autoresearch is about climbing *some* scalar; the transfer to an external leaderboard is a separate step.
- **Evaluation gaming is the real risk.** The coverage floor handles the "silently drop hard perturbations" failure. The pinned split handles test leakage. But a sufficiently motivated agent can still find holes. Skim `results.tsv` daily and spot-check the top-scoring commits' predictions for sanity (e.g., are all predictions identical? does the delta have plausible magnitude?).
- **Creativity ceiling is lower than in LM training.** Expect a plateau in 50–100 experiments, not 700. Don't conclude the pattern failed — it's supposed to plateau. The plateau is a useful signal about where you need to inject a new prior into `program.md`.
