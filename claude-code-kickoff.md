# Kickoff — paste this into Claude Code

Open Claude Code in the `autoresearch-perturbation/` directory. Paste the block below as your first message. Watch the first few experiments, then step away.

---

```
Read program.md in this directory. We're starting a new autoresearch run.

Follow the Setup section end to end: create the branch (use tag `nov1`), confirm data exists (run prepare_data.py if needed — note whether you got real or synthetic data), initialize results.tsv, and run the baseline once.

Report the baseline score to me, then ask for confirmation before starting the loop. Once I confirm, enter the experiment loop and do not stop until I interrupt you. Honor the NEVER STOP clause.

Constraints I want you to follow strictly:
- Only edit pipeline.py. Never touch harness.py, prepare_data.py, or program.md.
- Respect the 20-minute wallclock. Use `timeout 1200` on every harness run.
- Do not commit results.tsv or run.log.
- If you crash twice on the same idea, discard and try something different.
- If the loop has been at the same score for 20 experiments, write a note in NOTES_FOR_HUMAN.md and try a more radical change.

Start with Setup step 1.
```

---

## What to watch for in the first hour

1. **Setup completes cleanly.** The agent creates the branch, data loads (real or synthetic), baseline harness run produces a `score:` line. If any of these fail, fix before proceeding — a broken setup will poison every experiment.

2. **First experiment beats the baseline.** The starter baseline is deliberately weak. Experiment #1 should do something obvious (use perturbation identity as a feature) and score noticeably higher. If it doesn't, the agent is misunderstanding the contract — nudge it.

3. **Git discipline.** Every experiment is a single commit. Discards are clean `git reset --hard HEAD~1`. If the agent leaves uncommitted changes around, the ratchet breaks.

4. **Coverage stays at 1.0.** If coverage drops, the pipeline is silently failing on some perturbations. The 0.95 floor will zero the score automatically, but an upstream bug that causes partial coverage should be fixed, not hidden.

5. **Ideas stay diverse.** If the agent is cycling through minor variations of one approach, it's hit the creativity ceiling. Edit `program.md` to seed a new direction (a new feature source, a different loss function, a combining strategy).

## When to edit program.md

program.md is your iterated instruction file. Edit it between runs, not during. Good reasons to edit:

- The agent wasted cycles on an idea you already knew would fail → add to the "NOT worth trying early" list.
- The agent found something that worked but didn't push on it → codify that direction in the "Ideas worth trying" list.
- A class of bugs recurred → add a guardrail to "Domain priors."
- You got new domain knowledge from reading a paper → inject it.

The program.md at the end of the run — not the best model commit — is the portable artifact. Save it. Version it. That's what you take to the next benchmark.
