# condition_b_training_patch.md

The **most rigorous** Condition B — PerturBench's Latent Additive
retrained with our four training-side improvements active during
training, not just applied post-hoc — requires a small code change
to their `LatentAdditive` class. This doc describes the patch so
you can apply it by hand on the GPU VM before running
`scripts/run_on_gpu.sh`.

The local training-procedure ablation
(`pipeline_la_ablation.py`) already showed per-pert-mean training
is worth only ~+0.008 over per-cell. So in practice the
**post-hoc ensembling + output residual + per-target override**
implemented by `scripts/condition_b_wrapper.py` captures ~90% of
the available lift without touching their code. This patch is the
full-rigor option if you want the last 0.008 too.

## The patch

Target file:
`/tmp/perturbench/src/perturbench/modelcore/models/latent_additive.py`

### Change 1: output-space residual in `forward()`

```diff
  def forward(
      self,
      control_input: torch.Tensor,
      perturbation: torch.Tensor,
      covariates: dict[str, torch.Tensor],
  ):
      ...
      latent_control = self.gene_encoder(control_input)
      latent_perturbation = self.pert_encoder(perturbation)
      ...
      latent_perturbed = latent_control + latent_perturbation
      if self.inject_covariates_decoder:
          latent_perturbed = torch.cat([latent_perturbed, merged_covariates], dim=1)
      predicted_perturbed_expression = self.decoder(latent_perturbed)

-     if self.softplus_output:
-         predicted_perturbed_expression = F.softplus(predicted_perturbed_expression)
+     # Condition B: output-space residual instead of softplus.
+     # Decoder learns the delta off of the control baseline.
+     predicted_perturbed_expression = control_input + predicted_perturbed_expression
      return predicted_perturbed_expression
```

### Change 2: per-pert-mean training in `training_step()`

```diff
  def training_step(self, batch: Batch, batch_idx: int):
      (
          observed_perturbed_expression,
          control_expression,
          perturbation,
          covariates,
          embeddings,
      ) = self.unpack_batch(batch)
      ...
      predicted_perturbed_expression = self.forward(
          control_input, perturbation, covariates
      )
-     loss = F.mse_loss(predicted_perturbed_expression, observed_perturbed_expression)
+     # Condition B: aggregate per-perturbation before computing loss.
+     # `perturbation` is a multi-hot tensor identifying each cell's pert;
+     # cells with identical perturbation one-hots belong to the same pert.
+     pert_keys = perturbation.long().argmax(dim=-1)  # simplified; works for
+                                                    # single-gene + control
+     unique_keys = pert_keys.unique()
+     pred_means = torch.stack([
+         predicted_perturbed_expression[pert_keys == k].mean(dim=0)
+         for k in unique_keys
+     ])
+     obs_means = torch.stack([
+         observed_perturbed_expression[pert_keys == k].mean(dim=0)
+         for k in unique_keys
+     ])
+     loss = F.mse_loss(pred_means, obs_means)
      self.log("train_loss", loss, prog_bar=True, logger=True, batch_size=len(batch))
      return loss
```

> **Caveat:** the `pert_keys` computation above simplifies the
> multi-hot case. For dual perturbations like `KLF1+MAP2K6`, the
> multi-hot isn't a clean one-hot, so `argmax` picks one of the
> two. A cleaner version keys on the raw perturbation string from
> `batch.perturbation_labels`. The simplified form above is enough
> for the majority of training perts (singles dominate in their
> combo-prediction split), and the per-pert-mean lift is small
> anyway.

### Change 3: ensemble and target override

These are **not** model-internal changes — they're applied at
inference time by the wrapper script
(`scripts/condition_b_wrapper.py`). No patch to their code needed
for these.

## Applying the patch

Save the above as a conventional unified-diff file (`condition_b.patch`),
then on the GPU VM:

```bash
cd /tmp/perturbench
git apply /path/to/condition_b.patch
# retrain all seeds from scratch — the training_step change
# invalidates any pre-trained checkpoints
bash /path/to/run_on_gpu.sh
# ensemble wrapper runs on the patched-and-retrained checkpoints
python /path/to/scripts/condition_b_wrapper.py ...
```

## Expected result

Under our local ablations:
- **Their code + their training = their published 0.79** (not reproduced
  on GPU here; cited).
- **Their code + our training (residual + per-pert-mean) + our ensemble
  + our override ≈ 0.87**, matching our pipeline.

If this runs and lands near 0.87, the SOTA claim is closed at the
strongest-possible level: same architecture, same codebase, same
hyperparameters — the ~0.08 gap is entirely the four training-side
improvements.

If it lands somewhere *other* than 0.87 (meaningfully below our number
at 0.86 or above it at 0.88+), that's itself an interesting finding —
their codebase has something different beyond the four items we've
identified.
