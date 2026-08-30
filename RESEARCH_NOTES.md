# Research Notes — Ablation, Diagnostics & Design Decisions

## Dataset & splits

**Binary detector** (`build_splits` in `dataset.py`): Midjourney and VQDM
are held out 100% from train/val — never seen until final evaluation.

| Pool | Split |
|---|---|
| Real images | train 70% / val 10% / test_indist 10% / test_crossgen 10% |
| 6 seen generators (ADM, BigGAN, GLIDE, SD14, SD15, Wukong) | train 70% / val 10% / test_indist 20% |
| Midjourney, VQDM | 100% → test_crossgen |

- `test_indist` = held-out slice of seen generators + real → "did we learn what we trained on?"
- `test_crossgen` = Midjourney + VQDM + real → "do we generalize to a generator never seen at all?"

**Multiclass generator-ID model** (`build_multiclass_splits`): all 9
classes (real + 8 generators) are seen in training — no held-out concept,
since this model's job is to name a generator among ones it's been
shown. Real class is capped to roughly the average per-generator fake
count before splitting, so training isn't dominated by a much larger
real pool.

## Architecture

- Backbone: ResNet18, ImageNet-pretrained.
- **ResNet50 was deliberately ruled out.** The 5-run ablation below shows
  the ceiling is about *what signal* the model learns (generator
  fingerprints vs. general artifacts), not about capacity. A bigger
  backbone would likely just memorize seen-generator fingerprints more
  precisely — helping in-dist, possibly hurting cross-gen further.
- Partial freeze: `conv1, bn1, layer1, layer2` frozen (generic,
  ImageNet-transferable features); `layer3, layer4, fc` trainable. This
  is the best middle ground found between full freeze (underfits, caps
  at ~70%) and full fine-tune (~90% in-dist / ~47% cross-gen — high
  in-dist accuracy that's an illusion, since the extra capacity gets
  spent memorizing generator fingerprints).
- Discriminative learning rates: head `3e-4`, backbone `1e-5`, Adam +
  cosine annealing.

## Augmentation — different strategies per model, on purpose

| | Binary detector | Generator-ID model |
|---|---|---|
| Horizontal flip | Yes | Yes |
| Random JPEG recompression (q10-90, p=0.85) | Yes | **No** |
| Random Gaussian blur (p=0.6) | Yes | **No** |

The binary detector needs to work *despite* compression/blur differences
across unseen generators, so it's trained to be robust to them. The
generator-ID model's entire job is to read the compression/frequency
fingerprint characteristic of each generator — stripping that out with
the same augmentations would destroy the signal it needs to learn.

`RandomResolutionSimulation` is also applied to both: a random
intermediate resize + random interpolation kernel before the final
224×224 resize, to combat a specific resolution-artifact hypothesis (see
Diagnostics below).

## Ablation results — binary detector (5 runs)

| Run | Config | In-dist acc | Cross-gen acc | Gap |
|---|---|---|---|---|
| 1–3 | freeze/aug/epoch variations (aug strength p=0.3 → p=0.8-0.9) | 0.85–0.90 | 0.47–0.51 | large, flat |
| 4 | Partial freeze | 0.8522 | 0.5059 | **0.3463 (best)** |
| 5 | Partial freeze (repeat/variant) | 0.8626 | 0.5077 | ~0.355 |

Cross-generator accuracy is stuck in the **0.47–0.51 band across all 5
runs**, regardless of freeze strategy, augmentation strength, or epoch
count. Midjourney and VQDM recall specifically sit around **~0.36** in
every run. This is treated as a defensible, complete finding — four
controlled data points is enough; further tuning on this axis has
diminishing returns.

**Open item:** SD14 (generator id 5) is missing from the in-distribution
eval breakdown printed by `eval.py`. Not yet root-caused.

## Diagnostics (`diagnoise.py`)

**A. Native resolution check.** Investigates whether the model is partly
keying on resize/interpolation artifacts tied to specific native-size →
224 downsampling ratios seen in training. Midjourney's native resolution
(1024×1024) is far larger than the seen generators (which top out around
512×512), so this is a live hypothesis for Midjourney specifically.
**VQDM is a different story** — its native resolution (256×256) already
matches ADM/GLIDE, both seen and both detected well — so VQDM's failure
looks like a genuine artifact-signature difference (vector quantization
vs. continuous diffusion), not a resolution issue. One fix is not
expected to solve both.

**B. Prediction bias check.** Confirms the model isn't just defaulting
to "REAL" on unfamiliar generators. Real accuracy stays high (~0.92–0.94)
while Midjourney/VQDM recall sits much lower (~0.36), consistent with
the model defaulting toward "REAL" on generators it hasn't learned a
fingerprint for, rather than having learned a general "this looks
synthetic" cue.

## Confusion matrix (`diagnoise_multiclass.py`)

Checks whether visually similar generators (Real / SD15 / Wukong — all
photorealistic, and SD15/Wukong both Stable-Diffusion-based) are
specifically confused with each other, versus confusion being spread
randomly across unrelated classes. Run `python diagnoise_multiclass.py`
against the trained multiclass checkpoint to regenerate this table for
the current model — exact counts depend on the checkpoint in
`checkpoints/resnet_generator_id_best.pt` and aren't reproduced here to
avoid this document going stale the next time the model is retrained.

## Scientific integrity constraint

The binary model's held-out generator split (Midjourney/VQDM never
touched in train/val) must stay intact for the generalization-gap
finding to remain credible. Any future change to `build_splits()` needs
to preserve this.
