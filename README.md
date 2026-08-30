# AI-Generated Image Detector — Cross-Generator Generalization

**Live demo:** [PASTE YOUR HF SPACE URL HERE ONCE LIVE — see DEPLOYMENT.md]
*(Space runs on free CPU hardware and sleeps after inactivity — first
request after a while may take ~30-60s to wake up. This is normal, not
a bug.)*

A two-stage image classifier that detects whether an image is real or
AI-generated, and — if flagged as fake — identifies which generator
family likely produced it.

The core question this project investigates isn't "can a CNN tell real
from fake" (easy when train/test share generators) — it's whether a
detector trained on a handful of generators **generalizes to a
generator it has never seen**, which is the situation any real-world
deployment actually faces. Spoiler: it doesn't generalize well, and
this repo measures exactly how badly and why.

## How it works

```
                 ┌─────────────────────┐
  image  ──────▶ │  Stage 1: Binary     │──── "Real" ──▶ done
                 │  detector (ResNet18) │
                 └──────────┬───────────┘
                             │ "Fake"
                             ▼
                 ┌─────────────────────┐
                 │  Stage 2: Generator- │──▶ predicted generator
                 │  ID model (9-class)  │    (ADM / BigGAN / GLIDE /
                 │  ResNet18            │     Midjourney / SD15 /
                 └─────────────────────┘     VQDM / Wukong, etc.)
```

Stage 2 only runs when Stage 1 predicts "fake" — mirrors how the tool
would actually be used, and keeps each model's training objective
(and augmentation strategy) clean and separable.

## Headline result

| Split | Accuracy |
|---|---|
| In-distribution (6 seen generators, held-out slice) | 0.8522 – 0.8626 |
| Cross-generator (Midjourney + VQDM, **never seen in training**) | 0.5059 – 0.5077 |
| **Generalization gap** | **~0.35** |

Held across a 5-run controlled ablation (freeze strategy × augmentation
strength × epoch count) — the gap doesn't close with more tuning.
Midjourney/VQDM recall specifically sits around ~0.36. Full breakdown,
diagnostics, and the resolution-artifact hypothesis investigated for it
are in [`RESEARCH_NOTES.md`](./RESEARCH_NOTES.md).

## Dataset

[TheKernel01/Tiny-GenImage](https://huggingface.co/datasets/TheKernel01/Tiny-GenImage)
— real photos + fakes from 8 generators (ADM, BigGAN, GLIDE, Midjourney,
SD14, SD15, VQDM, Wukong). Midjourney and VQDM are **100% held out** from
binary-detector training — never touched until final evaluation — so the
cross-generator number is a genuine generalization test, not a relabeled
in-distribution number.

## Repo structure

```
config.py          hyperparameters, paths, generator ID maps
dataset.py          HF dataset loading, split builders, augmentations
model.py             ResNet18/50 builder with partial-freeze logic
train.py             binary detector training loop
train_multiclass.py  generator-ID model training loop
eval.py              in-dist / cross-gen evaluation with per-generator breakdown
diagnoise.py         resolution + prediction-bias diagnostics
diagnoise_multiclass.py  confusion matrix for the generator-ID model
app.py               Gradio app — the two-stage pipeline above
requirements.txt
```

Trained checkpoints (`checkpoints/*.pt`) are **not** in this repo — they
live in the Hugging Face Space, which is where inference actually runs.
See [`DEPLOYMENT.md`](./DEPLOYMENT.md) for how the Space is set up.

## Run locally

```bash
pip install -r requirements.txt
python dataset.py          # sanity-check splits
python train.py             # trains binary detector -> checkpoints/
python train_multiclass.py  # trains generator-ID model -> checkpoints/
python eval.py               # in-dist + cross-gen breakdown
python app.py                 # launches the Gradio app locally
```

## Known limitations

- Not trained on / will not reliably detect modern generators released
  after this dataset was built (DALL-E 3, Midjourney v6, Flux, SDXL,
  Bing/Copilot Image Creator).
- SD14 is excluded from the generator-ID guess (zero training samples
  for that class in this dataset — see `RESEARCH_NOTES.md`).
- Cross-generator accuracy (~0.51) is the honest number for unseen
  generators — this is the whole point of the project, not a bug to
  hide.

## Full writeup

See [`RESEARCH_NOTES.md`](./RESEARCH_NOTES.md) for the complete ablation
table, per-generator diagnostics, and the reasoning behind design
decisions (why ResNet50 was ruled out, why augmentation strategy differs
between the two models, etc.).
