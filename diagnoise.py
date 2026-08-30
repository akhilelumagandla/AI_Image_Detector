"""
diagnose.py
Investigates why cross-generator accuracy is stuck at ~0.47-0.51 across
5 different training configurations, while in-distribution accuracy varies
0.85-0.90 and augmentation strength (p=0.3 -> p=0.8-0.9) barely moves the
cross-gen number.

Checks two hypotheses:
  A. Resolution/preprocessing leakage: do real vs. per-generator images
     differ in native resolution before your Resize(224,224) step? If so,
     the model may be learning resize/interpolation artifacts as a
     real-vs-fake shortcut that happens to be generator-specific -- which
     would explain why it fails on Midjourney/VQDM regardless of
     augmentation strength.
  B. Prediction bias: is the model just defaulting to "real" on cross-gen
     images rather than genuinely confusing them? (Real acc staying ~0.92-
     0.94 while Midjourney/VQDM sit at 0.29-0.37 suggests this, but let's
     confirm directly via prediction distribution + confusion matrix.)

Run: python diagnose.py
Requires a trained checkpoint at config.CHECKPOINT_PATH (from your last run).
"""

import torch
from collections import defaultdict, Counter

import config
from dataset import load_raw_dataset, build_splits
from model import build_model


# ---------------------------------------------------------------------
# A. Native resolution check (no model needed -- run this part first,
#    it's fast and doesn't require a checkpoint)
# ---------------------------------------------------------------------
def check_native_resolutions(full, sample_per_gen=150):
    print("=" * 60)
    print("A. NATIVE RESOLUTION CHECK (before Resize(224,224))")
    print("=" * 60)
    print(f"Sampling up to {sample_per_gen} images per generator...\n")

    by_generator = defaultdict(list)
    for i, gen_id in enumerate(full["generator"]):
        by_generator[gen_id].append(i)

    print(f"{'Generator':<12} {'N sampled':>10} {'Width (mean/min/max)':>24} "
          f"{'Height (mean/min/max)':>24} {'Aspect ratios seen':>20}")

    for gen_id in sorted(by_generator.keys()):
        name = config.GENERATOR_ID_TO_NAME.get(gen_id, f"id_{gen_id}")
        idxs = by_generator[gen_id][:sample_per_gen]
        widths, heights = [], []
        for i in idxs:
            img = full[i]["image"]
            w, h = img.size
            widths.append(w)
            heights.append(h)

        n = len(widths)
        if n == 0:
            continue
        w_mean, w_min, w_max = sum(widths) / n, min(widths), max(widths)
        h_mean, h_min, h_max = sum(heights) / n, min(heights), max(heights)
        aspect_ratios = sorted(set(round(w / h, 2) for w, h in zip(widths, heights)))
        ar_display = aspect_ratios[:3] if len(aspect_ratios) > 3 else aspect_ratios
        ar_str = f"{ar_display}{'...' if len(aspect_ratios) > 3 else ''}"

        print(f"{name:<12} {n:>10} "
              f"{f'{w_mean:.0f}/{w_min}/{w_max}':>24} "
              f"{f'{h_mean:.0f}/{h_min}/{h_max}':>24} "
              f"{ar_str:>20}")

    print("\nWhat to look for:")
    print("  - If Real's resolution/aspect-ratio profile differs sharply from")
    print("    the 6 seen generators BUT matches Midjourney/VQDM (or vice versa),")
    print("    that's consistent with a resolution-based shortcut.")
    print("  - If Midjourney/VQDM have a distinctly different native resolution")
    print("    than the 6 seen generators (e.g. all seen gens are 256x256 but")
    print("    Midjourney is 1024x1024), that's a strong candidate: the model")
    print("    may have learned resize-artifact patterns tied to the seen gens'")
    print("    specific native size, which don't transfer.\n")


# ---------------------------------------------------------------------
# B. Prediction bias / confusion matrix check (requires trained checkpoint)
# ---------------------------------------------------------------------
def check_prediction_bias(model, loader, device, split_name):
    model.eval()
    pred_counter = Counter()
    confusion = defaultdict(lambda: defaultdict(int))  # confusion[true_label][pred_label]
    confusion_by_gen = defaultdict(lambda: defaultdict(int))  # confusion_by_gen[gen_id][pred_label]

    with torch.no_grad():
        for images, labels, gen_ids in loader:
            images = images.to(device)
            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)
            preds = outputs.argmax(dim=1).cpu()

            for pred, label, gen_id in zip(preds, labels, gen_ids):
                pred = pred.item()
                label = label.item() if torch.is_tensor(label) else label
                gen_id = int(gen_id)
                pred_counter[pred] += 1
                confusion[label][pred] += 1
                confusion_by_gen[gen_id][pred] += 1

    total = sum(pred_counter.values())
    print(f"\n--- {split_name}: overall prediction distribution ---")
    print(f"  Predicted REAL: {pred_counter[0]:>6} ({pred_counter[0]/total:.1%})")
    print(f"  Predicted FAKE: {pred_counter[1]:>6} ({pred_counter[1]/total:.1%})")

    print(f"\n--- {split_name}: per-generator prediction breakdown ---")
    print(f"  {'Generator':<12} {'N':>6} {'Pred REAL':>10} {'Pred FAKE':>10} {'%PredFake':>10}")
    for gen_id in sorted(confusion_by_gen.keys()):
        name = config.GENERATOR_ID_TO_NAME.get(gen_id, f"id_{gen_id}")
        n_real_pred = confusion_by_gen[gen_id][0]
        n_fake_pred = confusion_by_gen[gen_id][1]
        n_total = n_real_pred + n_fake_pred
        pct_fake = n_fake_pred / n_total if n_total else 0.0
        print(f"  {name:<12} {n_total:>6} {n_real_pred:>10} {n_fake_pred:>10} {pct_fake:>10.1%}")


def main():
    ds = load_raw_dataset()
    from datasets import concatenate_datasets
    full = concatenate_datasets([ds["train"], ds["validation"]])

    # Part A: resolution check (fast, no model needed)
    check_native_resolutions(full)

    # Part B: prediction bias check (needs the trained checkpoint)
    print("=" * 60)
    print("B. PREDICTION BIAS CHECK (using last trained checkpoint)")
    print("=" * 60)
    device = config.DEVICE
    model = build_model(backbone=config.BACKBONE, num_classes=config.NUM_CLASSES).to(device)
    model.load_state_dict(torch.load(config.CHECKPOINT_PATH, map_location=device))
    print(f"Loaded checkpoint: {config.CHECKPOINT_PATH}")

    splits = build_splits(full)
    from dataset import GenImageDataset
    from torch.utils.data import DataLoader

    test_indist_ds = GenImageDataset(full, splits["test_indist"], train=False, task="binary")
    test_crossgen_ds = GenImageDataset(full, splits["test_crossgen"], train=False, task="binary")
    test_indist_loader = DataLoader(test_indist_ds, batch_size=config.BATCH_SIZE, shuffle=False,
                                     num_workers=config.NUM_WORKERS, pin_memory=True)
    test_crossgen_loader = DataLoader(test_crossgen_ds, batch_size=config.BATCH_SIZE, shuffle=False,
                                       num_workers=config.NUM_WORKERS, pin_memory=True)

    check_prediction_bias(model, test_indist_loader, device, "IN-DISTRIBUTION")
    check_prediction_bias(model, test_crossgen_loader, device, "CROSS-GENERATOR")

    print("\nWhat to look for:")
    print("  - If %PredFake for Midjourney/VQDM is low (e.g. <40%) while it's high")
    print("    for the 6 seen generators, that confirms the model defaults to")
    print("    predicting REAL on unfamiliar generators rather than genuinely")
    print("    misclassifying them -- i.e. it never learned a general 'this looks")
    print("    synthetic' cue, only seen-generator-specific fingerprints.")


if __name__ == "__main__":
    main()