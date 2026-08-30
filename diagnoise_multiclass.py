"""
diagnose_multiclass.py
Confusion-matrix check for the 9-class generator-ID model. Confirms (or
refutes) the hypothesis that Real / SD15 / Wukong are confused with each
other specifically (all "photorealistic" -- SD15 and Wukong are both
Stable-Diffusion-based), rather than confusion being spread randomly
across unrelated classes.

Uses the existing checkpoint (config.MULTICLASS_CHECKPOINT_PATH) -- no
retraining needed.

Run: python diagnose_multiclass.py
"""

import torch
from collections import defaultdict

import config
from dataset import get_multiclass_dataloaders
from model import build_model


def build_confusion_matrix(model, loader, device):
    model.eval()
    # confusion[true_gen_id][pred_gen_id] = count
    confusion = defaultdict(lambda: defaultdict(int))

    with torch.no_grad():
        for images, labels, _ in loader:
            images = images.to(device)
            outputs = model(images)
            preds = outputs.argmax(dim=1).cpu()

            for pred, label in zip(preds, labels):
                pred = pred.item()
                label = label.item() if torch.is_tensor(label) else label
                confusion[label][pred] += 1

    return confusion


def print_matrix(confusion, class_ids):
    names = [config.GENERATOR_ID_TO_NAME[c][:8] for c in class_ids]
    header = "true\\pred".ljust(12) + "".join(n.rjust(9) for n in names)
    print(header)
    for true_id in class_ids:
        row = confusion.get(true_id, {})
        total = sum(row.values())
        if total == 0:
            continue
        name = config.GENERATOR_ID_TO_NAME[true_id][:11]
        cells = "".join(f"{row.get(p, 0):>9}" for p in class_ids)
        print(f"{name:<12}{cells}   (n={total})")


def print_top_confusions(confusion, class_ids, min_count=15):
    print("\nTop confusions (excluding correct predictions), n >= "
          f"{min_count}:")
    rows = []
    for true_id in class_ids:
        row = confusion.get(true_id, {})
        total = sum(row.values())
        if total == 0:
            continue
        for pred_id, count in row.items():
            if pred_id == true_id or count < min_count:
                continue
            pct = count / total
            rows.append((count, pct, true_id, pred_id))

    rows.sort(reverse=True)
    for count, pct, true_id, pred_id in rows:
        true_name = config.GENERATOR_ID_TO_NAME[true_id]
        pred_name = config.GENERATOR_ID_TO_NAME[pred_id]
        print(f"  True={true_name:<12} -> Predicted={pred_name:<12} "
              f"{count:>4} times ({pct:.1%} of true-{true_name} samples)")


def main():
    device = config.DEVICE
    print(f"Using device: {device}")

    _, _, test_loader = get_multiclass_dataloaders()

    model = build_model(
        backbone=config.BACKBONE, num_classes=config.NUM_CLASSES_MULTICLASS
    ).to(device)
    model.load_state_dict(
        torch.load(config.MULTICLASS_CHECKPOINT_PATH, map_location=device)
    )
    print(f"Loaded checkpoint: {config.MULTICLASS_CHECKPOINT_PATH}\n")

    confusion = build_confusion_matrix(model, test_loader, device)

    # Only include classes that actually have test samples (skips SD14, which
    # is empty in this dataset -- see prior finding).
    class_ids = sorted(
        cid for cid in config.GENERATOR_ID_TO_NAME.keys()
        if sum(confusion.get(cid, {}).values()) > 0
    )

    print("=" * 70)
    print("CONFUSION MATRIX (rows = true class, columns = predicted class)")
    print("=" * 70)
    print_matrix(confusion, class_ids)

    print("\n" + "=" * 70)
    print_top_confusions(confusion, class_ids)
    print("=" * 70)


if __name__ == "__main__":
    main()