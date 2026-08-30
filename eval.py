"""
eval.py
Reports accuracy on:
  1. test_indist    - held-out images from the 6 SEEN generators (+ real)
  2. test_crossgen  - Midjourney + VQDM (never seen in training) + real

For both, prints a PER-GENERATOR breakdown, not just an aggregate number.
The gap between test_indist and test_crossgen accuracy is the actual
resume-relevant result of this project.
"""

import torch
from collections import defaultdict

import config
from dataset import get_dataloaders
from model import build_model  # adjust if your model.py uses a different name


def evaluate_with_breakdown(model, loader, device):
    model.eval()
    correct_by_gen = defaultdict(int)
    total_by_gen = defaultdict(int)

    with torch.no_grad():
        for images, labels, gen_ids in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            preds = outputs.argmax(dim=1).cpu()

            for pred, label, gen_id in zip(preds, labels.cpu(), gen_ids):
                gen_id = int(gen_id)
                total_by_gen[gen_id] += 1
                if pred.item() == label.item():
                    correct_by_gen[gen_id] += 1

    overall_correct = sum(correct_by_gen.values())
    overall_total = sum(total_by_gen.values())
    overall_acc = overall_correct / overall_total if overall_total else 0.0

    print(f"  Overall accuracy: {overall_acc:.4f}  ({overall_correct}/{overall_total})")
    print(f"  {'Generator':<12} {'Accuracy':>10} {'N':>8}")
    for gen_id in sorted(total_by_gen.keys()):
        name = config.GENERATOR_ID_TO_NAME.get(gen_id, f"id_{gen_id}")
        acc = correct_by_gen[gen_id] / total_by_gen[gen_id]
        print(f"  {name:<12} {acc:>10.4f} {total_by_gen[gen_id]:>8}")

    return overall_acc


def main():
    device = config.DEVICE
    print(f"Using device: {device}")

    _, _, test_indist_loader, test_crossgen_loader = get_dataloaders()

    model = build_model(backbone = config.BACKBONE, num_classes = config.NUM_CLASSES).to(device)
    model.load_state_dict(torch.load(config.CHECKPOINT_PATH, map_location=device))
    print(f"Loaded checkpoint: {config.CHECKPOINT_PATH}\n")

    print("=" * 55)
    print("IN-DISTRIBUTION TEST  (real + 6 seen generators, held-out slice)")
    print("=" * 55)
    indist_acc = evaluate_with_breakdown(model, test_indist_loader, device)

    print("\n" + "=" * 55)
    print("CROSS-GENERATOR TEST  (real + Midjourney/VQDM, never seen in training)")
    print("=" * 55)
    crossgen_acc = evaluate_with_breakdown(model, test_crossgen_loader, device)

    print("\n" + "=" * 55)
    print("SUMMARY")
    print("=" * 55)
    print(f"In-distribution accuracy:   {indist_acc:.4f}")
    print(f"Cross-generator accuracy:   {crossgen_acc:.4f}")
    print(f"Generalization gap:         {indist_acc - crossgen_acc:.4f}")
    print("\nThis gap is the real finding. A small gap = genuine artifact-level "
          "detection. A large gap = the model learned generator-specific fingerprints, "
          "not general fake-image detection.")


if __name__ == "__main__":
    main()
