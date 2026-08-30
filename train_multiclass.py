"""
train_multiclass.py
Trains the SEPARATE 9-class generator-ID model (real=0, generators 1-8).
Unlike train.py, every generator is seen during training -- there's no
held-out concept here, because this model's job is to name the generator,
not to test generalization to unseen ones.

Structure mirrors train.py deliberately (same optimizer setup, discriminative
LRs, cosine schedule) so results between the two models are comparable.
Saves to a SEPARATE checkpoint (config.MULTICLASS_CHECKPOINT_PATH) so this
never overwrites the binary detector's checkpoint.
"""

import os
from collections import defaultdict

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR

import config
from dataset import get_multiclass_dataloaders
from model import build_model


def evaluate(model, loader, device):
    """Plain loss/accuracy eval, used during training for checkpoint selection."""
    model.eval()
    correct, total, loss_sum = 0, 0, 0.0
    criterion = nn.CrossEntropyLoss()
    with torch.no_grad():
        for images, labels, _ in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss_sum += loss.item() * images.size(0)
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    return loss_sum / total, correct / total


def evaluate_with_breakdown(model, loader, device):
    """Per-generator accuracy breakdown -- the actual point of this model.
    Label IS the generator_id for this task, so gen_id and label are the
    same value; this just gives a named, human-readable table."""
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
    os.makedirs(config.CHECKPOINT_DIR, exist_ok=True)
    device = config.DEVICE
    print(f"Using device: {device}")

    train_loader, val_loader, test_loader = get_multiclass_dataloaders()

    model = build_model(
        backbone=config.BACKBONE, num_classes=config.NUM_CLASSES_MULTICLASS
    ).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = Adam([
        {"params": model.fc.parameters(), "lr": config.LEARNING_RATE_HEAD},
        {"params": [p for n, p in model.named_parameters() if not n.startswith("fc")],
         "lr": config.LEARNING_RATE_BACKBONE},
    ])
    scheduler = CosineAnnealingLR(optimizer, T_max=config.MULTICLASS_NUM_EPOCHS)

    best_val_acc = 0.0

    for epoch in range(1, config.MULTICLASS_NUM_EPOCHS + 1):
        model.train()
        running_loss, correct, total = 0.0, 0, 0

        for images, labels, _ in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

        scheduler.step()

        train_loss = running_loss / total
        train_acc = correct / total
        val_loss, val_acc = evaluate(model, val_loader, device)

        current_lr = scheduler.get_last_lr()
        print(f"Epoch {epoch}/{config.MULTICLASS_NUM_EPOCHS} | "
              f"train_loss {train_loss:.4f} train_acc {train_acc:.4f} | "
              f"val_loss {val_loss:.4f} val_acc {val_acc:.4f} | "
              f"lr_head {current_lr[0]:.2e} lr_backbone {current_lr[1]:.2e}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), config.MULTICLASS_CHECKPOINT_PATH)
            print(f"  -> New best val_acc {val_acc:.4f}, checkpoint saved.")

    print(f"\nTraining done. Best val_acc: {best_val_acc:.4f}")
    print(f"Checkpoint: {config.MULTICLASS_CHECKPOINT_PATH}")

    # Load best checkpoint and report per-generator breakdown on the held-out test split
    model.load_state_dict(torch.load(config.MULTICLASS_CHECKPOINT_PATH, map_location=device))
    print("\n" + "=" * 55)
    print("TEST SET -- per-generator (9-class) breakdown")
    print("=" * 55)
    evaluate_with_breakdown(model, test_loader, device)

    print("\nNote: unlike the binary detector, there's no held-out generator here --")
    print("all 8 generators were seen in training. Expect visually similar generators")
    print("(e.g. SD14/SD15/Wukong, all Stable-Diffusion-based) to be harder to tell")
    print("apart than visually distinct ones (e.g. BigGAN vs. a diffusion model).")


if __name__ == "__main__":
    main()
