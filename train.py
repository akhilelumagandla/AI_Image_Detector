"""
train.py
Trains on the 6 SEEN generators + real images only.
Midjourney and VQDM are never touched here — they only appear in eval.py's
cross-generator test.

NOTE: assumes your existing model.py exposes a `build_model(backbone, num_classes)`
function returning a torch.nn.Module (this matched your original ResNet18 setup).
If your model.py uses a different function/class name, just adjust the import
below — nothing else in this file needs to change.
"""

import os
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR

import config
from dataset import get_dataloaders
from model import build_model  # adjust if your model.py uses a different name


def evaluate(model, loader, device):
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


def main():
    os.makedirs(config.CHECKPOINT_DIR, exist_ok=True)
    device = config.DEVICE
    print(f"Using device: {device}")

    train_loader, val_loader, test_indist_loader, test_crossgen_loader = get_dataloaders()

    model = build_model(backbone = config.BACKBONE, num_classes = config.NUM_CLASSES).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = Adam([
        {"params": model.fc.parameters(), "lr": config.LEARNING_RATE_HEAD},
        {"params": [p for n, p in model.named_parameters() if not n.startswith("fc")],
         "lr": config.LEARNING_RATE_BACKBONE},
    ])
    scheduler = CosineAnnealingLR(optimizer, T_max=config.NUM_EPOCHS)

    best_val_acc = 0.0

    for epoch in range(1, config.NUM_EPOCHS + 1):
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
        print(f"Epoch {epoch}/{config.NUM_EPOCHS} | "
              f"train_loss {train_loss:.4f} train_acc {train_acc:.4f} | "
              f"val_loss {val_loss:.4f} val_acc {val_acc:.4f} | "
              f"lr_head {current_lr[0]:.2e} lr_backbone {current_lr[1]:.2e}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), config.CHECKPOINT_PATH)
            print(f"  -> New best val_acc {val_acc:.4f}, checkpoint saved.")

    print(f"\nTraining done. Best val_acc: {best_val_acc:.4f}")
    print(f"Checkpoint: {config.CHECKPOINT_PATH}")
    print("Next step: run eval.py for in-distribution AND cross-generator "
          "(Midjourney/VQDM) breakdown — that cross-gen number is the one that matters.")


if __name__ == "__main__":
    main()
