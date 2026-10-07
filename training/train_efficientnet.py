"""
Training Script — EfficientNet-B4 Fine-tuning on FaceForensics++
Transfer learning from ImageNet pretrained weights.

Dataset: FaceForensics++ (c23 compression)
  - Download from: https://github.com/ondyari/FaceForensics
  - Expected structure:
      data/
        train/
          real/   (extracted face crops)
          fake/   (extracted face crops from all manipulation methods)
        val/
          real/
          fake/

Usage:
    python train_efficientnet.py --data_dir ./data --epochs 20 --batch_size 32
"""

import argparse
import os
import logging

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import timm

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("train_efficientnet")


def get_transforms(is_train: bool):
    """
    Get preprocessing transforms for EfficientNet-B4.
    Input: 299×299, ImageNet normalization.
    """
    if is_train:
        return transforms.Compose([
            transforms.Resize((320, 320)),
            transforms.RandomCrop(299),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
            transforms.RandomRotation(10),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
            transforms.RandomErasing(p=0.1),
        ])
    else:
        return transforms.Compose([
            transforms.Resize((299, 299)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ])


def train_one_epoch(model, loader, criterion, optimizer, device, epoch):
    """Train for one epoch."""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for batch_idx, (images, labels) in enumerate(loader):
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

        if batch_idx % 50 == 0:
            logger.info(
                f"  Epoch {epoch} [{batch_idx}/{len(loader)}] "
                f"Loss: {loss.item():.4f} | Acc: {100.*correct/total:.2f}%"
            )

    avg_loss = running_loss / len(loader)
    accuracy = 100. * correct / total
    return avg_loss, accuracy


@torch.no_grad()
def validate(model, loader, criterion, device):
    """Validate the model."""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        loss = criterion(outputs, labels)

        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

    avg_loss = running_loss / len(loader)
    accuracy = 100. * correct / total
    return avg_loss, accuracy


def main():
    parser = argparse.ArgumentParser(description="Fine-tune EfficientNet-B4 for deepfake detection")
    parser.add_argument("--data_dir", type=str, required=True, help="Path to dataset with train/val splits")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-5)
    parser.add_argument("--output_dir", type=str, default="./checkpoints")
    parser.add_argument("--freeze_backbone", action="store_true",
                        help="Freeze all layers except classifier head (linear probe)")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Load model with ImageNet pretrained weights
    model = timm.create_model("efficientnet_b4", pretrained=True, num_classes=2)
    model.to(device)
    logger.info("Loaded EfficientNet-B4 with ImageNet weights.")

    # Optionally freeze backbone
    if args.freeze_backbone:
        for name, param in model.named_parameters():
            if "classifier" not in name:
                param.requires_grad = False
        logger.info("Backbone frozen. Training classifier head only.")

    # Data loaders
    train_dataset = datasets.ImageFolder(
        os.path.join(args.data_dir, "train"),
        transform=get_transforms(is_train=True),
    )
    val_dataset = datasets.ImageFolder(
        os.path.join(args.data_dir, "val"),
        transform=get_transforms(is_train=False),
    )

    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size,
        shuffle=True, num_workers=4, pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size,
        shuffle=False, num_workers=4, pin_memory=True,
    )

    logger.info(f"Train: {len(train_dataset)} images, Val: {len(val_dataset)} images")
    logger.info(f"Classes: {train_dataset.classes}")

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    # Training loop
    best_val_acc = 0.0
    for epoch in range(1, args.epochs + 1):
        logger.info(f"\n{'='*60}")
        logger.info(f"Epoch {epoch}/{args.epochs}")
        logger.info(f"{'='*60}")

        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device, epoch
        )
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        scheduler.step()

        logger.info(
            f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}% | "
            f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%"
        )

        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            save_path = os.path.join(args.output_dir, "efficientnet_b4_best.pth")
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_acc": val_acc,
                "val_loss": val_loss,
            }, save_path)
            logger.info(f"✓ Saved best model (val_acc={val_acc:.2f}%) → {save_path}")

    logger.info(f"\nTraining complete. Best val accuracy: {best_val_acc:.2f}%")


if __name__ == "__main__":
    main()
