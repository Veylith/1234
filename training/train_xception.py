"""
Training Script — XceptionNet Fine-tuning for Deepfake Detection
Transfer learning from ImageNet pretrained weights on FaceForensics++.

Dataset: Same as EfficientNet (FF++ face crops, c23 compression).

Usage:
    python train_xception.py --data_dir ./data --epochs 20 --batch_size 24
"""

import argparse
import os
import logging

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import timm

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("train_xception")


def get_transforms(is_train):
    """XceptionNet preprocessing: 299×299, XceptionNet normalization."""
    if is_train:
        return transforms.Compose([
            transforms.Resize((320, 320)),
            transforms.RandomCrop(299),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.15, contrast=0.15),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])
    else:
        return transforms.Compose([
            transforms.Resize((299, 299)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])


def main():
    parser = argparse.ArgumentParser(description="Fine-tune XceptionNet for deepfake detection")
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=24)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--output_dir", type=str, default="./checkpoints")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.output_dir, exist_ok=True)

    # Model
    model = timm.create_model("xception", pretrained=True, num_classes=2)
    model.to(device)
    logger.info("XceptionNet loaded with ImageNet weights.")

    # Data
    train_dataset = datasets.ImageFolder(
        os.path.join(args.data_dir, "train"), transform=get_transforms(True)
    )
    val_dataset = datasets.ImageFolder(
        os.path.join(args.data_dir, "val"), transform=get_transforms(False)
    )

    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True
    )

    logger.info(f"Train: {len(train_dataset)}, Val: {len(val_dataset)}")

    # Optimizer with differential learning rates
    # Lower LR for backbone, higher for new classifier head
    backbone_params = [p for n, p in model.named_parameters() if "fc" not in n]
    head_params = [p for n, p in model.named_parameters() if "fc" in n]

    optimizer = optim.AdamW([
        {"params": backbone_params, "lr": args.lr * 0.1},
        {"params": head_params, "lr": args.lr},
    ], weight_decay=1e-4)

    criterion = nn.CrossEntropyLoss()
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_val_acc = 0.0
    for epoch in range(1, args.epochs + 1):
        # Train
        model.train()
        correct, total, running_loss = 0, 0, 0.0

        for batch_idx, (images, labels) in enumerate(train_loader):
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

        scheduler.step()
        train_acc = 100. * correct / total

        # Validate
        model.eval()
        val_correct, val_total = 0, 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, predicted = outputs.max(1)
                val_total += labels.size(0)
                val_correct += predicted.eq(labels).sum().item()

        val_acc = 100. * val_correct / val_total

        logger.info(
            f"Epoch {epoch}: Train Acc: {train_acc:.2f}% | Val Acc: {val_acc:.2f}% | "
            f"LR: {scheduler.get_last_lr()[0]:.6f}"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), os.path.join(args.output_dir, "xception_best.pth"))
            logger.info(f"✓ Saved best model (val_acc={val_acc:.2f}%)")

    logger.info(f"\nBest val accuracy: {best_val_acc:.2f}%")


if __name__ == "__main__":
    main()
