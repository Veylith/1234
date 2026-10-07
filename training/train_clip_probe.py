"""
Training Script — CLIP ViT-L/14 Linear Probe for AI-Generated Content Detection
Trains a linear classifier on top of frozen CLIP features.

Dataset: Mixture of real images + AI-generated outputs from multiple generators.
  - Recommended sources:
    - Real: LSUN, ImageNet validation, COCO
    - Fake: Stable Diffusion 1.5/2.1/XL, DALL-E 2/3, Midjourney, GAN outputs
  - Expected structure:
      data/
        train/
          real/
          fake/
        val/
          real/
          fake/

Usage:
    python train_clip_probe.py --data_dir ./data --epochs 10 --batch_size 64
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
import open_clip

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("train_clip_probe")


def get_clip_transforms():
    """
    Get CLIP preprocessing transforms.
    Input: 224×224, CLIP normalization.
    """
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.48145466, 0.4578275, 0.40821073],
            std=[0.26862954, 0.26130258, 0.27577711],
        ),
    ])


class CLIPFeatureExtractor:
    """Extract features from CLIP ViT-L/14 (frozen)."""

    def __init__(self, device):
        self.model, _, _ = open_clip.create_model_and_transforms(
            "ViT-L-14", pretrained="openai"
        )
        self.model.eval()
        self.model.to(device)
        self.device = device

        # Freeze all parameters
        for param in self.model.parameters():
            param.requires_grad = False

        logger.info("CLIP ViT-L/14 loaded and frozen.")

    @torch.no_grad()
    def extract(self, images):
        """Extract CLS token features from a batch of images."""
        features = self.model.encode_image(images.to(self.device))
        features = features / features.norm(dim=-1, keepdim=True)
        return features.float()


class LinearProbe(nn.Module):
    """Simple linear classifier on top of CLIP features."""

    def __init__(self, input_dim=768, num_classes=2):
        super().__init__()
        self.classifier = nn.Linear(input_dim, num_classes)

    def forward(self, x):
        return self.classifier(x)


def main():
    parser = argparse.ArgumentParser(description="Train CLIP linear probe for AI-generated detection")
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--output_dir", type=str, default="./checkpoints")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    os.makedirs(args.output_dir, exist_ok=True)

    # Initialize CLIP feature extractor (frozen)
    clip_extractor = CLIPFeatureExtractor(device)

    # Initialize linear probe
    probe = LinearProbe(input_dim=768, num_classes=2)
    probe.to(device)

    # Data
    transform = get_clip_transforms()
    train_dataset = datasets.ImageFolder(os.path.join(args.data_dir, "train"), transform=transform)
    val_dataset = datasets.ImageFolder(os.path.join(args.data_dir, "val"), transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)

    logger.info(f"Train: {len(train_dataset)}, Val: {len(val_dataset)}")
    logger.info(f"Classes: {train_dataset.classes}")

    # Loss and optimizer (only probe parameters)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(probe.parameters(), lr=args.lr)

    # Training loop
    best_val_acc = 0.0
    for epoch in range(1, args.epochs + 1):
        # Train
        probe.train()
        total_loss, correct, total = 0.0, 0, 0

        for batch_idx, (images, labels) in enumerate(train_loader):
            labels = labels.to(device)

            # Extract CLIP features (frozen, no grad)
            features = clip_extractor.extract(images)

            # Train probe
            optimizer.zero_grad()
            logits = probe(features)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            _, predicted = logits.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

            if batch_idx % 50 == 0:
                logger.info(f"  Epoch {epoch} [{batch_idx}/{len(train_loader)}] Loss: {loss.item():.4f}")

        train_acc = 100. * correct / total

        # Validate
        probe.eval()
        val_correct, val_total = 0, 0
        with torch.no_grad():
            for images, labels in val_loader:
                labels = labels.to(device)
                features = clip_extractor.extract(images)
                logits = probe(features)
                _, predicted = logits.max(1)
                val_total += labels.size(0)
                val_correct += predicted.eq(labels).sum().item()

        val_acc = 100. * val_correct / val_total

        logger.info(f"Epoch {epoch}: Train Acc: {train_acc:.2f}% | Val Acc: {val_acc:.2f}%")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            save_path = os.path.join(args.output_dir, "clip_probe_best.pth")
            torch.save(probe.state_dict(), save_path)
            logger.info(f"✓ Saved best probe (val_acc={val_acc:.2f}%) → {save_path}")

    logger.info(f"\nTraining complete. Best val accuracy: {best_val_acc:.2f}%")


if __name__ == "__main__":
    main()
