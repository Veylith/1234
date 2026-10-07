"""
Training Script — CDCN Liveness Detector
Fine-tunes the Central Difference Convolutional Network on FAS datasets.

Datasets:
  - OULU-NPU, SiW, CASIA-MFSD, Replay-Attack
  - Expected structure:
      data/
        train/
          live/   (live face crops)
          spoof/  (spoof face crops — printed, screen, mask)
        val/
          live/
          spoof/

  For depth map supervision, provide grayscale depth maps alongside RGB images:
      data/
        train/
          live/
            img_001.jpg
            img_001_depth.png  (PRNet-generated pseudo depth map)
          spoof/
            img_001.jpg
            img_001_depth.png  (all-zero / flat depth map)

Usage:
    python train_liveness.py --data_dir ./data --epochs 30 --batch_size 16
"""

import argparse
import os
import logging

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image

# Import CDCN model from our codebase
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "backend"))
from models.liveness_detector import CDCNModel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("train_liveness")


class FASDataset(Dataset):
    """
    Face Anti-Spoofing dataset with optional depth map supervision.

    Each sample is a face crop with a corresponding depth map label.
    Live faces have 3D depth structure; spoof faces have flat/zero depth.
    """

    def __init__(self, root_dir, is_train=True):
        self.root_dir = root_dir
        self.is_train = is_train
        self.samples = []

        # Collect samples from live/ and spoof/ subdirectories
        for label_name, label_id in [("live", 0), ("spoof", 1)]:
            dir_path = os.path.join(root_dir, label_name)
            if not os.path.isdir(dir_path):
                continue
            for fname in os.listdir(dir_path):
                if fname.endswith(("_depth.png", "_depth.jpg")):
                    continue
                if fname.lower().endswith((".jpg", ".png", ".jpeg", ".webp")):
                    img_path = os.path.join(dir_path, fname)
                    # Look for depth map
                    base = os.path.splitext(fname)[0]
                    depth_path = os.path.join(dir_path, f"{base}_depth.png")
                    if not os.path.exists(depth_path):
                        depth_path = None
                    self.samples.append((img_path, depth_path, label_id))

        logger.info(f"{'Train' if is_train else 'Val'} dataset: {len(self.samples)} samples")

        # Transforms
        self.img_transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.RandomHorizontalFlip() if is_train else transforms.Lambda(lambda x: x),
            transforms.ColorJitter(0.1, 0.1) if is_train else transforms.Lambda(lambda x: x),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ])

        self.depth_transform = transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
        ])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, depth_path, label = self.samples[idx]

        # Load image
        img = Image.open(img_path).convert("RGB")
        img = self.img_transform(img)

        # Load or generate depth map
        if depth_path and os.path.exists(depth_path):
            depth = Image.open(depth_path).convert("L")
            depth = self.depth_transform(depth)
        else:
            # Generate pseudo depth: live = ones (has depth), spoof = zeros (flat)
            if label == 0:  # live
                depth = torch.ones(1, 32, 32)
            else:  # spoof
                depth = torch.zeros(1, 32, 32)

        return img, depth, label


def main():
    parser = argparse.ArgumentParser(description="Train CDCN liveness detector")
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--output_dir", type=str, default="./checkpoints")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.output_dir, exist_ok=True)

    # Model
    model = CDCNModel()
    model.to(device)
    logger.info("CDCN model initialized (random weights).")

    # Data
    train_dataset = FASDataset(os.path.join(args.data_dir, "train"), is_train=True)
    val_dataset = FASDataset(os.path.join(args.data_dir, "val"), is_train=False)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)

    # Loss: MSE on depth map + binary cross-entropy on aggregated score
    mse_criterion = nn.MSELoss()
    bce_criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

    best_val_loss = float("inf")
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0

        for images, depths, labels in train_loader:
            images = images.to(device)
            depths = depths.to(device)
            labels = labels.float().to(device)

            optimizer.zero_grad()
            pred_depth = model(images)

            # Depth map supervision loss
            depth_loss = mse_criterion(pred_depth, depths)

            # Binary classification loss (aggregated depth mean)
            pred_score = 1.0 - pred_depth.mean(dim=[1, 2, 3])
            bce_loss = bce_criterion(pred_score, labels)

            loss = depth_loss + 0.5 * bce_loss
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        scheduler.step()
        avg_train_loss = total_loss / len(train_loader)

        # Validate
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for images, depths, labels in val_loader:
                images = images.to(device)
                depths = depths.to(device)
                labels = labels.to(device)

                pred_depth = model(images)
                loss = mse_criterion(pred_depth, depths)
                val_loss += loss.item()

                pred_labels = (1.0 - pred_depth.mean(dim=[1, 2, 3]) > 0.5).long()
                correct += (pred_labels == labels).sum().item()
                total += labels.size(0)

        avg_val_loss = val_loss / len(val_loader)
        val_acc = 100. * correct / total

        logger.info(f"Epoch {epoch}: Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | Val Acc: {val_acc:.2f}%")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), os.path.join(args.output_dir, "cdcn_best.pth"))
            logger.info(f"✓ Saved best model → cdcn_best.pth")


if __name__ == "__main__":
    main()
