import os
import glob
import argparse

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import nibabel as nib
from sklearn.metrics import classification_report


class MRIDataset(Dataset):
    def __init__(self, data_root, split="train", image_size=128):
        """
        data_root: folder that contains ADNI/AD and ADNI/CN
        split: "train" or "test"
        image_size: output slice size (image_size x image_size)
        """
        self.image_size = image_size
        self.split = split

        ad_dir = os.path.join(data_root, "AD")
        cn_dir = os.path.join(data_root, "CN")

        ad_files = sorted(glob.glob(os.path.join(ad_dir, "*.nii")))
        cn_files = sorted(glob.glob(os.path.join(cn_dir, "*.nii")))

        assert len(ad_files) == 9, f"Expected 9 AD files, found {len(ad_files)}"
        assert len(cn_files) == 9, f"Expected 9 CN files, found {len(cn_files)}"

        # 6 for train, 3 for test in each class
        if split == "train":
            self.files = ad_files[:6] + cn_files[:6]
            self.labels = [1] * 6 + [0] * 6  # AD = 1, CN = 0
        elif split == "test":
            self.files = ad_files[6:] + cn_files[6:]
            self.labels = [1] * 3 + [0] * 3
        else:
            raise ValueError("split must be 'train' or 'test'")

    def __len__(self):
        return len(self.files)

    def _load_middle_slice(self, path):
        nii = nib.load(path)
        vol = nii.get_fdata()  # shape: H x W x D (usually)
        vol = np.asarray(vol, dtype=np.float32)

        # pick the middle slice along the last axis to get a 2D image
        mid_idx = vol.shape[2] // 2
        img = vol[:, :, mid_idx]  # shape H x W

        # normalize: z-score then min-max to [0, 1]
        img = (img - img.mean()) / (img.std() + 1e-8)
        img = img - img.min()
        img = img / (img.max() + 1e-8)

        # convert to tensor and resize to image_size x image_size
        img_tensor = torch.from_numpy(img).unsqueeze(0).unsqueeze(0)  # 1 x 1 x H x W
        img_resized = F.interpolate(
            img_tensor,
            size=(self.image_size, self.image_size),
            mode="bilinear",
            align_corners=False,
        )
        img_resized = img_resized.squeeze(0)  # 1 x H x W
        return img_resized  # channel x H x W

    def __getitem__(self, idx):
        path = self.files[idx]
        label = self.labels[idx]
        img = self._load_middle_slice(path)
        return img, label


class MLP(nn.Module):
    def __init__(self, image_size=128, hidden1=256, hidden2=64, num_classes=2, dropout=0.3):
        super().__init__()
        input_dim = image_size * image_size  # flatten 1 x H x W
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(input_dim, hidden1),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden1, hidden2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden2, num_classes),
        )

    def forward(self, x):
        return self.net(x)


def train_epoch(model, loader, optimizer, device):
    model.train()
    criterion = nn.CrossEntropyLoss()
    total_loss = 0.0
    correct = 0
    total = 0

    for imgs, labels in loader:
        imgs = imgs.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * imgs.size(0)
        _, preds = torch.max(outputs, 1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    avg_loss = total_loss / total
    acc = correct / total
    return avg_loss, acc


def evaluate_model(model, loader, device, compute_report=False):
    model.eval()
    criterion = nn.CrossEntropyLoss()
    total_loss = 0.0
    correct = 0
    total = 0

    all_labels = []
    all_preds = []

    with torch.no_grad():
        for imgs, labels in loader:
            imgs = imgs.to(device)
            labels = labels.to(device)

            outputs = model(imgs)
            loss = criterion(outputs, labels)

            total_loss += loss.item() * imgs.size(0)
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

            all_labels.extend(labels.cpu().numpy().tolist())
            all_preds.extend(preds.cpu().numpy().tolist())

    avg_loss = total_loss / total
    acc = correct / total

    report = None
    if compute_report:
        report = classification_report(
            all_labels,
            all_preds,
            target_names=["CN", "AD"],
            zero_division=0,
        )
    return avg_loss, acc, report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_root",
        type=str,
        required=True,
        help="Path to ADNI folder that contains ADNI/AD and ADNI/CN",
    )
    parser.add_argument("--image_size", type=int, default=128)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--hidden1", type=int, default=256)
    parser.add_argument("--hidden2", type=int, default=64)
    parser.add_argument("--dropout", type=float, default=0.3)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_dataset = MRIDataset(
        os.path.join(args.data_root, "ADNI"),
        split="train",
        image_size=args.image_size,
    )
    test_dataset = MRIDataset(
        os.path.join(args.data_root, "ADNI"),
        split="test",
        image_size=args.image_size,
    )

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=len(test_dataset), shuffle=False)

    model = MLP(
        image_size=args.image_size,
        hidden1=args.hidden1,
        hidden2=args.hidden2,
        dropout=args.dropout,
    )
    model = model.to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    print("Start training")
    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, device)
        test_loss, test_acc, _ = evaluate_model(
            model, test_loader, device, compute_report=False
        )

        print(
            f"Epoch {epoch:03d} "
            f"Train loss: {train_loss:.4f}, Train acc: {train_acc:.3f} "
            f"Test loss: {test_loss:.4f}, Test acc: {test_acc:.3f}"
        )

    # final evaluation and detailed report
    test_loss, test_acc, report = evaluate_model(
        model, test_loader, device, compute_report=True
    )
    print("Final test results")
    print(f"Test loss: {test_loss:.4f}, Test accuracy: {test_acc:.3f}")
    print("Classification report:")
    print(report)


if __name__ == "__main__":
    main()
