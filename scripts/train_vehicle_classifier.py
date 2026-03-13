from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch import nn, optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, models, transforms


def train(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tfm = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    dataset = datasets.ImageFolder(args.data_dir, transform=tfm)
    class_names = dataset.classes
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_ds, val_ds = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=2)

    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    model.fc = nn.Linear(model.fc.in_features, len(class_names))
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    best_acc = 0.0
    for epoch in range(args.epochs):
        model.train()
        running_loss = 0.0
        for X, y in train_loader:
            X, y = X.to(device), y.to(device)
            optimizer.zero_grad()
            logits = model(X)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for X, y in val_loader:
                X, y = X.to(device), y.to(device)
                pred = model(X).argmax(dim=1)
                correct += (pred == y).sum().item()
                total += y.size(0)

        acc = correct / max(total, 1)
        print(f"epoch={epoch+1} loss={running_loss/ max(len(train_loader),1):.4f} val_acc={acc:.4f}")

        if acc > best_acc:
            best_acc = acc
            out = Path(args.output)
            out.parent.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), out)
            with out.parent.joinpath("vehicle_classifier_labels.json").open("w", encoding="utf-8") as f:
                json.dump(class_names, f)

    print(f"Best validation accuracy: {best_acc:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--output", default="models/vehicle_classifier.pt")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    train(parser.parse_args())
