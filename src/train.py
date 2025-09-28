import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader, Subset
import numpy as np

def get_data_loaders(data_dir, batch_size, num_workers=4, val_split=0.2):
    """
    Creates training and validation dataloaders with transforms and random split.
    """
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    full_dataset = datasets.ImageFolder(data_dir, transform=transform)
    num_classes = len(full_dataset.classes)

    # Build indices per class
    targets = np.array([s[1] for s in full_dataset.samples])
    train_indices = []
    val_indices = []

    for class_idx in range(num_classes):
        class_indices = np.where(targets == class_idx)[0]
        np.random.shuffle(class_indices)
        split = int(len(class_indices) * (1 - val_split))
        train_indices.extend(class_indices[:split])
        val_indices.extend(class_indices[split:])

    train_dataset = Subset(full_dataset, train_indices)
    val_dataset = Subset(full_dataset, val_indices)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, val_loader, num_classes


def build_model(num_classes, pretrained=True):
    """
    Loads a pretrained ResNet50 and adapts it for classification.
    """
    model = models.resnet50(pretrained=pretrained)
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    return model


def train_one_epoch(model, loader, criterion, optimizer, device, epoch, writer):
    """
    Train loop for one epoch.
    """
    model.train()
    running_loss, correct, total = 0.0, 0, 0

    for i, (inputs, labels) in enumerate(loader):
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

        if i % 50 == 0:
            print(f"Epoch {epoch} | Step {i}/{len(loader)} | Loss: {loss.item():.4f}")

    epoch_loss = running_loss / len(loader)
    epoch_acc = 100. * correct / total
    writer.add_scalar("Train/Loss", epoch_loss, epoch)
    writer.add_scalar("Train/Accuracy", epoch_acc, epoch)

    return epoch_loss, epoch_acc


def validate(model, loader, criterion, device, epoch, writer):
    """
    Validation loop.
    """
    model.eval()
    running_loss, correct, total = 0.0, 0, 0

    with torch.no_grad():
        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

    epoch_loss = running_loss / len(loader)
    epoch_acc = 100. * correct / total
    writer.add_scalar("Val/Loss", epoch_loss, epoch)
    writer.add_scalar("Val/Accuracy", epoch_acc, epoch)

    print(f"Validation | Loss: {epoch_loss:.4f} | Acc: {epoch_acc:.2f}%")

    return epoch_loss, epoch_acc


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    writer = SummaryWriter(log_dir=args.log_dir)

    train_loader, val_loader, num_classes = get_data_loaders(args.data_dir, args.batch_size, args.num_workers)
    model = build_model(num_classes, pretrained=True).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    best_acc = 0.0
    os.makedirs(args.checkpoint_dir, exist_ok=True)

    for epoch in range(args.epochs):
        print(f"\n--- Epoch {epoch+1}/{args.epochs} ---")

        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device, epoch, writer)
        val_loss, val_acc = validate(model, val_loader, criterion, device, epoch, writer)

        # Save best model
        if val_acc > best_acc:
            best_acc = val_acc
            ckpt_path = os.path.join(args.checkpoint_dir, "best_model.pth")
            torch.save(model.state_dict(), ckpt_path)
            print(f"✅ Saved new best model at {ckpt_path} (Val Acc: {best_acc:.2f}%)")

    writer.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Food-101 Classifier")
    parser.add_argument("--data_dir", type=str, default="dataset/food-101/images", help="Path to dataset")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints")
    parser.add_argument("--log_dir", type=str, default="runs")

    args = parser.parse_args()
    main(args)