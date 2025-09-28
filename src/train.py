import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

from src.data_loaders import get_dataloaders
from src.models import load_resnet_model
from src.utils import compute_accuracy, compute_batch_metrics, plot_training_curves
from src import config


def train_one_epoch(model, loader, criterion, optimizer, device, epoch, writer):
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
        top1, _ = compute_accuracy(outputs, labels)
        correct += (top1 / 100.0) * labels.size(0)  # accumulate correct predictions
        total += labels.size(0)

        if i % 50 == 0:
            print(f"Epoch {epoch} | Step {i}/{len(loader)} | Loss: {loss.item():.4f}")

    epoch_loss = running_loss / len(loader)
    epoch_acc = 100.0 * correct / total
    writer.add_scalar("Train/Loss", epoch_loss, epoch)
    writer.add_scalar("Train/Accuracy", epoch_acc, epoch)
    return epoch_loss, epoch_acc


def validate(model, loader, criterion, device, epoch, writer):
    model.eval()
    running_loss, correct, total = 0.0, 0, 0

    with torch.no_grad():
        for inputs, labels in loader:
            loss, top1, top5, preds = compute_batch_metrics(model, inputs, labels, criterion, device)
            correct += (top1 / 100.0) * labels.size(0)
            total += labels.size(0)

    epoch_loss = running_loss / len(loader)
    epoch_acc = 100.0 * correct / total
    writer.add_scalar("Val/Loss", epoch_loss, epoch)
    writer.add_scalar("Val/Accuracy", epoch_acc, epoch)
    print(f"Validation | Loss: {epoch_loss:.4f} | Acc: {epoch_acc:.2f}%")
    return epoch_loss, epoch_acc


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    writer = SummaryWriter(log_dir=config.LOG_DIR)

    # Load dataloaders
    train_loader, val_loader, test_loader, num_classes, class_names = get_dataloaders(
        config.FOOD101_SPLIT_DIR, batch_size=args.batch_size, num_workers=args.num_workers
    )

    # Load model
    model = load_resnet_model(18, num_classes,checkpoint_path=None, device=device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    best_acc = 0.0
    os.makedirs(config.CHECKPOINT_DIR, exist_ok=True)

    train_losses, val_losses = [], []
    train_accs, val_accs = [], []

    for epoch in range(args.epochs):
        epoch =+1
        print(f"\n--- Epoch {epoch}/{args.epochs} ---")
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device, epoch, writer)
        val_loss, val_acc = validate(model, val_loader, criterion, device, epoch, writer)
        train_losses.append(train_loss)
        train_accs.append(train_acc)
        val_losses.append(val_loss)
        val_accs.append(val_acc)

        # Save best model
        if val_acc > best_acc:
            best_acc = val_acc
            ckpt_path = os.path.join(config.CHECKPOINT_DIR, "best_model.pth")
            torch.save(model.state_dict(), ckpt_path)
            print(f"Saved new best model at {ckpt_path} (Val Acc: {best_acc:.2f}%)")

    # Final evaluation on TEST set using the best model ---
    print("\n=== Final Evaluation on Test Set ===")
    model.load_state_dict(torch.load(os.path.join(config.CHECKPOINT_DIR, "best_model.pth")))
    test_loss, test_acc = validate(model, test_loader, criterion, device, args.epochs, writer)
    print(f"Test Accuracy: {test_acc:.2f}%")

    plot_training_curves(train_losses, val_losses, train_accs, val_accs)

    writer.close()



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Food-101 Classifier")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--num_workers", type=int, default=8)

    args = parser.parse_args()
    main(args)