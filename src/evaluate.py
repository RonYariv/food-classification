import argparse
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

from datasets import get_dataloaders
from models import build_resnet50
from utils import plot_per_class_accuracy, forward_step


def evaluate(model, loader, criterion, device, class_names=None, save_plot_path="per_class_accuracy.png"):
    """
    Evaluate model on validation set, return loss, Top-1/Top-5 accuracy, and per-class accuracy.
    """
    model.eval()
    running_loss, total = 0.0, 0
    correct_top1, correct_top5 = 0, 0

    # Per-class accuracy
    class_correct = torch.zeros(len(class_names))
    class_total = torch.zeros(len(class_names))

    with torch.no_grad():
        for inputs, labels in loader:
            outputs = model(inputs)

            loss, top1, top5, preds = forward_step(model, inputs, labels, criterion, device)
            correct_top1 += (top1 / 100.0) * labels.size(0)
            correct_top5 += (top5 / 100.0) * labels.size(0)

            total += labels.size(0)

            # Update per-class accuracy
            _, pred_top1 = outputs.topk(1, dim=1)
            for i in range(labels.size(0)):
                label = labels[i].item()
                class_total[label] += 1
                if pred_top1[i].item() == label:
                    class_correct[label] += 1

    val_loss = running_loss / len(loader)
    top1_acc = 100.0 * correct_top1 / total
    top5_acc = 100.0 * correct_top5 / total
    class_acc = 100.0 * class_correct / class_total

    print(f"Validation Loss: {val_loss:.4f} | Top-1 Acc: {top1_acc:.2f}% | Top-5 Acc: {top5_acc:.2f}%")

    # Use utils function to plot per-class accuracy
    plot_per_class_accuracy(class_acc, class_names, save_path=save_plot_path)

    return val_loss, top1_acc, top5_acc, class_acc


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    writer = SummaryWriter(log_dir=args.log_dir)

    # Load dataloader
    val_loader, _, class_names = get_dataloaders(
        args.data_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        val_split=0.0  # Use full dataset as validation
    )[:3]

    # Load model
    num_classes = len(class_names)
    model = build_resnet50(num_classes, checkpoint_path=args.checkpoint, device=device)

    criterion = nn.CrossEntropyLoss()
    val_loss, top1_acc, top5_acc, class_acc = evaluate(model, val_loader, criterion, device, class_names)

    # Log metrics
    writer.add_scalar("Val/Loss", val_loss, 0)
    writer.add_scalar("Val/Top1_Accuracy", top1_acc, 0)
    writer.add_scalar("Val/Top5_Accuracy", top5_acc, 0)
    writer.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Food-101 Classifier")
    parser.add_argument("--data_dir", type=str, default="dataset/food-101/images", help="Path to dataset")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--checkpoint", type=str, default="checkpoints/best_model.pth", help="Path to model checkpoint")
    parser.add_argument("--log_dir", type=str, default="runs/eval")

    args = parser.parse_args()
    main(args)