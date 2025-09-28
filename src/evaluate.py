import argparse
import torch
import torch.nn as nn
import pandas as pd
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix

from src.data_loaders import get_dataloaders
from src.models import load_resnet_model
from src.utils import plot_metrics_table, plot_top_confusions
from src import config

def evaluate(model, loader, criterion, device, class_names=None, save_csv="metrics.csv"):
    """
    Evaluate model on a dataset, compute per-class metrics, save CSV, and plot table sorted by F1.
    """
    model.eval()
    all_labels = []
    all_preds = []
    running_loss = 0.0

    with torch.no_grad():
        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            running_loss += loss.item()
            _, preds = outputs.topk(1, dim=1)

            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy().flatten())

    val_loss = running_loss / len(loader)

    # Compute metrics per class
    metrics = {"Class": [], "Accuracy": [], "Precision": [], "Recall": [], "F1": []}

    for i, cls in enumerate(class_names):
        # Select only samples belonging to this class
        cls_indices = [j for j, label in enumerate(all_labels) if label == i]
        if not cls_indices:  # skip if no samples of this class
            continue

        cls_labels = [all_labels[j] for j in cls_indices]
        cls_preds  = [all_preds[j] for j in cls_indices]

        acc = sum([p == l for p, l in zip(cls_preds, cls_labels)]) / len(cls_labels) * 100
        prec = precision_score(cls_labels, cls_preds, labels=[i], average="macro", zero_division=0)
        rec  = recall_score(cls_labels, cls_preds, labels=[i], average="macro", zero_division=0)
        f1   = f1_score(cls_labels, cls_preds, labels=[i], average="macro", zero_division=0)

        metrics["Class"].append(cls)
        metrics["Accuracy"].append(acc)
        metrics["Precision"].append(prec)
        metrics["Recall"].append(rec)
        metrics["F1"].append(f1)

    df_metrics = pd.DataFrame(metrics)
    df_metrics = df_metrics.sort_values(by="F1", ascending=False).reset_index(drop=True)

    # Save CSV
    df_metrics.to_csv(save_csv, index=False)
    print(f"Saved per-class metrics to {save_csv}")

    # Plot metrics table
    plot_metrics_table(df_metrics)

    # Plot confusion matrix
    cm = confusion_matrix(all_labels, all_preds, labels=range(len(class_names)))
    plot_top_confusions(cm, class_names, top_k=20)

    # Overall accuracy
    overall_acc = sum([p == l for p, l in zip(all_preds, all_labels)]) / len(all_labels) * 100
    print(f"Validation Loss: {val_loss:.4f} | Overall Accuracy: {overall_acc:.2f}%")

    return val_loss, overall_acc, df_metrics

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load test dataloader
    _, _, test_loader, num_classes, class_names = get_dataloaders(
        config.FOOD101_SPLIT_DIR, batch_size=args.batch_size, num_workers=args.num_workers
    )

    # Load model
    model = load_resnet_model(config.RESNET_DEPTH, num_classes, checkpoint_path=args.checkpoint, device=device)

    criterion = nn.CrossEntropyLoss()
    val_loss, overall_acc, df_metrics = evaluate(
        model, test_loader, criterion, device, class_names,
        save_csv=args.save_csv
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Food-101 Classifier")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--checkpoint", type=str, default=f"{config.CHECKPOINT_DIR}/best_model.pth", help="Path to model checkpoint")
    parser.add_argument("--save_csv", type=str, default=f"{config.REPORTS_DIR}/metrics.csv", help="CSV file to save per-class metrics")

    args = parser.parse_args()
    main(args)