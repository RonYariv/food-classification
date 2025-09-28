import matplotlib.pyplot as plt
import os
import pandas as pd

from src import config


def plot_metrics_table(df_metrics: pd.DataFrame):
    """
    Plot a table of metrics (accuracy, precision, recall, F1).
    """
    fig, ax = plt.subplots(figsize=(12, max(6, int(len(df_metrics)*0.3))))
    ax.axis("off")

    table = ax.table(
        cellText=df_metrics.round(3).values,
        colLabels=df_metrics.columns,
        cellLoc="center",
        loc="center"
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.5)
    plt.tight_layout()
    plt.savefig(f"{config.REPORTS_DIR}/metrics_table.png")
    plt.close()
    print(f"Saved metrics table plot to {config.REPORTS_DIR}")

def plot_training_curves(train_losses, val_losses, train_accs, val_accs):
    os.makedirs(config.REPORTS_DIR, exist_ok=True)

    # Loss
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, len(train_losses) + 1), train_losses, label="Train Loss")
    plt.plot(range(1, len(val_losses) + 1), val_losses, label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training & Validation Loss")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(config.REPORTS_DIR, "loss_curve.png"))
    plt.close()

    # Accuracy
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, len(train_accs) + 1), train_accs, label="Train Accuracy")
    plt.plot(range(1, len(val_accs) + 1), val_accs, label="Validation Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy (%)")
    plt.title("Training & Validation Accuracy")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(config.REPORTS_DIR, "accuracy_curve.png"))
    plt.close()

    print(f"Saved training curves to {config.REPORTS_DIR}")