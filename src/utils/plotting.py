import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import os
import pandas as pd
import numpy as np

from src import config


def plot_top_confusions(cm: np.ndarray, class_names: list, top_k: int = 20):
    """
    Plot the top-k confusions from a confusion matrix.
    Each bar shows the number of times a true class was confused with a predicted class.
    """
    import matplotlib.pyplot as plt
    import numpy as np
    import os
    from src import config

    # Copy and zero out diagonal (we only want misclassifications)
    cm = cm.copy()
    np.fill_diagonal(cm, 0)

    # Collect all confusions: (true_class, predicted_class, count)
    confusions = []
    num_classes = len(class_names)
    for i in range(num_classes):
        for j in range(num_classes):
            if cm[i, j] > 0:
                confusions.append((class_names[i], class_names[j], cm[i, j]))

    # Sort by count descending and take top_k
    confusions.sort(key=lambda x: x[2], reverse=True)
    confusions = confusions[:top_k]

    if not confusions:
        print("No confusions to display!")
        return

    # Unpack lists
    true_labels, pred_labels, counts = zip(*confusions)

    # Plot
    plt.figure(figsize=(10, 6))
    cmap = plt.get_cmap("tab20")
    colors = [cmap(i) for i in range(cmap.N)]  # get discrete colors

    # Create horizontal bars
    for i, count in enumerate(counts):
        plt.barh(i, count, color=colors[i % len(colors)])

    # Set y-axis labels as "True → Predicted"
    plt.yticks(range(len(confusions)), [f"{t} → {p}" for t, p in zip(true_labels, pred_labels)])
    plt.xlabel("Count")
    plt.title(f"Top {len(confusions)} Confusions")
    plt.gca().invert_yaxis()  # highest confusion on top

    # Force x-axis to show only integer ticks
    plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))

    plt.tight_layout()
    os.makedirs(config.REPORTS_DIR, exist_ok=True)
    plt.savefig(os.path.join(config.REPORTS_DIR, "confusion_matrix.png"))
    plt.close()

    print(f"Saved confusion plot to {config.REPORTS_DIR}")

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

    epochs = list(range(1, len(train_losses) + 1))

    # Loss
    plt.figure(figsize=(10, 5))
    plt.plot(epochs, train_losses, label="Train Loss")
    plt.plot(epochs, val_losses, label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training & Validation Loss")
    plt.xticks(epochs)  # set x-axis ticks as natural numbers
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(config.REPORTS_DIR, "loss_curve.png"))
    plt.close()

    # Accuracy
    plt.figure(figsize=(10, 5))
    plt.plot(epochs, train_accs, label="Train Accuracy")
    plt.plot(epochs, val_accs, label="Validation Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy (%)")
    plt.title("Training & Validation Accuracy")
    plt.xticks(epochs)  # set x-axis ticks as natural numbers
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(config.REPORTS_DIR, "accuracy_curve.png"))
    plt.close()

    print(f"Saved training curves to {config.REPORTS_DIR}")