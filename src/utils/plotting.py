import matplotlib.pyplot as plt
import os
import config


def plot_per_class_accuracy(class_acc, class_names):
    plt.figure(figsize=(20,6))
    plt.bar(range(len(class_names)), class_acc)
    plt.xticks(range(len(class_names)), class_names, rotation=90)
    plt.ylabel("Accuracy (%)")
    plt.title("Per-Class Accuracy")
    plt.tight_layout()
    plt.savefig(os.path.join(config.REPORTS_DIR, "per_class_accuracy.png"))
    plt.close()
    print(f"Saved per-class accuracy plot: {config.REPORTS_DIR}")

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