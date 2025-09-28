import matplotlib.pyplot as plt

def compute_accuracy(outputs, labels, topk=(1,5)):
    """Compute Top-k accuracy"""
    maxk = max(topk)
    batch_size = labels.size(0)
    _, pred = outputs.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(labels.view(1, -1).expand_as(pred))
    res = []
    for k in topk:
        res.append(correct[:k].reshape(-1).float().sum(0).item() / batch_size * 100.0)
    return res  # [top1, top5]

def plot_per_class_accuracy(class_acc, class_names, save_path="per_class_accuracy.png"):
    plt.figure(figsize=(20,6))
    plt.bar(range(len(class_names)), class_acc)
    plt.xticks(range(len(class_names)), class_names, rotation=90)
    plt.ylabel("Accuracy (%)")
    plt.title("Per-Class Accuracy")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"Saved per-class accuracy plot: {save_path}")

def forward_step(model, inputs, labels, criterion, device):
    """
    Performs a forward pass and computes loss and top-1/top-5 accuracy.

    Returns:
        loss: torch scalar
        top1: float (%)
        top5: float (%)
        preds: torch.Tensor (predicted top-1 class indices)
    """
    inputs, labels = inputs.to(device), labels.to(device)
    outputs = model(inputs)
    loss = criterion(outputs, labels)
    top1, top5 = compute_accuracy(outputs, labels)
    _, preds = outputs.topk(1, dim=1)
    return loss, top1, top5, preds
