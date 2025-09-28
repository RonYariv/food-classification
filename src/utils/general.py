from torchvision import transforms
from PIL import Image

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

def compute_batch_metrics(model, inputs, labels, criterion, device):
    inputs, labels = inputs.to(device), labels.to(device)
    outputs = model(inputs)
    loss = criterion(outputs, labels)
    top1, top5 = compute_accuracy(outputs, labels)
    _, preds = outputs.topk(1, dim=1)
    return loss, top1, top5, preds

def preprocess_image(image_path):
    """Load and preprocess a single image for model prediction."""
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    image = Image.open(image_path).convert("RGB")
    return transform(image).unsqueeze(0)  # add batch dimension
