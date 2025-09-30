from torchvision import transforms
from PIL import Image
import os
from src import config

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


def get_transform():
    val_test_transform = transforms.Compose([
        transforms.Resize(256),  # keep aspect ratio
        transforms.CenterCrop(224),  # crop to 224x224
        transforms.ToTensor(),  # convert to tensor [C,H,W] and scale to [0,1]
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],  # pretrained ResNet normalization
            std=[0.229, 0.224, 0.225]
        )
    ])
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(224),  # random zooms/crops
        transforms.RandomHorizontalFlip(),  # simulate left/right variations
        transforms.ColorJitter(0.2, 0.2, 0.2, 0.1),  # brightness/contrast/saturation/hue
        transforms.RandomRotation(15),  # slight angle changes
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    return train_transform, val_test_transform


def preprocess_image(image_path):
    """Load and preprocess a single image for model prediction."""
    _, transform = get_transform()
    image = Image.open(image_path).convert("RGB")
    return transform(image).unsqueeze(0)  # add batch dimension


def get_class_names():
    classes_dir = os.path.join(config.SPLIT_DATA_DIR, "train")
    class_names = sorted(
        [
            folder_name
            for folder_name in os.listdir(classes_dir)
            if os.path.isdir(os.path.join(classes_dir, folder_name))
        ]
    )
    return class_names
