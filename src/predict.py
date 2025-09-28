import os
import torch
from torchvision import transforms
from PIL import Image
from models import load_resnet_model
import argparse

def load_class_names(train_dir):
    """Load class names from train folder structure."""
    from torchvision.datasets import ImageFolder
    dataset = ImageFolder(train_dir)
    return dataset.classes


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


def predict(model, image_tensor, class_names, device="cuda", topk=5):
    """Predict the top-k classes for the input image."""
    model.eval()
    image_tensor = image_tensor.to(device)

    with torch.no_grad():
        outputs = model(image_tensor)
        probs = torch.softmax(outputs, dim=1)
        top_probs, top_idxs = probs.topk(topk)
        top_probs = top_probs.cpu().numpy()[0]
        top_idxs = top_idxs.cpu().numpy()[0]
        top_classes = [class_names[i] for i in top_idxs]

    return list(zip(top_classes, top_probs))


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load class names
    class_names = load_class_names(args.data_dir)

    # Load model
    num_classes = len(class_names)
    model = load_resnet_model(depth=args.depth, num_classes=num_classes,
                             checkpoint_path=args.checkpoint, device=device)

    # Collect all image paths
    image_paths = [os.path.join(args.image_dir, f) for f in os.listdir(args.image_dir)
                   if f.lower().endswith((".png", ".jpg", ".jpeg"))]
    image_paths.sort()  # consistent order

    results = []

    for img_path in image_paths:
        image_tensor = preprocess_image(img_path)
        preds = predict(model, image_tensor, class_names, device=device, topk=args.topk)
        # Store result as dict
        results.append({
            "image": os.path.basename(img_path),
            "predictions": preds
        })
        print(f"Processed {img_path}: {preds[0][0]} ({preds[0][1] * 100:.2f}%)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predict multiple images using trained Food-101 model")
    parser.add_argument("--image_dir", type=str, required=True, help="Directory with images to predict")
    parser.add_argument("--data_dir", type=str, default="dataset/food-101/images",
                        help="Path to dataset train folder (for class names)")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/best_model.pth", help="Path to model checkpoint")
    parser.add_argument("--depth", type=int, default=18, help="ResNet depth")
    parser.add_argument("--topk", type=int, default=5, help="Top-K predictions to show")

    args = parser.parse_args()
    main(args)
