import os
import torch
import argparse

from src.models import load_resnet_model
from src import config
from src.utils import preprocess_image, get_class_names
from src.explanations import plot_grad_cam_heatmap


def predict(model, image_tensor, class_names, device="cuda"):
    """Predict the top-1 class for the input image."""
    model.eval()
    image_tensor = image_tensor.to(device)

    with torch.no_grad():
        outputs = model(image_tensor)
        probs = torch.softmax(outputs, dim=1)
        top_idx = probs.argmax(dim=1).item()
        top_class = class_names[top_idx]
        top_prob = probs[0, top_idx].item()

    return top_class, top_prob, top_idx


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load class names
    class_names = get_class_names()

    # Load model
    num_classes = len(class_names)
    model = load_resnet_model(
        depth=config.RESNET_DEPTH,
        num_classes=num_classes,
        checkpoint_path=args.checkpoint,
        device=device
    )

    # Collect all image paths
    image_paths = [
        os.path.join(args.image_dir, f)
        for f in os.listdir(args.image_dir)
        if f.lower().endswith((".png", ".jpg", ".jpeg"))
    ]
    image_paths.sort()  # consistent order

    results = []

    # Ensure Grad-CAM folder exists
    if args.explain:
        os.makedirs(args.gradcam_dir, exist_ok=True)

    for img_path in image_paths:
        image_tensor = preprocess_image(img_path)
        top_class, top_prob, top_idx = predict(model, image_tensor, class_names, device=device)
        results.append({
            "image": os.path.basename(img_path),
            "prediction": top_class,
            "probability": top_prob
        })
        print(f"Processed {img_path}: {top_class} ({top_prob * 100:.2f}%)")

        if args.explain:
            output_path = os.path.join(args.gradcam_dir, f"{os.path.basename(img_path)}_gradcam.png")
            plot_grad_cam_heatmap(
                img_path=img_path,
                output_path=output_path,
                checkpoint_path=args.checkpoint,
                target_class=top_idx
            )

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predict multiple images using trained model")
    parser.add_argument("--checkpoint", type=str, default=f"{config.CHECKPOINT_DIR}/best_model.pth",
                        help="Path to model checkpoint")
    parser.add_argument("--image-dir", type=str, default="inference_images",
                        help="Path to directory containing images")
    parser.add_argument("--explain",type=bool, default=False,
                        help="Generate Grad-CAM heatmaps for predictions")
    parser.add_argument("--gradcam-dir", type=str, default=f"{config.INFERENCE_IMAGES_DIR}/explanations",
                        help="Directory to save Grad-CAM heatmaps")

    args = parser.parse_args()
    main(args)