import torch
import numpy as np
import matplotlib.pyplot as plt
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image, preprocess_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from PIL import Image

from src.models import load_resnet_model
from src.utils import get_class_names, preprocess_image
import src.config as config


def plot_grad_cam_heatmap(img_path: str, output_path: str, checkpoint_path: str, target_class: int = None):
    """Generate and save a Grad-CAM heatmap for an input image."""
    # Load model
    classes_num = len(get_class_names())
    model = load_resnet_model(config.RESNET_DEPTH, classes_num, checkpoint_path=checkpoint_path)
    model.eval()

    # Load original image for visualization
    img = Image.open(img_path).convert("RGB")
    rgb_img = np.array(img).astype(np.float32) / 255.0  # (H,W,3), normalized 0-1

    input_tensor = preprocess_image(img_path)

    # Decide target class
    if target_class is None:
        with torch.no_grad():
            outputs = model(input_tensor)
            target_class = int(outputs.argmax(dim=1).cpu().numpy()[0])

    # Pick target layer (last conv block for ResNet)
    target_layer = model.layer4[-1]

    # Create Grad-CAM object
    cam = GradCAM(model=model, target_layers=[target_layer], use_cuda=torch.cuda.is_available())

    # Run CAM
    grayscale_cam = cam(input_tensor=input_tensor, targets=[ClassifierOutputTarget(target_class)])
    grayscale_cam = grayscale_cam[0, :]  # (H, W)

    # Overlay heatmap on image
    visualization = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)

    # Save as PNG
    plt.imsave(output_path, visualization)
    print(f"[Grad-CAM] Saved heatmap to {output_path}")
