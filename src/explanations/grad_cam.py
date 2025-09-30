import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image


def plot_grad_cam_heatmap(model, target_layers, img_path):
    """Generates and displays a Grad-CAM heatmap overlayed on the original image."""
    # Load image
    img = Image.open(img_path).convert("RGB")
    rgb_img = np.array(img).astype(np.float32) / 255.0  # Original size normalized [0,1]

    # Preprocess for model
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])
    input_tensor = transform(img).unsqueeze(0)  # Add batch dimension

    # Grad-CAM
    cam = GradCAM(model=model, target_layers=target_layers)
    grayscale_cam = cam(input_tensor=input_tensor, targets=None)  # shape: [1,224,224]
    grayscale_cam = grayscale_cam[0]

    # Resize CAM back to original image size using PIL
    cam_pil = Image.fromarray((grayscale_cam * 255).astype(np.uint8))
    cam_resized = cam_pil.resize((rgb_img.shape[1], rgb_img.shape[0]), Image.BILINEAR)
    grayscale_cam_resized = np.array(cam_resized).astype(np.float32) / 255.0

    # Overlay heatmap
    visualization = show_cam_on_image(rgb_img, grayscale_cam_resized, use_rgb=True)

    # Plot
    plt.figure(figsize=(8, 8))
    plt.imshow(visualization)
    plt.axis("off")
    plt.show()