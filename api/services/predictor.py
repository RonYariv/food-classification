import torch
from src.models import load_resnet_model
from src.utils import preprocess_image

model = load_resnet_model(18, num_classes=101, checkpoint_path="checkpoints/best_model.pth")
model.eval()

def predict_image(image_path):
    input_tensor = preprocess_image(image_path)
    with torch.no_grad():
        output = model(input_tensor)
        predicted_class = output.argmax(dim=1).item()
    return predicted_class
