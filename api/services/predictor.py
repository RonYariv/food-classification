import torch
from src.models import load_resnet_model
from src.utils import preprocess_image, get_class_names

# Load model
NUM_CLASSES = 101
MODEL_PATH = "checkpoints/best_model.pth"
model = load_resnet_model(18, num_classes=NUM_CLASSES, checkpoint_path=MODEL_PATH)
model.eval()

# Load class names
class_names = get_class_names("datasets/food-101/images")

def predict_image(image_path):
    input_tensor = preprocess_image(image_path)
    with torch.no_grad():
        output = model(input_tensor)
        probs = torch.softmax(output, dim=1)
        probability, class_id = torch.max(probs, dim=1)
        class_id = class_id.item()
        probability = probability.item()
        class_name = class_names[class_id]

    return class_id, class_name, probability
