import torch
from src.models import load_resnet_model
from src.utils import preprocess_image, get_class_names
from src import config

# Load model
class_names = get_class_names()
NUM_CLASSES = len(class_names)
MODEL_PATH = f"{config.CHECKPOINT_DIR}/best_model.pth"
model = load_resnet_model(config.RESNET_DEPTH, num_classes=NUM_CLASSES, checkpoint_path=MODEL_PATH)
model.eval()


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
