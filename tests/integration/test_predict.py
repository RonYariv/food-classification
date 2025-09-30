import torch
import pytest
from PIL import Image
from src.utils.general import preprocess_image, get_class_names
from src.models.models import load_resnet_model

@pytest.fixture
def dummy_image(tmp_path):
    # create a dummy RGB image
    img_path = tmp_path / "img.jpg"
    Image.new("RGB", (224, 224), color=(255, 0, 0)).save(img_path)
    return str(img_path)

@pytest.fixture
def dummy_model():
    model = load_resnet_model(depth=18, num_classes=2, checkpoint_path=None, device="cpu")
    return model

def test_single_image_prediction(dummy_model, dummy_image):
    dummy_model.eval()
    img_tensor = preprocess_image(dummy_image)  # shape [1,3,224,224]
    with torch.no_grad():
        outputs = dummy_model(img_tensor)
    # top-1 prediction
    _, preds = outputs.topk(1, dim=1)
    assert preds.shape == (1, 1)
