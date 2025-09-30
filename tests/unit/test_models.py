import torch
import pytest
from src.models.models import load_resnet_model

@pytest.fixture
def dummy_input():
    # batch size 2, 3 channels, 224x224
    return torch.randn(2, 3, 224, 224)

@pytest.fixture
def model():
    # Load small ResNet model with 2 classes (dummy)
    num_classes = 2
    # checkpoint_path=None to avoid loading weights
    return load_resnet_model(depth=18, num_classes=num_classes, checkpoint_path=None, device="cpu")

def test_forward_pass(model, dummy_input):
    model.eval()
    with torch.no_grad():
        outputs = model(dummy_input)
    assert outputs.shape[0] == dummy_input.shape[0]  # batch size matches
    assert outputs.shape[1] == 2  # number of classes