import os
import tempfile
import torch
from PIL import Image
import pytest

from src.data_loaders.data_loaders import get_dataloaders

# Helper to create dummy images
def create_dummy_image(path, size=(224, 224)):
    img = Image.new("RGB", size, color=(255, 0, 0))
    img.save(path)

# Fixture to create temporary dataset
@pytest.fixture
def temp_dataset():
    with tempfile.TemporaryDirectory() as tmpdir:
        for split in ["train", "val", "test"]:
            split_dir = os.path.join(tmpdir, split)
            os.makedirs(split_dir)
            # create two classes
            for cls in ["class1", "class2"]:
                cls_dir = os.path.join(split_dir, cls)
                os.makedirs(cls_dir)
                # add one dummy image per class
                create_dummy_image(os.path.join(cls_dir, "img1.jpg"))
        yield tmpdir  # provide the path to the test

def test_dataloaders_shapes(temp_dataset):
    batch_size = 2
    num_workers = 0  # keep CI simple

    train_loader, val_loader, test_loader, num_classes, class_names = get_dataloaders(
        temp_dataset, batch_size=batch_size, num_workers=num_workers
    )

    # Check number of classes
    assert num_classes == 2
    assert sorted(class_names) == ["class1", "class2"]

    # Check loader types
    from torch.utils.data import DataLoader
    assert isinstance(train_loader, DataLoader)
    assert isinstance(val_loader, DataLoader)
    assert isinstance(test_loader, DataLoader)

    # Check batch shapes (one batch per loader is enough)
    for loader in [train_loader, val_loader, test_loader]:
        inputs, labels = next(iter(loader))
        assert isinstance(inputs, torch.Tensor)
        assert isinstance(labels, torch.Tensor)
        # input shape [B, C, H, W]
        assert inputs.shape[0] <= batch_size
        assert inputs.shape[1] == 3
        assert inputs.shape[2] == 224
        assert inputs.shape[3] == 224
        # labels shape [B]
        assert labels.shape[0] <= batch_size