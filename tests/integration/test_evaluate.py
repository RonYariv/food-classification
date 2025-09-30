import torch
import pytest
from torch.utils.data import DataLoader, TensorDataset
from torch import nn
from src.evaluate import evaluate

@pytest.fixture
def dummy_dataset():
    # 4 samples, 3 channels, 224x224, 2 classes
    x = torch.randn(4, 3, 224, 224)
    y = torch.tensor([0, 1, 0, 1])
    dataset = TensorDataset(x, y)
    loader = DataLoader(dataset, batch_size=2)
    return loader

@pytest.fixture
def dummy_model():
    from src.models import load_resnet_model
    model = load_resnet_model(depth=18, num_classes=2, checkpoint_path=None, device="cpu")
    return model

def test_evaluate_pipeline(dummy_model, dummy_dataset):
    criterion = nn.CrossEntropyLoss()
    class_names = ["class0", "class1"]
    val_loss, overall_acc, df_metrics = evaluate(
        dummy_model,
        dummy_dataset,
        criterion,
        device="cpu",
        class_names=class_names,
        save_csv="metrics_dummy.csv"
    )
    # sanity checks
    assert isinstance(val_loss, float)
    assert isinstance(overall_acc, float)
    assert df_metrics.shape[0] == len(class_names)
