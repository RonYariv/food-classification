import os
import pytest
import torch
import numpy as np
import pandas as pd
from PIL import Image

from src import utils

# --------------------
# compute_accuracy
# --------------------
def test_compute_accuracy_top1_top5():
    outputs = torch.tensor([
        [0.1, 0.9, 0.0],
        [0.8, 0.1, 0.1],
    ])  # shape (2, 3)
    labels = torch.tensor([1, 0])  # correct classes

    top1, top5 = utils.compute_accuracy(outputs, labels, topk=(1, 2))
    assert isinstance(top1, float)
    assert isinstance(top5, float)
    assert 0 <= top1 <= 100
    assert 0 <= top5 <= 100
    # Check that top-1 is 100% correct
    assert pytest.approx(top1, rel=1e-3) == 100.0

# --------------------
# get_transform & preprocess_image
# --------------------
def test_get_transform_and_preprocess(tmp_path):
    train_t, val_t = utils.get_transform()
    assert callable(train_t)
    assert callable(val_t)

    # Create dummy image file
    img_path = tmp_path / "test.jpg"
    Image.new("RGB", (256, 256), color=(255, 0, 0)).save(img_path)

    tensor = utils.preprocess_image(str(img_path))
    assert tensor.shape[0] == 1  # batch dimension
    assert tensor.shape[1:] == (3, 224, 224)  # (C,H,W)

# --------------------
# get_class_names
# --------------------
def test_get_class_names(tmp_path, monkeypatch):
    # Simulate dataset dir
    train_dir = tmp_path / "train"
    os.makedirs(train_dir / "class_a")
    os.makedirs(train_dir / "class_b")

    monkeypatch.setattr("src.config.SPLIT_DATA_DIR", str(tmp_path))

    class_names = utils.get_class_names()
    assert sorted(class_names) == ["class_a", "class_b"]

# --------------------
# plot_top_confusions
# --------------------
def test_plot_top_confusions(tmp_path, monkeypatch):
    monkeypatch.setattr("src.config.REPORTS_DIR", str(tmp_path))

    cm = np.array([[0, 5], [3, 0]])
    class_names = ["cat", "dog"]

    utils.plot_top_confusions(cm, class_names, top_k=2)
    out_file = tmp_path / "confusion_matrix.png"
    assert out_file.exists()

# --------------------
# plot_metrics_table
# --------------------
def test_plot_metrics_table(tmp_path, monkeypatch):
    monkeypatch.setattr("src.config.REPORTS_DIR", str(tmp_path))

    df = pd.DataFrame({"accuracy": [0.9], "precision": [0.8], "recall": [0.7], "f1": [0.75]})
    utils.plot_metrics_table(df)
    out_file = tmp_path / "metrics_table.png"
    assert out_file.exists()

# --------------------
# plot_training_curves
# --------------------
def test_plot_training_curves(tmp_path, monkeypatch):
    monkeypatch.setattr("src.config.REPORTS_DIR", str(tmp_path))

    train_losses = [1.0, 0.8, 0.6]
    val_losses = [1.1, 0.9, 0.7]
    train_accs = [60, 70, 80]
    val_accs = [55, 65, 75]

    utils.plot_training_curves(train_losses, val_losses, train_accs, val_accs)

    assert (tmp_path / "loss_curve.png").exists()
    assert (tmp_path / "accuracy_curve.png").exists()
