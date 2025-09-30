import torch
import torchvision.models as models
import os


def load_resnet_model(
    depth: int, num_classes: int, checkpoint_path=None, device: str = "cuda"
):
    """
    Load a ResNet model of configurable depth.
    """
    resnet_map = {
        18: (models.resnet18, models.ResNet18_Weights),
        34: (models.resnet34, models.ResNet34_Weights),
        50: (models.resnet50, models.ResNet50_Weights),
        101: (models.resnet101, models.ResNet101_Weights),
    }

    if depth not in resnet_map:
        raise ValueError(
            f"Unsupported ResNet depth: {depth}. Choose from {list(resnet_map.keys())}"
        )

    # Load pretrained backbone
    resnet_func, weights_enum = resnet_map[depth]
    model = resnet_func(weights=weights_enum.DEFAULT)
    model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
    model = model.to(device)

    # Load checkpoint if provided
    if checkpoint_path and os.path.isfile(checkpoint_path):
        state_dict = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(state_dict)
        print(f"Loaded checkpoint from {checkpoint_path}")

    return model
