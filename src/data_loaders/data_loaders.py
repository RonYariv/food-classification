import os
from torch.utils.data import DataLoader
from torchvision import datasets
from src.utils import get_transform


def get_dataloaders(data_dir, batch_size=32, num_workers=4):
    train_t, val_test_t = get_transform()

    train_dataset = datasets.ImageFolder(
        root=os.path.join(data_dir, "train"), transform=train_t
    )
    val_dataset = datasets.ImageFolder(
        root=os.path.join(data_dir, "val"), transform=val_test_t
    )
    test_dataset = datasets.ImageFolder(
        root=os.path.join(data_dir, "test"), transform=val_test_t
    )

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers
    )

    num_classes = len(train_dataset.classes)
    class_names = train_dataset.classes

    return train_loader, val_loader, test_loader, num_classes, class_names
