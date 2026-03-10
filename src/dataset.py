import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader


def get_transforms(split="train"):
    """Get image transforms for the given split.
    
    Key design choices:
      - Resize to 256 then CenterCrop to 224: strips ~12% border area
        to remove text labels, black padding, and border artifacts common
        in the Kaggle chest X-ray dataset
      - Train split adds aggressive augmentation to prevent shortcut learning
    """
    if split == "train":
        return transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.3, contrast=0.3),
            transforms.RandomAffine(degrees=0, translate=(0.05, 0.05)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
            transforms.RandomErasing(p=0.3, scale=(0.02, 0.1)),
        ])
    else:
        return transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])


def get_dataloaders(base_path, batch_size=32):
    """Create DataLoaders for train, val, and test splits.
    
    Uses num_workers=2 (safer on Windows than 4).
    Class mapping: 0 = NORMAL, 1 = PNEUMONIA
    """
    loaders = {}
    for split in ["train", "val", "test"]:
        dataset = datasets.ImageFolder(
            root=f"{base_path}/{split}",
            transform=get_transforms(split)
        )
        loaders[split] = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=(split == "train"),
            num_workers=2,
            pin_memory=True
        )
    return loaders
