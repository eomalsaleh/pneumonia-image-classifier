import torch.nn as nn
from torchvision import models


class BaseCNN(nn.Module):
    """Simple 3-layer CNN for baseline comparison.
    
    Expected accuracy: ~80-85% — used to show improvement
    when switching to transfer learning with EfficientNet.
    """
    def __init__(self, num_classes=2):
        super(BaseCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 28 * 28, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        return self.classifier(self.features(x))


def get_efficientnet(num_classes=2, freeze_features=True):
    """Load pretrained EfficientNet-B0 and replace the classifier head.
    
    Args:
        num_classes: Number of output classes (2 for Normal/Pneumonia)
        freeze_features: If True, freeze all feature extraction layers
                         (only train the classifier head)
    
    Why EfficientNet:
        - Strong performance with fewer parameters than ResNet
        - Pretrained on ImageNet — already understands edges, textures, shapes
        - Fine-tuning adapts this knowledge to X-ray features
    """
    model = models.efficientnet_b0(weights="IMAGENET1K_V1")

    if freeze_features:
        for param in model.features.parameters():
            param.requires_grad = False

    # Replace final classifier
    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(0.3),
        nn.Linear(in_features, num_classes)
    )

    return model


def unfreeze_last_n_blocks(model, n=3):
    """Unfreeze the last N blocks of EfficientNet's feature extractor.
    
    This enables two-stage fine-tuning:
      Stage 1: Train with all features frozen (learn classifier head)
      Stage 2: Unfreeze last N blocks and fine-tune with lower LR
               (learn X-ray-specific features for better Grad-CAM)
    
    Args:
        model: EfficientNet model
        n: Number of feature blocks to unfreeze from the end
    """
    # EfficientNet-B0 has 9 blocks in model.features (indices 0-8)
    total_blocks = len(model.features)
    for i in range(total_blocks - n, total_blocks):
        for param in model.features[i].parameters():
            param.requires_grad = True
    
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"  Unfroze last {n} blocks: {trainable:,}/{total:,} params trainable")
    return model
