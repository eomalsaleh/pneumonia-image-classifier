import os
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm
import matplotlib.pyplot as plt


def train_model(model, loaders, num_epochs=15, device="cuda", save_path="outputs/models/best_model.pth", lr=1e-4):
    """Train a model with weighted cross-entropy loss to handle class imbalance.
    
    Features:
        - Weighted loss (PNEUMONIA >> NORMAL in training set)
        - Adam optimizer with lr=1e-4
        - ReduceLROnPlateau scheduler (patience=3, factor=0.5)
        - Best model checkpointing based on validation accuracy
    
    Args:
        model: PyTorch model to train
        loaders: Dict with 'train' and 'val' DataLoaders
        num_epochs: Number of training epochs
        device: 'cuda' or 'cpu'
        save_path: Path to save best model checkpoint
    
    Returns:
        model: Trained model (best weights loaded)
        history: Dict with train/val loss and accuracy per epoch
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    model = model.to(device)

    # Weighted loss to handle class imbalance (PNEUMONIA >> NORMAL)
    class_weights = torch.tensor([2.0, 1.0]).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)
    scheduler = ReduceLROnPlateau(optimizer, patience=3, factor=0.5)

    best_val_acc = 0.0
    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}

    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch + 1}/{num_epochs}")
        print("-" * 40)

        for phase in ["train", "val"]:
            if phase == "train":
                model.train()
            else:
                model.eval()

            running_loss, correct, total = 0.0, 0, 0

            for inputs, labels in tqdm(loaders[phase], desc=f"  {phase}"):
                inputs, labels = inputs.to(device), labels.to(device)
                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == "train"):
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    if phase == "train":
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                _, preds = torch.max(outputs, 1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)

            epoch_loss = running_loss / total
            epoch_acc = correct / total
            history[f"{phase}_loss"].append(epoch_loss)
            history[f"{phase}_acc"].append(epoch_acc)
            print(f"  {phase} Loss: {epoch_loss:.4f} | Acc: {epoch_acc:.4f}")

            if phase == "val":
                scheduler.step(epoch_loss)
                if epoch_acc > best_val_acc:
                    best_val_acc = epoch_acc
                    torch.save(model.state_dict(), save_path)
                    print(f"  * Best model saved (val acc: {best_val_acc:.4f})")

    # Load best weights
    model.load_state_dict(torch.load(save_path, map_location=device))
    print(f"\nTraining complete. Best val accuracy: {best_val_acc:.4f}")
    return model, history


def plot_history(history, save_path="outputs/figures/training_history.png"):
    """Plot and save training loss and accuracy curves."""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(history["train_loss"], label="Train", linewidth=2)
    axes[0].plot(history["val_loss"], label="Val", linewidth=2)
    axes[0].set_title("Loss Over Epochs")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(history["train_acc"], label="Train", linewidth=2)
    axes[1].plot(history["val_acc"], label="Val", linewidth=2)
    axes[1].set_title("Accuracy Over Epochs")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Training history saved to {save_path}")
