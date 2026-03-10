"""
Medical Image Classifier — Full Training Pipeline
===================================================
Runs the complete pipeline end-to-end on local GPU:
1. Verify dataset
2. Train baseline CNN (~10 epochs)
3. Train EfficientNet-B0 (~15 epochs)
4. Evaluate both models on test set
5. Generate Grad-CAM visualizations

Usage:
    python run_training.py

Requirements:
    - Dataset in data/chest_xray/ (train/val/test splits)
    - NVIDIA GPU with CUDA support
    - All dependencies installed (see requirements.txt)
"""

import os
import sys
import torch

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.dataset import get_dataloaders
from src.model import BaseCNN, get_efficientnet, unfreeze_last_n_blocks
from src.train import train_model, plot_history
from src.evaluate import evaluate_model
from src.gradcam import generate_gradcam_examples


def verify_dataset(base_path):
    """Check that the dataset exists and print class counts."""
    print("=" * 50)
    print("DATASET VERIFICATION")
    print("=" * 50)

    total = 0
    for split in ["train", "val", "test"]:
        for label in ["NORMAL", "PNEUMONIA"]:
            path = os.path.join(base_path, split, label)
            if not os.path.exists(path):
                print(f"ERROR: Missing directory: {path}")
                print("Please download the dataset from:")
                print("https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia")
                sys.exit(1)
            count = len([f for f in os.listdir(path)
                        if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
            print(f"  {split}/{label}: {count} images")
            total += count

    print(f"\n  Total images: {total}")
    print()


def main():
    # Configuration
    DATA_PATH = "data/chest_xray"
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    BATCH_SIZE = 32

    print(f"\nDevice: {DEVICE}")
    if DEVICE == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    print()

    # Step 1: Verify dataset
    verify_dataset(DATA_PATH)

    # Step 2: Create data loaders
    print("Loading data...")
    loaders = get_dataloaders(DATA_PATH, batch_size=BATCH_SIZE)
    print(f"  Train batches: {len(loaders['train'])}")
    print(f"  Val batches:   {len(loaders['val'])}")
    print(f"  Test batches:  {len(loaders['test'])}")
    print()

    # Step 3: Train Baseline CNN
    print("=" * 50)
    print("PHASE 1: BASELINE CNN TRAINING")
    print("=" * 50)
    baseline_model = BaseCNN(num_classes=2)
    baseline_model, baseline_history = train_model(
        baseline_model, loaders,
        num_epochs=10,
        device=DEVICE,
        save_path="outputs/models/baseline_cnn.pth"
    )
    plot_history(baseline_history, "outputs/figures/baseline_history.png")

    # Step 4: Evaluate Baseline CNN
    print("\n" + "=" * 50)
    print("BASELINE CNN — TEST SET EVALUATION")
    print("=" * 50)
    baseline_auc, _ = evaluate_model(baseline_model, loaders["test"], device=DEVICE)

    # Step 5: Train EfficientNet — Stage 1 (frozen features, train classifier head)
    print("\n" + "=" * 50)
    print("PHASE 2a: EFFICIENTNET-B0 — STAGE 1 (FROZEN FEATURES)")
    print("=" * 50)
    efficientnet = get_efficientnet(num_classes=2, freeze_features=True)
    efficientnet, eff_history_s1 = train_model(
        efficientnet, loaders,
        num_epochs=10,
        device=DEVICE,
        save_path="outputs/models/best_model.pth"
    )
    plot_history(eff_history_s1, "outputs/figures/training_history_stage1.png")

    # Step 6: Train EfficientNet — Stage 2 (unfreeze last 3 blocks, lower LR)
    print("\n" + "=" * 50)
    print("PHASE 2b: EFFICIENTNET-B0 — STAGE 2 (FINE-TUNE FEATURES)")
    print("=" * 50)
    print("Unfreezing last 4 feature blocks for fine-tuning...")
    efficientnet = unfreeze_last_n_blocks(efficientnet, n=4)
    efficientnet, eff_history_s2 = train_model(
        efficientnet, loaders,
        num_epochs=10,
        device=DEVICE,
        save_path="outputs/models/best_model.pth",
        lr=5e-6  # Very low LR to preserve pretrained features
    )
    plot_history(eff_history_s2, "outputs/figures/training_history.png")

    # Step 7: Evaluate EfficientNet
    print("\n" + "=" * 50)
    print("EFFICIENTNET-B0 — TEST SET EVALUATION")
    print("=" * 50)
    eff_auc, _ = evaluate_model(efficientnet, loaders["test"], device=DEVICE)

    # Step 7: Generate Grad-CAM examples
    print("\n" + "=" * 50)
    print("GENERATING GRAD-CAM VISUALIZATIONS")
    print("=" * 50)
    generate_gradcam_examples(
        efficientnet,
        os.path.join(DATA_PATH, "test"),
        device=DEVICE,
        n_examples=3
    )

    # Summary
    print("\n" + "=" * 50)
    print("TRAINING COMPLETE — RESULTS SUMMARY")
    print("=" * 50)
    print(f"  Baseline CNN AUC-ROC:    {baseline_auc:.4f}")
    print(f"  EfficientNet AUC-ROC:    {eff_auc:.4f}")
    print()
    print("  Saved files:")
    print("    outputs/models/baseline_cnn.pth")
    print("    outputs/models/best_model.pth")
    print("    outputs/figures/baseline_history.png")
    print("    outputs/figures/training_history.png")
    print("    outputs/figures/confusion_matrix.png")
    print("    outputs/figures/roc_curve.png")
    print("    outputs/figures/gradcam_*.png")
    print()
    print("  Next step: Run the Streamlit dashboard:")
    print("    streamlit run app/dashboard.py")


if __name__ == "__main__":
    main()
