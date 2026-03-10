import os
import torch
import numpy as np
from sklearn.metrics import (classification_report, confusion_matrix,
                              roc_auc_score, roc_curve)
import matplotlib.pyplot as plt
import seaborn as sns


def evaluate_model(model, test_loader, device="cuda", save_dir="outputs/figures"):
    """Evaluate a trained model on the test set with clinical metrics.
    
    Computes and saves:
        - Classification report (precision, recall, F1-score)
        - AUC-ROC score
        - Confusion matrix heatmap
        - ROC curve
    
    Key metrics for clinical AI:
        - Recall (sensitivity): most important — missing pneumonia is dangerous
        - AUC-ROC: overall discrimination ability
        - Precision: how often a positive prediction is correct
        - F1-score: balance between precision and recall
    """
    os.makedirs(save_dir, exist_ok=True)
    model.eval()
    all_preds, all_labels, all_probs = [], [], []

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            probs = torch.softmax(outputs, dim=1)[:, 1]
            preds = torch.argmax(outputs, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())
            all_probs.extend(probs.cpu().numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)

    # Classification report
    report = classification_report(all_labels, all_preds,
                                   target_names=["NORMAL", "PNEUMONIA"])
    print("\n" + "=" * 50)
    print("CLASSIFICATION REPORT")
    print("=" * 50)
    print(report)

    # AUC-ROC
    auc = roc_auc_score(all_labels, all_probs)
    print(f"AUC-ROC: {auc:.4f}")

    # Confusion Matrix
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["NORMAL", "PNEUMONIA"],
                yticklabels=["NORMAL", "PNEUMONIA"])
    plt.title("Confusion Matrix")
    plt.ylabel("Actual")
    plt.xlabel("Predicted")
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "confusion_matrix.png"), dpi=150)
    plt.close()
    print(f"Confusion matrix saved to {save_dir}/confusion_matrix.png")

    # ROC Curve
    fpr, tpr, _ = roc_curve(all_labels, all_probs)
    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, label=f"AUC = {auc:.4f}", linewidth=2)
    plt.plot([0, 1], [0, 1], "k--", alpha=0.5)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate (Recall)")
    plt.title("ROC Curve")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "roc_curve.png"), dpi=150)
    plt.close()
    print(f"ROC curve saved to {save_dir}/roc_curve.png")

    return auc, report
