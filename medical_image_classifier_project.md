# Medical Image Classifier
### CNN-Based Pneumonia Detection with Grad-CAM Explainability

---

## Project Overview

Build a deep learning pipeline that classifies chest X-ray images as **Normal** or **Pneumonia** using a Convolutional Neural Network (CNN). You will fine-tune a pretrained model (EfficientNet), evaluate it using clinical-grade metrics, and add Grad-CAM visualizations to show *where* in the image the model is looking — a critical requirement for medical AI credibility.

**Target Dataset:** [Chest X-Ray Images (Pneumonia) — Kaggle](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia)
- 5,863 X-ray images (JPEG)
- 2 classes: NORMAL / PNEUMONIA
- Pre-split into train, val, test folders

**Tech Stack:**
- **Deep Learning:** PyTorch + torchvision
- **Explainability:** Grad-CAM (pytorch-grad-cam library)
- **Visualization:** matplotlib, seaborn
- **Dashboard:** Streamlit
- **Version Control:** Git + GitHub

---

## Project Structure

```
medical-image-classifier/
│
├── data/
│   └── chest_xray/
│       ├── train/
│       │   ├── NORMAL/
│       │   └── PNEUMONIA/
│       ├── val/
│       │   ├── NORMAL/
│       │   └── PNEUMONIA/
│       └── test/
│           ├── NORMAL/
│           └── PNEUMONIA/
│
├── notebooks/
│   ├── 01_eda.ipynb                  # Dataset exploration & visualization
│   ├── 02_baseline_cnn.ipynb         # Build & train custom CNN
│   └── 03_transfer_learning.ipynb    # Fine-tune EfficientNet
│
├── src/
│   ├── dataset.py        # Custom PyTorch Dataset + transforms
│   ├── model.py          # Model definitions
│   ├── train.py          # Training loop
│   ├── evaluate.py       # Metrics & confusion matrix
│   └── gradcam.py        # Grad-CAM visualization
│
├── app/
│   └── dashboard.py      # Streamlit app
│
├── outputs/
│   ├── models/           # Saved .pth checkpoints
│   └── figures/          # Saved plots & Grad-CAM images
│
├── requirements.txt
└── README.md
```

---

## Step-by-Step Breakdown

---

### PHASE 1 — Environment Setup

#### Step 1: Set Up Your Environment
Create a dedicated virtual environment and install all dependencies.

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate        # Mac/Linux
venv\Scripts\activate           # Windows

# Install dependencies
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install pytorch-grad-cam scikit-learn matplotlib seaborn streamlit pillow pandas tqdm
```

Check if GPU is available (highly recommended — use Google Colab if your machine has no GPU):

```python
import torch
print(torch.cuda.is_available())    # True = GPU available
print(torch.cuda.get_device_name(0))
```

> **Tip:** If you don't have a local GPU, run training notebooks on **Google Colab** (free T4 GPU) and save the model weights. Then run the Streamlit dashboard locally.

**Goal:** A working Python environment ready for deep learning.

---

#### Step 2: Download the Dataset
- Go to [Kaggle Chest X-Ray Dataset](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia)
- Download and unzip into `data/chest_xray/`
- Verify the folder structure matches the project layout above

```python
import os

base = "data/chest_xray"
for split in ["train", "val", "test"]:
    for label in ["NORMAL", "PNEUMONIA"]:
        path = os.path.join(base, split, label)
        count = len(os.listdir(path))
        print(f"{split}/{label}: {count} images")
```

Expected output:
```
train/NORMAL: 1341
train/PNEUMONIA: 3875
val/NORMAL: 8
val/PNEUMONIA: 8
test/NORMAL: 234
test/PNEUMONIA: 390
```

**Goal:** Dataset downloaded, organized, and verified.

---

### PHASE 2 — Exploratory Data Analysis (EDA)

#### Step 3: Visualize the Dataset
In `notebooks/01_eda.ipynb`, explore the data visually before touching any model.

```python
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os, random

def show_samples(base_path, label, n=5):
    folder = os.path.join(base_path, "train", label)
    images = random.sample(os.listdir(folder), n)
    fig, axes = plt.subplots(1, n, figsize=(15, 4))
    for ax, img_name in zip(axes, images):
        img = mpimg.imread(os.path.join(folder, img_name))
        ax.imshow(img, cmap="gray")
        ax.set_title(label)
        ax.axis("off")
    plt.tight_layout()
    plt.show()

show_samples("data/chest_xray", "NORMAL")
show_samples("data/chest_xray", "PNEUMONIA")
```

**Also analyze:**
- Class imbalance (PNEUMONIA >> NORMAL in the training set)
- Image size distribution (X-rays vary in resolution)
- Pixel intensity distributions for each class
- Visual differences between normal and pneumonia X-rays

```python
import numpy as np
from PIL import Image

def get_image_stats(folder):
    sizes = []
    for fname in os.listdir(folder):
        img = Image.open(os.path.join(folder, fname)).convert("L")
        sizes.append(img.size)
    widths, heights = zip(*sizes)
    print(f"Width  — min: {min(widths)}, max: {max(widths)}, avg: {int(np.mean(widths))}")
    print(f"Height — min: {min(heights)}, max: {max(heights)}, avg: {int(np.mean(heights))}")

get_image_stats("data/chest_xray/train/NORMAL")
get_image_stats("data/chest_xray/train/PNEUMONIA")
```

**Goal:** Understand class imbalance and image variability before modeling — this informs your augmentation and loss function choices.

---

### PHASE 3 — Data Pipeline

#### Step 4: Build the PyTorch Dataset and DataLoaders
In `src/dataset.py`, define transforms and loaders:

```python
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

def get_transforms(split="train"):
    if split == "train":
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])
    else:
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

def get_dataloaders(base_path, batch_size=32):
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
            num_workers=4
        )
    return loaders

# Class mapping: 0 = NORMAL, 1 = PNEUMONIA
```

**Why these augmentations?**
- `RandomHorizontalFlip` — X-rays can be mirrored
- `RandomRotation` — slight patient positioning variation
- `ColorJitter` — accounts for different X-ray machine exposures
- ImageNet normalization — required for pretrained models

**Goal:** A robust data pipeline with augmentation to combat overfitting and class imbalance.

---

### PHASE 4 — Baseline CNN (Optional but Recommended)

#### Step 5: Build and Train a Simple Custom CNN
In `notebooks/02_baseline_cnn.ipynb`, build a small CNN from scratch first. This gives you a baseline to compare against the pretrained model and makes your results table more compelling.

```python
import torch.nn as nn

class BaseCNN(nn.Module):
    def __init__(self):
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
            nn.Linear(256, 2)
        )

    def forward(self, x):
        return self.classifier(self.features(x))
```

**Goal:** Establish a baseline accuracy (~80–85%) to show clear improvement when you switch to transfer learning.

---

### PHASE 5 — Transfer Learning with EfficientNet

#### Step 6: Fine-Tune EfficientNet-B0
In `src/model.py`, load a pretrained EfficientNet and replace the classifier head:

```python
import torch.nn as nn
from torchvision import models

def get_efficientnet(num_classes=2, freeze_features=True):
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
```

**Why EfficientNet?**
- Strong performance with fewer parameters than ResNet
- Pretrained on ImageNet — already understands edges, textures, shapes
- Fine-tuning reuses this knowledge and adapts it to X-ray features

**Goal:** A model architecture ready for clinical-grade training.

---

#### Step 7: Write the Training Loop
In `src/train.py`:

```python
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm

def train_model(model, loaders, num_epochs=15, device="cuda"):
    model = model.to(device)

    # Weighted loss to handle class imbalance (PNEUMONIA >> NORMAL)
    class_weights = torch.tensor([3.0, 1.0]).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-4)
    scheduler = ReduceLROnPlateau(optimizer, patience=3, factor=0.5)

    best_val_acc = 0.0
    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}

    for epoch in range(num_epochs):
        for phase in ["train", "val"]:
            model.train() if phase == "train" else model.eval()
            running_loss, correct, total = 0.0, 0, 0

            for inputs, labels in tqdm(loaders[phase], desc=f"Epoch {epoch+1} [{phase}]"):
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
                    torch.save(model.state_dict(), "outputs/models/best_model.pth")
                    print(f"  Best model saved (val acc: {best_val_acc:.4f})")

    return model, history
```

**Goal:** A trained model checkpoint saved to disk, with training history logged for plotting.

---

### PHASE 6 — Evaluation

#### Step 8: Evaluate with Clinical Metrics
In `src/evaluate.py`, evaluate on the held-out test set:

```python
import torch
import numpy as np
from sklearn.metrics import (classification_report, confusion_matrix,
                              roc_auc_score, roc_curve)
import matplotlib.pyplot as plt
import seaborn as sns

def evaluate_model(model, test_loader, device="cuda"):
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

    print(classification_report(all_labels, all_preds,
                                 target_names=["NORMAL", "PNEUMONIA"]))

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
    plt.savefig("outputs/figures/confusion_matrix.png")
    plt.show()

    # ROC Curve
    fpr, tpr, _ = roc_curve(all_labels, all_probs)
    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, label=f"AUC = {auc:.4f}")
    plt.plot([0, 1], [0, 1], "k--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate (Recall)")
    plt.title("ROC Curve")
    plt.legend()
    plt.tight_layout()
    plt.savefig("outputs/figures/roc_curve.png")
    plt.show()

    return auc
```

**Key metrics to highlight:**
- **Recall (Sensitivity)** — most important in clinical settings; missing a pneumonia case is more dangerous than a false alarm
- **AUC-ROC** — overall model discrimination ability
- **Precision** — how often a positive prediction is correct
- **F1-Score** — balance between precision and recall

**Goal:** A complete, documented evaluation showing you understand *clinical* metrics, not just accuracy.

---

### PHASE 7 — Grad-CAM Explainability

#### Step 9: Implement Grad-CAM Visualizations
This is the most impressive part of the project. In `src/gradcam.py`:

```python
import torch
import numpy as np
import matplotlib.pyplot as plt
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from PIL import Image
from torchvision import transforms

def generate_gradcam(model, image_path, true_label, device="cuda"):
    # Load and preprocess image
    raw_img = Image.open(image_path).convert("RGB").resize((224, 224))
    raw_np = np.array(raw_img) / 255.0

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])
    input_tensor = transform(raw_img).unsqueeze(0).to(device)

    # Target the last conv layer of EfficientNet
    target_layers = [model.features[-1]]

    with GradCAM(model=model, target_layers=target_layers) as cam:
        targets = [ClassifierOutputTarget(1)]  # 1 = PNEUMONIA class
        grayscale_cam = cam(input_tensor=input_tensor, targets=targets)[0]

    cam_image = show_cam_on_image(raw_np.astype(np.float32), grayscale_cam, use_rgb=True)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].imshow(raw_img, cmap="gray")
    axes[0].set_title(f"Original X-Ray\nTrue Label: {true_label}")
    axes[0].axis("off")

    axes[1].imshow(cam_image)
    axes[1].set_title("Grad-CAM Heatmap\n(Red = High Activation)")
    axes[1].axis("off")

    plt.tight_layout()
    plt.savefig(f"outputs/figures/gradcam_{true_label}.png")
    plt.show()
```

**Generate examples for:**
- A correctly classified PNEUMONIA case
- A correctly classified NORMAL case
- A misclassified case (if any) — discuss *why* in your README

**What good Grad-CAM looks like:** The heatmap should light up the **lung fields** (center of the image), not the corners, image labels, or borders. If it lights up irrelevant areas, your model is overfitting to artifacts.

**Goal:** Visual proof that your model attends to clinically meaningful regions — this is what separates serious biomedical AI work from a basic Kaggle notebook.

---

### PHASE 8 — Streamlit Dashboard

#### Step 10: Build the Interactive Web App
In `app/dashboard.py`:

```python
import streamlit as st
import torch
from torchvision import transforms
from PIL import Image
import numpy as np
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from src.model import get_efficientnet

@st.cache_resource
def load_model():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = get_efficientnet(num_classes=2)
    model.load_state_dict(torch.load("outputs/models/best_model.pth",
                                      map_location=device))
    model.eval()
    return model.to(device), device

model, device = load_model()

st.title("Chest X-Ray Pneumonia Detector")
st.markdown("Upload a chest X-ray image to classify it as **Normal** or **Pneumonia**.")
st.caption("For educational purposes only — not for clinical diagnosis.")

uploaded = st.file_uploader("Upload Chest X-Ray (JPG/PNG)", type=["jpg", "jpeg", "png"])

if uploaded:
    image = Image.open(uploaded).convert("RGB")
    st.image(image, caption="Uploaded X-Ray", use_column_width=True)

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    input_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(input_tensor)
        prob = torch.softmax(output, dim=1)[0]
        pred_class = torch.argmax(prob).item()

    label = "PNEUMONIA" if pred_class == 1 else "NORMAL"
    confidence = prob[pred_class].item()

    st.subheader(f"Prediction: {label}")
    st.metric("Confidence", f"{confidence:.1%}")
    st.progress(confidence)

    # Grad-CAM
    st.subheader("Grad-CAM Explanation")
    st.caption("Highlighted regions show where the model focused its attention.")

    raw_np = np.array(image.resize((224, 224))) / 255.0
    target_layers = [model.features[-1]]

    with GradCAM(model=model, target_layers=target_layers) as cam:
        grayscale_cam = cam(input_tensor=input_tensor,
                            targets=[ClassifierOutputTarget(pred_class)])[0]

    cam_image = show_cam_on_image(raw_np.astype(np.float32), grayscale_cam, use_rgb=True)

    col1, col2 = st.columns(2)
    col1.image(image.resize((224, 224)), caption="Original")
    col2.image(cam_image, caption="Grad-CAM Heatmap")
```

Run with: `streamlit run app/dashboard.py`

**Goal:** A polished, deployable web app that any non-technical reviewer can interact with.

---

### PHASE 9 — Documentation & GitHub

#### Step 11: Plot and Save Training History
```python
import matplotlib.pyplot as plt

def plot_history(history):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(history["train_loss"], label="Train")
    axes[0].plot(history["val_loss"], label="Val")
    axes[0].set_title("Loss Over Epochs")
    axes[0].set_xlabel("Epoch")
    axes[0].legend()

    axes[1].plot(history["train_acc"], label="Train")
    axes[1].plot(history["val_acc"], label="Val")
    axes[1].set_title("Accuracy Over Epochs")
    axes[1].set_xlabel("Epoch")
    axes[1].legend()

    plt.tight_layout()
    plt.savefig("outputs/figures/training_history.png")
    plt.show()
```

Include this in your README to show convergence and that the model isn't overfitting.

---

#### Step 12: Write a Professional README
Your `README.md` should include:

- **Problem Statement** — frame it clinically: "Pneumonia causes over 2 million deaths annually. Automated X-ray analysis can accelerate diagnosis..."
- **Dataset details** — source, size, class distribution bar chart
- **Model architecture** — EfficientNet-B0 fine-tuned, why transfer learning over training from scratch
- **Results comparison table:**

| Model | Accuracy | AUC-ROC | Recall (Pneumonia) |
|-------|----------|---------|-------------------|
| Baseline CNN | ~82% | ~0.87 | ~0.88 |
| EfficientNet-B0 | ~92% | ~0.97 | ~0.96 |

- **Grad-CAM screenshots** — side-by-side original vs. heatmap
- **Dashboard screenshot**
- **How to run locally** — clear setup instructions
- **Disclaimer** — always include "not for clinical use"

---

## Resume Bullet Points (copy-paste ready)

Once complete, here's how to list this on your resume:

> **Medical Image Classifier — Pneumonia Detection** *(Python, PyTorch, EfficientNet, Grad-CAM, Streamlit)*
> Fine-tuned EfficientNet-B0 on 5,800+ labeled chest X-ray images to detect pneumonia with AUC-ROC of 0.XX and recall of 0.XX. Implemented Grad-CAM heatmap visualizations to localize clinically relevant lung regions and deployed an interactive diagnostic interface via Streamlit.

---

## Estimated Timeline

| Phase | Task | Time Estimate |
|-------|------|---------------|
| 1 | Environment setup + dataset download | 1–2 hours |
| 2 | EDA notebook | 2–3 hours |
| 3 | Data pipeline (Dataset + DataLoaders) | 1–2 hours |
| 4 | Baseline CNN (optional) | 2–3 hours |
| 5–6 | EfficientNet fine-tuning + evaluation | 4–6 hours |
| 7 | Grad-CAM implementation | 2–3 hours |
| 8 | Streamlit dashboard | 3–4 hours |
| 9 | README + GitHub | 2 hours |
| **Total** | | **~17–25 hours** |

> Training time: ~20–40 min per run on Google Colab T4 GPU (free tier)

---

## Key Libraries to Install

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install pytorch-grad-cam scikit-learn matplotlib seaborn streamlit pillow pandas tqdm
```

---

## Resources

- [Chest X-Ray Dataset — Kaggle](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia)
- [PyTorch Documentation](https://pytorch.org/docs/)
- [pytorch-grad-cam Library](https://github.com/jacobgil/pytorch-grad-cam)
- [EfficientNet Paper (arxiv)](https://arxiv.org/abs/1905.11946)
- [Google Colab — Free GPU](https://colab.research.google.com/)
- [Streamlit Documentation](https://docs.streamlit.io/)

---

## Important Notes

- Always include a **disclaimer** that this model is for educational purposes and not for clinical diagnosis
- The val set in this dataset is very small (16 images total) — monitor training loss carefully and rely primarily on the test set for final evaluation
- If your model predicts only PNEUMONIA for every image, your weighted loss isn't working — try adjusting the class weights upward for NORMAL
- Grad-CAM heatmaps should highlight the **lung fields**, not image corners, borders, or text labels — if they don't, something is wrong with your model or target layer selection
