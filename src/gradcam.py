import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
from pytorch_grad_cam import GradCAMPlusPlus
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from PIL import Image
from torchvision import transforms


def generate_gradcam(model, image_path, true_label, device="cuda",
                     save_dir="outputs/figures"):
    """Generate Grad-CAM heatmap overlay for a chest X-ray image.
    
    Shows where the model is focusing its attention. In a good model,
    heatmaps should highlight the lung fields (center), not corners,
    borders, or text labels.
    
    Args:
        model: Trained EfficientNet model
        image_path: Path to chest X-ray image
        true_label: Ground truth label string (e.g. "NORMAL" or "PNEUMONIA")
        device: 'cuda' or 'cpu'
        save_dir: Directory to save the Grad-CAM figure
    """
    os.makedirs(save_dir, exist_ok=True)

    # Load and preprocess image (match training: Resize 256 -> CenterCrop 224)
    raw_img = Image.open(image_path).convert("RGB")
    raw_img = transforms.Resize(256)(raw_img)
    raw_img = transforms.CenterCrop(224)(raw_img)
    raw_np = np.array(raw_img) / 255.0

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])
    input_tensor = transform(raw_img).unsqueeze(0).to(device)

    # Get model prediction
    model.eval()
    with torch.no_grad():
        output = model(input_tensor)
        prob = torch.softmax(output, dim=1)[0]
        pred_class = torch.argmax(prob).item()
        pred_label = "PNEUMONIA" if pred_class == 1 else "NORMAL"
        confidence = prob[pred_class].item()

    # Target an intermediate layer — features[7] balances spatial resolution
    # and semantic meaning for focused, meaningful heatmaps
    target_layers = [model.features[6]]

    with GradCAMPlusPlus(model=model, target_layers=target_layers) as cam:
        grayscale_cam = cam(input_tensor=input_tensor)[0]

    # Smooth the heatmap to reduce scatter and produce cleaner attention regions
    grayscale_cam = gaussian_filter(grayscale_cam, sigma=10)
    grayscale_cam = (grayscale_cam - grayscale_cam.min()) / (grayscale_cam.max() - grayscale_cam.min() + 1e-8)

    cam_image = show_cam_on_image(raw_np.astype(np.float32), grayscale_cam,
                                  use_rgb=True)

    # Plot side-by-side
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].imshow(raw_img)
    axes[0].set_title(f"Original X-Ray\nTrue: {true_label}")
    axes[0].axis("off")

    axes[1].imshow(cam_image)
    axes[1].set_title(f"Grad-CAM Heatmap\nPred: {pred_label} ({confidence:.1%})")
    axes[1].axis("off")

    plt.tight_layout()
    save_name = f"gradcam_{true_label.lower()}_{os.path.basename(image_path)}"
    save_path = os.path.join(save_dir, save_name)
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Grad-CAM saved to {save_path}")

    return pred_label, confidence


def generate_gradcam_examples(model, test_dir, device="cuda",
                              save_dir="outputs/figures", n_examples=3):
    """Generate Grad-CAM examples for both classes.
    
    Picks random correctly-classified and misclassified examples
    to demonstrate model attention patterns.
    """
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])

    for label in ["NORMAL", "PNEUMONIA"]:
        label_dir = os.path.join(test_dir, label)
        if not os.path.exists(label_dir):
            continue

        images = os.listdir(label_dir)
        count = 0
        for img_name in images:
            if count >= n_examples:
                break
            img_path = os.path.join(label_dir, img_name)
            try:
                generate_gradcam(model, img_path, label, device, save_dir)
                count += 1
            except Exception as e:
                print(f"Skipping {img_name}: {e}")
                continue

    print(f"\nGenerated Grad-CAM examples in {save_dir}")
