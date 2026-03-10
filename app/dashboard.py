import streamlit as st
import torch
from torchvision import transforms
from PIL import Image
import numpy as np
from pytorch_grad_cam import GradCAMPlusPlus
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
import sys
import os
from scipy.ndimage import gaussian_filter

# Add project root to path so we can import from src
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.model import get_efficientnet


@st.cache(allow_output_mutation=True)
def load_model():
    """Load the trained EfficientNet model from checkpoint."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = get_efficientnet(num_classes=2, freeze_features=False)

    model_path = os.path.join(os.path.dirname(__file__), "..",
                              "outputs", "models", "best_model.pth")
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model.to(device), device


def get_transform():
    """Standard inference transform (no augmentation)."""
    return transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])


def main():
    # Page config
    st.set_page_config(
        page_title="Pneumonia Detector",
        page_icon="+",
        layout="wide"
    )

    # Header
    st.title("Chest X-Ray Pneumonia Detector")
    st.markdown(
        "Upload a chest X-ray image to classify it as **Normal** or **Pneumonia**. "
        "The model uses EfficientNet-B0 fine-tuned on 5,800+ labeled X-ray images."
    )
    st.caption("Disclaimer: For educational purposes only — not for clinical diagnosis.")

    st.markdown("---")

    # Load model
    try:
        model, device = load_model()
        st.sidebar.success(f"Model loaded on **{device.upper()}**")
    except Exception as e:
        st.error(f"Failed to load model: {e}")
        st.info("Make sure you've trained the model first by running `python run_training.py`")
        return

    # File uploader
    uploaded = st.file_uploader(
        "Upload Chest X-Ray (JPG/PNG)",
        type=["jpg", "jpeg", "png"],
        help="Upload a frontal chest X-ray image for classification"
    )

    if uploaded:
        image = Image.open(uploaded).convert("RGB")

        # Classify
        transform = get_transform()
        input_tensor = transform(image).unsqueeze(0).to(device)

        with torch.no_grad():
            output = model(input_tensor)
            prob = torch.softmax(output, dim=1)[0]
            pred_class = torch.argmax(prob).item()

        label = "PNEUMONIA" if pred_class == 1 else "NORMAL"
        confidence = prob[pred_class].item()

        # Results layout
        col_left, col_right = st.columns([1, 1])

        with col_left:
            st.subheader("Classification Result")
            if label == "PNEUMONIA":
                st.error(f"**Prediction: {label}**")
            else:
                st.success(f"**Prediction: {label}**")

            st.metric("Confidence", f"{confidence:.1%}")
            st.progress(confidence)

            st.markdown("**Class Probabilities:**")
            import pandas as pd
            st.bar_chart(pd.DataFrame(
                {"Probability": [prob[0].item(), prob[1].item()]},
                index=["Normal", "Pneumonia"]
            ))

        with col_right:
            st.subheader("Grad-CAM Explanation")
            st.caption("Highlighted regions show where the model focused its attention.")

            # Generate Grad-CAM
            raw_img = image.copy()
            raw_img = transforms.Resize(256)(raw_img)
            raw_img = transforms.CenterCrop(224)(raw_img)
            raw_np = np.array(raw_img) / 255.0
            target_layers = [model.features[6]]

            with GradCAMPlusPlus(model=model, target_layers=target_layers) as cam:
                grayscale_cam = cam(input_tensor=input_tensor)[0]

            # Smooth heatmap for cleaner attention regions
            grayscale_cam = gaussian_filter(grayscale_cam, sigma=10)
            grayscale_cam = (grayscale_cam - grayscale_cam.min()) / (grayscale_cam.max() - grayscale_cam.min() + 1e-8)

            cam_image = show_cam_on_image(
                raw_np.astype(np.float32), grayscale_cam, use_rgb=True
            )

            st.image(raw_img, caption="Original X-Ray", width=224)
            st.image(cam_image, caption="Grad-CAM Heatmap", width=224)

        st.markdown("---")
        st.markdown(
            "**How to interpret Grad-CAM:** Red/warm regions indicate areas the model "
            "considers most important for its prediction. In a well-trained model, these "
            "regions should overlay the **lung fields**, not image borders or labels."
        )


if __name__ == "__main__":
    main()
