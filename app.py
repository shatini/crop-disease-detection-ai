"""
Gradio demo — Skin Cancer Detection (HAM10000).
Upload best_model.pth (MobileNetV2, 7 classes) to the Space root.
"""

import gradio as gr
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image

CLASS_NAMES = ["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"]

CLASS_DESCRIPTIONS = {
    "akiec": "Actinic Keratoses (precancerous)",
    "bcc":   "Basal Cell Carcinoma",
    "bkl":   "Benign Keratosis",
    "df":    "Dermatofibroma",
    "mel":   "Melanoma (malignant)",
    "nv":    "Melanocytic Nevi (mole)",
    "vasc":  "Vascular Lesion",
}

RISK_LEVELS = {
    "akiec": "Medium — precancerous, monitor closely",
    "bcc":   "Medium — most common skin cancer, usually treatable",
    "bkl":   "Low — benign growth",
    "df":    "Low — benign fibrous nodule",
    "mel":   "HIGH — potentially deadly, seek immediate evaluation",
    "nv":    "Low — common mole, usually benign",
    "vasc":  "Low — benign vascular growth",
}

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load model
ckpt = torch.load("best_model.pth", map_location=DEVICE, weights_only=True)
arch = ckpt.get("arch", "mobilenet_v2")

if arch == "mobilenet_v2":
    model = models.mobilenet_v2(weights=None)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, len(CLASS_NAMES))
elif arch == "efficientnet_b0":
    model = models.efficientnet_b0(weights=None)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, len(CLASS_NAMES))
elif arch == "resnet18":
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, len(CLASS_NAMES))

model.load_state_dict(ckpt["model_state_dict"])
model.to(DEVICE)
model.eval()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


def predict(image):
    if image is None:
        return {}, ""
    img = Image.fromarray(image).convert("RGB")
    tensor = transform(img).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        logits = model(tensor)
        probs = torch.softmax(logits, dim=1)[0]

    pred_idx = probs.argmax().item()
    pred_class = CLASS_NAMES[pred_idx]

    # Build warning message
    desc = CLASS_DESCRIPTIONS[pred_class]
    risk = RISK_LEVELS[pred_class]
    confidence = probs[pred_idx].item()

    if pred_class == "mel":
        warning = (
            f"### Prediction: {desc}\n"
            f"**Confidence: {confidence:.1%}**\n\n"
            f"**Risk: {risk}**\n\n"
            "> **WARNING:** Melanoma detected. "
            "Please consult a dermatologist immediately."
        )
    elif pred_class in ("bcc", "akiec"):
        warning = (
            f"### Prediction: {desc}\n"
            f"**Confidence: {confidence:.1%}**\n\n"
            f"**Risk: {risk}**\n\n"
            "> **NOTICE:** Potentially concerning lesion. "
            "Professional evaluation recommended."
        )
    else:
        warning = (
            f"### Prediction: {desc}\n"
            f"**Confidence: {confidence:.1%}**\n\n"
            f"**Risk: {risk}**"
        )

    labels = {CLASS_DESCRIPTIONS[CLASS_NAMES[i]]: float(probs[i]) for i in range(len(CLASS_NAMES))}
    return labels, warning


with gr.Blocks(title="Skin Cancer Detector", theme=gr.themes.Soft()) as demo:
    gr.Markdown(
        "# Skin Cancer Detection\n"
        "Upload a dermoscopic image of a skin lesion to classify it.\n\n"
        "Model: MobileNetV2 | Dataset: HAM10000 | 7 diagnostic categories\n\n"
        "> **Disclaimer:** This tool is for educational purposes only. "
        "It does NOT replace professional medical diagnosis."
    )

    with gr.Row():
        with gr.Column():
            image_input = gr.Image(label="Dermoscopic Image")
            btn = gr.Button("Analyze Lesion", variant="primary", size="lg")
        with gr.Column():
            output = gr.Label(num_top_classes=7, label="Classification")
            warning_box = gr.Markdown(label="Assessment")

    btn.click(fn=predict, inputs=image_input, outputs=[output, warning_box])

    with gr.Accordion("Lesion Types Reference", open=False):
        gr.Markdown(
            "\n".join(
                f"- **{k.upper()}** — {v} | Risk: {RISK_LEVELS[k]}"
                for k, v in CLASS_DESCRIPTIONS.items()
            )
        )

    gr.Markdown(
        "---\n"
        "Built by [Nikolai Shatikhin](https://github.com/shatini) "
        "| [Source Code](https://github.com/shatini/skin-cancer-detection-ai)"
    )

if __name__ == "__main__":
    demo.launch()
