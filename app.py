"""
Gradio demo — Crop Disease Detection.
Upload best_model.pth (ResNet18, 38 classes) and class_names.txt to the Space root.
"""

import gradio as gr
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
from pathlib import Path

CLASS_NAMES = [
    "Apple___Apple_scab", "Apple___Black_rot", "Apple___Cedar_apple_rust", "Apple___healthy",
    "Blueberry___healthy",
    "Cherry_(including_sour)___Powdery_mildew", "Cherry_(including_sour)___healthy",
    "Corn_(maize)___Cercospora_leaf_spot", "Corn_(maize)___Common_rust",
    "Corn_(maize)___Northern_Leaf_Blight", "Corn_(maize)___healthy",
    "Grape___Black_rot", "Grape___Esca_(Black_Measles)",
    "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)", "Grape___healthy",
    "Orange___Haunglongbing_(Citrus_greening)",
    "Peach___Bacterial_spot", "Peach___healthy",
    "Pepper,_bell___Bacterial_spot", "Pepper,_bell___healthy",
    "Potato___Early_blight", "Potato___Late_blight", "Potato___healthy",
    "Raspberry___healthy",
    "Soybean___healthy",
    "Squash___Powdery_mildew",
    "Strawberry___Leaf_scorch", "Strawberry___healthy",
    "Tomato___Bacterial_spot", "Tomato___Early_blight", "Tomato___Late_blight",
    "Tomato___Leaf_Mold", "Tomato___Septoria_leaf_spot",
    "Tomato___Spider_mites Two-spotted_spider_mite", "Tomato___Target_Spot",
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus", "Tomato___Tomato_mosaic_virus",
    "Tomato___healthy",
]

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = models.resnet18(weights=None)
model.fc = nn.Linear(model.fc.in_features, len(CLASS_NAMES))
model.load_state_dict(torch.load("best_model.pth", map_location=DEVICE))
model.to(DEVICE)
model.eval()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


def format_label(name):
    crop, condition = name.split("___")
    crop = crop.replace("_", " ")
    condition = condition.replace("_", " ")
    return f"{crop} — {condition}"


def predict(image):
    if image is None:
        return {}
    img = Image.fromarray(image).convert("RGB")
    tensor = transform(img).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        logits = model(tensor)
        probs = torch.softmax(logits, dim=1)[0]

    top5_probs, top5_idx = torch.topk(probs, 5)
    return {format_label(CLASS_NAMES[i]): float(p) for p, i in zip(top5_probs, top5_idx)}


with gr.Blocks(title="Crop Disease Detector", theme=gr.themes.Soft()) as demo:
    gr.Markdown(
        "# Crop Disease Detection\n"
        "Upload a leaf photo to identify the plant disease.\n\n"
        "Model: ResNet18 | Accuracy: 98.8% | 38 categories | 14 crop species"
    )

    with gr.Row():
        with gr.Column():
            image_input = gr.Image(label="Leaf Photo")
            btn = gr.Button("Identify Disease", variant="primary", size="lg")
        with gr.Column():
            output = gr.Label(num_top_classes=5, label="Top-5 Predictions")

    btn.click(fn=predict, inputs=image_input, outputs=output)

    gr.Markdown(
        "**Supported crops:** Apple, Blueberry, Cherry, Corn, Grape, Orange, "
        "Peach, Pepper, Potato, Raspberry, Soybean, Squash, Strawberry, Tomato\n\n---\n"
        "Built by [Nikolai Shatikhin](https://github.com/shatini) "
        "| [Source Code](https://github.com/shatini/crop-disease-detection-ai)"
    )

if __name__ == "__main__":
    demo.launch()
