"""
Gradio demo — Metal Surface Defect Detection.
Upload best_model.pth (EfficientNet-B0, 6 classes) to the Space root.
"""

import gradio as gr
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image

CLASS_NAMES = [
    "Crazing", "Inclusion", "Patches",
    "Pitted_Surface", "Rolled-in_Scale", "Scratches",
]

CLASS_DESCRIPTIONS = {
    "Crazing": "Fine network of surface cracks",
    "Inclusion": "Foreign material embedded in surface",
    "Patches": "Irregular surface discoloration",
    "Pitted_Surface": "Small holes or cavities",
    "Rolled-in_Scale": "Oxide scale pressed into surface",
    "Scratches": "Linear surface damage",
}

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load model
model = models.efficientnet_b0(weights=None)
model.classifier[1] = nn.Linear(model.classifier[1].in_features, len(CLASS_NAMES))
model.load_state_dict(torch.load("best_model.pth", map_location=DEVICE))
model.to(DEVICE)
model.eval()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


def predict(image):
    if image is None:
        return {}
    img = Image.fromarray(image).convert("RGB")
    tensor = transform(img).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        logits = model(tensor)
        probs = torch.softmax(logits, dim=1)[0]

    return {CLASS_NAMES[i]: float(probs[i]) for i in range(len(CLASS_NAMES))}


with gr.Blocks(title="Metal Defect Detector", theme=gr.themes.Soft()) as demo:
    gr.Markdown(
        "# Metal Surface Defect Detection\n"
        "Upload a steel surface image to identify the defect type.\n\n"
        "Model: EfficientNet-B0 | Accuracy: 96.7% | 6 defect classes"
    )

    with gr.Row():
        with gr.Column():
            image_input = gr.Image(label="Steel Surface Image")
            btn = gr.Button("Detect Defect", variant="primary", size="lg")
        with gr.Column():
            output = gr.Label(num_top_classes=6, label="Prediction")
            gr.Markdown(
                "### Defect Types\n" +
                "\n".join(f"- **{k}** — {v}" for k, v in CLASS_DESCRIPTIONS.items())
            )

    btn.click(fn=predict, inputs=image_input, outputs=output)

    gr.Markdown(
        "---\n"
        "Built by [Nikolai Shatikhin](https://github.com/shatini) "
        "| [Source Code](https://github.com/shatini/metal-defects-ai)"
    )

if __name__ == "__main__":
    demo.launch()
