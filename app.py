"""
Gradio web app — Plant Disease Detection.
Runs locally: python app.py
"""

import gradio as gr
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
from pathlib import Path

import config
from model import build_model

CHECKPOINT = config.CHECKPOINT_DIR / "best_model.pth"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------------------------------------------------------------------------
# Load model
# ---------------------------------------------------------------------------
ckpt = torch.load(CHECKPOINT, map_location=DEVICE, weights_only=True)
arch = ckpt.get("arch", "resnet18")
num_classes = ckpt.get("num_classes", config.NUM_CLASSES)
CLASS_NAMES = ckpt.get("class_names") or [f"class_{i}" for i in range(num_classes)]

model = build_model(arch=arch, num_classes=num_classes, pretrained=False)
model.load_state_dict(ckpt["model_state_dict"])
model.to(DEVICE).eval()

transform = transforms.Compose([
    transforms.Resize((config.IMG_SIZE, config.IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(config.IMAGENET_MEAN, config.IMAGENET_STD),
])

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def is_healthy(class_name: str) -> bool:
    return "healthy" in class_name.lower()

def format_label(raw: str) -> str:
    """Apple___Black_rot  →  Apple — Black rot"""
    return raw.replace("___", " — ").replace("_", " ").title()

# ---------------------------------------------------------------------------
# Prediction
# ---------------------------------------------------------------------------
def predict(image):
    if image is None:
        return {}, "Загрузите фотографию листика."

    img = Image.fromarray(image).convert("RGB")
    tensor = transform(img).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        probs = torch.softmax(model(tensor), dim=1).squeeze()

    top_probs, top_idx = probs.topk(5)
    top_predictions = {
        format_label(CLASS_NAMES[i]): float(p)
        for i, p in zip(top_idx.tolist(), top_probs.tolist())
    }

    best_class = CLASS_NAMES[top_idx[0].item()]
    confidence = top_probs[0].item()
    label = format_label(best_class)

    if is_healthy(best_class):
        status = f"## Растение здорово\n**{label}**\nУверенность: {confidence:.1%}"
    else:
        status = f"## Обнаружена болезнь\n**{label}**\nУверенность: {confidence:.1%}"

    return top_predictions, status


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------
with gr.Blocks(title="Определение болезней растений", theme=gr.themes.Soft()) as demo:
    gr.Markdown(
        "# Определение болезней растений\n"
        "Загрузите фото листика — модель определит культуру и состояние здоровья.\n\n"
        "**Поддерживается:** 14 культур, 38 категорий | Точность: 98.8%"
    )

    with gr.Row():
        with gr.Column():
            img_input = gr.Image(label="Фото листика")
            btn = gr.Button("Определить", variant="primary", size="lg")
        with gr.Column():
            result_label = gr.Label(num_top_classes=5, label="Топ-5 предсказаний")
            result_text = gr.Markdown(label="Диагноз")

    btn.click(fn=predict, inputs=img_input, outputs=[result_label, result_text])
    img_input.change(fn=predict, inputs=img_input, outputs=[result_label, result_text])

    gr.Markdown(
        "---\n"
        "Модель: ResNet18 | Датасет: PlantVillage (54 305 изображений) | "
        "[GitHub](https://github.com/shatini/crop-disease-detection-ai)"
    )

if __name__ == "__main__":
    demo.launch()
