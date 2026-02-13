Plant Disease Classifier — AgroAI
A deep learning model that looks at a photo of a plant leaf and tells you what’s wrong with it. Trained on 43,000+ images across 38 disease categories.
Final accuracy: 98.8% on validation set.
 
Why this exists
Farmers lose billions every year because diseases get caught too late. This model can identify 38 different plant conditions from a single photo — no lab, no expert, no waiting.
 
What it can detect
38 classes across crops including tomato, potato, corn, grape, apple, pepper and more.
Examples:
 
How it was built
Dataset: PlantVillage — 43,444 training images, 10,861 validation images
Model: ResNet18 pretrained on ImageNet, fine-tuned for plant diseases
Framework: PyTorch
Training: 15 epochs on GPU (Google Colab T4)
Results
Epoch	Train Acc	Val Acc
1	90.9%	95.5%
5	97.9%	97.7%
10	98.8%	98.0%
12	99.0%	98.8%
15	99.3%	98.5%
 
Project structure
agro-ai-portfolio/
├── train.py          # Training script
├── README.md
└── models/
    └── agro_ai_resnet18.pth   # Trained weights
    
## Training visualization

![Training Progress](https://github.com/JohnDeepInside/agro-ai-portfolio/raw/main/training_plot.png)

Run it yourself
# Install dependencies pip install torch torchvision
# Train from scratch python train.py

## Model Weights

Trained model weights (`agro_ai_resnet18.pth`) are not included in this repository due to file size (44MB).

To obtain the trained model:
- Train from scratch using `train.py` (recommended for learning)
- Contact for pre-trained weights: Available upon request

Tech stack
Python 3.12
 
Author
Built by Nikolai Shatikhin as part of an AI/ML portfolio. Open to freelance projects in computer vision and image classification.
Reach out via GitHub issues or direct message.
