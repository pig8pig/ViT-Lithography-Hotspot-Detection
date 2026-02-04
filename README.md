# ViT-Lithography-Hotspot-Detection

Semiconductor Hotspot Detection with Vision Transformers
This repository contains the implementation of a Vision Transformer (ViT) pipeline designed to detect manufacturing defects ("hotspots") in semiconductor lithography layouts.

The project addresses the challenge of identifying rare defects in high-dimensional image data, utilizing a pre-trained vit_base_patch16_224 model fine-tuned on the ICCAD industrial dataset.

🚀 Key Features
Architecture: Fine-tuned Vision Transformer (ViT-B/16) using the timm library.

Transfer Learning: Leverages pre-trained ImageNet weights to extract robust features from lithography patterns, adapting a general-purpose vision model to a specialized industrial task.

Data Pipeline: Custom PyTorch DataLoader with torchvision.transforms for resizing (224x224) and normalization.

Optimization: Implemented a custom training loop with Adam optimizer and CrossEntropyLoss, achieving convergence on a binary classification task (Hotspot vs. Non-Hotspot).

🛠️ Tech Stack
Language: Python

Deep Learning Framework: PyTorch, timm (Torch Image Models)

Data Processing: torchvision, numpy, PIL

Visualization: matplotlib

Environment: Designed for Google Colab (GPU accelerated)

📂 Dataset Structure
The project utilizes the ICCAD (International Conference on Computer-Aided Design) contest dataset. The code expects the following directory structure:

iccad1_modified/
├── train/
│   ├── Hotspot/       # Images containing lithography defects
│   └── Not_Hotspot/   # Clean layout images
└── validation/
    ├── Hotspot/
    └── Not_Hotspot/
(Note: The notebook currently includes code to mount Google Drive and unzip the dataset automatically. Ensure paths match your local or cloud environment.)

⚙️ Installation & Usage
Clone the repository:

Bash
git clone https://github.com/your-username/semiconductor-vit-research.git
Install dependencies:

Bash
pip install torch torchvision timm tqdm matplotlib
Run the Notebook:
Open the .ipynb file in Google Colab or Jupyter Notebook.
Note: Enable GPU Runtime for efficient training.

📊 Methodology
The pipeline follows a standard Transfer Learning approach:

Preprocessing: Images are resized to 224x224 and normalized to mean/std (0.5).

Model Definition: A vit_base_patch16_224 model is loaded from timm with pre-trained weights.

Feature Extraction: The backbone parameters are frozen (requires_grad = False) to prevent overfitting on the smaller dataset.

Fine-Tuning: The classification head (net.head) is replaced and trained to output class probabilities for the specific Hotspot/Non-Hotspot classes.

📈 Training Configuration
Epochs: 15

Batch Size: 100

Learning Rate: 1e-3

Loss Function: CrossEntropyLoss

📝 Author
Yuzhong (WeiWei) Luo Engineering Science, University of Oxford Research focus: AI for Industrial Defect Detection & Optimization
