#🔍 Caltech-101 Object Recognition App
An end-to-end image classification system capable of recognizing 101 object categories. This project utilizes Transfer Learning with a ResNet-18 architecture and includes a user-friendly web interface for real-time predictions via image upload or webcam.

📂 Dataset
The model is trained on the Caltech-101 Dataset.


Source: Caltech-101 (CaltechData) 

Description: Pictures of objects belonging to 101 categories (e.g., airplanes, cameras, pandas, wild cats) plus a background category.

Setup:

Download the dataset from the link above.

Extract the contents.

Ensure the folder 101_ObjectCategories is placed in your project directory (or update the path in ai-project.ipynb before training).

🌟 Features
Deep Learning Model: Fine-tuned ResNet-18 (pre-trained on ImageNet).

Advanced Training: Implements "Freeze-Backbone" strategy followed by fine-tuning of deep layers (layer4).

Data Augmentation: Uses Random Rotation, Color Jitter, and Random Resized Crop to improve model generalization.


Interactive Web App: Built with Streamlit.

Upload Mode: Drag and drop images for instant classification.

Webcam Mode: Real-time object detection using your computer's camera.

CPU Optimization: The inference app is optimized to run smoothly on standard CPUs.

🛠️ Installation
Clone the repository:

Bash

git clone https://github.com/YOUR-USERNAME/Caltech101-Object-Recognition.git
cd Caltech101-Object-Recognition
Install dependencies: (Ensure you have renamed requiremnets.txt to requirements.txt first)

Bash

pip install -r requirements.txt
Model Setup:

If the trained model file resnet18_caltech101_generalized.pth is not in the repo (due to size limits), you must run the ai-project.ipynb notebook to train it and generate the file.

🚀 Usage
1. Running the Web App
To start the interface, run the following command in your terminal:

Bash

streamlit run app.py
The app will open automatically in your browser at http://localhost:8501.

2. Retraining the Model
If you want to improve the model or train it from scratch:

Open ai-project.ipynb in Jupyter Notebook or VS Code.

Update the dataset_root variable to point to your local 101_ObjectCategories folder.

Run all cells. This will generate a new resnet18_caltech101_generalized.pth file.

📁 Project Structure
Plaintext

Caltech101-Object-Recognition/
│
├── ai-project.ipynb      # Training notebook (Data prep, Fine-tuning, Evaluation)
├── app.py                # Streamlit deployment script (Inference logic)
├── requirements.txt      # List of dependencies
├── README.md             # Project documentation
├── .gitignore            # Files to exclude from Git
└── resnet18_caltech101_generalized.pth  # The trained model weights
🧠 Model Performance & Architecture

Architecture: ResNet-18 

Input Size: 224x224 pixels

Classes: 101 (Background class removed during preprocessing)

Training Method:

Phase 1: Freeze feature extractor, train classifier head (FC layer) for 8 epochs.

Phase 2: Unfreeze layer4 and fine-tune with a lower learning rate (1e-4) for 10 epochs.

Accuracy: The model achieves high accuracy (approx 98-99% on validation split) due to the generalized pre-training on ImageNet.

📜 Credits
Dataset: Fei-Fei Li, Marco Andreetto, and Marc 'Aurelio Ranzato. Caltech 101. 

Frameworks: PyTorch, Torchvision, Streamlit.
