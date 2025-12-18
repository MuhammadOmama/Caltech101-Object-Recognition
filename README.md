# 🔍 Caltech-101 Object Recognition App

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-ResNet18-red)
![Streamlit](https://img.shields.io/badge/Streamlit-App-FF4B4B)

An end-to-end image classification system capable of recognizing **101 object categories**. This project utilizes **Transfer Learning** with a ResNet-18 architecture and includes a user-friendly web interface for real-time predictions via image upload or webcam.

## 📂 Dataset

The model is trained on the **Caltech-101 Dataset**.

* **Source:** [Caltech-101 (CaltechData)](https://data.caltech.edu/records/mzrjq-6wc02)
* **Description:** Pictures of objects belonging to 101 categories (e.g., airplanes, cameras, pandas, wild cats) plus a background category.
* **Setup:**
    1.  Download the dataset from the link above.
    2.  Extract the contents.
    3.  Ensure the folder `101_ObjectCategories` is placed in your project directory (or update the path in `object-classifier-model.ipynb` before training).

## 🌟 Features

* **Deep Learning Model:** Fine-tuned **ResNet-18** (pre-trained on ImageNet).
* **Advanced Training:** Implements "Freeze-Backbone" strategy followed by fine-tuning of deep layers (`layer4`).
* **Data Augmentation:** Uses Random Rotation, Color Jitter, and Random Resized Crop to improve model generalization.
* **Interactive Web App:** Built with **Streamlit**.
    * **Upload Mode:** Drag and drop images for instant classification.
    * **Webcam Mode:** Real-time object detection using your computer's camera.
* **CPU Optimization:** The inference app is optimized to run smoothly on standard CPUs.

## 🛠️ Installation

### 1. Clone the repository
```bash
git clone https://github.com/MuhammadOmama/Caltech101-Object-Recognition
cd Caltech101-Object-Recognition
```
### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Model Setup
If the trained model file `resnet18_caltech101_generalized.pth` is not in the repo (due to size limits), you must run the `object-classifier-model.ipynb` notebook to train it and generate the file.

## 🚀 Usage

### 1. Running the Web App
To start the interface, run the following command in your terminal:

```bash
streamlit run app.py
```
The app will open automatically in your browser at `http://localhost:8501`

### 2. Retraining the Model
If you want to improve the model or train it from scratch:

1.  Open `object-classifier-model.ipynb` in Jupyter Notebook or VS Code or Kaggle.
2.  Update the `dataset_root` variable to point to your local `101_ObjectCategories` folder.
3.  Run all cells. This will generate a new `resnet18_caltech101_generalized.pth` file.

## 📁 Project Structure

```text
Caltech101-Object-Recognition/
│
├── object-classifier-model.ipynb      # Training notebook (Data prep, Fine-tuning, Evaluation)
├── app.py                             # Streamlit deployment script (Inference logic)
├── requirements.txt                   # List of dependencies
├── README.md                          # Project documentation
├── .gitignore                         # Files to exclude from Git
└── resnet18_caltech101_generalized.pth  # The trained model weights
```

## 🧠 Model Performance & Architecture

* **Architecture:** ResNet-18
* **Input Size:** 224x224 pixels
* **Classes:** 101 (Background class removed during preprocessing)
* **Training Method:**
    * **Phase 1:** Freeze feature extractor, train classifier head (FC layer) for 8 epochs.
    * **Phase 2:** Unfreeze `layer4` and fine-tune with a lower learning rate (`1e-4`) for 10 epochs.
* **Accuracy:** The model achieves high accuracy (approx 98-99% on validation split) due to the generalized pre-training on ImageNet.

## 📜 Credits

* **Dataset:** Fei-Fei Li, Marco Andreetto, and Marc 'Aurelio Ranzato. [Caltech 101](https://data.caltech.edu/records/mzrjq-6wc02).
* **Frameworks:** PyTorch, Torchvision, Streamlit.
