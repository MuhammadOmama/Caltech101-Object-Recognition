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
    3.  Ensure the folder `101_ObjectCategories` is placed in your project directory (or update the path in `ai-project.ipynb` before training).

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
git clone [https://github.com/YOUR-USERNAME/Caltech101-Object-Recognition.git](https://github.com/YOUR-USERNAME/Caltech101-Object-Recognition.git)
cd Caltech101-Object-Recognition
