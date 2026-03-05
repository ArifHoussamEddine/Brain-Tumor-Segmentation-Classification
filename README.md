# High-Accuracy Segmentation and Classification of Brain Tumors in MRI Using Genetically-Tuned Deep Learning

![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)
![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![Framework](https://img.shields.io/badge/Framework-Keras%20/%20TensorFlow-orange)

## 📄 Overview
This repository contains the official implementation of our research paper. The project presents an end-to-end automated pipeline for brain tumor diagnosis from MRI scans, combining **Semantic Segmentation** (U-Net) and **Multi-class Classification** optimized by **Genetic Algorithms (GA)**.

## 🚀 Key Features
- **Hybrid Architecture:** Integration of U-Net for precise lesion localization and CNNs for tumor type classification.
- **Evolutionary Optimization:** Use of Genetic Algorithms to automate hyperparameter tuning (Learning Rate, Dropout, Batch Size).
- **Comprehensive Pipeline:** Includes image preprocessing (Normalization, Augmentation), Training, Evaluation, and a Deployment-ready Web App.
- **High Performance:** Achieved state-of-the-art results in Dice Coefficient and Accuracy as detailed in the publication.

## 📂 Project Structure
- `Dataset/`: Sample MRI slices (Images & Masks) following the Kaggle structure.
- `Classification/`: Baseline CNN models for tumor categorization.
- `Segmentation/`: Progressive versions (v1-v6) of the U-Net implementation.
- `Classification GA/`: The core innovation—GA-based optimization scripts.
- `Web App/`: Flask-based server for real-time tumor prediction.

## 🔬 Methodology
Our approach follows three main stages:
1. **Preprocessing:** Intensity normalization and spatial resizing.
2. **Segmentation:** U-Net captures fine-grained spatial features to outline the tumor.
3. **GA-Optimization:** A population-based search to find the most efficient neural network configuration.



## 🎓 Citation
If you use this code or refer to our paper, please cite:
> **Arif, H. E., et al.** "High-Accuracy Segmentation and Classification of Brain Tumors in MRI Using a Genetically-Tuned Deep Learning Model." (Submitted to International Conference).

---
**Contact:** Houssam Eddine Arif - h.arif.inf@lagh-univ.dz
