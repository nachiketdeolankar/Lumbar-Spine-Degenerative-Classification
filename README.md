# Lumbar-Spine-Degenerative-Classification

This repository contains Python scripts and resources for classifying degenerative conditions in the lumbar spine using MRI images. Leveraging a Visual Transformer (ViT) model, the project aims to enhance the accuracy and efficiency of medical image analysis for conditions like spinal stenosis and neural foraminal narrowing.

## Project Overview

According to the World Health Organization, back pain is a leading cause of disability, affecting approximately 620 million people worldwide in 2020. This project focuses on classifying five lumbar spine degenerative conditions across various severity levels using AI-powered solutions.

### Conditions Analyzed
- Left Neural Foraminal Narrowing
- Right Neural Foraminal Narrowing
- Left Subarticular Stenosis
- Right Subarticular Stenosis
- Spinal Canal Stenosis

### Dataset
The dataset, provided by the American Society of Neuroradiology (ASNR) and Kaggle, includes:
- **1976 studies**
- **155,488 `.dcm` image files**
- Severity labels: Normal/Mild, Moderate, Severe
- Intervertebral disc levels: L1/L2 through L5/S1

[Kaggle Dataset Link](https://www.kaggle.com/competitions/rsna-2024-lumbar-spine-degenerative-classification)

---

## Methodology

### Model Architecture: Visual Transformer (ViT)
- Utilizes self-attention mechanisms to process images as patches.
- Inspired by the [original ViT paper](https://arxiv.org/abs/2010.11929), this architecture eliminates the need for convolutional layers, enabling scalable and efficient classification.

### Key Steps
1. Data preprocessing: Loading and normalizing `.dcm` files.
2. Image patching: Segmenting MRIs into manageable patches.
3. Model training: Optimizing using loss minimization.
4. Evaluation: Achieving high accuracy with significant insights into degenerative patterns.

---

## Results
- **Accuracy**: 95.69%
- **Loss**: 0.1683
- **Key Findings**:
  - Strong correlation between left and right subarticular stenosis across multiple levels.
  - Potential for real-world integration into clinical workflows to aid radiologists.

---

## Repository Contents
- **`main.py`**: Primary script to manage data preprocessing, model training, and evaluation.
- **`data_analytics.py`**: Contains data preprocessing and visualization functions, including correlation matrices.
- **`visual_transformer.py`**: Implements the Visual Transformer model and training pipeline.

---

## Setup Instructions
1. Clone the repository:
   ```bash
   git clone https://github.com/nachiketdeolankar/lumbar-spine-degenerative-classification.git

2. Navigate to the project directory:
   ```bash
   cd lumbar-spine-classification

3. Install required packages:
   ```bash
   pip install -r requirements.txt

4. Download the dataset from Kaggle and place it in the data/ directory.

5. Run the main script:
  ```bash
  python main.py

---

## **Future Work**
- **Integration with clinical diagnostic tools**: Exploring real-world applications to assist radiologists in diagnostic workflows.
- **Expansion to classify other spinal conditions**: Broadening the scope to include additional conditions and severities.
- **Incorporation of larger datasets**: Improving model generalization and robustness with more diverse datasets.
- **Exploration of hybrid models**: Combining transformer-based architectures with convolutional neural networks (CNNs) for enhanced feature extraction.

---

## **Acknowledgments**
- **American Society of Neuroradiology (ASNR)** and **Kaggle** for providing the dataset and organizing the competition.
- **Visual Transformer (ViT) development team** for their foundational research ([ViT paper](https://arxiv.org/abs/2010.11929)).
- Special thanks to contributors and collaborators for their insights and support in advancing this project.

  
