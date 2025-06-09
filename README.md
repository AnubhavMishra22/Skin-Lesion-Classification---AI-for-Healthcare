Hybrid CNN–ViT for Skin Lesion Classification for ECS 289L AI for Healthcare

📚 Project Overview
This project explores a hybrid deep learning model that combines Convolutional Neural Networks (CNNs) and Vision Transformers (ViTs) for the classification of skin lesions. CNNs specialise in capturing local features like textures and edges, while ViTs excel at modelling long-range dependencies and global context. By integrating these two architectures, the hybrid model aims to leverage their complementary strengths for improved lesion classification accuracy.
The model is trained and evaluated on the BCN20000 dataset, a large and diverse collection of dermoscopic images. We also investigate the role of data augmentation techniques in enhancing model generalisation, particularly in the context of large-scale datasets.

📦 Key Features
Hybrid CNN–ViT Architecture: Combines local and global feature extraction.
Baseline Comparisons: Performance compared against ResNet50, EfficientNetB0, and standalone ViT models.
BCN20000 Dataset: Used for training and evaluation to ensure diversity and realism.
Data Augmentation: Evaluated to understand its impact on model generalisation.
Evaluation Metrics: Accuracy, sensitivity (recall), specificity, confusion matrix.

🛠️ Technologies Used
Python 3.8+
PyTorch
Transformers
Scikit-learn
Matplotlib / Seaborn

🚀 How to Run

1. Clone the repository:
git clone https://github.com/AnubhavMishra22/Skin-Lesion-Classification---AI-for-Healthcare
cd skin-lesion-hybrid-cnn-vit

2. Install dependencies:
pip install -r requirements.txt

3. Download the BCN20000 dataset and place it in the data/ directory.

4. Run the Jupyter Notebook:
Jupyter notebook vit-bcn20000-project.ipynb


📈 Results
Hybrid CNN–ViT models outperform standalone architectures.
Data augmentation improves generalisation, especially for underrepresented classes.
Detailed classification reports and confusion matrices are provided.


📚 Primary References
[Dosovitskiy et al., 2020] — Vision Transformer (ViT)

[Jasil and Ulagamuthalvi, 2023] — Hybrid CNN Architectures

[Perez et al., 2023] — BCN20000 Dataset

[Debelee et al., 2023] — Survey on Skin Lesion Classification
