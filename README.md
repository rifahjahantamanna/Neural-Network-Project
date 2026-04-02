# 🎵 Multimodal Music Genre Classification using VAE

This project explores **music genre classification** using a **multimodal deep learning approach**, combining both **audio features** and **song lyrics**. A Variational Autoencoder (VAE / Beta-VAE) is used to learn meaningful latent representations, followed by clustering and evaluation.

---

## 🚀 Project Overview

The goal of this project is to:
- Learn **latent representations** from music data
- Combine **audio (MFCC features)** and **lyrics (TF-IDF features)**
- Perform **unsupervised clustering**
- Evaluate clustering quality using standard metrics

---

## 📂 Datasets Used

### 1. GTZAN Dataset
- Audio dataset with 10 genres
- Extracted features:
  - MFCCs (Mel-frequency cepstral coefficients)
- Used for:
  - Audio representation learning
  - Clustering baseline

### 2. Song Lyrics Dataset
- English song lyrics dataset
- Columns include:
  - `title`, `artist`, `tag`, `lyrics`
- Used for:
  - Text feature extraction (TF-IDF)
  - Lyrics embedding via Autoencoder

---

## 🧠 Methodology

### 🔹 Audio Processing
- Extracted MFCC features
- Used as input to a **Variational Autoencoder (VAE)**

### 🔹 Lyrics Processing
- Cleaned text (lowercase, remove punctuation, etc.)
- Converted to numerical form using:
  - **TF-IDF Vectorization**
- Encoded using a **Neural Autoencoder**

### 🔹 Multimodal Fusion
- Combined:
  - Audio latent vectors
  - Lyrics latent vectors
- Created a **joint latent space**

---

## 🔬 Models Used

- **Beta-VAE (for audio features)**
- **Autoencoder (for lyrics)**
- **K-Means Clustering** on latent representations

---

## 📊 Evaluation Metrics

Clustering performance is evaluated using:

- **Silhouette Score**
- **Normalized Mutual Information (NMI)**
- **Adjusted Rand Index (ARI)**
- **Cluster Purity**

---

## 📈 Visualizations

The project includes:
- 2D latent space visualization (PCA)
- Cluster distribution across genres
- Reconstruction examples (MFCC)

---

## ⚖️ Baseline Comparisons

We compare the proposed method with:

- Raw audio features + K-Means
- PCA + K-Means
- Autoencoder + K-Means
- VAE-based clustering (proposed method)

---

## 🛠️ Technologies Used

- Python
- PyTorch
- Scikit-learn
- Pandas / NumPy
- Matplotlib / Seaborn

---

## 📁 Project Structure
