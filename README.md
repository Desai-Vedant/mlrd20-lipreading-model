# MLRD-20 Lip Reading Model

This repository contains the training and evaluation code for a lip reading model built on the **MLRD-20** Marathi Lip Reading Dataset. The model uses a hybrid **3D CNN + BLSTM** architecture to classify isolated Marathi words from short video clips of mouth movements.

## 📦 Dataset

The dataset used for training is publicly available on Kaggle:

👉 [MLRD-20 Marathi Lip Reading Dataset on Kaggle](https://www.kaggle.com/datasets/desaivedantanil/mlrd-20)

The dataset includes:

* 38 speakers
* 20 unique Marathi words
* 2,280 videos (with augmentations)
* Cropped mouth region (64x64px, 2 seconds @ 25fps)

> Please download and extract the dataset, then update the path in `dataset.py` or notebook accordingly.

---

## 🧠 Model Architecture

* **Feature Extractor:** 3D Convolutional Neural Network (3D-CNN)
* **Temporal Modeling:** Bidirectional LSTM (BLSTM)
* **Classification:** Fully connected layers

---

## 🚀 Training

To train the model, run:

```bash
python src/train.py
```

You can also use the provided Jupyter notebook under `notebooks/` for experimentation.

---

## 📁 Directory Structure

```
mlrd20-lipreading-model/
├── data/                  # Dataset path or symbolic link
├── models/                # Saved checkpoints
├── notebooks/             # Training exploration notebook
├── src/                   # Source code
│   ├── train.py           # Training pipeline
│   ├── model.py           # Model architecture
│   ├── dataset.py         # PyTorch Dataset loader
├── LICENSE                # MIT License
└── README.md              # This file
```

---

## 📊 Results

Final model accuracy: *95.28%*

---

## 👥 Authors

* **Vedant Desai**
* **Muaaj Nesarikar**
* **Aditya Patil**
* **Sushant More**

---

## 📄 License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
