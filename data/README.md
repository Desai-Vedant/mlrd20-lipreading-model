# 📁 Data Folder – MLRD-20 Dataset

This folder is expected to contain the **MLRD-20 Marathi Lip Reading Dataset**, used for training and evaluation of our lipreading model.

## 📦 Structure

```
data/
├── videos/                # Contains all cropped mouth region videos (.mp4)
│   ├── s1_w1_base.mp4
│   ├── s1_w2_base.mp4
│   ├── ...
│   └── s38_w20_darken.mp4
├── annotation.csv         # CSV file containing metadata and labels
└── README.md              # This file
```

## 📋 annotation.csv Format

The `annotation.csv` file provides labels and metadata for each video. It includes the following columns:

| Column Name  | Description                                  |
| ------------ | -------------------------------------------- |
| filename     | Video filename (e.g., s1\_w1\_base.mp4)      |
| speaker\_id  | Unique speaker ID (e.g., s1, s2, ...)        |
| word\_id     | Word ID (e.g., w1 to w20)                    |
| word\_text   | Actual Marathi word (e.g., शिवाजी)           |
| augmentation | Augmentation type (base, lighten, darken)    |
| frames       | Total number of frames in video (usually 50) |
| fps          | Frame rate (usually 25 fps)                  |
| duration     | Duration in seconds (2.0)                    |

## ⚠️ Note

* All videos are cropped to 64x64 pixels, grayscale, and 2 seconds long at 25fps.

## 🔗 Source

The full dataset is available on [Kaggle](https://www.kaggle.com/datasets/desaivedantanil/mlrd-20) as part of the MLRD-20 project.

---

For any questions, refer to the main project repository or contact the authors.
