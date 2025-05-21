import os
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split

from dataset import tf_load_video
from model import build_model

# Load annotations
df = pd.read_csv("data/annotation.csv")

# Encode labels
word_to_idx = {word: i for i, word in enumerate(sorted(df['word_text'].unique()))}
idx_to_word = {i: word for word, i in word_to_idx.items()}
df["label"] = df["word_text"].map(word_to_idx)

# Train-test split
train_df, val_df = train_test_split(df, test_size=0.2, stratify=df["label"], random_state=42)

# Data generator
def build_dataset(df, batch_size=8, shuffle=True):
    paths = df["filename"].apply(lambda x: os.path.join("data","videos", x)).tolist()
    labels = df["label"].tolist()

    path_ds = tf.data.Dataset.from_tensor_slices(paths)
    video_ds = path_ds.map(tf_load_video, num_parallel_calls=tf.data.AUTOTUNE)

    label_ds = tf.data.Dataset.from_tensor_slices(tf.keras.utils.to_categorical(labels, num_classes=len(word_to_idx)))

    ds = tf.data.Dataset.zip((video_ds, label_ds))
    if shuffle:
        ds = ds.shuffle(buffer_size=len(df))
    return ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)

train_ds = build_dataset(train_df)
val_ds = build_dataset(val_df, shuffle=False)

# Build model
model = build_model(num_classes=len(word_to_idx))
model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

# Callbacks
checkpoint_cb = tf.keras.callbacks.ModelCheckpoint("models/best_model.weights.h5", save_best_only=True)
early_stopping_cb = tf.keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True)

# Train
model.fit(train_ds, validation_data=val_ds, epochs=50, callbacks=[checkpoint_cb, early_stopping_cb])

model.save('models/lipreading_model.h5')