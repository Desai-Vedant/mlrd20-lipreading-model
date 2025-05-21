import os
import cv2
import numpy as np
import tensorflow as tf

def load_video(video_path):
    path = video_path.numpy().decode("utf-8")
    cap = cv2.VideoCapture(path)
    frames = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.resize(frame, (64, 64))
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        frame = np.expand_dims(frame, axis=-1)
        frames.append(frame)

    cap.release()

    if len(frames) == 0:
        return np.zeros((50, 64, 64, 1), dtype=np.float32)

    frames = np.stack(frames).astype(np.float32)

    if frames.shape[0] > 50:
        frames = frames[:50]
    elif frames.shape[0] < 50:
        padding = np.zeros((50 - frames.shape[0], 64, 64, 1), dtype=np.float32)
        frames = np.concatenate([frames, padding], axis=0)

    mean = np.mean(frames)
    std = np.std(frames)
    frames = (frames - mean) / (std + 1e-6)

    return frames.astype(np.float32)

def tf_load_video(path):
    video = tf.py_function(func=load_video, inp=[path], Tout=tf.float32)
    video.set_shape((50, 64, 64, 1))
    return video
