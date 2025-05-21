import tensorflow as tf
from tensorflow.keras import layers, models

def build_model(input_shape=(50, 64, 64, 1), num_classes=20):
    inputs = tf.keras.Input(shape=input_shape)

    # 3D CNN Feature Extractor
    x = layers.Conv3D(32, kernel_size=(3, 3, 3), activation='relu', padding='same')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling3D(pool_size=(1, 2, 2))(x)

    x = layers.Conv3D(64, kernel_size=(3, 3, 3), activation='relu', padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling3D(pool_size=(2, 2, 2))(x)

    x = layers.Conv3D(128, kernel_size=(3, 3, 3), activation='relu', padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling3D(pool_size=(2, 2, 2))(x)  # reduces temporal dim

    # Shape before reshape: (batch, time, h, w, c)
    # We collapse spatial dims and keep temporal as sequence
    x = layers.Reshape((x.shape[1], -1))(x)  # (time, features)

    # BiLSTM Layers
    x = layers.Bidirectional(layers.LSTM(128, return_sequences=True))(x)
    x = layers.Bidirectional(layers.LSTM(64))(x)

    # Classifier
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)

    model = models.Model(inputs, outputs)
    return model
