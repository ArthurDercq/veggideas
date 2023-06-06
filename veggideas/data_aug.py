import tensorflow as tf
from keras import layers

def preprocessing_image():
    resize_and_rescale = tf.keras.Sequential([
        layers.Rescaling(1./255)
    ])

    data_augmentation = tf.keras.Sequential([
        layers.RandomFlip("horizontal_and_vertical"),
        layers.RandomRotation(0.2),
        ])
    return resize_and_rescale, data_augmentation
