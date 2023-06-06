import tensorflow as tf
from keras import layers
from veggideas.load_data import load_train_data
from veggideas.load_data import load_val_data
import numpy as np
from keras import models
from keras import Sequential, layers
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense


train_data = load_train_data()
val_data = load_val_data()

resize_and_rescale = tf.keras.Sequential([
    layers.Rescaling(1./255)])

data_augmentation = tf.keras.Sequential([
    layers.RandomFlip("horizontal_and_vertical"),
    layers.RandomRotation(0.2),
    ])

model = tf.keras.Sequential([
  # Add the preprocessing layers you created earlier.
  resize_and_rescale,
  data_augmentation,
  layers.Conv2D(16, 3, padding='same', activation='relu'),
  layers.MaxPooling2D(),
  # Rest of your model.
])


#### 1. ARCHITECTURE
def create_model():
    model = tf.keras.Sequential()
    model.add(resize_and_rescale)
    model.add(data_augmentation)
    model.add(layers.Conv2D(, 3, activation='relu', padding='same', input_shape=(224, 224, 3)))
    model.add(layers.Dense(15, activation='softmax'))

    return model




#### 2. COMPILATION
model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy', 'precision'])
