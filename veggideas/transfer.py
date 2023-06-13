from veggideas.load_data import load_train_data, load_val_data, load_test_data
from keras.applications import VGG16, InceptionV3
from veggideas.registry import save_model
from keras import layers, models, regularizers
from keras import optimizers, callbacks
import tensorflow as tf
import pandas as pd
import numpy as np


def load_model_VGG16():

    model = VGG16(weights="imagenet", include_top=False, input_shape=(224,224,3))
    model.trainable = False
    return model

def load_model_Inception():

    model = InceptionV3(weights="imagenet", include_top=False, input_shape=(224,224, 3))
    model.trainable = False

    print("✅ Inception loaded")

    return model

def initialize_model():
    '''Take a pre-trained model, set its parameters as non-trainable, and add additional trainable layers on top'''

    print("✅Initializing the model")

    i = tf.keras.layers.Input([None, None, 3], dtype = tf.uint8)
    x = tf.cast(i, tf.float32)
    x = tf.keras.applications.inception_v3.preprocess_input(x, data_format=None)
    core = load_model_Inception()

    x = tf.keras.layers.RandomFlip("horizontal_and_vertical")(x)
    x = tf.keras.layers.RandomRotation(0.2)(x)

    x = core(x)

    x = layers.Flatten()(x)
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dense(64, activation='relu')(x)
    reg_l2 = regularizers.L2(0.01)
    x = layers.Dense(128, activation='relu', bias_regularizer=reg_l2)(x)
    x = layers.Dense(12, activation='softmax')(x)

    model = tf.keras.Model(inputs=[i], outputs=[x])

    print("✅ Model initiated")

    opt = optimizers.Adam(learning_rate=1e-3)
    model.compile(loss='categorical_crossentropy',
                  optimizer=opt,
                  metrics=['accuracy'])

    print("✅ Model compiled")

    return model


def get_trained():
    model = initialize_model()

    es = callbacks.EarlyStopping(patience=1, restore_best_weights=True)

    model.fit(train_data, batch_size=32, epochs=10, validation_data=val_data, callbacks=[es])

    print("Model trained ✅")

    return model


def evaluate_model(model):
    test_data = load_test_data()
    print("Evaluating model...")

    if model is None:
        print(f"\n❌ No model to evaluate")
        return None

    metrics = model.evaluate(test_data,
        batch_size=32,
        verbose=0,
        return_dict=True)

    loss = metrics["loss"]
    accuracy = metrics["accuracy"]

    print(f"✅ Model evaluated, ACCURACY: {round(accuracy, 2)}, LOSS: {loss}")


if __name__ == '__main__':

    train_data = load_train_data()
    val_data = load_val_data()


    history = get_trained()

    save_model(history)

    evaluate_model(history)
