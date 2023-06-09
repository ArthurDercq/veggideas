from veggideas.load_data import load_train_data, load_val_data, load_test_data
from keras.applications.vgg16 import VGG16
from veggideas.registry import save_model
from keras import layers, models, regularizers
from keras import optimizers, callbacks
import tensorflow as tf
import pandas as pd
import numpy as np


def load_non_trainable_model():

    model = VGG16(weights="imagenet", include_top=False, input_shape=(224,224,3))
    model.trainable = False
    return model


def add_last_layers():
    '''Take a pre-trained model, set its parameters as non-trainable, and add additional trainable layers on top'''
    resize_and_rescale = tf.keras.Sequential([
        layers.Rescaling(1./255)])

    data_augmentation = tf.keras.Sequential([
    layers.RandomFlip("horizontal_and_vertical"),
    layers.RandomRotation(0.2),
    ])

    base_model = load_non_trainable_model()
    flattening_layer = layers.Flatten()
    dense_layer = layers.Dense(128, activation='relu')
    dense_layer_2 = layers.Dense(64, activation='relu')
    reg_l2 = regularizers.L2(0.01)
    dense_layer_reg = layers.Dense(128, activation='relu', bias_regularizer=reg_l2)
    maxpool_layer = layers.MaxPool2D(pool_size=(2,2))
    prediction_layer = layers.Dense(15, activation='softmax')

    model = models.Sequential([
        resize_and_rescale,
        data_augmentation,
        base_model,
        flattening_layer,
        dense_layer,
        dense_layer_reg,
        dense_layer_2,
        prediction_layer
        ])


    opt = optimizers.Adam(learning_rate=1e-3)
    model.compile(loss='categorical_crossentropy',
                  optimizer=opt,
                  metrics=['accuracy'])
    return model


def get_trained():
    model = add_last_layers()

    es = callbacks.EarlyStopping(patience=2, restore_best_weights=True)

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

    history_df = pd.DataFrame(history.history)
    print(history_df)
