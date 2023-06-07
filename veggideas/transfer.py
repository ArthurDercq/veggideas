from veggideas.load_data import load_train_data, load_val_data
from keras.applications.vgg16 import VGG16
from keras import layers, models, regularizers
from keras import optimizers
import tensorflow as tf




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
    model.fit(train_data, batch_size=32, epochs=1, validation_data=val_data)
    return model


#load data
train_data = load_train_data()
val_data = load_val_data()

print("Starting to train the model")

#train the final model
history = get_trained()

print("model trained")
