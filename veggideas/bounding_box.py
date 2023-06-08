import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf
from veggideas.transfer import load_non_trainable_model
from keras import regularizers, layers
from keras import models
from keras import optimizers
from veggideas.load_data import load_train_data, load_val_data



train_data = load_train_data()
val_data = load_val_data()
base_model = load_non_trainable_model()


def data_augmentation():
    datagen = ImageDataGenerator(
        rotation_range=20,  # Rotate images by 20 degrees
        width_shift_range=0.2,  # Shift images horizontally by 20% of the total width
        height_shift_range=0.2,  # Shift images vertically by 20% of the total height
        shear_range=0.2,  # Apply shear transformation with a shear intensity of 20%
        zoom_range=0.2,  # Apply zoom transformation with a zoom range of 20%
        horizontal_flip=True,  # Flip images horizontally
        fill_mode='nearest'  # Fill newly created pixels during transformations with the nearest value
    )

    # Create an empty list to store augmented images and labels
    augmented_images = []
    augmented_labels = []

    # Iterate over each batch in the training dataset
    for images, labels in train_data:
        # Apply data augmentation to the batch
        augmented_batch = datagen.flow(images, labels, batch_size=images.shape[0])
        augmented_images.extend(augmented_batch[0][0])
        augmented_labels.extend(augmented_batch[0][1])

    # Convert the augmented images and labels to TensorFlow tensors
    augmented_images = tf.convert_to_tensor(augmented_images)
    augmented_labels = tf.convert_to_tensor(augmented_labels)

    # Create a new dataset using the augmented images and labels
    augmented_train_data = tf.data.Dataset.from_tensor_slices((augmented_images, augmented_labels))

    return augmented_train_data



""""
BOUNDING BOX

"""


def building_the_model():
    def add_last_layers(base_model):
        '''Take a pre-trained model, set its parameters as non-trainable, and add additional trainable layers on top'''
        resize_and_rescale = tf.keras.Sequential([
            layers.Rescaling(1./255)])

        base_model = base_model
        flattening_layer = layers.Flatten()
        dense_layer = layers.Dense(128, activation=tf.keras.layers.LeakyReLU(alpha=0.1))
        dense_layer_2 = layers.Dense(64, activation='relu')
        reg_l2 = regularizers.L2(0.001)
        dense_layer_reg = layers.Dense(128, activation='relu', bias_regularizer=reg_l2)
        conv2D_256 = layers.Conv2D(512, 3, padding='same', activation='relu')
        conv2D_512 = layers.Conv2D(512, 3, padding='same', activation='relu')
        maxpool_layer = layers.MaxPool2D(pool_size=(2,2))
        prediction_layer = layers.Dense(15, activation='softmax')

        model = models.Sequential([
            resize_and_rescale,
            base_model,
            conv2D_256,
            maxpool_layer,
            conv2D_512,
            maxpool_layer,
            flattening_layer,
            dense_layer,
            dense_layer_reg,
            dense_layer_2,
            prediction_layer
            ])


        opt = optimizers.Adam(learning_rate=1e-5)
        model.compile(loss='categorical_crossentropy',
                    optimizer=opt,
                    metrics=['accuracy'])
        return model

    augmented_train_data = data_augmentation()
    model_transfer = add_last_layers(base_model)
    model_transfer.fit(augmented_train_data, batch_size=32, epochs=10, validation_data=val_data)

    return model_transfer


def bounding_box():
