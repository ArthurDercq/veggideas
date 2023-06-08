import tensorflow as tf
from veggideas.params import *



data_dir = LOCAL_DATA_PATH
batch_size = 32
image_size = (224, 224)


def load_train_data():

    train_data = tf.keras.utils.image_dataset_from_directory(
    data_dir+'/train',
    batch_size=batch_size,
    image_size=image_size,
    labels='inferred',
    label_mode= "categorical",
    shuffle=True,
    seed=42)

    print("Loading training data")
    print("Training data successfully loaded ✅")
    return train_data



def load_val_data():
    val_data = tf.keras.utils.image_dataset_from_directory(
    data_dir+'/validation',
    batch_size=batch_size,
    image_size=image_size,
    labels='inferred',
    label_mode= "categorical",
    shuffle=True,
    seed=42)

    print("Loading validation data")
    print("Validation data successfully loaded ✅")
    return val_data


def load_test_data():
    test_data = tf.keras.utils.image_dataset_from_directory(
    data_dir+'/test',
    batch_size=batch_size,
    image_size=image_size,
    labels='inferred',
    label_mode= "categorical",
    shuffle=True,
    seed=42)

    print("Loading test data")
    print("Testing data successfully loaded ✅")
    return test_data


if __name__ == '__main__':

    train_data = load_train_data()
    val_data = load_val_data()
    test_data = load_test_data()
