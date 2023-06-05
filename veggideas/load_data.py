import tensorflow as tf

data_dir ="/Users/arthurdercq/code/ArthurDercq/veggideas/raw_data"
batch_size = 32
image_size = (224, 224)


def load_train_data():

    train_data = tf.keras.utils.image_dataset_from_directory(
    data_dir+'/train_data',
    batch_size=batch_size,
    image_size=image_size,
    labels='inferred',
    label_mode= "categorical",
    shuffle=True,
    seed=42)

    print("Loading training data")
    return train_data

load_train_data()
print("Training data successfully loaded ✅")


def load_val_data():
    val_data = tf.keras.utils.image_dataset_from_directory(
    data_dir+'/val_data',
    batch_size=batch_size,
    image_size=image_size,
    labels='inferred',
    label_mode= "categorical",
    shuffle=True,
    seed=42)

    print("Loading validation data")
    return val_data

def load_test_data():
    test_data = tf.keras.utils.image_dataset_from_directory(
    data_dir+'/test_data',
    batch_size=batch_size,
    image_size=image_size,
    labels='inferred',
    label_mode= "categorical",
    shuffle=True,
    seed=42)

    print("Loading test data")
    return test_data
