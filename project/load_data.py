import tensorflow as tf

data_dir ="/Users/arthurdercq/code/ArthurDercq/veggideas/raw_data"

def load_train_data():

    batch_size = 32
    image_size = (224, 224)

    train_data = tf.keras.utils.image_dataset_from_directory(
    data_dir+'/train_data',
    batch_size=batch_size,
    image_size=image_size,
    labels='inferred',
    label_mode= "categorical",
    shuffle=True,
    seed=42)

    print("Loading data")
    return train_data

load_train_data()
print("Training data successfully loaded ✅")
