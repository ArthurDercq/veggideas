import tensorflow as tf
from veggideas.load_data import load_train_data
from veggideas.load_data import load_val_data

train_data = load_train_data()
val_data = load_val_data()

def split_data(data):
    X = []
    y = []


    for images, labels in data:
        X.append(images)
        y.append(labels)
    X = tf.concat(X, axis=0)
    y = tf.concat(y, axis=0)

    return X, y

X_train, y_train = split_data(train_data)
