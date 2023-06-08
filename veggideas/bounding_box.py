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
model = load_non_trainable_model()

# Define the path to save the TFRecord file
tfrecord_file = 'augmented_dataset.tfrecord'

# Create an instance of ImageDataGenerator for data augmentation
datagen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
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

# Create a TFRecord writer to save the augmented dataset
writer = tf.data.experimental.TFRecordWriter(tfrecord_file)

# Iterate over each example in the augmented dataset and write it to the TFRecord file
for image, label in augmented_train_data:
    # Serialize the image and label tensors
    image_bytes = tf.io.serialize_tensor(image)
    label_bytes = tf.io.serialize_tensor(label)

    # Create a feature dictionary
    feature_dict = {
        'image': tf.train.Feature(bytes_list=tf.train.BytesList(value=[image_bytes.numpy()])),
        'label': tf.train.Feature(bytes_list=tf.train.BytesList(value=[label_bytes.numpy()]))
    }

    # Create an Example proto and serialize it to a string
    example = tf.train.Example(features=tf.train.Features(feature=feature_dict))
    example_string = example.SerializeToString()

    # Write the serialized Example to the TFRecord file
    writer.write(example_string)

# Close the TFRecord writer
writer.close()
